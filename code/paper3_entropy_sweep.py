#!/usr/bin/env python3
"""Paper 3: the state-of-the-art baseline the paper is missing, measured.

The paper compares its thermal controller against always-exit1, always-exit2 and
always-final. Those are its own static bounds, not published work. The baseline
the early-exit literature actually uses, from BranchyNet onward, is a confidence
threshold on the side-branch output. TETC asks for a comparison against
state-of-the-art work, and that is the comparison.

Design notes that matter for believing the result.

ALL ARMS RUN IN ONE SESSION. The paper's four policies are re-measured here
alongside the new ones rather than compared against the September run. An earlier
campaign on this hardware found an unexplained between-session offset of about
7%, and the rule that came out of it was to interleave arms within a session.
Comparing a new arm measured today against baselines measured on 16 September
would be exactly that mistake.

THE CASCADE IS CHARGED MORE THAN A REAL IMPLEMENTATION WOULD BE. The three exits
are separate ONNX graphs, so escalating from exit1 to exit2 recomputes the shared
backbone instead of resuming from it. A real branching network taps one backbone
and pays only the deepest exit it reaches. This harness therefore handicaps the
baseline it is trying to give a fair hearing. Both numbers are recorded: the
measured wall-clock cost, and an ideal shared-backbone accounting that charges
each request only the service time of the exit it actually took. The measured
temperature is an upper bound on what a real implementation would produce, so if
the entropy arm still looks good it looks good in spite of the handicap.

Protocol is copied from paper3_controller_exp.run_policy and must stay identical:
rate 1400/s, cap 49.0 C, 90 s, 4 workers, cool below 50.5 C before every arm,
15 s between arms, fresh controller per repetition.

Run on an idle board, one heavy job at a time, under nohup.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper3_controller_exp as P3  # noqa: E402

REPS = int(os.environ.get("P3_REPS", "5"))
CAP = 49.0
RATE = 1400
DUR = 90
NWORKERS = 4

# Chosen from paper3_entropy_calib.py, which computed accuracy and exit mix
# exactly over the whole test set. These five span the compute range from
# "82% still reach the final exit" to "94% stop at the first".
#   tau   acc%    exit1/exit2/final %
#   1.50  34.17    3.7 / 14.1 / 82.2   <- accuracy peak, above always-final
#   2.50  31.47   19.1 / 29.2 / 51.7
#   3.00  29.06   33.2 / 39.5 / 27.3
#   3.50  25.05   59.2 / 36.0 /  4.7
#   4.00  20.30   94.5 /  5.5 /  0.0
TAUS = [1.5, 2.5, 3.0, 3.5, 4.0]

OUT = Path.home() / "paper3_entropy_sweep.json"
EXITS = ["exit1", "exit2", "final"]


def entropy_nats(z):
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def run_entropy_policy(tau, rate_hz, cap, dur=DUR, nworkers=NWORKERS):
    """Per-sample cascaded entropy exit. Mirrors P3.run_policy exactly otherwise."""
    import queue
    import threading

    stop = threading.Event()
    q = queue.Queue(maxsize=nworkers * 4)
    lat, exit_used = [], []
    correct, total, ideal_ms = [0], [0], [0.0]
    lock = threading.Lock()

    def worker():
        while not stop.is_set():
            try:
                idx = q.get(timeout=0.5)
            except queue.Empty:
                continue
            x = P3.img(idx)
            t0 = time.perf_counter()
            out = P3.SESS["exit1"].run(None, {P3.INAME: x})[0]
            ex = "exit1"
            if entropy_nats(out[0]) > tau:
                out = P3.SESS["exit2"].run(None, {P3.INAME: x})[0]
                ex = "exit2"
                if entropy_nats(out[0]) > tau:
                    out = P3.SESS["final"].run(None, {P3.INAME: x})[0]
                    ex = "final"
            dt = (time.perf_counter() - t0) * 1000
            with lock:
                lat.append(dt)
                total[0] += 1
                ideal_ms[0] += P3.SVC[ex]
                exit_used.append(ex)
                if int(out.argmax(1)[0]) == P3.Y[idx]:
                    correct[0] += 1

    ws = [threading.Thread(target=worker, daemon=True) for _ in range(nworkers)]
    for w in ws:
        w.start()

    temps, viol = [], [0]

    def control():
        while not stop.is_set():
            T = P3.temp()
            temps.append(T)
            if T > cap:
                viol[0] += 1
            time.sleep(1.0)

    threading.Thread(target=control, daemon=True).start()

    t0 = time.time()
    i = 0
    interval = 1.0 / rate_hz
    while time.time() - t0 < dur:
        try:
            q.put(i % 10000, timeout=0.01)
        except queue.Full:
            pass
        i += 1
        nxt = t0 + i * interval
        time.sleep(max(0, nxt - time.time()))
    stop.set()
    time.sleep(1)

    lat = np.array(lat) if lat else np.array([0.0])
    mix = Counter(exit_used)
    n = max(total[0], 1)
    return dict(policy="entropy-%.2f" % tau, tau=tau, n=total[0],
                real_acc=100.0 * correct[0] / n,
                mean_temp=float(np.mean(temps)), max_temp=float(np.max(temps)),
                cap_viol_s=viol[0],
                p50=float(np.percentile(lat, 50)), p99=float(np.percentile(lat, 99)),
                mean_lat_ms=float(lat.mean()),
                ideal_mean_ms=ideal_ms[0] / n,
                exit_mix={k: round(100.0 * v / len(exit_used), 1)
                          for k, v in mix.items()} if exit_used else {})


def arms():
    """Fresh per repetition: the controller carries state and must not be reused."""
    static = [("always-exit1", lambda T: "exit1"),
              ("always-exit2", lambda T: "exit2"),
              ("always-final", lambda T: "final"),
              ("adaptive-ctrl", P3.make_controller(CAP))]
    return static, list(TAUS)


def main():
    print("Paper 3 entropy baseline sweep: %d reps, 4 paper arms + %d entropy arms, "
          "rate=%d/s, cap=%.1f C, %ds each" % (REPS, len(TAUS), RATE, CAP, DUR),
          flush=True)
    print("start temp %.1f C" % P3.temp(), flush=True)
    runs = []
    t_begin = time.time()
    for rep in range(1, REPS + 1):
        static, taus = arms()
        for name, fn in static:
            while P3.temp() > 50.5:
                time.sleep(2)
            t_start = P3.temp()
            r = P3.run_policy(name, fn, RATE, CAP, DUR)
            r.update(rep=rep, start_temp=t_start, kind="paper")
            # the paper's arms run one exit per request, so ideal == measured
            r["ideal_mean_ms"] = P3.SVC[max(r["exit_mix"], key=r["exit_mix"].get)] \
                if r["exit_mix"] else None
            runs.append(r)
            print("[rep %d] %-14s n=%7d acc=%6.2f%% meanT=%5.1f maxT=%5.1f viol=%3ds "
                  "p99=%7.2f startT=%.1f mix=%s"
                  % (rep, r["policy"], r["n"], r["real_acc"], r["mean_temp"],
                     r["max_temp"], r["cap_viol_s"], r["p99"], t_start, r["exit_mix"]),
                  flush=True)
            json.dump(runs, open(OUT, "w"), indent=1)
            time.sleep(15)

        for tau in taus:
            while P3.temp() > 50.5:
                time.sleep(2)
            t_start = P3.temp()
            r = run_entropy_policy(tau, RATE, CAP, DUR)
            r.update(rep=rep, start_temp=t_start, kind="entropy")
            runs.append(r)
            print("[rep %d] %-14s n=%7d acc=%6.2f%% meanT=%5.1f maxT=%5.1f viol=%3ds "
                  "p99=%7.2f lat=%5.3f ideal=%5.3f startT=%.1f mix=%s"
                  % (rep, r["policy"], r["n"], r["real_acc"], r["mean_temp"],
                     r["max_temp"], r["cap_viol_s"], r["p99"], r["mean_lat_ms"],
                     r["ideal_mean_ms"], t_start, r["exit_mix"]), flush=True)
            json.dump(runs, open(OUT, "w"), indent=1)
            time.sleep(15)

        print("--- rep %d done, %.1f min elapsed ---"
              % (rep, (time.time() - t_begin) / 60.0), flush=True)

    json.dump(runs, open(OUT, "w"), indent=1)
    print("wrote %s after %.1f min" % (OUT, (time.time() - t_begin) / 60.0), flush=True)


if __name__ == "__main__":
    main()
