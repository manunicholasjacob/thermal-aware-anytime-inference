#!/usr/bin/env python3
"""Paper 3, second campaign: remove the run-order leak from the first one.

The first campaign (paper3_entropy_sweep.py, 2 Oct) found that the harness gate,
`while temp() > 50.5`, never fired: none of 45 runs started above 49.0 C. Starting
temperature was therefore set by whatever ran before each arm, spread 44.6 to
49.0 C by sequence position, and the static policies' cap-violation counts carried
standard deviations as large as +-28 s partly as an artifact of that. It also made
the lowest-compute arm the hottest, which is physically absurd.

Three changes, and nothing else, so the two campaigns stay comparable:

  1. ARM ORDER IS RANDOMISED per repetition, from a recorded seed. This is the
     change that actually removes the bias, and unlike a temperature gate it works
     at any ambient.
  2. THE GATE BINDS, or says so. It waits for the board to reach GATE_C, gives up
     after GATE_MAX_S, and records both the wait and whether it bound. A gate that
     silently never fires is worse than no gate, because it reads as control.
  3. THE CAMPAIGN WAITS FOR AMBIENT before it starts anything. The board idles at
     about 46.6 C at night and 48.0 C in the afternoon, against a 49.0 C cap, so
     starting in the afternoon would compress the whole experiment into the top
     degree of its range and would not be comparable with the first campaign. The
     preamble blocks until the board is at or below AMBIENT_C, for up to
     AMBIENT_MAX_H hours, then runs. Launch it whenever; it starts when the room
     is cool enough.

Everything else is held identical to the first campaign: nine arms, five
repetitions, 90 s per run, 1400 req/s, 49.0 C cap, four workers, 15 s between arms,
a fresh controller per repetition.

Run on an idle board, one heavy job at a time, under nohup.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper3_controller_exp as P3  # noqa: E402
import paper3_entropy_sweep as S1  # noqa: E402

REPS = int(os.environ.get("P3_REPS", "5"))
# Not an arbitrary seed. With only five repetitions a plain shuffle leaves real
# residual imbalance: the obvious seed put always-exit1 at mean position 7.0 and
# adaptive-ctrl at 3.4, which is the very bias this campaign exists to remove.
# This seed was selected by search so that every arm's mean position is within
# 0.40 of the ideal 5.0 and every arm occupies at least four distinct positions.
SEED = int(os.environ.get("P3_SEED", "20319103"))
CAP, RATE, DUR = 49.0, 1400, 90
TAUS = S1.TAUS

GATE_C = float(os.environ.get("P3_GATE", "46.0"))      # binding, unlike 50.5
GATE_MAX_S = float(os.environ.get("P3_GATE_MAX", "240"))
AMBIENT_C = float(os.environ.get("P3_AMBIENT", "46.5"))
AMBIENT_MAX_H = float(os.environ.get("P3_AMBIENT_MAX_H", "10"))

OUT = Path.home() / "paper3_entropy_sweep2.json"


def wait_ambient():
    """Block until the room is cool enough that the 49 C cap still has room."""
    t0 = time.time()
    while True:
        t = P3.temp()
        if t <= AMBIENT_C:
            print("ambient gate: board at %.1f C <= %.1f, starting after %.1f min"
                  % (t, AMBIENT_C, (time.time() - t0) / 60.0), flush=True)
            return
        if time.time() - t0 > AMBIENT_MAX_H * 3600:
            print("ambient gate: GAVE UP after %.1f h, board still %.1f C. Running "
                  "anyway; results are NOT comparable with campaign 1 and the "
                  "start temps below say so." % (AMBIENT_MAX_H, t), flush=True)
            return
        if int(time.time() - t0) % 600 < 20:
            print("  ambient gate: %.1f C, waiting for <= %.1f (%.0f min so far)"
                  % (t, AMBIENT_C, (time.time() - t0) / 60.0), flush=True)
        time.sleep(20)


def gate():
    """Cool to GATE_C. Returns (start_temp, seconds_waited, did_it_bind)."""
    t0 = time.time()
    while P3.temp() > GATE_C and time.time() - t0 < GATE_MAX_S:
        time.sleep(2)
    waited = time.time() - t0
    t = P3.temp()
    return t, waited, bool(t <= GATE_C)


def main():
    rng = random.Random(SEED)
    print("Paper 3 campaign 2: %d reps, 9 arms, RANDOMISED order, seed %d"
          % (REPS, SEED), flush=True)
    print("gate %.1f C (max %.0f s), ambient gate %.1f C (max %.1f h), cap %.1f C"
          % (GATE_C, GATE_MAX_S, AMBIENT_C, AMBIENT_MAX_H, CAP), flush=True)
    print("board now %.1f C" % P3.temp(), flush=True)

    wait_ambient()
    runs = []
    t_begin = time.time()

    for rep in range(1, REPS + 1):
        names = ["always-exit1", "always-exit2", "always-final", "adaptive-ctrl"] + \
                ["entropy-%.2f" % t for t in TAUS]
        rng.shuffle(names)
        print("[rep %d] order: %s" % (rep, ", ".join(names)), flush=True)

        # fresh controller each repetition: it carries state between runs
        static = {"always-exit1": lambda T: "exit1",
                  "always-exit2": lambda T: "exit2",
                  "always-final": lambda T: "final",
                  "adaptive-ctrl": P3.make_controller(CAP)}

        for pos, name in enumerate(names, 1):
            t_start, waited, bound = gate()
            if name.startswith("entropy"):
                r = S1.run_entropy_policy(float(name.split("-")[1]), RATE, CAP, DUR)
                r["kind"] = "entropy"
            else:
                r = P3.run_policy(name, static[name], RATE, CAP, DUR)
                r["kind"] = "paper"
            r.update(rep=rep, position=pos, start_temp=t_start,
                     gate_wait_s=round(waited, 1), gate_bound=bound, seed=SEED)
            runs.append(r)
            print("[rep %d/%d] %-14s n=%7d acc=%6.2f%% meanT=%5.1f maxT=%5.1f "
                  "viol=%3ds startT=%.1f gate=%.0fs%s"
                  % (rep, pos, r["policy"], r["n"], r["real_acc"], r["mean_temp"],
                     r["max_temp"], r["cap_viol_s"], t_start, waited,
                     "" if bound else " GATE-DID-NOT-BIND"), flush=True)
            json.dump(runs, open(OUT, "w"), indent=1)
            time.sleep(15)

        print("--- rep %d done, %.1f min elapsed ---"
              % (rep, (time.time() - t_begin) / 60.0), flush=True)

    json.dump(runs, open(OUT, "w"), indent=1)
    nb = sum(1 for r in runs if not r["gate_bound"])
    print("wrote %s after %.1f min. Gate failed to bind on %d of %d runs."
          % (OUT, (time.time() - t_begin) / 60.0, nb, len(runs)), flush=True)


if __name__ == "__main__":
    main()
