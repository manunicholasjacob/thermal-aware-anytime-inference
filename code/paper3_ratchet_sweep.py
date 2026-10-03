#!/usr/bin/env python3
"""Paper 3, campaign 3: measure the ratchet instead of inferring it.

Campaigns 1 and 2 differ only in arm ordering, and the controller's accuracy fell
from 33.49% to 26.13% between them while every other arm reproduced to the
hundredth. That establishes that the controller depends on run order, but order
is a proxy. The thing order was varying is the temperature the controller starts
from, and campaign 2 shows the raw shape:

    start 45.8 C -> 33.67%, exit mix 100% final
    start 45.2 C -> 25.57%, 54% exit1
    start 48.5 C -> 26.43%, 50% exit1
    start 48.5 C -> 24.47%, 65% exit1
    start 48.5 C -> 20.51%, 92% exit1

Five points, and two of them start within 0.6 C of each other and land 8 accuracy
points apart. So start temperature is not the whole story either, and a paper
that claims "where it ends up depends on where it started" owes a measurement
that separates the two.

THIS CAMPAIGN CONTROLS THE STARTING TEMPERATURE DIRECTLY. Each run is driven to a
target start temperature, by heating with the real workload or by waiting to cool,
then the policy runs for the usual 90 s. Four targets, five repetitions, and a
stateless control arm.

  adaptive-ctrl   the stateful policy under test
  entropy-3.00    the arm that dominated it in campaign 2. Its ACCURACY is
                  deterministic by construction, so it cannot vary with start
                  temperature; any accuracy spread it shows would be measurement
                  noise and bounds the noise floor for the controller's spread.
                  Its temperature does respond, which calibrates the thermal
                  carryover separately from the policy effect.

What the result will mean:

  - controller accuracy varies systematically with start temperature, control arm
    flat  ->  the ratchet is a function of starting temperature. Quantifiable, and
    the paper can state a slope.
  - controller accuracy varies but NOT with start temperature  ->  the dependence
    is on path, not on starting point, and the paper's current wording ("depends
    on where it started") is wrong in an interesting way.
  - both flat  ->  campaigns 1 and 2 differed for some other reason and this needs
    rethinking before anything is claimed.

Order of the eight (arm, target) cells is randomised within each repetition, with
the same balanced-seed discipline as campaign 2, so this campaign cannot acquire
the ordering artifact it exists to study.

Run on an idle board, one heavy job at a time, under nohup.
"""
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper3_controller_exp as P3  # noqa: E402
import paper3_entropy_sweep as S1  # noqa: E402

REPS = int(os.environ.get("P3_REPS", "5"))
SEED = int(os.environ.get("P3_SEED", "20319103"))
CAP, RATE, DUR = 49.0, 1400, 90
TARGETS = [float(x) for x in os.environ.get("P3_TARGETS", "45.5,47.0,48.5,50.0").split(",")]
ARMS = ["adaptive-ctrl", "entropy-3.00"]
TOL = 0.3
COOL_MAX_S = float(os.environ.get("P3_COOL_MAX", "420"))
HEAT_MAX_S = float(os.environ.get("P3_HEAT_MAX", "300"))
OUT = Path.home() / "paper3_ratchet_sweep.json"


def temp_avg(n=5, dt=0.4):
    """Median of n reads.

    The bare sensor spikes: a first attempt at heating to 50.0 C exited after two
    seconds on a single high read and then measured 46.3 C, below where it began.
    A 4 C fall in 1.5 s is not physical, so the high read was noise. Every
    decision here is made on a median, never on one sample.
    """
    xs = []
    for _ in range(n):
        xs.append(P3.temp())
        time.sleep(dt)
    xs.sort()
    return xs[len(xs) // 2]


def heat_to(target):
    """Heat until the board SETTLES at target once the load stops.

    The quantity this campaign controls is the temperature at the moment a run
    begins, which is after heating has stopped. Those are not the same number.
    Under four burn threads the die genuinely reads about 50 C, and it falls
    roughly 3 C within two seconds of the load being removed: a first version
    targeted the under-load reading and started its runs 3 C below target every
    time. So heat in bursts, stop, let it settle, measure, and repeat. Bursts
    lengthen as they go, because a longer burn raises the settled floor rather
    than just the peak.
    """
    t0 = time.time()
    burst = 8.0
    while time.time() - t0 < HEAT_MAX_S:
        stop = threading.Event()

        def burn():
            x = P3.img(0)
            while not stop.is_set():
                P3.SESS["final"].run(None, {P3.INAME: x})

        ws = [threading.Thread(target=burn, daemon=True) for _ in range(4)]
        for w in ws:
            w.start()
        time.sleep(min(burst, max(0.0, HEAT_MAX_S - (time.time() - t0))))
        stop.set()
        time.sleep(2.5)                      # let the die settle before judging
        if temp_avg(n=3) >= target:
            break
        burst += 4.0
    return time.time() - t0


def cool_to(target):
    t0 = time.time()
    while temp_avg() > target and time.time() - t0 < COOL_MAX_S:
        time.sleep(2)
    return time.time() - t0


def set_start(target):
    """Drive the board to target. Returns (achieved, seconds, hit, how)."""
    t = temp_avg()
    how, secs = "none", 0.0
    if t < target - TOL:
        how, secs = "heat", heat_to(target)
    elif t > target + TOL:
        how, secs = "cool", cool_to(target)
    t = temp_avg()
    return t, round(secs, 1), bool(abs(t - target) <= TOL + 0.5), how


def main():
    rng = random.Random(SEED)
    cells = [(a, g) for a in ARMS for g in TARGETS]
    print("Paper 3 campaign 3: the ratchet, with starting temperature CONTROLLED.",
          flush=True)
    print("targets %s C, arms %s, %d reps, cap %.1f C, %d s per run"
          % (TARGETS, ARMS, REPS, CAP, DUR), flush=True)
    print("board now %.1f C" % P3.temp(), flush=True)

    runs = []
    t_begin = time.time()
    for rep in range(1, REPS + 1):
        order = list(cells)
        rng.shuffle(order)
        print("[rep %d] order: %s" % (rep, ", ".join("%s@%.1f" % c for c in order)),
              flush=True)
        # the controller carries state between runs and must be rebuilt each time
        for pos, (arm, target) in enumerate(order, 1):
            achieved, secs, hit, how = set_start(target)
            if arm == "adaptive-ctrl":
                r = P3.run_policy(arm, P3.make_controller(CAP), RATE, CAP, DUR)
            else:
                r = S1.run_entropy_policy(float(arm.split("-")[1]), RATE, CAP, DUR)
            r.update(rep=rep, position=pos, arm=arm, target_start=target,
                     start_temp=achieved, start_secs=secs, start_hit=hit,
                     start_how=how, seed=SEED)
            runs.append(r)
            print("[rep %d/%d] %-14s target=%.1f start=%.1f(%s %.0fs%s) acc=%6.2f%% "
                  "meanT=%5.1f viol=%3ds mix=%s"
                  % (rep, pos, arm, target, achieved, how, secs,
                     "" if hit else " MISSED", r["real_acc"], r["mean_temp"],
                     r["cap_viol_s"], r["exit_mix"]), flush=True)
            json.dump(runs, open(OUT, "w"), indent=1)
            time.sleep(10)
        print("--- rep %d done, %.1f min elapsed ---"
              % (rep, (time.time() - t_begin) / 60.0), flush=True)

    json.dump(runs, open(OUT, "w"), indent=1)
    missed = sum(1 for r in runs if not r["start_hit"])
    print("wrote %s after %.1f min. Start target missed on %d of %d runs."
          % (OUT, (time.time() - t_begin) / 60.0, missed, len(runs)), flush=True)


if __name__ == "__main__":
    main()
