#!/usr/bin/env python3
"""Paper 3: repetitions, which every review of this work has asked for.

TSUSC's three reviews of the earlier version (TSUSC-2025-10-0304) converged on
the same gap. R1: "statistical analysis across multiple trials, including
variability measures (e.g., standard deviation, confidence intervals)". R3: "All
results come from a single 10-minute run on one device... No multi-run
statistics". The AE: "single-device single-run results without baselines or
statistics".

The released data has exactly one run per policy, so the gap is real. This runs
the same four policies R times and writes every repetition, so the paper can
report mean and standard deviation instead of a point estimate.

Two things the wrapper has to get right that a naive loop would not:

  1. `make_controller` closes over mutable state (`idx=[2]`, the current exit
     level). Reusing one controller across repetitions would carry the ending
     state of run N into the start of run N+1, so each repetition gets a FRESH
     controller. The static policies are stateless and can be reused.
  2. The harness cools to below 50.5 C before each policy. That is kept, because
     starting temperature is exactly the variable the earlier campaigns showed
     selects between regimes.

Run it on an idle board, one heavy job at a time.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper3_controller_exp as P3  # noqa: E402  (module-level load of CIFAR + sessions)

REPS = int(os.environ.get("P3_REPS", "5"))
CAP = 49.0
RATE = 1400
DUR = 90
OUT = Path.home() / "paper3_controller_reps.json"


def policies():
    """Fresh policy list per repetition; the controller must not carry state."""
    return [
        ("always-exit1", lambda T: "exit1"),
        ("always-exit2", lambda T: "exit2"),
        ("always-final", lambda T: "final"),
        ("adaptive-ctrl", P3.make_controller(CAP)),
    ]


def main():
    print("Paper 3 repetitions: %d reps x 4 policies, rate=%d/s, cap=%.1f C, %ds each"
          % (REPS, RATE, CAP, DUR), flush=True)
    runs = []
    for rep in range(1, REPS + 1):
        for name, fn in policies():
            while P3.temp() > 50.5:
                time.sleep(2)
            t_start = P3.temp()
            r = P3.run_policy(name, fn, RATE, CAP, DUR)
            r["rep"] = rep
            r["start_temp"] = t_start
            runs.append(r)
            print("[rep %d] %-14s n=%7d acc=%6.2f%% meanT=%5.1f maxT=%5.1f "
                  "viol=%3ds p50=%6.2f p99=%7.2f startT=%.1f"
                  % (rep, r["policy"], r["n"], r["real_acc"], r["mean_temp"],
                     r["max_temp"], r["cap_viol_s"], r["p50"], r["p99"], t_start),
                  flush=True)
            json.dump(runs, open(OUT, "w"), indent=1)
            time.sleep(15)
    json.dump(runs, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
