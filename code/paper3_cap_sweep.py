#!/usr/bin/env python3
"""Paper 3: how wide is the regime where the thermal-aware controller pays?

WHY. All three TSUSC reviews of the earlier version asked for a sensitivity
analysis, and the paper does not have one. It also makes a claim it never
measures: that the regime in which the controller earns its complexity is
narrow, needing a cap that actually binds. A cap sweep turns that from an
assertion into a measurement, and it is the sensitivity analysis the reviewers
asked for, so one experiment answers both.

WHAT IT SHOULD SHOW. At a slack cap nothing binds, every policy sits below it,
and the controller has nothing to do: it should converge on always-final and its
accuracy advantage over always-exit1 should be the whole gap. At a cap so tight
that even the shallowest exit violates it, the controller cannot help either. In
between there should be a window where it holds the cap and keeps most of the
accuracy. The width of that window IS the paper's thesis, and nobody has
measured it.

CAP ORDER IS DELIBERATELY NOT MONOTONIC. Walking caps from low to high would
make cap collinear with elapsed time and with board temperature, so thermal
drift would be indistinguishable from a cap effect, and it would bias in the
direction that flatters the hypothesis: later runs are hotter, so a high cap
measured last would look harder to hold than it is. Paper 17 in this portfolio
spent a full campaign discovering that a sequential sweep cannot separate the
swept variable from elapsed time, and this is the same trap. The order below
alternates around the middle, and the reference cap is measured first and last
so drift is visible rather than assumed absent.

Run on the Pi. One policy at a time, with a cooldown between, nothing else on
the board.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~"))
import paper3_controller_exp as P3       # noqa: E402  (loads models on import)

RATE = 1400.0        # same fixed arrival rate as the main experiment
DUR = 60
COOL = 20
OUT = os.path.expanduser("~/paper3_cap_sweep.json")

# Alternating about the middle, with 49 C (the paper's operating point) first
# and repeated last as a drift check.
CAPS = [49.0, 46.0, 51.0, 47.0, 50.0, 48.0, 49.0]

results = []
print("cap sweep: rate=%.0f/s dur=%ds order=%s" % (RATE, DUR, CAPS), flush=True)

for run_idx, cap in enumerate(CAPS):
    policies = [
        ("always-exit1", lambda T: "exit1"),
        ("always-final", lambda T: "final"),
        ("adaptive-ctrl", P3.make_controller(cap)),
    ]
    for name, fn in policies:
        t0 = P3.temp()
        r = P3.run_policy(name, fn, RATE, cap, DUR)
        r["cap"] = cap
        r["run_index"] = run_idx
        r["is_repeat_reference"] = (run_idx == len(CAPS) - 1)
        r["temp_before"] = t0
        results.append(r)
        print("cap=%.1f %-14s acc=%6.2f%% viol=%4ds meanT=%5.1f maxT=%5.1f mix=%s"
              % (cap, name, r["real_acc"], r["cap_viol_s"], r["mean_temp"],
                 r["max_temp"], r["exit_mix"]), flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
        time.sleep(COOL)

print("\nwrote", OUT, flush=True)

# Drift check: the same cap measured first and last, identical configuration.
first = [r for r in results if r["run_index"] == 0]
last = [r for r in results if r.get("is_repeat_reference")]
if first and last:
    print("\ndrift check at cap 49 C (first pass vs last pass):", flush=True)
    for a in first:
        b = [x for x in last if x["policy"] == a["policy"]]
        if b:
            print("   %-14s acc %6.2f -> %6.2f   viol %4d -> %4d   meanT %5.1f -> %5.1f"
                  % (a["policy"], a["real_acc"], b[0]["real_acc"],
                     a["cap_viol_s"], b[0]["cap_viol_s"],
                     a["mean_temp"], b[0]["mean_temp"]), flush=True)
