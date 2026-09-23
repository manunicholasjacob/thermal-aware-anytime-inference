#!/usr/bin/env python3
"""Paper 3: how sensitive is the controller to its dead band?

The controller has two parameters and the paper has now measured one. The cap
sweep found a three-degree window where the machinery pays. The second
parameter, the dead band in `make_controller(cap, band=3.0)`, has never been
varied at all: it was set to 3.0 C once and every result in the paper uses it.

The band decides how far below the cap the temperature must fall before the
controller steps back to a deeper exit. Too narrow and it should oscillate,
stepping down and up repeatedly and losing accuracy to hysteresis it does not
have. Too wide and it should be sluggish, staying shallow long after the board
has cooled and giving away accuracy for nothing. Whether either happens within a
plausible range is an open question that one sweep closes, and it is the obvious
follow-up to the cap sweep for a referee who has just read it.

BAND ORDER ALTERNATES, for the same reason the cap order did: a monotonic sweep
would make the band collinear with elapsed time and board temperature, so drift
would be indistinguishable from a band effect. The default 3.0 C is measured
first and last as a drift check.

Run on the Pi, one policy at a time, nothing else on the board.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~"))
import paper3_controller_exp as P3       # noqa: E402  (loads models on import)

RATE = 1400.0
DUR = 60
COOL = 20
CAP = 49.0                                # held at the paper's operating point
BANDS = [3.0, 1.0, 5.0, 2.0, 4.0, 3.0]    # alternating about the default
OUT = os.path.expanduser("~/paper3_band_sweep.json")

results = []
print("band sweep: cap=%.1f rate=%.0f/s dur=%ds order=%s"
      % (CAP, RATE, DUR, BANDS), flush=True)

for run_idx, band in enumerate(BANDS):
    fn = P3.make_controller(CAP, band=band)
    t0 = P3.temp()
    r = P3.run_policy("adaptive-band%.1f" % band, fn, RATE, CAP, DUR)
    r["cap"] = CAP
    r["band"] = band
    r["run_index"] = run_idx
    r["is_repeat_reference"] = (run_idx == len(BANDS) - 1)
    r["temp_before"] = t0
    results.append(r)
    print("band=%.1f acc=%6.2f%% viol=%4ds meanT=%5.1f maxT=%5.1f mix=%s"
          % (band, r["real_acc"], r["cap_viol_s"], r["mean_temp"],
             r["max_temp"], r["exit_mix"]), flush=True)
    json.dump(results, open(OUT, "w"), indent=2)
    time.sleep(COOL)

print("wrote " + OUT, flush=True)

first = [r for r in results if r["run_index"] == 0]
last = [r for r in results if r.get("is_repeat_reference")]
if first and last:
    a, b = first[0], last[0]
    print("drift check at band %.1f C: acc %.2f -> %.2f, viol %d -> %d, "
          "meanT %.1f -> %.1f"
          % (a["band"], a["real_acc"], b["real_acc"], a["cap_viol_s"],
             b["cap_viol_s"], a["mean_temp"], b["mean_temp"]), flush=True)
