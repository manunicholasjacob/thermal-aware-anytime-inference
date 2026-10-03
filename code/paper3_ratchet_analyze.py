#!/usr/bin/env python3
"""Campaign 3: is the ratchet a function of starting temperature, or of path?

Reads paper3_ratchet_sweep.json. Two arms at four controlled start temperatures,
five repetitions each.

The test is a comparison of two slopes, not a single regression. entropy-3.00 is
stateless and its accuracy is deterministic by construction, so its accuracy
slope against start temperature MUST be zero. Any non-zero value it shows is the
measurement noise floor, and the controller's slope only means something if it
clears that floor.

Three possible readings, decided here rather than argued:

  controller slope significant, control flat
      the ratchet IS a function of starting temperature. The paper can state a
      slope in accuracy points per degree.
  controller spread large but slope not significant
      the dependence is on path, not on starting point, and the paper's wording
      "where it ends up depends on where it started" is wrong in an interesting
      way: it depends on the route, not the origin.
  both flat and controller spread small
      campaigns 1 and 2 differed for some other reason. Nothing should be
      claimed until that is found.
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

SWEEP = Path(sys.argv[1] if len(sys.argv) > 1 else "paper3_ratchet_sweep.json")
runs = json.load(open(SWEEP))


def fit(xs, ys):
    """Least squares slope with its standard error and a t statistic."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icept = my - slope * mx
    resid = [y - (icept + slope * x) for x, y in zip(xs, ys)]
    if n <= 2:
        return None
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    t = slope / se if se > 0 else float("inf")
    sy = st.pstdev(ys)
    r2 = 1 - (sum(r * r for r in resid) / (n * sy * sy)) if sy > 0 else float("nan")
    return dict(slope=slope, se=se, t=t, r2=r2, n=n)


by_arm = {}
for r in runs:
    by_arm.setdefault(r["arm"], []).append(r)

print("Campaign 3: the ratchet with starting temperature controlled")
print("runs: %s" % {a: len(v) for a, v in by_arm.items()})
print()

for arm in sorted(by_arm):
    rs = by_arm[arm]
    print("=" * 78)
    print(arm)
    print("=" * 78)
    print("  %8s %8s %10s %10s %10s %8s" %
          ("target", "n", "startT", "accuracy", "meanT", "viol s"))
    for tgt in sorted({r["target_start"] for r in rs}):
        g = [r for r in rs if r["target_start"] == tgt]
        print("  %8.1f %8d %10.2f %7.2f+-%4.2f %10.2f %8.1f"
              % (tgt, len(g), st.mean(r["start_temp"] for r in g),
                 st.mean(r["real_acc"] for r in g),
                 st.pstdev([r["real_acc"] for r in g]) if len(g) > 1 else 0.0,
                 st.mean(r["mean_temp"] for r in g),
                 st.mean(r["cap_viol_s"] for r in g)))
    xs = [r["start_temp"] for r in rs]
    f_acc = fit(xs, [r["real_acc"] for r in rs])
    f_tmp = fit(xs, [r["mean_temp"] for r in rs])
    print()
    if f_acc:
        print("  accuracy vs start temp : slope %+.3f pp/C  se %.3f  t %+.2f  R2 %.3f  n %d"
              % (f_acc["slope"], f_acc["se"], f_acc["t"], f_acc["r2"], f_acc["n"]))
    if f_tmp:
        print("  mean temp vs start temp: slope %+.3f C/C   se %.3f  t %+.2f  R2 %.3f"
              % (f_tmp["slope"], f_tmp["se"], f_tmp["t"], f_tmp["r2"]))
    print("  accuracy spread over all runs: sd %.2f pp, range %.2f to %.2f"
          % (st.pstdev([r["real_acc"] for r in rs]),
             min(r["real_acc"] for r in rs), max(r["real_acc"] for r in rs)))
    print()

print("=" * 78)
print("VERDICT")
print("=" * 78)
ctrl = by_arm.get("adaptive-ctrl", [])
ctl = by_arm.get("entropy-3.00", [])
if ctrl and ctl:
    fc = fit([r["start_temp"] for r in ctrl], [r["real_acc"] for r in ctrl])
    fk = fit([r["start_temp"] for r in ctl], [r["real_acc"] for r in ctl])
    sd_c = st.pstdev([r["real_acc"] for r in ctrl])
    sd_k = st.pstdev([r["real_acc"] for r in ctl])
    print("controller accuracy sd %.2f pp   stateless control sd %.2f pp (noise floor)"
          % (sd_c, sd_k))
    if fc and fk:
        print("controller slope %+.3f pp/C (t %+.2f)   control slope %+.3f pp/C (t %+.2f)"
              % (fc["slope"], fc["t"], fk["slope"], fk["t"]))
        sig = abs(fc["t"]) >= 2.1          # ~5% two-sided at these dof
        big = sd_c > 3 * max(sd_k, 0.05)
        print()
        if sig:
            print("READING: the ratchet IS a function of starting temperature.")
            print("  Every degree hotter at the start costs %.2f accuracy points."
                  % (-fc["slope"]))
            print("  The paper can state that as a measured slope.")
        elif big:
            print("READING: the controller varies far beyond the noise floor but NOT")
            print("  systematically with starting temperature. The dependence is on")
            print("  PATH, not on origin. The paper's wording, that where it ends up")
            print("  depends on where it started, is wrong in an interesting way and")
            print("  should say route rather than starting point.")
        else:
            print("READING: neither effect is resolved. Campaigns 1 and 2 differed for")
            print("  some reason this design does not capture. Claim nothing yet.")
    print()
    print("controller exit mixes, which is the mechanism:")
    for r in sorted(ctrl, key=lambda x: x["start_temp"]):
        print("   start %5.1f C -> acc %6.2f%%  mix %s"
              % (r["start_temp"], r["real_acc"], r["exit_mix"]))
