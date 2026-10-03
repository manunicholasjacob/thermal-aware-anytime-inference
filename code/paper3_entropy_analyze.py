#!/usr/bin/env python3
"""Paper 3: does the thermal controller sit on the accuracy-temperature frontier?

Consumes paper3_entropy_sweep.json, which measured the paper's four policies and
five entropy-threshold arms in one interleaved session.

The question the paper has to answer for TETC is not "is the controller better
than its own static bounds", which it obviously is, but "is it better than what
the early-exit literature already does". The entropy arms trace a frontier in
accuracy against mean temperature. The controller either sits on that frontier,
in which case it is competitive, or inside it, in which case a published baseline
dominates it and the paper has to say so.

Two accountings are reported, and they can disagree.

  MEASURED. What the board actually did. The cascade pays for every exit it
  touches, because the three exits are separate ONNX graphs and escalating
  recomputes the shared backbone.

  IDEAL. What a real branching network would pay, charging each request only the
  service time of the exit it reached. Temperature under this accounting cannot
  be measured here, so it is predicted from the three always-X arms in this same
  session, which give three (utilisation, temperature) points. That prediction is
  a model, it is labelled as one everywhere it appears, and it is not a
  substitute for a measurement.
"""
import json
import statistics as st
import sys
from pathlib import Path

SWEEP = Path(sys.argv[1] if len(sys.argv) > 1 else "paper3_entropy_sweep.json")
CALIB = Path(sys.argv[2] if len(sys.argv) > 2 else "paper3_entropy_calib.json")
RATE, NWORKERS = 1400, 4

# The SVC table in paper3_controller_exp.py is NOT used here, and the
# ideal_mean_ms recorded by the sweep is superseded by what follows.
#
# That table (exit1 0.086, exit2 0.168, final 0.594 ms) was measured with
# intra_op_num_threads=4, in paper3_int8_ab.py and paper3_final_accuracy.py.
# The thermal experiments run with intra_op_num_threads=1 and four concurrent
# worker threads instead, which is a different configuration, so charging the
# ideal accounting with those constants mixes two threading regimes.
#
# Per-exit service time is therefore taken from THIS session: the p50 of the
# always-exitN arms is the service time of exit N under exactly the
# configuration every arm here ran under.
runs = json.load(open(SWEEP))
by = {}
for r in runs:
    by.setdefault(r["policy"], []).append(r)


def session_svc():
    svc = {}
    for name, ex in (("always-exit1", "exit1"), ("always-exit2", "exit2"),
                     ("always-final", "final")):
        if name in by:
            svc[ex] = st.mean(r["p50"] for r in by[name])
    return svc


SVC = session_svc()


def ideal_from_mix(mix):
    """Shared-backbone cost: charge only the exit the request actually reached."""
    if not mix or len(SVC) < 3:
        return float("nan")
    return sum(SVC[k] * v / 100.0 for k, v in mix.items() if k in SVC)


def agg(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)


order = ["always-exit1", "always-exit2", "always-final", "adaptive-ctrl"]
order += sorted([p for p in by if p.startswith("entropy")],
                key=lambda p: float(p.split("-")[1]))

print("Paper 3: paper policies vs the entropy-threshold baseline, one session")
print("reps per arm: %s" % {p: len(rs) for p, rs in by.items()})
print()
print("%-15s %14s %14s %9s %10s %10s" %
      ("arm", "accuracy %", "mean temp C", "viol s", "meas ms", "ideal ms"))
print("-" * 80)
rows = []
for p in order:
    if p not in by:
        continue
    rs = by[p]
    a, asd = agg(rs, "real_acc")
    t, tsd = agg(rs, "mean_temp")
    v, vsd = agg(rs, "cap_viol_s")
    ml, _ = agg(rs, "mean_lat_ms")
    # recomputed from this session's own service times, not the sweep's field
    mixes = [r["exit_mix"] for r in rs if r.get("exit_mix")]
    il = st.mean([ideal_from_mix(m) for m in mixes]) if mixes else float("nan")
    rows.append(dict(arm=p, acc=a, acc_sd=asd, temp=t, temp_sd=tsd, viol=v,
                     viol_sd=vsd, meas_ms=ml, ideal_ms=il,
                     kind=rs[0].get("kind", "paper")))
    print("%-15s %7.2f +-%4.2f %7.2f +-%4.2f %5.1f+-%3.1f %10s %10s"
          % (p, a, asd, t, tsd, v, vsd,
             ("%.3f" % ml) if ml == ml else "n/a",
             ("%.3f" % il) if il == il else "n/a"))

ctrl = next((r for r in rows if r["arm"] == "adaptive-ctrl"), None)
ent = [r for r in rows if r["kind"] == "entropy"]
final = next((r for r in rows if r["arm"] == "always-final"), None)

print()
print("=" * 80)
print("DOES A PUBLISHED BASELINE DOMINATE THE CONTROLLER?")
print("=" * 80)
if ctrl and ent:
    print("controller: %.2f%% at %.2f C, %.1f s over cap"
          % (ctrl["acc"], ctrl["temp"], ctrl["viol"]))
    dom = [e for e in ent if e["acc"] >= ctrl["acc"] and e["temp"] <= ctrl["temp"]]
    print()
    if dom:
        print("DOMINATED. These entropy arms are at least as accurate AND no hotter:")
        for e in dom:
            print("   %-15s %.2f%% at %.2f C  (%+.2f pp, %+.2f C)"
                  % (e["arm"], e["acc"], e["temp"], e["acc"] - ctrl["acc"],
                     e["temp"] - ctrl["temp"]))
    else:
        print("NOT DOMINATED on measured temperature. No entropy arm is both at least")
        print("as accurate and no hotter. Nearest contenders:")
        for e in sorted(ent, key=lambda e: abs(e["acc"] - ctrl["acc"]))[:3]:
            print("   %-15s %.2f%% (%+.2f pp) at %.2f C (%+.2f C)"
                  % (e["arm"], e["acc"], e["acc"] - ctrl["acc"], e["temp"],
                     e["temp"] - ctrl["temp"]))
    print()
    more_acc = [e for e in ent if e["acc"] > ctrl["acc"]]
    if more_acc:
        print("More accurate than the controller, whatever the temperature cost:")
        for e in more_acc:
            print("   %-15s %.2f%%  (%+.2f pp) at %.2f C"
                  % (e["arm"], e["acc"], e["acc"] - ctrl["acc"], e["temp"]))
    else:
        print("No entropy arm beats the controller on accuracy at any temperature.")

# ---- ideal-cost temperature model, from this session's three static arms ----
print()
print("=" * 80)
print("IDEAL SHARED-BACKBONE ACCOUNTING (MODEL, NOT MEASUREMENT)")
print("=" * 80)
print("per-exit service time measured in THIS session (p50 of the always-exitN arms):")
for ex in ("exit1", "exit2", "final"):
    if ex in SVC:
        print("   %-6s %.3f ms" % (ex, SVC[ex]))
print()
pts = []
for name, ex in (("always-exit1", "exit1"), ("always-exit2", "exit2"),
                 ("always-final", "final")):
    r = next((x for x in rows if x["arm"] == name), None)
    if r and ex in SVC:
        pts.append((RATE * SVC[ex] / 1000.0 / NWORKERS, r["temp"]))
if len(pts) >= 2:
    print("calibration points, utilisation -> mean temp, from this session:")
    for u, t in pts:
        print("   util %.4f -> %.2f C" % (u, t))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    icept = my - slope * mx
    print("   linear fit: T = %.2f + %.2f * util" % (icept, slope))
    print()
    print("   %-15s %9s %9s %12s" % ("arm", "ideal ms", "util", "predicted C"))
    for e in ent:
        if e["ideal_ms"] != e["ideal_ms"]:
            continue
        u = RATE * e["ideal_ms"] / 1000.0 / NWORKERS
        print("   %-15s %9.3f %9.4f %12.2f" % (e["arm"], e["ideal_ms"], u,
                                               icept + slope * u))
    if ctrl:
        print()
        print("   controller measured: %.2f%% at %.2f C" % (ctrl["acc"], ctrl["temp"]))
        print("   An entropy arm whose PREDICTED temperature is below that and whose")
        print("   accuracy is above it would dominate a real implementation. The")
        print("   prediction is a two-parameter fit to three points and nothing more.")
else:
    print("not enough static arms to fit a model")

if CALIB.exists():
    c = json.load(open(CALIB))
    print()
    print("=" * 80)
    print("CROSS-CHECK: measured accuracy vs the exact offline curve")
    print("=" * 80)
    curve = {round(x["tau"], 2): x for x in c["curve"]}
    for e in ent:
        tau = round(float(e["arm"].split("-")[1]), 2)
        if tau in curve:
            print("   tau %.2f   measured %6.2f%%   exact %6.2f%%   delta %+5.2f pp"
                  % (tau, e["acc"], curve[tau]["acc"], e["acc"] - curve[tau]["acc"]))
    print("   A delta beyond a few tenths means the online run is not sampling the")
    print("   test set the way the offline pass did.")

json.dump(rows, open("paper3_entropy_summary.json", "w"), indent=1)
print()
print("wrote paper3_entropy_summary.json")
