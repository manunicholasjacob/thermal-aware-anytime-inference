#!/usr/bin/env python3
"""Paper 3, third figure: the controller loses, and this is why.

Replaces an earlier version of this file that drew campaign 1 and claimed two
empty quadrants. Campaign 2 randomised arm order and the controller's result did
not survive, so that framing is gone. The old generator is in git history.

(a) campaign 2, randomised arm order. Accuracy against mean board temperature.
    The controller is no longer outside the baseline's frontier, it is inside it:
    entropy tau=3.0 is both more accurate and cooler.

(b) campaign 3, starting temperature controlled. The controller holds full
    final-exit accuracy on the cool side of the cap and loses a third of it on
    the warm side, while the stateless baseline is a flat line because its
    accuracy cannot depend on anything. The cap is the latch.

Data: data/paper3_entropy_sweep2.json (45 runs) and
data/paper3_ratchet_sweep.json (40 runs), both on a Pi 5, Oct 2026.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# DATE rejects submissions containing Type-3 fonts; 42 embeds TrueType instead.
plt.rcParams.update({"font.size": 7, "font.family": "serif", "figure.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "dejavuserif"})

ENT = "#2b6cb0"
CTRL = "#c05621"
STAT = "#a0aec0"
INK = "#1a1a1a"
CAP = 49.0

ROOT = Path(__file__).resolve().parent.parent / "data"
c2 = json.load(open(ROOT / "paper3_entropy_sweep2.json"))
c3 = json.load(open(ROOT / "paper3_ratchet_sweep.json"))

by2 = {}
for r in c2:
    by2.setdefault(r["policy"], []).append(r)


def m2(p, k):
    return st.mean(x[k] for x in by2[p])


def sd2(p, k):
    return st.pstdev([x[k] for x in by2[p]])


ent = sorted([p for p in by2 if p.startswith("entropy")],
             key=lambda p: float(p.split("-")[1]))
statics = ["always-exit1", "always-exit2", "always-final"]
SLAB = {"always-exit1": "exit1", "always-exit2": "exit2", "always-final": "final"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ---------------- (a) campaign 2: the controller is dominated --------------
ax1.errorbar([m2(p, "mean_temp") for p in ent], [m2(p, "real_acc") for p in ent],
             xerr=[sd2(p, "mean_temp") for p in ent], fmt="o", color=ENT, ms=3.4,
             mec=INK, mew=0.4, elinewidth=0.7, capsize=1.8, ls="none",
             label="entropy threshold (BranchyNet policy)", zorder=3)
for p, off in zip(ent, ((4, -8), (5, -8), (7, 3), (5, 5), (5, 3))):
    ax1.annotate(r"$\tau$=%.1f" % float(p.split("-")[1]),
                 (m2(p, "mean_temp"), m2(p, "real_acc")), textcoords="offset points",
                 xytext=off, fontsize=5.6, color=ENT, zorder=5)
ax1.errorbar([m2(p, "mean_temp") for p in statics], [m2(p, "real_acc") for p in statics],
             xerr=[sd2(p, "mean_temp") for p in statics], fmt="s", color=STAT, ms=3.2,
             mec=INK, mew=0.4, elinewidth=0.7, capsize=1.8, ls="none",
             label="static policies", zorder=2)
for p, off in zip(statics, ((-21, -2), (-7, 6), (-2, 6))):
    ax1.annotate(SLAB[p], (m2(p, "mean_temp"), m2(p, "real_acc")),
                 textcoords="offset points", xytext=off, fontsize=5.6,
                 color="#4a5568", zorder=5)

cx, cy = m2("adaptive-ctrl", "mean_temp"), m2("adaptive-ctrl", "real_acc")
ax1.errorbar([cx], [cy], xerr=[sd2("adaptive-ctrl", "mean_temp")],
             yerr=[sd2("adaptive-ctrl", "real_acc")], fmt="*", color=CTRL, ms=11,
             mec=INK, mew=0.5, elinewidth=0.8, capsize=2.0,
             label="thermal-aware controller", zorder=4)

dx, dy = m2("entropy-3.00", "mean_temp"), m2("entropy-3.00", "real_acc")
ax1.annotate("", xy=(dx, dy), xytext=(cx, cy),
             arrowprops=dict(arrowstyle="->", color=CTRL, lw=0.8,
                             shrinkA=8, shrinkB=4), zorder=6)
ax1.text(46.9, 22.6, "$\\tau$=3.0 is more accurate\nAND cooler", fontsize=5.8,
         color=CTRL, ha="center", zorder=6)

ax1.set_xlim(45.6, 51.0)
ax1.set_ylim(18.5, 36.2)
ax1.set_xlabel("mean board temperature ($^\\circ$C)")
ax1.set_ylabel("accuracy (\\%)")
ax1.set_title("(a) randomised arm order: the controller is dominated",
              fontsize=6.8, pad=4)
ax1.grid(alpha=0.22, lw=0.4)

# ---------------- (b) campaign 3: the latch --------------------------------
ctrl = [r for r in c3 if r["arm"] == "adaptive-ctrl"]
ctl = [r for r in c3 if r["arm"] != "adaptive-ctrl"]

ax2.axvspan(CAP, 51.2, color=CTRL, alpha=0.10, lw=0, zorder=0)
ax2.axvline(CAP, color=CTRL, lw=0.8, ls="--", zorder=1)
ax2.text(CAP + 0.10, 19.0, "starts above the cap", fontsize=5.8, color=CTRL,
         rotation=90, va="bottom", zorder=5)

ax2.plot([r["start_temp"] for r in ctl], [r["real_acc"] for r in ctl], "o",
         color=ENT, ms=3.0, mec=INK, mew=0.35, ls="none", zorder=3,
         label="entropy $\\tau$=3.0 (stateless), sd 0.00")
ax2.plot([r["start_temp"] for r in ctrl], [r["real_acc"] for r in ctrl], "*",
         color=CTRL, ms=9, mec=INK, mew=0.45, ls="none", zorder=4,
         label="thermal-aware controller")

ax2.annotate("14 of 15 hold\nfull accuracy", (46.0, 33.67),
             textcoords="offset points", xytext=(0, -30), fontsize=5.8,
             color=CTRL, ha="left", zorder=5)
ax2.annotate("0 of 5 do", (49.9, 27.2), textcoords="offset points",
             xytext=(0, 34), fontsize=5.8, color=CTRL, ha="center", zorder=5)
odd = [r for r in ctrl if r["start_temp"] < CAP and r["real_acc"] < 30]
if odd:
    o = odd[0]
    ax2.annotate("the one exception", (o["start_temp"], o["real_acc"]),
                 textcoords="offset points", xytext=(-7, -10), fontsize=5.5,
                 color="#4a5568", ha="right", zorder=5)

ax2.set_xlim(45.0, 51.2)
ax2.set_ylim(18.5, 36.2)
ax2.set_xlabel("board temperature at the start of the run ($^\\circ$C)")
ax2.set_ylabel("accuracy (\\%)")
ax2.set_title("(b) controlled start: the cap is the latch", fontsize=6.8, pad=4)
ax2.grid(alpha=0.22, lw=0.4)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + [h2[0]], l1 + [l2[0]], fontsize=5.8, loc="lower center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, -0.03), handletextpad=0.5,
           columnspacing=1.4)
fig.tight_layout(pad=0.6, rect=(0, 0.07, 1, 1))
OUT = Path(__file__).resolve().parent.parent / "paper" / "fig_p3c.pdf"
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
print("campaign 2 controller : %.2f%% at %.2f C" % (cy, cx))
print("campaign 2 entropy-3.0: %.2f%% at %.2f C  -> dominates by %+.2f pp, %+.2f C"
      % (dy, dx, dy - cy, dx - cx))
below = [r for r in ctrl if r["start_temp"] < CAP]
above = [r for r in ctrl if r["start_temp"] >= CAP]
print("campaign 3: below cap %d runs, %d held >=33%%; at/above cap %d runs, %d held"
      % (len(below), len([r for r in below if r["real_acc"] >= 33]),
         len(above), len([r for r in above if r["real_acc"] >= 33])))
