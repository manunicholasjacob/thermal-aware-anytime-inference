#!/usr/bin/env python3
"""Paper 3, second figure: (a) INT8 speedup over FP32 by export format and exit
depth -- static quantization is a slowdown at exit1 and pays only from exit2 onward
(0.83, 1.26, 2.47x), and dynamic quantization is a
pessimization everywhere; (b) the same effect seen as a compression of the exit
ladder's latency span, which is the range the thermal controller has to work with.
All numbers from data/paper3_clean_dataset.json (measured on Pi 5)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# DATE rejects submissions containing Type-3 fonts; 42 embeds TrueType instead.
plt.rcParams.update({"font.size": 7, "font.family": "serif", "figure.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "dejavuserif"})
QOP = "#a0aec0"; QDQ = "#2b6cb0"; FP = "#c05621"; INK = "#1a1a1a"

D = Path(__file__).resolve().parent.parent / "data" / "paper3_clean_dataset.json"
rows = json.load(open(D))
names = [r["exit"] for r in rows]
qop = [r["int8-QOperator"]["speedup"] for r in rows]
qdq = [r["int8-QDQ-clean"]["speedup"] for r in rows]
lat_fp = [r["fp32"]["lat_mean"] for r in rows]
lat_qdq = [r["int8-QDQ-clean"]["lat_mean"] for r in rows]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.5))

# (a) speedup by format
x = np.arange(len(names)); w = 0.36
ax1.bar(x - w/2, qop, w, color=QOP, edgecolor=INK, lw=.4, label="INT8 dynamic")
ax1.bar(x + w/2, qdq, w, color=QDQ, edgecolor=INK, lw=.4, label="INT8 static (QDQ)")
ax1.axhline(1.0, color=FP, lw=1.0, ls="--", zorder=1)
ax1.text(-0.44, 1.07, "FP32 baseline", fontsize=6, color=FP, ha="left")
for xi, v in zip(x - w/2, qop):
    ax1.text(xi, v + .05, f"{v:.2f}", ha="center", fontsize=5.5, color=INK)
for xi, v in zip(x + w/2, qdq):
    ax1.text(xi, v + .05, f"{v:.2f}", ha="center", fontsize=5.5, color=INK)
ax1.set_xticks(x); ax1.set_xticklabels(names)
ax1.set_ylabel("speedup over FP32 ($\\times$)")
ax1.set_ylim(0, 2.95)
ax1.set_title("(a) Static INT8 pays only from exit2 onward", fontsize=7.5)
ax1.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2)
ax1.spines[["top", "right"]].set_visible(False)

# (b) the exit ladder, FP32 vs QDQ
ax2.plot(lat_fp, [1]*3, "o-", color=FP, ms=5, mec=INK, mew=.4, lw=1.0, label="FP32")
ax2.plot(lat_qdq, [0]*3, "o-", color=QDQ, ms=5, mec=INK, mew=.4, lw=1.0, label="INT8 static")
for l, n in zip(lat_fp, names):
    ax2.annotate(n, (l, 1), fontsize=5.5, xytext=(l, 1.18), ha="center", color=INK)
for l, n in zip(lat_qdq, names):
    ax2.annotate(n, (l, 0), fontsize=5.5, xytext=(l, -0.32), ha="center", color=INK)
ax2.text(0.28, 1.52, f"span {lat_fp[-1]/lat_fp[0]:.1f}$\\times$", fontsize=6.5, color=FP, ha="center")
ax2.text(0.20, -0.72, f"span {lat_qdq[-1]/lat_qdq[0]:.1f}$\\times$", fontsize=6.5, color=QDQ, ha="center")
ax2.set_xscale("log"); ax2.set_xlim(0.055, 2.2); ax2.set_ylim(-0.95, 1.75)
ax2.set_yticks([]); ax2.set_xlabel("latency per inference (ms, log)")
ax2.set_xticks([0.1, 0.3, 1.0]); ax2.set_xticklabels(["0.1", "0.3", "1.0"])
ax2.set_title("(b) Quantizing compresses the exit ladder", fontsize=7.5)
ax2.legend(frameon=False, fontsize=6, loc="center right", handlelength=1.2)
ax2.spines[["top", "right", "left"]].set_visible(False)

fig.tight_layout(pad=0.4, h_pad=1.4)
out = Path(__file__).resolve().parent.parent / "paper" / "fig_p3b.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
print("wrote", out)
