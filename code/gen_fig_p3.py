#!/usr/bin/env python3
"""Paper 3 figures: (a) controller operating regime under a binding cap -- the
adaptive controller uniquely occupies the thermally-safe, higher-accuracy corner;
(b) the accuracy-latency Pareto of early-exit policies (QDQ-clean). Measured on Pi 5."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
# DATE rejects submissions containing Type-3 fonts; 42 embeds TrueType instead.
plt.rcParams.update({"font.size":8,"font.family":"serif","figure.dpi":300,
                     "pdf.fonttype":42,"ps.fonttype":42})
CTRL="#c05621"; ST="#2b6cb0"; INK="#1a1a1a"

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.0,2.7))

# (a) accuracy vs cap-violations at binding cap 49C
pts={"always-exit1":(1,19.48),"always-exit2":(41,32.75),
     "always-final":(80,33.67),"adaptive":(2,23.91)}
for name,(v,a) in pts.items():
    c=CTRL if name=="adaptive" else ST
    ax1.scatter(v,a,s=70,color=c,zorder=3,edgecolor=INK,lw=.5)
    dy = 0.9 if name!="always-exit2" else -1.6
    ax1.annotate(name.replace("always-",""),(v,a),fontsize=6.5,
                 xytext=(v+2,a+dy),color=c)
ax1.axvspan(-3,5,color="green",alpha=0.06)
ax1.text(6.5,17.6,"thermally safe",fontsize=6,color="green",ha="left")
ax1.set_xlabel("cap violations (s of 90 s)"); ax1.set_ylabel("achieved accuracy (%)")
ax1.set_title("(a) Under a binding 49°C cap",fontsize=8)
ax1.set_xlim(-4,88); ax1.set_ylim(17,36)
ax1.spines[["top","right"]].set_visible(False)

# (b) accuracy-latency Pareto (QDQ-clean)
lat=[0.086,0.168,0.594]; acc=[19.45,32.73,33.65]; names=["exit1","exit2","final"]
ax2.plot(lat,acc,"-o",color=ST,ms=6,zorder=3,mec=INK,mew=.5)
for l,a,n in zip(lat,acc,names):
    dy = 0.9 if n=="exit1" else -1.4          # exit1 sits on the axis; label it above
    ax2.annotate(n,(l,a),fontsize=7,xytext=(l*1.05,a+dy),color=INK)
ax2.set_xscale("log"); ax2.set_xlabel("latency per inference (ms, log)")
ax2.set_ylabel("top-1 accuracy (%)")
ax2.set_title("(b) Early-exit accuracy–latency Pareto",fontsize=8)
ax2.set_xticks([0.1,0.2,0.5]); ax2.set_xticklabels(["0.1","0.2","0.5"])
# the data spans less than a decade, so matplotlib labels the minor log ticks too
ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.spines[["top","right"]].set_visible(False)

fig.tight_layout(pad=0.6)
fig.savefig("fig_p3.pdf",bbox_inches="tight"); fig.savefig("fig_p3.png",bbox_inches="tight",dpi=150)
print("wrote fig_p3.pdf/.png")
