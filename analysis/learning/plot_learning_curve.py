#!/usr/bin/env python3
"""Learning-curve figure (v4): median and 10th-90th percentiles of |L4| and |L6| of the O4-constrained BLUE weights over
the 30 random subsets at each N (percentiles of the absolute values, so bands are not clipped when L changes sign),
full-sample values as stars, and the fixed-O4 values |L4|=4, |L6|=20 as reference lines.
Usage: plot_learning_curve.py SUBSETS_CSV FRONTIER_CSV"""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
SUB = sys.argv[1] if len(sys.argv) > 1 else "poly11_small_learning_curve_exact_subsets.csv"
FULL = sys.argv[2] if len(sys.argv) > 2 else "frontier.csv"
s = pd.read_csv(SUB); full = pd.read_csv(FULL); full = full[(full.tau == 'unrestricted') & (full.k == 0)]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for i, particle in enumerate(["P", "N"]):
    d = s[(s.particle == particle) & (s.N <= 800)]; Nfull = 992 if particle == "P" else 999
    for j, (stem, marker) in enumerate((("L4", "o"), ("L6", "s"))):
        g = d.groupby("N")[stem].apply(lambda v: np.abs(v).quantile([.1, .5, .9])).unstack()
        c = f"C{2*i+j}"
        ax.plot(g.index, g[0.5], marker=marker, color=c, label=rf"{particle} $|L_{stem[1]}|$")
        ax.fill_between(g.index, g[0.1], g[0.9], color=c, alpha=0.15, linewidth=0)
        ax.plot([Nfull], [abs(full[full.particle == particle].iloc[0][stem])], marker="*", ms=11, color=c, ls="none")
for val, lab in ((4, r"fixed O4 $|L_4|=4$"), (20, r"fixed O4 $|L_6|=20$")):
    ax.axhline(val, color="0.4", ls=":" if val == 4 else "--", lw=1); ax.text(560, val * 1.25, lab, fontsize=8, color="0.3")
ax.set_xlabel("Subset size $N$"); ax.set_ylabel(r"$|L_4|$, $|L_6|$ of O4-constrained BLUE")
ax.set_yscale("log"); ax.set_ylim(1, 3e6); ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.3), ncol=2); ax.grid(True, which="both", alpha=0.25)
fig.tight_layout(); fig.savefig("fig_learning_curve_leverage_v4.pdf", bbox_inches="tight"); fig.savefig("fig_learning_curve_leverage_v4.png", dpi=200, bbox_inches="tight")
print(pd.concat({p: s[s.particle == p].groupby("N")[["L4", "L6"]].apply(lambda v: np.abs(v).quantile(.1)) for p in "PN"}))
