"""Extended Data Fig. 1 | Pearson correlation structure of optimization trajectories."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from _style import C, LABEL, WIDTH_DOUBLE, apply_style, panel_label, save

OUT = Path(__file__).resolve().parent / "ed1_correlation"
VARS = ["Gd", r"$k_{\mathrm{eff}}$", r"$F_q$", r"$F_{\Delta H}$", "fitness"]


def valid(df):
    m = (df["k_eff"] > 0.5) & (df["fq"] < 50) & (df["fdh"] < 50) & (df["fitness"] < 50)
    return df.loc[m].reset_index(drop=True)


def records():
    out = {}
    out["GA16"] = valid(pd.concat([d[["g_count", "k_eff", "fq", "fdh", "fitness"]] for d in D.ga(True)]))
    out["GA"] = valid(pd.concat([d[["g_count", "k_eff", "fq", "fdh", "fitness"]] for d in D.ga(False)]))
    rows = []
    for d in D.dpo_single():
        s = d[["chosen_g_count", "chosen_k_eff", "chosen_fq", "chosen_fdh", "chosen_fitness"]].copy()
        s.columns = ["g_count", "k_eff", "fq", "fdh", "fitness"]; rows.append(s)
    out["DPO"] = valid(pd.concat(rows))
    out["GRPO"] = valid(pd.concat([d[["g_count", "k_eff", "fq", "fdh", "fitness"]] for d in D.grpo_single()]))
    return out


def main():
    apply_style()
    R = records()
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.35, hspace=0.42, left=0.08, right=0.94, top=0.93, bottom=0.07)
    im = None
    for i, key in enumerate(["GA16", "GA", "DPO", "GRPO"]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        corr = R[key].corr().to_numpy()
        cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#E5E5E5")
        im = ax.imshow(np.ma.masked_invalid(corr), cmap=cmap, vmin=-1, vmax=1)
        for s in ax.spines.values():
            s.set_visible(True); s.set_edgecolor(C[key]); s.set_linewidth(1.4)
        ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(VARS); ax.set_yticklabels(VARS)
        ax.tick_params(length=0)
        for r in range(5):
            for c in range(5):
                v = corr[r, c]
                ax.text(c, r, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (not np.isnan(v) and abs(v) > 0.55) else "black")
        ax.set_title(LABEL[key], color=C[key], fontweight="bold", fontsize=8, loc="left")
        panel_label(ax, "abcd"[i], x=-0.2)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 2])); cb.set_label("Pearson correlation"); cb.outline.set_linewidth(0.5)
    save(fig, OUT)


if __name__ == "__main__":
    main()
