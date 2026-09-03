"""Fig. 7 | Reward-only support expansion on measured Buchwald-Hartwig yields.

a,c  expected absolute yield error (single target 90; ten balanced targets)
b,d  probability mass outside the CPT/SFT training support
Thin grey lines: same seed before/after alignment.  Dotted: best attainable
error within the restricted support and over the full catalogue.
Same style as the reactor figures (CPT+SFT arm only).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from _style import C, WIDTH_DOUBLE, apply_style, mean_ci, panel_label, refline, save

OUT = Path(__file__).resolve().parent / "fig7_empirical"
METHODS = ["sft_base", "dpo", "grpo"]
LAB = {"sft_base": "SFT base", "dpo": "DPO", "grpo": "GRPO"}
COL = {"sft_base": "#7A7A7A", "dpo": C["DPO"], "grpo": C["GRPO"]}


def panel(ax, df, metric, ylabel, fmt, refs=None):
    wide = df.pivot(index="seed", columns="algorithm", values=metric).loc[:, METHODS]
    x = np.arange(3)
    for _, row in wide.iterrows():
        ax.plot(x, row.to_numpy(float), color=C["seed"], linewidth=0.7, marker="o", markersize=2, zorder=1)
    means = wide.mean(0).to_numpy(float); _, lo, hi = mean_ci(wide.to_numpy(float), axis=0)
    ax.plot(x, means, color="#333", linewidth=1.0, zorder=2)
    for i, m in enumerate(METHODS):
        ax.errorbar(x[i], means[i], yerr=[[means[i] - lo[i]], [hi[i] - means[i]]], fmt="o", color=COL[m],
                    markersize=5.5, markeredgecolor="white", markeredgewidth=0.6, capsize=2.5, linewidth=1.2, zorder=3)
        ax.annotate(format(means[i], fmt), (x[i], hi[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=COL[m], fontweight="bold")
    top = max(float(np.nanmax(hi)), float(wide.to_numpy(float).max()))
    if refs:
        for lab, v in refs.items():
            refline(ax, v, f"best {lab} {v:{fmt}}", va="bottom", x_text=0.98)
            top = max(top, v)
    ax.set_ylim(0, top * 1.18); ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in METHODS]); ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel(ylabel)


def main():
    apply_style()
    df = D.empirical_seed_summary()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_DOUBLE, 5.2), gridspec_kw=dict(hspace=0.45, wspace=0.3))
    for row, regime in enumerate(("Single target", "Multi target")):
        sub = df[df["regime"] == regime]
        panel(axes[row, 0], sub, "expected_abs_error", "Expected absolute yield error", ".1f", D.empirical_reference(regime))
        panel(axes[row, 1], sub, "expansion_mass", "Probability outside CPT/SFT support", ".2f")
        axes[row, 0].set_title(f"{regime}: yield matching", loc="left", fontsize=7.5)
        axes[row, 1].set_title(f"{regime}: support expansion", loc="left", fontsize=7.5)
    for ax, L in zip(axes.ravel(), "abcd"):
        panel_label(ax, L, x=-0.15)
    fig.tight_layout()
    save(fig, OUT)


if __name__ == "__main__":
    main()
