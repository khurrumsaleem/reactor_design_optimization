"""Fig. 5 | Effect of continued pre-training on alignment (n = 5 seeds).

a,c  single-target DPO / GRPO: cumulative best fitness (log y)
b,d  multi-target DPO / GRPO: per-step fitness, 25-step rolling mean
e    best composite fitness per seed by initialization arm
f    Gd inventory of the best design per seed by initialization arm
Solid = CPT+SFT, dashed = SFT-only (same hue per algorithm).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from _style import C, WIDTH_DOUBLE, apply_style, band, bottom_legend, ci_proxy, dots_with_mean, panel_label, refline, rolling_mean, save

OUT = Path(__file__).resolve().parent / "fig5_cpt"
ROLL = 25


def single(algo, base):
    steps = [D.dpo_steps(d) for d in D.dpo_single(base)] if algo == "DPO" else [D.grpo_steps(d) for d in D.grpo_single(base)]
    return steps, D.matrix(steps, "cum_best_fit")


def multi(algo, base):
    steps = [D.dpo_steps(d) for d in D.dpo_multi(base)] if algo == "DPO" else [D.grpo_steps(d) for d in D.grpo_multi(base)]
    xs = [s["x"].to_numpy() for s in steps]
    ys = [rolling_mean(s["step_fit"].to_numpy(), ROLL) for s in steps]
    return D.align(xs, ys)


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 7.6))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.32)
    axes = {L: fig.add_subplot(gs[i // 2, i % 2]) for i, L in enumerate("abcdef")}
    finals = {}
    for algo, (Ls, Lm) in (("DPO", ("a", "b")), ("GRPO", ("c", "d"))):
        for base, ls, al in (("sft_only", "--", 0.08), ("cpt_sft", "-", 0.15)):
            steps, (x, m) = single(algo, base)
            band(axes[Ls], x, m, color=C[algo], ls=ls, alpha=al)
            finals[(algo, base)] = D.best_records(steps)
            xm, mm = multi(algo, base)
            band(axes[Lm], xm, mm, color=C[algo], ls=ls, alpha=al)
        axes[Ls].set_yscale("log"); axes[Ls].set_ylim(1.2, 40); axes[Ls].set_ylabel("Cumulative best fitness")
        axes[Ls].set_yticks([1.5, 2, 3, 5, 10]); axes[Ls].yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axes[Ls].set_title(f"{algo}, single target", loc="left", fontsize=7.5, color=C[algo], fontweight="bold")
        axes[Lm].set_ylabel(f"Per-step fitness\n({ROLL}-step rolling mean)"); axes[Lm].set_ylim(0, 30)
        axes[Lm].set_title(f"{algo}, multi target", loc="left", fontsize=7.5, color=C[algo], fontweight="bold")
    for L in "abcd":
        axes[L].set_xlabel("OpenMC evaluations"); axes[L].set_xlim(0, 2000)

    # e, f
    xk = [("DPO", "sft_only"), ("DPO", "cpt_sft"), ("GRPO", "sft_only"), ("GRPO", "cpt_sft")]
    xt = np.arange(4); labs = ["DPO\nSFT only", "DPO\nCPT + SFT", "GRPO\nSFT only", "GRPO\nCPT + SFT"]
    for xi, (algo, base) in zip(xt, xk):
        hollow = base == "sft_only"
        dots_with_mean(axes["e"], xi, [r["fit"] for r in finals[(algo, base)]], color=C[algo], hollow=hollow)
        dots_with_mean(axes["f"], xi, [r["gd"] for r in finals[(algo, base)]], color=C[algo], hollow=hollow)
    axes["e"].set_ylabel("Best composite fitness"); axes["e"].set_ylim(1.5, 1.95)
    axes["f"].set_ylabel("Gd inventory of best design (rods)")
    refline(axes["f"], 16, "training inventory = 16", va="bottom")
    for L in "ef":
        axes[L].set_xticks(xt); axes[L].set_xticklabels(labs); axes[L].set_xlim(-0.6, 3.6)
    for L, ax in axes.items():
        panel_label(ax, L, x=-0.15)
    handles = [Line2D([], [], color="#444", linestyle="-", linewidth=1.4, label="CPT + SFT base (filled markers)"),
               Line2D([], [], color="#444", linestyle="--", linewidth=1.4, label="SFT-only base (hollow markers)"),
               ci_proxy()]
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    bottom_legend(fig, handles, ncol=3, y=0.0)
    save(fig, OUT)


if __name__ == "__main__":
    main()
