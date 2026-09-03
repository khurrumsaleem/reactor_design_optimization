"""Fig. 2 | Alignment and absorber-inventory dynamics (n = 5 seeds).

a  cumulative best composite fitness vs OpenMC evaluations (log y)
b  Gd inventory of the cumulative-best design (dotted: 16-rod training inventory)
c  k_eff of the cumulative-best design (dotted: target 1.05)
Solid = CPT+SFT base, dashed = SFT-only base, GA baselines in green/purple.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from _style import C, LABEL, WIDTH_DOUBLE, apply_style, band, bottom_legend, ci_proxy, panel_label, refline, save

OUT = Path(__file__).resolve().parent / "fig2_dynamics"

SERIES = [  # key, loader, color, linestyle, label
    ("GA16", lambda: [D.ga_steps(d) for d in D.ga(True)], C["GA16"], "-", LABEL["GA16"]),
    ("GA", lambda: [D.ga_steps(d) for d in D.ga(False)], C["GA"], "-", LABEL["GA"]),
    ("DPO", lambda: [D.dpo_steps(d) for d in D.dpo_single()], C["DPO"], "-", LABEL["DPO"]),
    ("DPO_s", lambda: [D.dpo_steps(d) for d in D.dpo_single("sft_only")], C["DPO"], "--", "DPO (SFT only)"),
    ("GRPO", lambda: [D.grpo_steps(d) for d in D.grpo_single()], C["GRPO"], "-", LABEL["GRPO"]),
    ("GRPO_s", lambda: [D.grpo_steps(d) for d in D.grpo_single("sft_only")], C["GRPO"], "--", "GRPO (SFT only)"),
]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 2.55))
    data = {k: f() for k, f, *_ in SERIES}
    for key, _, col, ls, lab in SERIES:
        steps = data[key]
        for ax, colname in zip(axes, ("cum_best_fit", "cum_best_gd", "cum_best_k")):
            x, m = D.matrix(steps, colname)
            band(ax, x, m, color=col, ls=ls, alpha=0.13 if ls == "-" else 0.07)
    a, b, c = axes
    a.set_yscale("log"); a.set_ylabel("Cumulative best composite fitness")
    a.set_ylim(1.2, 40); a.set_yticks([1.5, 2, 5, 10, 20]); a.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    b.set_ylabel("Gd inventory of best design (rods)")
    refline(b, 16, "training inventory = 16", va="top", x_text=0.98)
    c.set_ylabel(r"$k_{\mathrm{eff}}$ of best design")
    refline(c, 1.05, r"target $k_{\mathrm{eff}}$ = 1.05", va="bottom")
    c.set_ylim(0.97, 1.22); c.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for ax, L in zip(axes, "abc"):
        ax.set_xlabel("OpenMC evaluations")
        ax.set_xlim(0, 2000); ax.set_xticks([0, 500, 1000, 1500, 2000])
        panel_label(ax, L, x=-0.22)
    handles = [Line2D([], [], color=col, linestyle=ls, linewidth=1.4, label=lab) for _, _, col, ls, lab in SERIES]
    handles.append(ci_proxy())
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.6)
    bottom_legend(fig, handles, ncol=4, y=0.0)
    save(fig, OUT)


if __name__ == "__main__":
    main()
