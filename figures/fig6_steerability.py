"""Fig. 6 | Steerability-fitness trade-off under simulator-in-the-loop alignment.

a  cumulative best fitness, single- vs multi-target DPO and GRPO (CPT+SFT)
b  per-step fitness (25-step rolling mean), the cost of multi-target exposure
c  multi-target criticality error |k_eff - k_target| (pcm, rolling mean)
d  generated Gd inventory vs prompted k_eff for the CPT+SFT checkpoints
   (mean +/- 95 % CI over ~100 samples per target, truncated generations excluded)
e  bootstrap 95 % CI (10,000 resamples) of the slope for all ten checkpoints
Panels d-e use the registered steerability analysis files (data/reevaluation).
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
from _style import C, WIDTH_DOUBLE, apply_style, band, bottom_legend, ci_proxy, panel_label, refline, rolling_mean, save

OUT = Path(__file__).resolve().parent / "fig6_steerability"
ROLL = 25

STYLE = {  # model_name -> (color, linestyle, label)
    "CPT_SFT_Base":        (C["SFT_ONLY"], ":",  "Base (CPT + SFT)"),
    "DPO_Single_CPT_SFT":  (C["DPO"], "-",  "DPO single"),
    "DPO_Multi_CPT_SFT":   (C["DPO_multi"], "--", "DPO multi"),
    "GRPO_Single_CPT_SFT": (C["GRPO"], "-",  "GRPO single"),
    "GRPO_Multi_CPT_SFT":  (C["GRPO_multi"], "--", "GRPO multi"),
    "SFT_Only_Base":       (C["SFT_ONLY"], ":",  "Base (SFT only)"),
    "DPO_Single_SFT":      (C["DPO"], "-",  "DPO single"),
    "DPO_Multi_SFT":       (C["DPO_multi"], "--", "DPO multi"),
    "GRPO_Single_SFT":     (C["GRPO"], "-",  "GRPO single"),
    "GRPO_Multi_SFT":      (C["GRPO_multi"], "--", "GRPO multi"),
}
CPT_MODELS = ["CPT_SFT_Base", "DPO_Single_CPT_SFT", "DPO_Multi_CPT_SFT", "GRPO_Single_CPT_SFT", "GRPO_Multi_CPT_SFT"]
SLOPE_ORDER = ["SFT_Only_Base", "DPO_Single_SFT", "DPO_Multi_SFT", "GRPO_Single_SFT", "GRPO_Multi_SFT",
               "CPT_SFT_Base", "DPO_Single_CPT_SFT", "DPO_Multi_CPT_SFT", "GRPO_Single_CPT_SFT", "GRPO_Multi_CPT_SFT"]


def traj():
    out = {}
    out["DPO_single"] = [D.dpo_steps(d) for d in D.dpo_single()]
    out["GRPO_single"] = [D.grpo_steps(d) for d in D.grpo_single()]
    out["DPO_multi"] = [D.dpo_steps(d) for d in D.dpo_multi()]
    out["GRPO_multi"] = [D.grpo_steps(d) for d in D.grpo_multi()]
    return out


def main():
    apply_style()
    T = traj()
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 5.6))
    outer = fig.add_gridspec(2, 1, hspace=0.5, height_ratios=[1, 1.15])
    top = outer[0].subgridspec(1, 3, wspace=0.55)
    bot = outer[1].subgridspec(1, 2, wspace=0.62, width_ratios=[1.0, 1.05])
    a = fig.add_subplot(top[0, 0]); b = fig.add_subplot(top[0, 1]); c = fig.add_subplot(top[0, 2])
    d = fig.add_subplot(bot[0, 0]); e = fig.add_subplot(bot[0, 1])

    series = [("DPO_single", C["DPO"], "-"), ("GRPO_single", C["GRPO"], "-"),
              ("DPO_multi", C["DPO_multi"], "--"), ("GRPO_multi", C["GRPO_multi"], "--")]
    for key, col, ls in series:
        steps = T[key]
        x, m = D.matrix(steps, "cum_best_fit"); band(a, x, m, color=col, ls=ls)
        x, m = D.align([s["x"].to_numpy() for s in steps], [rolling_mean(s["step_fit"].to_numpy(), ROLL) for s in steps])
        band(b, x, m, color=col, ls=ls)
        if key.endswith("multi"):
            err = [rolling_mean(np.abs(s["step_k"].to_numpy() - s["target_k"].to_numpy()) * 1e5, ROLL) for s in steps]
            x, m = D.align([s["x"].to_numpy() for s in steps], err); band(c, x, m, color=col, ls=ls)
    a.set_yscale("log"); a.set_yticks([1.5, 2, 3, 5, 10]); a.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    a.set_ylabel("Cumulative best fitness"); a.set_ylim(1.2, 30)
    b.set_yscale("log"); b.set_yticks([2, 5, 10, 20]); b.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    b.set_ylabel(f"Per-step fitness ({ROLL}-step rolling mean)"); b.set_ylim(1.5, 40)
    c.set_yscale("log"); c.set_ylim(300, 3e4); c.set_ylabel(r"$|k_{\mathrm{eff}} - k_{\mathrm{target}}|$ (pcm, rolling mean)")
    for ax in (a, b, c):
        ax.set_xlabel("OpenMC evaluations"); ax.set_xlim(0, 2000); ax.set_xticks([0, 1000, 2000])

    # d: Gd vs prompted target for CPT+SFT checkpoints
    sm = D.steerability_means()
    for name in CPT_MODELS:
        sub = sm[sm["model_name"] == name].sort_values("target_k")
        col, ls, lab = STYLE[name]
        d.fill_between(sub["target_k"], sub["ci_lower"], sub["ci_upper"], color=col, alpha=0.15, linewidth=0)
        d.plot(sub["target_k"], sub["mean_gd"], color=col, linestyle=ls, marker="o", markersize=3, linewidth=1.2, label=lab)
    d.set_xlabel(r"Prompted target $k_{\mathrm{eff}}$"); d.set_ylabel("Generated Gd inventory (rods)")
    d.set_xticks(np.round(np.arange(1.02, 1.081, 0.01), 2))
    refline(d, 16, "training inventory = 16", va="bottom", x_text=0.98)
    d.legend(loc="center left", bbox_to_anchor=(0.01, 0.30), fontsize=6.0, ncol=1)
    d.set_title("CPT + SFT checkpoints", loc="left", fontsize=7.5)

    # e: slopes
    sl = D.steerability_slopes().set_index("model_name")
    y = np.arange(len(SLOPE_ORDER))[::-1]
    for yi, name in zip(y, SLOPE_ORDER):
        r = sl.loc[name]; col, ls, lab = STYLE[name]
        sig = bool(r["ci_excludes_zero"]) if not isinstance(r["ci_excludes_zero"], str) else r["ci_excludes_zero"] == "True"
        e.errorbar(r["point_slope"], yi, xerr=[[r["point_slope"] - r["ci_lower"]], [r["ci_upper"] - r["point_slope"]]],
                   fmt="o", color=col, ecolor=col, capsize=2, markersize=4.2 if sig else 3.4, linewidth=1.0,
                   markerfacecolor=col if sig else "white", markeredgewidth=0.9)
    e.axvline(0, color=C["ref"], linestyle=":", linewidth=0.8)
    e.set_yticks(y); e.set_yticklabels([STYLE[n][2] for n in SLOPE_ORDER], fontsize=6.3)
    e.axhspan(4.5, 9.7, color="#F2F2F2", zorder=0); e.axhspan(-0.5, 4.5, color="white", zorder=0)
    e.text(0.015, 0.985, "SFT-only base", transform=e.transAxes, ha="left", va="top", fontsize=6.3, color="#555", fontweight="bold")
    e.text(0.015, 0.48, "CPT + SFT base", transform=e.transAxes, ha="left", va="top", fontsize=6.3, color="#555", fontweight="bold")
    e.set_xlim(-215, 95)
    e.set_xlabel(r"Slope of $\langle$Gd$\rangle$ vs target $k_{\mathrm{eff}}$ (rods per unit $k_{\mathrm{eff}}$)")
    e.set_ylim(-0.6, 9.7)

    for ax, L, xo in ((a, "a", -0.32), (b, "b", -0.32), (c, "c", -0.32), (d, "d", -0.17), (e, "e", -0.47)):
        panel_label(ax, L, x=xo)
    handles = [Line2D([], [], color=col, linestyle=ls, linewidth=1.4, label=lab) for col, ls, lab in
               ((C["DPO"], "-", "DPO single"), (C["GRPO"], "-", "GRPO single"),
                (C["DPO_multi"], "--", "DPO multi"), (C["GRPO_multi"], "--", "GRPO multi"))]
    handles += [ci_proxy(r"a–c: mean $\pm$ 95% CI ($n$ = 5 seeds); d: $\pm$ 95% CI over samples; e: bootstrap 95% CI, filled = excludes zero")]
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    bottom_legend(fig, handles, ncol=5, y=0.0, fontsize=6.3)
    save(fig, OUT)


if __name__ == "__main__":
    main()
