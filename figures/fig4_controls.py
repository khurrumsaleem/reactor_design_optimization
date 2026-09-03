"""Fig. 4 | Objective-dependent adaptation and attribution of the gain.

a  Gd inventory of the chosen design (25-step rolling mean) under the full
   objective (DPO, 5 seeds) and with the criticality penalty removed (1 run).
b  per-seed best composite fitness of fixed-inventory GA, unconstrained GA,
   informed random search supplied with the [20, 40] window, DPO and GRPO.
c  criticality error and  d  peaking-only fitness for the three methods that
   reach the target region.  Panels b-d absorb the former oracle table.
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
from _style import (C, LABEL, WIDTH_DOUBLE, apply_style, band, ci_proxy, dots_with_mean,
                    panel_label, refline, rolling_mean, save)

OUT = Path(__file__).resolve().parent / "fig4_controls"
ROLL = 25


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.6, wspace=0.35)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1])

    # ---- a: inventory trajectories, full objective vs no penalty
    full = [D.dpo_steps(df) for df in D.dpo_single()]
    xs = [s["x"].to_numpy() for s in full]
    ys = [rolling_mean(s["step_gd"].to_numpy(), ROLL) for s in full]
    x, m = D.align(xs, ys)
    band(a, x, m, color=C["DPO"], label="full objective (DPO, 5 seeds)")
    npn = D.no_penalty()
    a.plot(npn["simulation_count"], rolling_mean(npn["chosen_g_count"].to_numpy(), ROLL),
           color=C["NOPEN"], linewidth=1.2, linestyle="-", label="criticality penalty removed (1 run)")
    refline(a, 16, "training inventory = 16", va="top", x_text=0.98)
    a.set_xlabel("OpenMC evaluations"); a.set_ylabel(f"Gd inventory of chosen design\n({ROLL}-step rolling mean, rods)")
    a.set_xlim(0, 2000); a.set_ylim(-1, 45)
    a.annotate("↑ suppress excess reactivity", xy=(1500, 30), fontsize=6.3, color=C["DPO"], ha="center")
    a.annotate("↓ flatten power only", xy=(1500, 3.2), fontsize=6.3, color=C["NOPEN"], ha="center")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), fontsize=6.0)

    # ---- b-d: attribution
    recs = {
        "GA16": D.best_records([D.ga_steps(df) for df in D.ga(True)]),
        "GA": D.best_records([D.ga_steps(df) for df in D.ga(False)]),
        "RS": D.informed_best_records(),
        "DPO": D.best_records([D.dpo_steps(df) for df in D.dpo_single()]),
        "GRPO": D.best_records([D.grpo_steps(df) for df in D.grpo_single()]),
    }
    keys = ["GA16", "GA", "RS", "DPO", "GRPO"]
    xt = np.arange(len(keys))
    labels = ["GA\nGd = 16", "GA\nfree", "Informed\nrandom\nsearch", "DPO", "GRPO"]
    source = ["prescribed", "unguided\nmutation", "supplied\npost hoc", "identified\nby policy", "identified\nby policy"]
    for xi, k in zip(xt, keys):
        dots_with_mean(b, xi, [r["fit"] for r in recs[k]], color=C[k])
    b.set_yscale("log"); b.set_ylim(1.3, 16)
    b.set_yticks([1.5, 2, 3, 5, 10]); b.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    b.set_ylabel("Best composite fitness")
    b.set_xticks(xt); b.set_xticklabels(labels); b.set_xlim(-0.6, 4.6)
    for xi, s in zip(xt, source):
        b.text(xi, -0.36, s, transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=5.6, color="#666666", style="italic")
    b.text(-0.55, -0.36, "window\nsource:", transform=b.get_xaxis_transform(), ha="right", va="top", fontsize=5.6, color="#666666", style="italic")

    keys3 = ["RS", "DPO", "GRPO"]; xt3 = np.arange(3); lab3 = ["Informed\nrandom search", "DPO", "GRPO"]
    for xi, k in zip(xt3, keys3):
        dots_with_mean(c, xi, [D.dk_pcm(r["k"]) for r in recs[k]], color=C[k])
        dots_with_mean(d, xi, [D.peaking_only(r["fq"], r["fdh"]) for r in recs[k]], color=C[k])
    c.set_yscale("log"); c.set_ylim(3, 1000); refline(c, 150, r"online $\sigma_k$ ≈ 150 pcm", va="bottom", x_text=0.98)
    c.set_ylabel(r"$|k_{\mathrm{eff}} - 1.05|$ (pcm)")
    d.set_ylabel(r"Peaking-only fitness  $0.6F_q + 0.4F_{\Delta H}$"); d.set_ylim(1.45, 1.80)
    for ax in (c, d):
        ax.set_xticks(xt3); ax.set_xticklabels(lab3); ax.set_xlim(-0.6, 2.6)
    for ax, L, xo in ((a, "a", -0.17), (b, "b", -0.22), (c, "c", -0.17), (d, "d", -0.22)):
        panel_label(ax, L, x=xo)
    save(fig, OUT)


if __name__ == "__main__":
    main()
