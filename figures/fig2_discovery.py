"""Fig. 2 | Physics alignment identifies an inventory regime absent from training (n = 5 seeds).

Merges the former fig2_dynamics (a-c) and fig3_designs (d-l) into one full-page figure
so that Results 2.2 refers to a single display item.

a-c  cumulative best composite fitness (log y), Gd inventory and k_eff of the
     cumulative-best design vs OpenMC evaluations.  Solid = CPT+SFT base,
     dashed = SFT-only base; GA baselines in green/purple.
d-i  best-of-seed lattices: 16-Gd and 24-Gd symmetric references, GA (Gd=16),
     GA (unconstrained), DPO, GRPO.  Pins drawn to scale; guide tubes hollow.
j    per-seed best composite fitness (log y); dotted lines = references
k    criticality error |k_eff - 1.05| in pcm (log y); filled = online 4e5-history
     estimate, hollow = post-hoc 2e7-history re-evaluation; dotted = online sigma_k
l    peaking-only fitness 0.6 Fq + 0.4 FdH; dotted = references
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
                    draw_lattice, lattice_legend_handles, panel_label, refline, save)

OUT = Path(__file__).resolve().parent / "fig2_discovery"
SIGMA_ONLINE = 150.0  # pcm, mean OpenMC-reported s.d. at 4e5 histories

SERIES = [  # key, loader, color, linestyle, label
    ("GA16", lambda: [D.ga_steps(d) for d in D.ga(True)], C["GA16"], "-", LABEL["GA16"]),
    ("GA", lambda: [D.ga_steps(d) for d in D.ga(False)], C["GA"], "-", LABEL["GA"]),
    ("DPO", lambda: [D.dpo_steps(d) for d in D.dpo_single()], C["DPO"], "-", LABEL["DPO"]),
    ("DPO_s", lambda: [D.dpo_steps(d) for d in D.dpo_single("sft_only")], C["DPO"], "--", "DPO (SFT only)"),
    ("GRPO", lambda: [D.grpo_steps(d) for d in D.grpo_single()], C["GRPO"], "-", LABEL["GRPO"]),
    ("GRPO_s", lambda: [D.grpo_steps(d) for d in D.grpo_single("sft_only")], C["GRPO"], "--", "GRPO (SFT only)"),
]


def collect():
    data = {k: f() for k, f, *_ in SERIES}
    nu = D.nuscale_records()
    recs = {
        "NU16": [nu["NU16"]], "NU24": [nu["NU24"]],
        "GA16": D.best_records(data["GA16"]),
        "GA": D.best_records(data["GA"]),
        "DPO": D.best_records(data["DPO"]),
        "GRPO": D.best_records(data["GRPO"]),
    }
    hs = D.high_stat()
    recheck = {m: hs.loc[hs["method"] == m, "recheck_dk_pcm"].to_numpy(float) for m in ("DPO", "GRPO")}
    return data, recs, recheck


def lattice_caption(ax, key, r):
    ax.text(0.5, -0.03, LABEL[key], transform=ax.transAxes, ha="center", va="top",
            fontsize=7.0, fontweight="bold", color=C[key])
    ax.text(0.5, -0.13,
            f"fit {r['fit']:.3f}  ·  Gd {r['gd']}  ·  $k_{{\\mathrm{{eff}}}}$ {r['k']:.4f}\n"
            f"$F_q$ {r['fq']:.2f}  ·  $F_{{\\Delta H}}$ {r['fdh']:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.1, color="#333333", linespacing=1.35)


def main():
    apply_style()
    data, recs, recheck = collect()

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.7))
    # rows: trajectories | legend spacer | lattices | caption spacer | decomposition
    outer = fig.add_gridspec(5, 1, height_ratios=[1.0, 0.48, 1.95, 0.24, 1.05], hspace=0.10,
                             top=0.985, bottom=0.045, left=0.075, right=0.985)

    # ---------------------------------------------------------------- a-c
    top = outer[0].subgridspec(1, 3, wspace=0.42)
    axes = [fig.add_subplot(top[0, k]) for k in range(3)]
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
        panel_label(ax, L, x=-0.30)
    handles = [Line2D([], [], color=col, linestyle=ls, linewidth=1.4, label=lab) for _, _, col, ls, lab in SERIES]
    handles.append(ci_proxy())
    # legend for a-c sits just below the trajectory row
    sp = outer[1].get_position(fig)
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, sp.y1 - 0.030), ncol=4, frameon=False,
               fontsize=6.6, handlelength=1.8, columnspacing=1.2)

    # ---------------------------------------------------------------- d-i
    mid = outer[2].subgridspec(2, 3, hspace=0.40, wspace=0.10)
    order = ["NU16", "NU24", "GA16", "GA", "DPO", "GRPO"]
    for i, key in enumerate(order):
        ax = fig.add_subplot(mid[i // 3, i % 3])
        best = min(recs[key], key=lambda r: r["fit"])
        draw_lattice(ax, D.parse_grid(best["grid"]), C[key])
        lattice_caption(ax, key, best)
        panel_label(ax, "defghi"[i], x=-0.02, y=1.0)
    fig.legend(handles=lattice_legend_handles(), loc="lower center",
               bbox_to_anchor=(0.5, sp.y0), ncol=3, frameon=False, fontsize=6.8)

    # ---------------------------------------------------------------- j-l
    bot = outer[4].subgridspec(1, 3, wspace=0.42)
    g, h, i_ = [fig.add_subplot(bot[0, k]) for k in range(3)]
    xkeys = ["GA16", "GA", "DPO", "GRPO"]
    xt = np.arange(len(xkeys))
    tick_labels = ["GA\nGd = 16", "GA\nfree", "DPO", "GRPO"]

    for x, key in zip(xt, xkeys):
        dots_with_mean(g, x, [r["fit"] for r in recs[key]], color=C[key])
    g.set_yscale("log"); g.set_ylim(1.2, 20)
    g.set_yticks([1.5, 2, 3, 5, 10, 15]); g.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    refline(g, recs["NU16"][0]["fit"], "16-Gd ref.", color=C["NU16"], va="bottom")
    refline(g, recs["NU24"][0]["fit"], "24-Gd ref.", color=C["NU24"], va="bottom")
    g.set_ylabel("Best composite fitness")

    for x, key in zip(xt, xkeys):
        vals = [D.dk_pcm(r["k"]) for r in recs[key]]
        dots_with_mean(h, x - (0.12 if key in recheck else 0), vals, color=C[key])
        if key in recheck:
            dots_with_mean(h, x + 0.20, recheck[key], color=C[key], hollow=True, mean_width=0.14, jitter=0.05)
    h.set_yscale("log"); h.set_ylim(2.5, 3e4)
    refline(h, D.dk_pcm(recs["NU16"][0]["k"]), "16-Gd ref.", color=C["NU16"], va="bottom")
    refline(h, D.dk_pcm(recs["NU24"][0]["k"]), "24-Gd ref.", color=C["NU24"], va="bottom")
    refline(h, SIGMA_ONLINE, r"online $\sigma_k$", va="top", x_text=0.02, ha="left")
    h.set_ylabel(r"$|k_{\mathrm{eff}} - 1.05|$ (pcm)")

    for x, key in zip(xt, xkeys):
        dots_with_mean(i_, x, [D.peaking_only(r["fq"], r["fdh"]) for r in recs[key]], color=C[key])
    refline(i_, D.peaking_only(recs["NU16"][0]["fq"], recs["NU16"][0]["fdh"]), "16-Gd ref.", color=C["NU16"], va="top", x_text=0.02, ha="left")
    refline(i_, D.peaking_only(recs["NU24"][0]["fq"], recs["NU24"][0]["fdh"]), "24-Gd ref.", color=C["NU24"], va="bottom", x_text=0.02, ha="left")
    i_.set_ylabel(r"Peaking-only fitness ($0.6F_q + 0.4F_{\Delta H}$)")
    i_.set_ylim(1.45, 1.85)

    for ax, L in zip((g, h, i_), "jkl"):
        ax.set_xticks(xt); ax.set_xticklabels(tick_labels); ax.set_xlim(-0.6, len(xkeys) - 0.4)
        panel_label(ax, L, x=-0.30)
        ax.tick_params(axis="x", labelsize=6.4)
    hh = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="#555", markeredgecolor="#555", markersize=4, label="online, 4 × 10$^5$ histories"),
          Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#555", markersize=4, label="re-evaluated, 2 × 10$^7$")]
    h.legend(handles=hh, loc="lower left", bbox_to_anchor=(-0.02, -0.02), fontsize=5.8, handletextpad=0.3, labelspacing=0.3)
    save(fig, OUT)


if __name__ == "__main__":
    main()
