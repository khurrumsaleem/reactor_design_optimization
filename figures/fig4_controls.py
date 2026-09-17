"""Fig. 4 (manuscript Fig. 3, label fig:controls) | Objective-dependent
adaptation and fixed-search controls, five seeds throughout.

a  chosen-design Gd inventory during DPO alignment (25-step rolling mean):
   full objective vs criticality penalty removed; thin = seeds, thick = mean.
b  matched seeds: final-100-step mean inventory under both objectives.
c-e best-of-budget composite fitness, criticality error and peaking-only
   fitness for GA (Gd = 16), GA (free), uninformed random search (0-264),
   informed random search (20-40), DPO and GRPO.
f  objective-removal statistics per seed.   g  best-of-budget metrics table.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from _style import (C, apply_style, dots_with_mean, mean_sd, panel_title, rolling_mean, save, table_panel)

OUT = Path(__file__).resolve().parent / "fig4_controls"
ROLL = 25


def best(dfs, dpo=False):
    out = []
    for d in dfs:
        pre = "chosen_" if dpo else ""
        b = d.loc[d[pre + "fitness"].idxmin()]
        out.append({"fit": b[pre + "fitness"], "error": abs(b[pre + "k_eff"] - 1.05) * 1e5,
                    "peak": 0.6 * b[pre + "fq"] + 0.4 * b[pre + "fdh"]})
    return out


def main():
    apply_style()
    plt.rcParams.update({"savefig.bbox": None, "font.size": 7.5})
    full, no = D.dpo_single(), D.no_penalty_all()
    grpo, ga, ga16 = D.grpo_single(), D.ga(False), D.ga(True)
    urs, irs = D.uninformed_search(), D.informed_search()
    for d in full + no:
        assert len(d) == 1000 and d.simulation_count.iloc[-1] == 2000
    for d in grpo + ga + ga16 + urs + irs:
        assert len(d) == 2000
    assert all(len(v) == 5 for v in (full, no, grpo, ga, ga16, urs, irs)), "five seeds required"
    recs = {"GA16": best(ga16), "GA": best(ga), "URS": best(urs), "RS": best(irs),
            "DPO": best(full, True), "GRPO": best(grpo)}

    fig = plt.figure(figsize=(7.2, 8.25))
    g = fig.add_gridspec(4, 6, height_ratios=[2.1, 1.65, 1.32, 1.25], left=.09, right=.98, top=.96,
                         bottom=.065, hspace=.60, wspace=1.1)
    a, b = fig.add_subplot(g[0, :3]), fig.add_subplot(g[0, 3:])
    axes = [fig.add_subplot(g[1, i:i + 2]) for i in (0, 2, 4)]

    for ds, color, label in ((full, C["DPO"], "Full objective"), (no, C["NOPEN"], "No criticality penalty")):
        ys = np.array([rolling_mean(d.chosen_g_count, ROLL) for d in ds])
        x = ds[0].simulation_count.to_numpy()
        for y in ys:
            a.plot(x, y, color=color, alpha=.22, lw=.65)
        a.plot(x, ys.mean(axis=0), color=color, lw=1.6, label=label)
    a.axhline(16, color="#888", ls=":", lw=.8)
    a.set(xlim=(0, 2000), ylim=(0, None), xlabel="OpenMC evaluations",
          ylabel=f"Chosen-design Gd inventory\n({ROLL}-step rolling mean)")
    a.legend(fontsize=6.5, loc="upper left")
    panel_title(a, "a", "Objective-dependent inventory trajectories")

    fv = np.array([d.chosen_g_count.tail(100).mean() for d in full])
    nv = np.array([d.chosen_g_count.tail(100).mean() for d in no])
    for s, (v, w) in enumerate(zip(fv, nv)):
        off = (s - 2) * .06
        b.plot([off, 1 + off], [v, w], c="#B7B7B7", lw=.8, zorder=1)
        b.scatter([off, 1 + off], [v, w], c=[C["DPO"], C["NOPEN"]], s=23, zorder=3)
        b.annotate(str(s), (off, v), xytext=(-4, 5), textcoords="offset points", fontsize=6.5)
        b.annotate(str(s), (1 + off, w), xytext=(-2, 5), textcoords="offset points", fontsize=6.5)
    b.set(xticks=[0, 1], xticklabels=["Full objective", "No penalty"], xlim=(-.35, 1.35), ylim=(-4, 65),
          ylabel="Mean Gd inventory\n(final 100 steps)")
    panel_title(b, "b", "Matched seeds (labels: seed ID)")

    keys = list(recs)
    labs = ["GA\n16 Gd", "GA\nfree", "U-rand.", "I-rand.", "DPO", "GRPO"]
    spec = [("fit", "c", "Composite objective", "Best composite fitness"),
            ("error", "d", "Multiplication-factor error", r"$|k_{\mathrm{eff}}-1.05|$ (pcm)"),
            ("peak", "e", "Power peaking", "Peaking-only fitness")]
    for ax, (metric, letter, title, ylabel) in zip(axes, spec):
        ks = keys if metric == "fit" else keys[1:]
        ls = labs if metric == "fit" else labs[1:]
        for i, k in enumerate(ks):
            dots_with_mean(ax, i, [r[metric] for r in recs[k]], color=C[k], ms=3, jitter=.08, mean_width=.20)
        ax.set(xticks=range(len(ks)), xticklabels=ls, xlim=(-.55, len(ks) - .45), ylabel=ylabel)
        ax.tick_params(axis="x", labelsize=5.7)
        if metric == "fit":
            ax.set_yscale("log"); ax.set_ylim(1.3, 15); ax.set_yticks([1.5, 2, 5, 10])
            ax.yaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
        elif metric == "error":
            ax.set_yscale("log"); ax.set_ylim(2, 650); ax.set_yticks([10, 100, 500])
            ax.yaxis.set_major_formatter(ScalarFormatter()); ax.axhline(150, color="#999", ls=":", lw=.7)
        else:
            ax.set_ylim(1.50, 1.82)
        panel_title(ax, letter, title)

    peak100 = [(0.6 * d.chosen_fq + 0.4 * d.chosen_fdh).tail(100).mean() for d in no]
    peak_last = [0.6 * d.chosen_fq.iloc[-1] + 0.4 * d.chosen_fdh.iloc[-1] for d in no]
    peak_min = [d.chosen_fitness.min() for d in no]
    rows = [[str(s), f"{fv[s]:.2f}", f"{nv[s]:.2f}", f"{peak100[s]:.4f}", f"{peak_last[s]:.4f}", f"{peak_min[s]:.4f}"]
            for s in range(5)]
    rows.append(["Mean ± SD", mean_sd(fv), mean_sd(nv), mean_sd(peak100, 4), mean_sd(peak_last, 4), mean_sd(peak_min, 4)])
    table_panel(fig.add_subplot(g[2, :]), "f", "Objective-removal results (five seeds)", rows,
                ["Seed", "Full: Gd\nfinal 100", "No penalty: Gd\nfinal 100", "No penalty: peak\nfinal 100",
                 "No penalty: peak\nfinal step", "No penalty: peak\nrun minimum"],
                widths=[.12, .16, .17, .19, .18, .18], fontsize=6.8)
    names = ["GA (16 Gd)", "GA (free inventory)", "Random (0–264)", "Random (20–40)", "DPO", "GRPO"]
    rows = [[n, mean_sd([r["fit"] for r in recs[k]], 3), mean_sd([r["error"] for r in recs[k]], 1),
             mean_sd([r["peak"] for r in recs[k]], 3)] for k, n in zip(keys, names)]
    table_panel(fig.add_subplot(g[3, :]), "g", "Best-of-budget design metrics (mean ± SD across five seeds)", rows,
                ["Method", "Composite fitness", "Error (pcm)", "Peaking-only fitness"], widths=[.31, .23, .23, .23])
    fig.text(.09, .018, "a: thin lines, individual seeds; thick lines, means.  c–e: seed points, mean bars and ±1 SD.\n"
             "2,000 evaluations per run. U-rand.: uniform Gd 0–264; I-rand.: uniform Gd 20–40.", fontsize=6.6)
    save(fig, OUT)


if __name__ == "__main__":
    main()
