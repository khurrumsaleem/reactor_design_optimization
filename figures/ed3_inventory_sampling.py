"""Extended Data Fig. 3 (label fig:ed_inventory_sampling) | Inventory
trajectories and candidate occurrence under full-space sampling.

a  evaluations at the first batch containing a candidate with >= 20 Gd rods
   (batch completion counts; segments = possible within-batch interval).
b  per-seed count of uninformed random candidates within ±500 pcm of target.
c, d  cumulative-best inventory trajectories of DPO and GRPO, five seeds.
e  first-passage and monotonicity statistics.   f  random-sampling counts.
Inputs: $DATA_ROOT/analysis/first_passage_per_seed.csv (analysis/first_passage.py),
oracle/uninformed_random_seed{K}_results.csv, alignment/ DPO and GRPO logs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from _style import C, apply_style, mean_sd, panel_title, save, table_panel

OUT = Path(__file__).resolve().parent / "ed3_inventory_sampling"


def inv_sequence(d, method):
    if method == "DPO":
        return d.simulation_count.to_numpy(), d.best_g_count.to_numpy()
    d = d.sort_values(["step", "group_idx"])
    fit = d.fitness.to_numpy()
    idx, b = [], 0
    for i, v in enumerate(fit):
        if v < fit[b]:
            b = i
        idx.append(b)
    return d.sim_count.to_numpy(), d.g_count.to_numpy()[idx]


def main():
    apply_style()
    plt.rcParams.update({"savefig.bbox": None, "font.size": 7.5})
    logstats, urs = D.first_passage(), D.uninformed_search()
    full, grpo = D.dpo_single(), D.grpo_single()
    assert len(urs) == 5 and all(len(d) == 2000 for d in urs)
    counts = []
    for s, d in enumerate(urs):
        near = (d.k_eff - 1.05).abs() <= .005
        inside = d.g_count.between(28, 35)
        counts.append([s, int(inside.sum()), int(near.sum()), int((inside & near).sum())])
    tot_in, tot_near = sum(r[1] for r in counts), sum(r[2] for r in counts)
    n_total = sum(len(d) for d in urs)

    fig = plt.figure(figsize=(7.2, 8.6))
    g = fig.add_gridspec(3, 2, height_ratios=[1.7, 2, 1.75], left=.09, right=.98, top=.95, bottom=.065, hspace=.7, wspace=.45)
    a, b = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1])
    for i, k in enumerate(["DPO", "GRPO", "GA"]):
        d = logstats[logstats.method == k]
        for j, r in enumerate(d.itertuples()):
            yy = i + (j - 2) * .10
            a.plot([r.first_ge20_eval_lower, r.first_ge20_eval_upper], [yy, yy], color=C[k], lw=1)
            a.scatter(r.first_ge20_eval_upper, yy, s=17, color=C[k])
            a.annotate(str(r.seed), (r.first_ge20_eval_upper, yy), xytext=(4, 0), textcoords="offset points",
                       fontsize=6, va="center")
    a.set(yticks=range(3), yticklabels=["DPO", "GRPO", "GA"], xlabel="Evaluations at first qualifying batch", xlim=(0, 650))
    a.invert_yaxis()
    panel_title(a, "a", "First generated candidate with Gd ≥ 20")
    b.bar(range(5), [r[2] for r in counts], color=C["URS"], width=.65)
    for s, _, v, _ in counts:
        b.text(s, v + .25, str(v), ha="center", fontsize=8)
    b.set(xticks=range(5), xlabel="Random-search seed (2,000 candidates each)", ylabel="Candidates within ±500 pcm",
          ylim=(0, max(r[2] for r in counts) + 2))
    panel_title(b, "b", "Full-space random sampling")

    for ax, ds, k, letter in ((fig.add_subplot(g[1, 0]), full, "DPO", "c"), (fig.add_subplot(g[1, 1]), grpo, "GRPO", "d")):
        for s, d in enumerate(ds):
            x, y = inv_sequence(d, k)
            ax.step(x, y, where="post", lw=.8, alpha=.8, color=plt.get_cmap("Blues" if k == "DPO" else "Oranges")(.4 + .13 * s),
                    label=f"Seed {s}")
            chk = logstats[(logstats.method == k) & (logstats.seed == s)].iloc[0]
            assert max(y) == chk.maximum_cumulative_best_gd and y[-1] == chk.final_cumulative_best_gd
            assert (np.diff(y) < 0).sum() == chk.downward_changes
        ax.axhline(16, color="#999", ls=":", lw=.7)
        ax.set(xlim=(0, 2000), ylim=(0, 40), xlabel="OpenMC evaluations", ylabel="Cumulative-best Gd inventory")
        ax.legend(ncol=3, fontsize=6, loc="lower right")
        panel_title(ax, letter, k + " inventory trajectories")

    rows = []
    for k in ["DPO", "GRPO", "GA"]:
        d = logstats[logstats.method == k]
        rows.append([k, mean_sd(d.first_ge20_eval_upper, 1), f"{(~d.nondecreasing).sum()}/5", f"{d.overshot_final.sum()}/5",
                     f"{d.maximum_cumulative_best_gd.min()}–{d.maximum_cumulative_best_gd.max()}",
                     f"{d.final_cumulative_best_gd.min()}–{d.final_cumulative_best_gd.max()}"])
    sub = g[2, :].subgridspec(2, 1, height_ratios=[1, .85], hspace=.7)
    table_panel(fig.add_subplot(sub[0, 0]), "e", "First passage and inventory dynamics", rows,
                ["Method", "First batch end\n(mean ± SD)", "Any decrease\n(runs)", "Max > final\n(runs)", "Maximum Gd\n(range)", "Final Gd\n(range)"],
                widths=[.11, .24, .17, .17, .16, .15], fontsize=6.8)
    rr = [["Candidates: Gd 28–35"] + [str(r[1]) for r in counts] + [f"{tot_in} / {n_total:,} ({100 * tot_in / n_total:.2f}%)"],
          ["Candidates: within ±500 pcm"] + [str(r[2]) for r in counts] + [f"{tot_near} / {n_total:,} ({100 * tot_near / n_total:.2f}%)"]]
    both = sum(r[3] for r in counts)
    title = ("Random sampling: all target-region candidates contained 28–35 Gd rods" if both == tot_near
             else f"Random sampling: {both} of {tot_near} target-region candidates contained 28–35 Gd rods")
    table_panel(fig.add_subplot(sub[1, 0]), "f", title, rr,
                ["Count", "Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4", "Total"],
                widths=[.30, .085, .085, .085, .085, .085, .275], fontsize=6.7)
    fig.text(.09, .014, "First passage uses batch-completion counts (DPO: 2; GRPO: 4; GA: 40 candidates).\n"
             "Gd ≥ 20 is an inventory threshold; target-region counts use evaluated |k − 1.05| ≤ 0.005.", fontsize=6.6)
    save(fig, OUT)


if __name__ == "__main__":
    main()
