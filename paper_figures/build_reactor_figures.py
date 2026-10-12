#!/usr/bin/env python3
"""Rebuild the main reactor figures from completed, archived experiments.

Only reads source files. No simulations, fitting or resampling are performed.
Usage
    python build_reactor_figures.py --source-root /path/to/reactor_genome \
        --output-dir "/path/to/reactor_genome/NMI_submit/1st appeal/figure"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


BLUE = "#387CA3"
ORANGE = "#BD713D"
GRAY = "#858A8D"
LIGHT = "#C7CBCE"
INK = "#252525"
SEEDS = range(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-figure2", action="store_true", help="Update Figure 2 without rewriting Figure 3 artwork")
    args = parser.parse_args()
    root, out = args.source_root.resolve(), args.output_dir.resolve()
    authorized = (root / "NMI_submit/1st appeal/figure").resolve()
    if out == root or (root in out.parents and out != authorized and authorized not in out.parents):
        raise ValueError("Outputs inside the source project are permitted only in NMI_submit/1st appeal/figure")
    out.mkdir(parents=True, exist_ok=True)
    dataout = out / "reactor_data"
    dataout.mkdir(exist_ok=True)
    provenance = {}

    def read(rel):
        path = root / rel
        payload = path.read_bytes()
        provenance[rel] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        return pd.read_csv(path)

    full = [read(f"after/training/dpo/single_target/dpo_single_target_seed{s}_results.csv") for s in SEEDS]
    grpo = [read(f"after/training/grpo/single/grpo_cpt_sft_single_seed{s}_results.csv") for s in SEEDS]
    no = [read(f"alam_additional_expriment/results/no_penalty_dpo_seed{s}_results.csv") for s in SEEDS]
    uniform = [read(f"alam_additional_expriment/results/uninformed_random_seed{s}_results.csv") for s in SEEDS]
    informed = [read("professor/Test/t3_random_search/random_search_results.csv")] + [
        read(f"professor/Test/t3_random_search/oracle_seed{s}_results.csv") for s in range(1, 5)]
    ga = [read(f"after/ga/ga_baseline_single_seed{s}_results.csv") for s in SEEDS]
    ga16 = [read(f"after/ga/ga_baseline_single_gd16_seed{s}_results.csv") for s in SEEDS]
    samples = read("professor/Test/t6_generation_yield/t6_samples_v2.csv")
    evaluated = read("professor/Test/t6_generation_yield/t6_openmc_subsample.csv")
    high = read("chakraborty/results/openmc_high_stat_all-per-seed_summary.csv")
    no_summary = read("alam_additional_expriment/results/no_penalty_seed_summary.csv")
    log_summary = read("alam_additional_expriment/results/existing_log_per_seed.csv")

    assert all(len(d) == 1000 and d.simulation_count.iloc[-1] == 2000 for d in full + no)
    assert all(len(d) == 2000 for d in grpo + uniform + informed + ga + ga16)
    assert samples.groupby(["algo", "seed"]).size().eq(1000).all() and len(samples) == 10000
    assert evaluated.groupby(["algo", "seed"]).size().eq(50).all() and len(evaluated) == 500
    assert len(high) == 10 and high.replicates.eq(1).all()
    assert samples.grid.str.count("g").eq(samples.gd_count).all()
    assert samples.grid.str.len().eq(289).all()

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.7, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "figure.dpi": 150,
    })

    def panel(ax, letter, title):
        ax.text(-.18, 1.09, letter, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", ha="left")
        ax.set_title(title, loc="left", pad=13)
        ax.tick_params(pad=3)

    def save(fig, name):
        if args.only_figure2 and name == "fig3_reactor_controls":
            plt.close(fig)
            return
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches=None)
        plt.close(fig)

    trajectory_rows = []
    paths = {}
    for method, dfs in [("DPO", full), ("GRPO", grpo)]:
        method_paths = []
        for seed, d in enumerate(dfs):
            if method == "DPO":
                x = d.simulation_count.to_numpy()
                fit = d.chosen_fitness.to_numpy()
                inventory = d.chosen_g_count.to_numpy()
            else:
                d = d.sort_values(["step", "group_idx"]).reset_index(drop=True)
                # Update only after every member of a GRPO group is evaluated.
                chosen = d.loc[d.groupby("step").fitness.idxmin()].sort_values("step")
                x = chosen.sim_count.to_numpy()
                fit = chosen.fitness.to_numpy()
                inventory = chosen.g_count.to_numpy()
            idx = np.zeros(len(fit), dtype=int)
            best = 0
            for j in range(len(fit)):
                if fit[j] < fit[best]:
                    best = j
                idx[j] = best
            y = inventory[idx]
            method_paths.append((x, y))
            log = log_summary[(log_summary.method == method) & (log_summary.seed == seed)].iloc[0]
            assert y[-1] == log.final_cumulative_best_gd
            # GRPO group-completion view preserves endpoints; within-group transient maxima
            # are retained in the source logs and existing supplementary analyses.
            for j in range(len(x)):
                trajectory_rows.append(dict(method=method, seed=seed, evaluations=int(x[j]),
                                            best_objective=float(fit[idx[j]]), best_inventory=int(y[j])))
        paths[method] = method_paths
    pd.DataFrame(trajectory_rows).to_csv(dataout / "fig2a_selected_best_trajectories.csv", index=False)

    high = high.copy()
    high["signed_target_error_pcm"] = (high.recheck_k_mean - 1.05) * 1e5
    high.to_csv(dataout / "fig2b_selected_high_statistics.csv", index=False)
    assert high.recheck_dk_pcm.le(500).all()

    distrows = []
    for method in ("DPO", "GRPO"):
        values = samples.loc[samples.algo == method, "gd_count"]
        counts = values.value_counts().reindex(range(265), fill_value=0).sort_index()
        for inventory, n in counts.items():
            distrows.append(dict(method=method, inventory=int(inventory), count=int(n),
                                 fraction=float(n / len(values)),
                                 cumulative_fraction=float(counts.loc[:inventory].sum() / len(values))))
    distribution = pd.DataFrame(distrows)
    distribution.to_csv(dataout / "fig2c_generation_inventory_distribution.csv", index=False)
    samples.drop(columns="grid").to_csv(dataout / "final_generation_sample_metrics.csv", index=False)
    evaluated["within_500_pcm"] = (evaluated.k_eff - 1.05).abs().le(.005)
    evaluated.to_csv(dataout / "final_generation_evaluations.csv", index=False)
    ur_hits = [int((d.k_eff - 1.05).abs().le(.005).sum()) for d in uniform]
    assert ur_hits == [6, 11, 10, 5, 4]
    assert int(evaluated.within_500_pcm.sum()) == 101
    incidences = [
        dict(sampler="Aligned policies pooled", method="Pooled", hits=101, n=500),
        *[dict(sampler=f"{m} policy", method=m,
               hits=int(evaluated.loc[evaluated.algo == m, "within_500_pcm"].sum()), n=250)
          for m in ("DPO", "GRPO")],
        dict(sampler="Informed random", method="Informed", hits=sum(int((d.k_eff - 1.05).abs().le(.005).sum()) for d in informed), n=10000),
        dict(sampler="Inventory-uniform random", method="Uniform", hits=sum(ur_hits), n=10000),
    ]
    incidence = pd.DataFrame(incidences)
    incidence["percent"] = 100 * incidence.hits / incidence.n
    incidence.to_csv(dataout / "fig2d_target_window_incidence.csv", index=False)
    conditional_rows = []
    for label, frame, count_col in [
        ("Inventory-uniform random", pd.concat(uniform), "g_count"),
        ("Informed random", pd.concat(informed), "g_count"),
        ("Aligned policies pooled", evaluated, "gd"),
        *[(f"{method} policy", evaluated[evaluated.algo == method], "gd") for method in ("DPO", "GRPO")],
    ]:
        hits = (frame.k_eff - 1.05).abs().le(.005)
        inside = frame[count_col].between(28, 35)
        conditional_rows.append(dict(sampler=label, n_all=len(frame), hits_all=int(hits.sum()),
                                     n_28_35=int(inside.sum()), hits_28_35=int((hits & inside).sum()),
                                     percent_given_28_35=float(100 * hits[inside].mean()),
                                     hits_outside_28_35=int((hits & ~inside).sum())))
    pd.DataFrame(conditional_rows).to_csv(dataout / "inventory_28_35_conditional_incidence.csv", index=False)
    seed_count_rows = []
    for label, frames in [("Inventory-uniform random", uniform), ("Informed random", informed)]:
        for seed, frame in enumerate(frames):
            seed_count_rows.append(dict(sampler=label, seed=seed, n=len(frame),
                                        hits=int((frame.k_eff - 1.05).abs().le(.005).sum())))
    for (method, seed), frame in evaluated.groupby(["algo", "seed"]):
        seed_count_rows.append(dict(sampler=f"{method} policy", seed=int(seed), n=len(frame),
                                    hits=int((frame.k_eff - 1.05).abs().le(.005).sum())))
    pd.DataFrame(seed_count_rows).to_csv(dataout / "target_incidence_by_seed.csv", index=False)

    # Figure 2 follows the evidence chain from search dynamics to selected quality
    # and then to the complete corrected generation distribution.
    fig = plt.figure(figsize=(7.2, 5.55))
    g = fig.add_gridspec(2, 2, left=.095, right=.975, bottom=.115, top=.91,
                        hspace=.64, wspace=.40)
    a, b, c, d = [fig.add_subplot(g[i, j]) for i in range(2) for j in range(2)]
    for method, color in [("DPO", BLUE), ("GRPO", ORANGE)]:
        for x, y in paths[method]:
            a.step(x, y, where="post", lw=.85, alpha=.65, color=color)
        a.plot([], [], color=color, lw=1.4, label=method)
    a.axhline(16, color=GRAY, lw=.8, ls=(0, (3, 2)))
    a.text(.98, 16, "Corpus 16", transform=a.get_yaxis_transform(), va="top", ha="right",
           fontsize=7.5, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    a.set(xlim=(0, 2050), ylim=(0, 41), xlabel="OpenMC evaluations",
          ylabel="Absorbers in best-so-far layout", xticks=[0, 1000, 2000], yticks=[0, 10, 20, 30, 40])
    a.legend(loc="lower right", frameon=False, ncol=2, handlelength=1.3, columnspacing=1)
    panel(a, "a", "Selection moves beyond the corpus")

    b.axhspan(-500, 500, color="#F2F3F4", zorder=0)
    b.axhline(-500, lw=.6, ls=(0, (3, 2)), color=GRAY)
    b.axhline(500, lw=.6, ls=(0, (3, 2)), color=GRAY)
    b.axhline(0, lw=.6, color=LIGHT, zorder=0)
    for i, (method, color) in enumerate([("DPO", BLUE), ("GRPO", ORANGE)]):
        sub = high[high.method == method].sort_values("training_seed")
        xx = np.arange(5) + 1 + i * 6
        b.errorbar(xx, sub.signed_target_error_pcm, yerr=sub.mean_openmc_k_std_pcm,
                   fmt="o", color=color, ms=3.5, elinewidth=.8, capsize=2, mew=.65)
    b.set(xlim=(.2, 11.8), ylim=(-580, 580), ylabel=r"$k - 1.05$ (pcm)",
          xticks=[3, 9], xticklabels=["DPO", "GRPO"], yticks=[-500, 0, 500])
    b.text(.97, .85, "All 10 layouts within 500 pcm", transform=b.transAxes,
           ha="right", va="top", fontsize=7.5)
    b.set_xlabel(r"Selected layouts reevaluated with $2\times10^7$ histories", fontsize=7.4)
    panel(b, "b", "Independent target confirmation")

    for method, color in [("DPO", BLUE), ("GRPO", ORANGE)]:
        sub = distribution[distribution.method == method]
        c.step(sub.inventory, 100 * sub.cumulative_fraction, where="post", color=color,
               lw=1.4, label=method)
    c.axvline(16, color=GRAY, lw=.8, ls=(0, (3, 2)))
    c.text(16, 96, "16", fontsize=7.5, ha="right", va="top",
           bbox=dict(facecolor="white", edgecolor="none", pad=1))
    c.set(xlim=(0, 175), ylim=(0, 103), xticks=[0, 50, 100, 150], yticks=[0, 50, 100],
          xlabel="Absorbers in corrected output", ylabel="Cumulative fraction of outputs (%)")
    c.text(.98, .08, "5,000 outputs per method", transform=c.transAxes, ha="right", fontsize=7.5)
    c.legend(loc="center right", frameon=False)
    panel(c, "c", "Final-generation inventory distribution")

    labels = ["Aligned pooled", "DPO", "GRPO", "Informed random", "Inventory-uniform"]
    colors = [INK, BLUE, ORANGE, "#62686C", GRAY]
    for y, row in incidence.iterrows():
        d.barh(y, row.percent, color=colors[y], height=.53)
        d.text(row.percent + .7, y,
               f"{row.percent:.1f}%  ({row.hits}/{row.n:,})" if y < 3 else f"{row.percent:.2f}%  ({row.hits}/{row.n:,})",
               va="center", fontsize=7.2)
    d.set(yticks=range(5), yticklabels=labels, xlim=(0, 43), ylim=(4.6, -.6),
          xticks=[0, 10, 20, 30, 40], xlabel="Outputs within 500 pcm (%)")
    d.tick_params(axis="y", length=0, labelsize=7.5)
    d.spines.left.set_visible(False)
    panel(d, "d", "Unselected target-window incidence")
    fig.text(.095, .028, "Five seeds per method. Selected-layout error bars show one transport standard deviation.", fontsize=7.2)
    save(fig, "fig2_reactor_discovery")

    # Full objective and objective-removal data retain the high-inventory fifth seed.
    rows, rolling = [], []
    for seed, (fd, nd) in enumerate(zip(full, no)):
        for label, frame in [("Full objective", fd), ("Target penalty removed", nd)]:
            vals = frame.chosen_g_count
            smoothed = vals.rolling(25, min_periods=1).mean()
            for x, value in zip(frame.simulation_count, smoothed):
                rolling.append(dict(condition=label, seed=seed, evaluations=int(x), rolling25_inventory=float(value)))
            rows.append(dict(condition=label, seed=seed,
                             final100_inventory=float(vals.tail(100).mean())))
        r = no_summary[no_summary.seed == seed].iloc[0]
        assert np.isclose(nd.chosen_g_count.tail(100).mean(), r.final100_inventory_mean)
        assert np.isclose(fd.chosen_g_count.tail(100).mean(), r.full_objective_final100_inventory_mean)
    endpoint = pd.DataFrame(rows)
    rolling = pd.DataFrame(rolling)
    endpoint.to_csv(dataout / "fig3b_objective_removal_endpoint.csv", index=False)
    rolling.to_csv(dataout / "fig3a_objective_removal_trajectories.csv", index=False)

    results = []
    method_specs = [("GA fixed 16", ga16, False), ("GA free", ga, False),
                    ("Inventory-uniform", uniform, False), ("Informed random", informed, False),
                    ("DPO", full, True), ("GRPO", grpo, False)]
    for method, frames, is_dpo in method_specs:
        for seed, frame in enumerate(frames):
            p = "chosen_" if is_dpo else ""
            r = frame.loc[frame[p + "fitness"].idxmin()]
            results.append(dict(method=method, seed=seed, objective=float(r[p + "fitness"]),
                                target_error_pcm=float(abs(r[p + "k_eff"] - 1.05) * 1e5),
                                peaking=float(.6 * r[p + "fq"] + .4 * r[p + "fdh"]),
                                inventory=int(r[p + "g_count"])))
    best = pd.DataFrame(results)
    best.to_csv(dataout / "fig3cde_best_of_budget.csv", index=False)
    summary = best.groupby("method", sort=False)[["objective", "target_error_pcm", "peaking", "inventory"]].agg(["mean", "std"])
    summary.to_csv(dataout / "best_of_budget_summary.csv")

    fig = plt.figure(figsize=(7.2, 6.15))
    top = fig.add_gridspec(1, 2, left=.10, right=.975, top=.92, bottom=.595,
                          wspace=.45)
    a, b = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    for condition, color, ls in [("Full objective", BLUE, "-"), ("Target penalty removed", GRAY, "--")]:
        matrix = []
        for seed in SEEDS:
            sub = rolling[(rolling.condition == condition) & (rolling.seed == seed)]
            a.plot(sub.evaluations, sub.rolling25_inventory, color=color, lw=.55, alpha=.4, ls=ls)
            matrix.append(sub.rolling25_inventory.to_numpy())
        a.plot(sub.evaluations, np.mean(matrix, axis=0), color=color, lw=1.6, ls=ls, label=condition)
    a.axhline(16, color=LIGHT, lw=.7, ls=(0, (3, 2)))
    a.set(xlim=(0, 2000), ylim=(-2, 90), xticks=[0, 1000, 2000], yticks=[0, 30, 60, 90],
          xlabel="OpenMC evaluations", ylabel="Chosen-layout absorber count")
    a.legend(loc="upper left", frameon=False, fontsize=7.1, handlelength=1.6, labelspacing=.4)
    panel(a, "a", "Inventory responds to the objective")

    fv = endpoint[endpoint.condition == "Full objective"].sort_values("seed").final100_inventory.to_numpy()
    nv = endpoint[endpoint.condition == "Target penalty removed"].sort_values("seed").final100_inventory.to_numpy()
    for seed, (v, w) in enumerate(zip(fv, nv)):
        offset = (seed - 2) * .05
        b.plot([offset, 1 + offset], [v, w], color="#B7BABC", lw=.85, zorder=1)
        b.scatter([offset, 1 + offset], [v, w], s=21, c=[BLUE, GRAY], zorder=3, edgecolors="white", linewidths=.4)
        b.text(offset - .07, v + 1.8, str(seed), fontsize=6.5, ha="center")
        if seed == 4:
            b.text(1 + offset + .05, w, "Seed 4", fontsize=7.2, va="center")
    b.set(xlim=(-.35, 1.65), ylim=(-5, 64), xticks=[0, 1],
          xticklabels=["Full objective", "Target penalty\nremoved"],
          ylabel="Mean absorbers in final 100 steps", yticks=[0, 20, 40, 60])
    b.text(.97, .17, "Four seeds reach zero", transform=b.transAxes, ha="right", fontsize=7.4)
    panel(b, "b", "Matched response across five seeds")

    lower = fig.add_gridspec(1, 3, left=.175, right=.975, top=.415, bottom=.145,
                            wspace=.34, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(lower[0, i]) for i in range(3)]
    names = [m for m, _, _ in method_specs]
    colors = [GRAY, GRAY, GRAY, INK, BLUE, ORANGE]
    for ax, metric, letter, title, xlabel in zip(
            axes, ["objective", "target_error_pcm", "peaking"], "cde",
            ["Best objective", "Target error", "Power peaking"],
            ["Composite objective", "Error (pcm)", "Peaking component"]):
        for y, (method, color) in enumerate(zip(names, colors)):
            vals = best.loc[best.method == method, metric].to_numpy()
            yy = y + np.linspace(-.13, .13, len(vals))
            ax.scatter(vals, yy, s=13, color=color, edgecolors="none", zorder=3)
            mean, sd = vals.mean(), vals.std(ddof=1)
            ax.errorbar(mean, y, xerr=sd, fmt="|", markersize=11, color=color,
                        elinewidth=.85, capsize=2.5, mew=1.2, zorder=4)
        ax.set(ylim=(5.65, -.65), yticks=range(6), xlabel=xlabel)
        ax.set_yticklabels(names if metric == "objective" else [])
        ax.tick_params(axis="y", length=0, labelsize=7.5)
        ax.spines.left.set_visible(False)
        ax.axhline(3.5, color="#E5E7E8", lw=.6)
        if metric == "objective":
            ax.set_xscale("log"); ax.set_xlim(1.45, 14.8); ax.set_xticks([1.5, 2, 5, 10])
            ax.xaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
        elif metric == "target_error_pcm":
            ax.set_xscale("log"); ax.set_xlim(2, 15000); ax.set_xticks([10, 100, 1000, 10000])
            ax.set_xticklabels(["10", "100", "1,000", "10,000"]); ax.minorticks_off()
            ax.axvline(150, color=LIGHT, lw=.7, ls=(0, (3, 2)), zorder=0)
        else:
            ax.set_xlim(1.48, 1.83); ax.set_xticks([1.5, 1.6, 1.7, 1.8])
        panel(ax, letter, title)
    fig.text(.10, .055, "2,000 evaluations per run. Lower panels show individual seeds and mean ±1 SD.", fontsize=7.2)
    fig.text(.10, .032, "Inventory-uniform samples 0-264 absorbers. Informed random receives the interval 20-40 after alignment.", fontsize=7.1)
    save(fig, "fig3_reactor_controls")

    captions = r"""Figure 2. Scientific feedback finds productive designs beyond the training inventory.
a, Absorber counts of the best-so-far layouts during single-target DPO and GRPO alignment. Each line is one of five training seeds. The supervised corpus contains exactly 16 absorbers per lattice. Final selected counts range from 28 to 35 for DPO and 28 to 29 for GRPO. GRPO is displayed at group-completion times. b, Independent high-statistics reevaluation of all ten selected layouts. Signed deviations from the requested multiplication factor 1.05 are plotted with one Monte Carlo standard deviation. Every layout remains inside the 500-pcm target window. Each reevaluation uses 20 million active histories. c, Empirical cumulative distributions of absorber counts in 5,000 corrected, unselected outputs from each method, with 1,000 outputs per checkpoint at temperature 1.0. The dashed line marks the corpus inventory. d, Target-window incidence in a stratified random sample of 500 corrected policy outputs, with 50 per checkpoint, compared with 10,000 candidates from each random procedure. Aligned outputs give 101/500 hits (20.2%), informed random sampling over inventories 20-40 gives 521/10,000 (5.21%), and inventory-uniform sampling over 0-264 gives 36/10,000 (0.36%). Individual policy results are 33/250 for DPO and 68/250 for GRPO. All three sampling procedures are defined in Methods.

Figure 3. Objective dependence and information transfer explain the discovered search region.
a, Chosen-layout absorber trajectories for full-objective DPO and matched runs with the multiplication-factor penalty removed. Thin lines show individual seeds and thick lines the five-seed mean, after a 25-step rolling mean. b, Final-100-step mean inventories for each matched seed. Four removed-penalty runs reach zero inventory and seed 4 retains a mean of 39.29. c-e, Best-of-budget composite objective, absolute target error and the peaking-only component from five independent runs per method with 2,000 OpenMC evaluations per run. Points show seeds and bars show means and sample standard deviations. Inventory-uniform random search samples absorber count uniformly over 0-264 and placement uniformly conditional on count. Informed random search receives the interval 20-40 identified after alignment. It recovers objective quality comparable to DPO and GRPO. The dashed reference in d marks approximate online transport uncertainty of 150 pcm. The fixed-inventory GA uses 16 absorbers. Full per-seed measurements and supplementary trajectory analyses remain in Supplementary Information.
"""
    (out / "reactor_captions.txt").write_text(captions)
    validation = {
        "figures": ["fig2_reactor_discovery", "fig3_reactor_controls"],
        "source_files": len(provenance),
        "source_files_unchanged": all(hashlib.sha256((root / rel).read_bytes()).hexdigest() == p["sha256"] for rel, p in provenance.items()),
        "runs_per_method": 5, "online_evaluations_per_run": 2000,
        "generation_outputs": len(samples), "generation_evaluated": len(evaluated),
        "generation_target_hits": 101, "uniform_target_hits": 36,
        "uniform_total": 10000, "uniform_hits_by_seed": ur_hits,
        "informed_target_hits": int(incidence.loc[incidence.method == "Informed", "hits"].iloc[0]),
        "informed_total": 10000,
        "informed_hits_by_seed": [int((d.k_eff-1.05).abs().le(.005).sum()) for d in informed],
        "conditional_inventory_28_35": conditional_rows,
        "aligned_to_uniform_incidence_ratio": (101 / 500) / (36 / 10000),
        "generation_inventory_mean": float(samples.gd_count.mean()),
        "generation_inventory_sd": float(samples.gd_count.std(ddof=1)),
        "selected_DPO_inventory": high[high.method == "DPO"].g_count.tolist(),
        "selected_GRPO_inventory": high[high.method == "GRPO"].g_count.tolist(),
        "all_selected_high_statistics_within_500_pcm": bool(high.recheck_dk_pcm.le(500).all()),
        "no_penalty_final100_inventory": nv.tolist(),
        "full_objective_final100_inventory": fv.tolist(),
        "transport_uncertainty_is_not_training_seed_variability": True,
        "all_data_are_completed_campaigns": True,
    }
    assert validation["source_files_unchanged"]
    (out / "reactor_source_provenance.json").write_text(json.dumps({"source_root": str(root), "files": provenance}, indent=2))
    (out / "reactor_validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
