#!/usr/bin/env python3
"""Render the five existing prompt-control panels from archived experiment data.

Example
    python build_prompt_control.py --source-root /path/to/reactor_genome \
        --output-dir "/path/to/reactor_genome/NMI_submit/1st appeal/figure"

No experiments or bootstrap resampling are run. Original source files are read-only.
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
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import stats

BLUE, ORANGE, GRAY = "#387CA3", "#BD713D", "#858A8D"
CPT_MODELS = ["CPT_SFT_Base", "DPO_Single_CPT_SFT", "DPO_Multi_CPT_SFT",
              "GRPO_Single_CPT_SFT", "GRPO_Multi_CPT_SFT"]
SLOPE_ORDER = ["SFT_Only_Base", "DPO_Single_SFT", "DPO_Multi_SFT", "GRPO_Single_SFT", "GRPO_Multi_SFT",
               "CPT_SFT_Base", "DPO_Single_CPT_SFT", "DPO_Multi_CPT_SFT", "GRPO_Single_CPT_SFT", "GRPO_Multi_CPT_SFT"]


def style(name):
    if "Base" in name:
        return GRAY, ":", "Base"
    method = "DPO" if "DPO" in name else "GRPO"
    multi = "Multi" in name or "multi" in name
    return BLUE if method == "DPO" else ORANGE, "--" if multi else "-", method + (" multi" if multi else " single")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.source_root.resolve(), args.output_dir.resolve()
    authorized = (root / "NMI_submit/1st appeal/figure").resolve()
    if out == root or (root in out.parents and out != authorized and authorized not in out.parents):
        raise ValueError("Outputs inside the source project are permitted only in NMI_submit/1st appeal/figure")
    out.mkdir(parents=True, exist_ok=True)
    dataout = out / "prompt_control_data"
    dataout.mkdir(exist_ok=True)
    provenance = {}

    def source(rel, csv=True):
        p = root / rel
        payload = p.read_bytes()
        provenance[rel] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        return pd.read_csv(p) if csv else payload.decode()

    # Record the source algorithms used to preserve numerical semantics.
    for rel in ["final plot/fig6_steerability.py", "final plot/_data.py", "final plot/_style.py"]:
        source(rel, csv=False)

    templates = {
        "DPO_single": "after/training/dpo/single_target/dpo_single_target_seed{seed}_results.csv",
        "DPO_multi": "after/training/dpo/multi_target/dpo_multi_lhs_seed{seed}_results.csv",
        "GRPO_single": "after/training/grpo/single/grpo_cpt_sft_single_seed{seed}_results.csv",
        "GRPO_multi": "after/training/grpo/multi/grpo_cpt_sft_multi_seed{seed}_results.csv",
    }

    def steps(frame, method):
        if method.startswith("DPO"):
            assert len(frame) == 1000 and frame.simulation_count.iloc[-1] == 2000
            d = frame.reset_index(drop=True)
            result = pd.DataFrame(dict(x=d.simulation_count, step_fit=d.chosen_fitness,
                                       step_k=d.chosen_k_eff, cum_best_fit=d.chosen_fitness.cummin()))
        else:
            assert len(frame) == 2000 and frame.sim_count.iloc[-1] == 2000
            d = frame.loc[frame.groupby("step").fitness.idxmin()].sort_values("step").reset_index(drop=True)
            result = pd.DataFrame(dict(x=frame.groupby("step").sim_count.max().sort_index().to_numpy(),
                                       step_fit=d.fitness, step_k=d.k_eff, cum_best_fit=d.fitness.cummin()))
        if "target_k" in d:
            result["target_k"] = d.target_k
        return result

    T = {name: [steps(source(template.format(seed=s)), name) for s in range(5)]
         for name, template in templates.items()}
    means = source("chakraborty/results/steerability_target_mean_95ci.csv")
    slopes = source("chakraborty/results/steerability_bootstrap_summary.csv")
    sm = means[means.truncation_policy == "exclude_truncated"].copy()
    sl = slopes[slopes.truncation_policy == "exclude_truncated"].copy().set_index("model_name")
    assert len(sm) == 70 and set(sm.model_name) == set(SLOPE_ORDER)
    assert len(sl) == 10 and sl.n_bootstrap.eq(10000).all()
    assert sm.groupby("model_name").size().eq(7).all()
    assert sm.n.between(94, 100).all()
    assert sl.ci_excludes_zero.sum() == 1 and bool(sl.loc["GRPO_Multi_CPT_SFT", "ci_excludes_zero"])
    sm[sm.model_name.isin(CPT_MODELS)].to_csv(dataout / "fig5d_prompt_response.csv", index=False)
    sl.loc[SLOPE_ORDER].reset_index().to_csv(dataout / "fig5e_checkpoint_slopes.csv", index=False)
    means.to_csv(dataout / "registered_target_means_both_truncation_policies.csv", index=False)
    slopes.to_csv(dataout / "registered_slopes_both_truncation_policies.csv", index=False)

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
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "figure.dpi": 150,
    })

    def panel(ax, letter, title, dx=-.18):
        ax.text(dx, 1.09, letter, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", ha="left")
        ax.set_title(title, loc="left", pad=13)

    def rolling(arr):
        # Same prefix behavior and arithmetic as the original 25-step helper.
        s = np.asarray(arr, dtype=float)
        return np.asarray([np.nanmean(s[max(0, i-24):i+1]) for i in range(len(s))])

    def align(xs, ys):
        x = np.linspace(max(np.nanmin(v) for v in xs), min(np.nanmax(v) for v in xs), 400)
        m = []
        for xv, yv in zip(xs, ys):
            o = np.argsort(xv)
            m.append(np.interp(x, np.asarray(xv)[o], np.asarray(yv)[o]))
        return x, np.vstack(m)

    plotrows, seedrows = [], []

    def band(ax, method, metric, arrays):
        x, m = align([d.x.to_numpy() for d in T[method]], arrays)
        mean = np.nanmean(m, axis=0)
        n = np.sum(~np.isnan(m), axis=0)
        sem = stats.sem(m, axis=0, nan_policy="omit")
        half = stats.t.ppf(.975, n-1) * sem
        low, high = mean-half, mean+half
        color, linestyle, _ = style(method)
        ax.fill_between(x, low, high, color=color, alpha=.14, linewidth=0)
        ax.plot(x, mean, color=color, ls=linestyle, lw=1.3)
        for i in range(len(x)):
            plotrows.append(dict(method=method, metric=metric, evaluations=float(x[i]),
                                 mean=float(mean[i]), ci_lower=float(low[i]), ci_upper=float(high[i]), n_seeds=int(n[i])))
            for seed in range(5):
                seedrows.append(dict(method=method, metric=metric, seed=seed, evaluations=float(x[i]), value=float(m[seed, i])))

    fig = plt.figure(figsize=(7.2, 6.15))
    top = fig.add_gridspec(1, 3, left=.095, right=.98, top=.925, bottom=.67, wspace=.54)
    a, b, c = [fig.add_subplot(top[0, i]) for i in range(3)]
    for key in ["DPO_single", "GRPO_single", "DPO_multi", "GRPO_multi"]:
        band(a, key, "best_so_far_objective", [d.cum_best_fit.to_numpy() for d in T[key]])
        band(b, key, "rolling25_step_objective", [rolling(d.step_fit.to_numpy()) for d in T[key]])
        if key.endswith("multi"):
            band(c, key, "rolling25_requested_target_error_pcm",
                 [rolling(np.abs(d.step_k.to_numpy()-d.target_k.to_numpy())*1e5) for d in T[key]])
    for ax in [a, b, c]:
        ax.set(xlim=(0, 2000), xticks=[0, 1000, 2000], xlabel="OpenMC evaluations")
        ax.set_yscale("log")
        ax.minorticks_off()
    a.set(ylim=(1.2, 30), yticks=[1.5, 3, 10, 30], ylabel="Best-so-far objective")
    a.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    b.set(ylim=(1.5, 40), yticks=[2, 5, 10, 30], ylabel="Per-step objective")
    b.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    c.set(ylim=(300, 3e4), yticks=[1000, 10000], ylabel="Target error (pcm)")
    c.set_yticklabels(["1,000", "10,000"])
    panel(a, "a", "Best objective", -.29)
    panel(b, "b", "Per-step objective", -.29)
    panel(c, "c", "Requested-target error", -.29)
    c.text(.97, .94, "Multi-target schedules", transform=c.transAxes, ha="right", va="top", fontsize=7)

    handles = [Line2D([], [], color=style(k)[0], ls=style(k)[1], lw=1.5, label=style(k)[2])
               for k in ["DPO_single", "DPO_multi", "GRPO_single", "GRPO_multi", "CPT_SFT_Base"]]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, .56), ncol=5,
               frameon=False, handlelength=2.1, columnspacing=1.4, fontsize=7.5)

    d = fig.add_axes([.095, .17, .325, .29])
    e = fig.add_axes([.655, .17, .315, .29])
    for name in CPT_MODELS:
        sub = sm[sm.model_name == name].sort_values("target_k")
        color, linestyle, label = style(name)
        d.fill_between(sub.target_k, sub.ci_lower, sub.ci_upper, color=color, alpha=.14, linewidth=0)
        d.plot(sub.target_k, sub.mean_gd, color=color, ls=linestyle, marker="o", ms=3,
               lw=1.2, label=label)
    d.axhline(16, color=GRAY, lw=.7, ls=(0, (3, 2)))
    d.text(.98, 16.5, "Corpus 16", transform=d.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.2)
    d.set(xlabel=r"Requested target $k$", ylabel="Generated absorber count", xticks=[1.02, 1.04, 1.06, 1.08])
    panel(d, "d", "Prompt-conditioned generation", -.20)
    d.text(.02, .96, "CPT + SFT checkpoints", transform=d.transAxes, ha="left", va="top", fontsize=7.4)

    ypos = [11, 10, 9, 8, 7, 4, 3, 2, 1, 0]
    for y, name in zip(ypos, SLOPE_ORDER):
        r = sl.loc[name]
        color, _, _ = style(name)
        sig = bool(r.ci_excludes_zero)
        e.errorbar(r.point_slope, y, xerr=[[r.point_slope-r.ci_lower], [r.ci_upper-r.point_slope]],
                   fmt="o", color=color, capsize=2, ms=4.2 if sig else 3.5,
                   markerfacecolor=color if sig else "white", mew=.9, lw=1)
    e.axvline(0, color=GRAY, ls=(0, (2, 2)), lw=.8)
    e.axhspan(5.9, 12.6, color="#F3F4F5", zorder=0)
    e.text(.02, 12, "SFT-only initialization", transform=e.get_yaxis_transform(), fontsize=7.5, va="center")
    e.text(.02, 5, "CPT + SFT initialization", transform=e.get_yaxis_transform(), fontsize=7.5, va="center")
    e.set(xlim=(-215, 95), ylim=(-.65, 12.6), yticks=ypos,
          yticklabels=[style(n)[2] for n in SLOPE_ORDER], xticks=[-200, -100, 0],
          xlabel=r"Inventory slope (rods per unit $k$)")
    e.tick_params(axis="y", labelsize=7.4, length=0)
    e.spines.left.set_visible(False)
    panel(e, "e", "Control across checkpoints", -.42)
    fig.text(.095, .085, "a-c  Five training seeds with pointwise 95% intervals. b,c  25-step rolling means.", fontsize=7.2)
    fig.text(.095, .060, "d  95% intervals across generated responses. e  95% bootstrap intervals within each checkpoint.", fontsize=7.2)
    fig.text(.095, .035, "Filled point indicates an interval excluding zero. Each checkpoint is evaluated at seven requested targets.", fontsize=7.2)

    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out/f"fig5_prompt_control.{ext}", dpi=300, bbox_inches=None)
    plt.close(fig)
    pd.DataFrame(plotrows).to_csv(dataout/"fig5abc_mean_and_95ci.csv", index=False)
    pd.DataFrame(seedrows).to_csv(dataout/"fig5abc_interpolated_seed_values.csv", index=False)
    rawrows = []
    for name, frames in T.items():
        for seed, frame in enumerate(frames):
            frame = frame.copy(); frame["method"] = name; frame["seed"] = seed
            rawrows.append(frame)
    pd.concat(rawrows, ignore_index=True).to_csv(dataout/"fig5abc_step_records.csv", index=False)
    main_slope = sl.loc["GRPO_Multi_CPT_SFT"]
    validation = dict(source_files=len(provenance), training_seeds=5, evaluations_per_run=2000,
                      trajectory_grid_points=400, rolling_steps=25, checkpoints=10, requested_targets=7,
                      generated_samples_per_target_before_filter=100, retained_per_target_min=int(sm.n.min()),
                      retained_per_target_max=int(sm.n.max()), bootstrap_resamples_per_checkpoint=10000,
                      significant_checkpoint="GRPO_Multi_CPT_SFT", slope=float(main_slope.point_slope),
                      slope_ci_lower=float(main_slope.ci_lower), slope_ci_upper=float(main_slope.ci_upper),
                      slope_unit="absorber rods per unit k", source_files_unchanged=all(
                          hashlib.sha256((root/rel).read_bytes()).hexdigest()==v["sha256"] for rel,v in provenance.items()))
    assert validation["source_files_unchanged"]
    (out/"prompt_control_source_provenance.json").write_text(json.dumps(dict(source_root=str(root), files=provenance), indent=2))
    (out/"prompt_control_validation.json").write_text(json.dumps(validation, indent=2))
    (out/"prompt_control_caption.txt").write_text(
        "Figure 5. Optimization quality and prompt-conditioned control. "
        "Single-target and multi-target variants use CPT + SFT initialization and 2,000 online evaluations. "
        "a,b, Best-so-far and per-step objective trajectories. Each multi-target step evaluates the target requested at that step. "
        "c, Absolute error relative to each requested target for multi-target variants. "
        "Bands in a-c show pointwise two-sided 95% Student-t intervals across five training runs after interpolation onto 400 common evaluation counts. "
        "Panels b,c use 25-step rolling means. "
        "d, Mean generated absorber count at CPT + SFT checkpoints for seven requested targets with initially 100 responses per cell. "
        "Bands show two-sided 95% Student-t intervals across generated responses after severe-truncation filtering. "
        "e, Inventory slopes for all ten checkpoints with 95% percentile intervals calculated using 10,000 within-target bootstrap resamples at each checkpoint. "
        "The filled point marks the multi-target GRPO checkpoint with CPT + SFT initialization, whose interval excludes zero. "
        "Its slope is -162.87 rods per unit k with interval [-193.74, -134.35]. "
        "Supplementary Information reports SFT-only response curves and the all-response sensitivity check.\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
