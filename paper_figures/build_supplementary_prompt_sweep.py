#!/usr/bin/env python3
"""Build the supplementary response sweep of archived policy checkpoints.

This script only reads archived response statistics. It runs no experiments,
fits or bootstrap draws and never imports the original plotting code.
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
import numpy as np
import pandas as pd

BLUE, ORANGE, GRAY = "#387CA3", "#BD713D", "#858A8D"
CPT = ["CPT_SFT_Base", "DPO_Single_CPT_SFT", "DPO_Multi_CPT_SFT", "GRPO_Single_CPT_SFT", "GRPO_Multi_CPT_SFT"]
SFT = ["SFT_Only_Base", "DPO_Single_SFT", "DPO_Multi_SFT", "GRPO_Single_SFT", "GRPO_Multi_SFT"]


def style(name):
    if "Base" in name:
        return GRAY, ":", "Base"
    method = "DPO" if "DPO" in name else "GRPO"
    return BLUE if method == "DPO" else ORANGE, "--" if "Multi" in name else "-", method + (" multi" if "Multi" in name else " single")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.source_root.resolve(), args.output_dir.resolve()
    allowed = (root / "NMI_submit/1st appeal/figure").resolve()
    if out == root or (root in out.parents and out != allowed and allowed not in out.parents):
        raise ValueError("Within the source project, outputs are allowed only under NMI_submit/1st appeal/figure")
    out.mkdir(parents=True, exist_ok=True)
    dataout = out / "si_prompt_sweep_data"
    dataout.mkdir(exist_ok=True)
    provenance = {}

    def source(rel, is_csv=True):
        p = root / rel
        payload = p.read_bytes()
        provenance[rel] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        return pd.read_csv(p) if is_csv else payload.decode()

    for rel in ["after/eval_prompt_sensitivity/eval_prompt_sensitivity_safe.py", "chakraborty/steerability_bootstrap.py"]:
        source(rel, False)
    means = source("chakraborty/results/steerability_target_mean_95ci.csv")
    slopes = source("chakraborty/results/steerability_bootstrap_summary.csv")
    raw = source("after/eval_prompt_sensitivity/prompt_sensitivity_results.csv")
    sm = means[means.truncation_policy == "exclude_truncated"].copy()
    sl = slopes[slopes.truncation_policy == "exclude_truncated"].set_index("model_name")
    assert len(sm) == 70 and len(sl) == 10
    assert set(sm.model_name) == set(CPT + SFT) == set(sl.index)
    assert sm.groupby("model_name").size().eq(7).all()
    assert sl.n_bootstrap.eq(10000).all()
    assert sm.n.between(94, 100).all()
    assert int(sl.ci_excludes_zero.sum()) == 1
    assert bool(sl.loc["GRPO_Multi_CPT_SFT", "ci_excludes_zero"])
    # Independently confirm archived target means and counts against the retained responses.
    retained = raw[raw.is_truncated == 0]
    checked = retained.groupby(["model_name", "target_k"]).gd_count.agg(["count", "mean"])
    merged = sm.merge(checked, left_on=["model_name", "target_k"], right_index=True)
    assert (merged.n == merged["count"]).all()
    assert np.allclose(merged.mean_gd, merged["mean"], atol=1e-12, rtol=1e-12)
    for letter, names in [("a", CPT), ("b", SFT)]:
        sm[sm.model_name.isin(names)].to_csv(dataout / f"panel_{letter}_response_curves.csv", index=False)
    sl.loc[CPT + SFT].reset_index().to_csv(dataout / "panel_c_slopes.csv", index=False)
    means.to_csv(dataout / "registered_target_means_both_filters.csv", index=False)
    slopes.to_csv(dataout / "registered_slopes_both_filters.csv", index=False)
    raw.drop(columns=["grid_str"]).to_csv(dataout / "archived_response_metrics.csv", index=False)

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.7, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    })
    fig = plt.figure(figsize=(7.2, 5.4))
    a = fig.add_axes([.10, .60, .335, .285])
    b = fig.add_axes([.10, .20, .335, .285])
    c = fig.add_axes([.665, .20, .31, .685])

    def panel(ax, letter, title, dx, label_y=1.08):
        ax.text(dx, label_y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
        ax.set_title(title, loc="left", pad=12)

    for ax, names, letter, title in [(a, CPT, "a", "CPT + SFT initialization"), (b, SFT, "b", "SFT-only initialization")]:
        for name in names:
            sub = sm[sm.model_name == name].sort_values("target_k")
            color, linestyle, _ = style(name)
            ax.fill_between(sub.target_k, sub.ci_lower, sub.ci_upper, color=color, alpha=.14, linewidth=0)
            ax.plot(sub.target_k, sub.mean_gd, color=color, ls=linestyle, lw=1.25, marker="o", ms=3.1)
        ax.axhline(16, color=GRAY, lw=.65, ls=(0, (3, 2)))
        ax.text(.98, 16.5, "Corpus 16", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7)
        ax.set(xlim=(1.018, 1.082), ylim=(8, 48), xticks=[1.02, 1.04, 1.06, 1.08], yticks=[10, 20, 30, 40],
               ylabel="Generated absorber count", xlabel=r"Requested target $k$")
        panel(ax, letter, title, -.22)
    a.set_xlabel("")

    yvalues = [11, 10, 9, 8, 7, 4, 3, 2, 1, 0]
    for y, name in zip(yvalues, CPT + SFT):
        row = sl.loc[name]
        color, _, _ = style(name)
        sig = bool(row.ci_excludes_zero)
        c.errorbar(row.point_slope, y, xerr=[[row.point_slope-row.ci_lower], [row.ci_upper-row.point_slope]],
                   fmt="o", color=color, capsize=2, ms=4.3 if sig else 3.6,
                   mfc=color if sig else "white", mew=.9, lw=1)
    c.axhspan(5.9, 12.65, color="#F3F4F5", zorder=0)
    c.axvline(0, color=GRAY, ls=(0, (2, 2)), lw=.75)
    c.text(.03, 12.05, "CPT + SFT", transform=c.get_yaxis_transform(), fontsize=8, va="center")
    c.text(.03, 5.05, "SFT only", transform=c.get_yaxis_transform(), fontsize=8, va="center")
    c.set(xlim=(-215, 95), ylim=(-.7, 12.65), yticks=yvalues,
          yticklabels=[style(n)[2] for n in CPT + SFT], xticks=[-200, -100, 0],
          xlabel=r"Inventory slope (rods per unit $k$)")
    c.tick_params(axis="y", length=0, labelsize=7.8)
    c.spines.left.set_visible(False)
    panel(c, "c", "Response slopes", -.45, label_y=1.03)

    handles = [Line2D([], [], color=style(n)[0], ls=style(n)[1], lw=1.5, label=style(n)[2])
               for n in ["DPO_Single_CPT_SFT", "DPO_Multi_CPT_SFT", "GRPO_Single_CPT_SFT", "GRPO_Multi_CPT_SFT", "CPT_SFT_Base"]]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.535, .095), ncol=5,
               frameon=False, handlelength=2, columnspacing=1.3, fontsize=7.5)
    fig.text(.10, .965, "One archived checkpoint per condition, preceding the five-seed campaign", fontsize=8)
    fig.text(.10, .034, "Response-level 95% intervals. Filled point marks a slope interval excluding zero.", fontsize=7.6)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out / f"si_prompt_sweep.{ext}", dpi=300, bbox_inches=None)
    plt.close(fig)

    caption = (
        "Prompt responses of archived policy checkpoints. One archived checkpoint represents each of ten conditions. "
        "These checkpoints precede the five-seed optimization campaign. "
        "a,b, Mean parsed absorber counts across seven requested multiplication factors for CPT + SFT and SFT-only initialization. "
        "Each condition initially contributes 100 generated responses per target. The registered primary analysis retains 94-100 responses per cell after excluding severe truncation. "
        "Bands show two-sided 95% Student-t intervals across retained generated responses. Counts follow filtering to f/g/c and length normalization, before guide-coordinate correction. "
        "c, Response slopes for all ten checkpoints. Intervals are two-sided 95% percentile bootstrap intervals based on 10,000 resamples within target cells of each checkpoint. "
        "The filled point marks the multi-target GRPO checkpoint with CPT + SFT initialization, whose slope is -162.87 rods per unit k with interval [-193.74, -134.35]. "
        "All-response sensitivity intervals and exact checkpoint paths are reported separately in Supplementary Information.\n")
    (out / "si_prompt_sweep_caption.txt").write_text(caption)
    validation = {
        "checkpoints": 10, "one_archived_checkpoint_per_condition": True,
        "campaign_relation": "Archived checkpoints predate the five-seed optimization campaign, confirmed by the user",
        "same_hyperparameters_as_five_seed_campaign_claimed": False,
        "target_cells": 70, "targets_per_checkpoint": 7,
        "initial_responses_per_cell": 100, "retained_min": int(sm.n.min()), "retained_max": int(sm.n.max()),
        "curve_interval_unit": "Generated responses within one checkpoint and requested target",
        "slope_interval_unit": "Generated responses resampled independently within target cells of one checkpoint",
        "bootstrap_resamples": 10000,
        "raw_means_and_counts_match_registered_summaries": True,
        "registered_CIs_used_without_recomputation": True,
        "slope": float(sl.loc["GRPO_Multi_CPT_SFT", "point_slope"]),
        "slope_ci": [float(sl.loc["GRPO_Multi_CPT_SFT", "ci_lower"]), float(sl.loc["GRPO_Multi_CPT_SFT", "ci_upper"])],
        "source_files_unchanged": all(hashlib.sha256((root/rel).read_bytes()).hexdigest() == info["sha256"] for rel,info in provenance.items()),
    }
    assert validation["source_files_unchanged"]
    (out / "si_prompt_sweep_provenance.json").write_text(json.dumps({"source_root": str(root), "files": provenance}, indent=2))
    (out / "si_prompt_sweep_validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
