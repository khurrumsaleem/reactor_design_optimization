#!/usr/bin/env python3
"""Rebuild the main chemistry figure from the registered, completed records.

Read-only source inputs. No model loading, training, new reward evaluations, or
imports from the original plotting implementation. Requirements are matplotlib,
numpy, and pandas. The fixed t critical value is for the registered five seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nmi_appeal_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("sft_base", "dpo", "grpo")
COLORS = {"sft_base": "#929BA0", "dpo": "#387CA3", "grpo": "#BD713D"}
METRICS = ("expected_abs_error", "expansion_mass")
T95_DF4 = 2.7764451051977987
PREFIX = "fig4_chemistry"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_catalog_references(path: Path, targets: list[float]) -> tuple[pd.DataFrame, dict]:
    """Exact uniform expectations and per-task minima, averaged over conditions."""
    catalog = pd.read_csv(path)
    assert len(catalog) == 3955
    assert not catalog.duplicated(["Ligand", "Additive", "Base", "Aryl halide"]).any()
    tasks = sorted(catalog["Aryl halide"].unique())
    ligands = sorted(catalog["Ligand"].unique())
    assert len(tasks) == 15 and len(ligands) == 4
    rows = []
    restricted_count = 0
    for task_id, task in enumerate(tasks):
        task_rows = catalog.loc[catalog["Aryl halide"] == task]
        anchor = task_rows["Ligand"] == ligands[task_id % 4]
        restricted_count += int(anchor.sum())
        for target in targets:
            error = (task_rows["Output"] - target).abs()
            rows.append({
                "aryl_id": task_id, "target": float(target),
                "random_full_error": float(error.mean()),
                "random_restricted_error": float(error.loc[anchor].mean()),
                "random_full_mass": float((~anchor).mean()),
                "random_restricted_mass": 0.0,
                "full_oracle_abs_error": float(error.min()),
                "restricted_oracle_abs_error": float(error.loc[anchor].min()),
            })
    assert restricted_count == 990
    table = pd.DataFrame(rows)
    values = table.drop(columns=["aryl_id", "target"]).mean().to_dict()
    return table, values


def load_regime(source: Path, name: str) -> tuple[pd.DataFrame, dict, list[Path], pd.DataFrame, dict]:
    location = source / "empirical/results" / f"registered_{name}"
    metrics_path = location / "metrics.csv"
    summary_path = location / "summary.json"
    catalog_path = source / "empirical/data/buchwald_hartwig.csv"
    metrics = pd.read_csv(metrics_path)
    summary = json.loads(summary_path.read_text())
    targets = summary["config"]["targets"]
    assert summary["config"]["seeds"] == [0, 1, 2, 3, 4]
    assert len(targets) == (1 if name == "single" else 10)
    assert summary["config"]["dpo_steps"] * 2 == 2000
    assert summary["config"]["grpo_steps"] * summary["config"]["grpo_group_size"] == 2000
    sub = metrics.loc[(metrics["arm"] == "cpt_sft") & metrics["algorithm"].isin(METHODS)].copy()
    sub["seed"] = sub["seed"].astype(int)
    assert len(sub) == 5 * 3 * 15 * len(targets)
    assert not sub.duplicated(["seed", "algorithm", "aryl_id", "target"]).any()
    assert sub.groupby(["seed", "algorithm"]).size().eq(15 * len(targets)).all()
    seeds = sub.groupby(["seed", "algorithm"], as_index=False)[list(METRICS)].mean()
    seeds.insert(0, "regime", name)
    for method in METHODS:
        for metric in METRICS:
            found = seeds.loc[seeds["algorithm"] == method, metric].mean()
            np.testing.assert_allclose(found, summary["aggregates"][f"cpt_sft/{method}"][metric], atol=1e-10, rtol=0)
    derived, refs = derive_catalog_references(catalog_path, targets)
    check_map = {
        "random_full_error": ("random_full", "expected_abs_error"),
        "random_restricted_error": ("random_restricted", "expected_abs_error"),
        "random_full_mass": ("random_full", "expansion_mass"),
        "random_restricted_mass": ("random_restricted", "expansion_mass"),
        "full_oracle_abs_error": ("random_full", "full_oracle_abs_error"),
        "restricted_oracle_abs_error": ("random_full", "restricted_oracle_abs_error"),
    }
    for key, (method, metric) in check_map.items():
        np.testing.assert_allclose(refs[key], summary["baselines"][method][metric], atol=1e-10, rtol=0)
        baseline_rows = metrics.loc[(metrics["arm"] == "baseline") & (metrics["algorithm"] == method)]
        np.testing.assert_allclose(refs[key], baseline_rows[metric].mean(), atol=1e-10, rtol=0)
    derived.insert(0, "regime", name)
    wide_error = seeds.pivot(index="seed", columns="algorithm", values="expected_abs_error")
    wide_mass = seeds.pivot(index="seed", columns="algorithm", values="expansion_mass")
    checks = {method: {
        "all_five_error_changes_negative": bool((wide_error[method] < wide_error["sft_base"]).all()),
        "all_five_mass_changes_positive": bool((wide_mass[method] > wide_mass["sft_base"]).all()),
    } for method in ("dpo", "grpo")}
    assert all(all(x.values()) for x in checks.values())
    return seeds, refs, [metrics_path, summary_path, catalog_path], derived, checks


def draw_panel(ax, seeds: pd.DataFrame, refs: dict, metric: str, regime: str, letter: str) -> list[dict]:
    wide = seeds.pivot(index="seed", columns="algorithm", values=metric).loc[:, list(METHODS)].sort_index()
    # Each trace starts at the shared supervised base; the two updates are parallel.
    jitter = np.linspace(-0.085, 0.085, 5)
    for j, (_, row) in zip(jitter, wide.iterrows()):
        for end in (1, 2):
            ax.plot([j, end + j], [row.iloc[0], row.iloc[end]], color="#C8CDD0", lw=0.55, zorder=1)
        for i, method in enumerate(METHODS):
            ax.scatter(i + j, row[method], s=8, color=COLORS[method], alpha=.7, edgecolor="white", linewidth=.25, zorder=3)
    means = wide.mean()
    ci = wide.std(ddof=1) / np.sqrt(5) * T95_DF4
    entries = []
    for i, method in enumerate(METHODS):
        ax.errorbar(i, means[method], yerr=ci[method], fmt="o", ms=5.2,
                    mfc=COLORS[method], mec="white", mew=.65, color=COLORS[method],
                    elinewidth=1.1, capsize=2.5, capthick=.9, zorder=5)
        val = f"{means[method]:.1f}" if metric == "expected_abs_error" else f"{means[method]:.3f}"
        upper = max(means[method] + ci[method], wide[method].max())
        ax.annotate(val, (i, upper), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="black", zorder=6)
        entries.append({"regime": regime, "metric": metric, "procedure": method, "type": "mean_and_95pct_t_interval",
                        "mean": means[method], "ci_low": means[method]-ci[method], "ci_high": means[method]+ci[method], "n_seeds": 5})
    for i, name in ((3, "random_full"), (4, "random_restricted")):
        suffix = "error" if metric == "expected_abs_error" else "mass"
        value = refs[f"{name}_{suffix}"]
        ax.scatter(i, value, marker="s" if i == 3 else "D", s=26, facecolor="white", edgecolor="#555C60", linewidth=1.0, zorder=5)
        val = f"{value:.1f}" if metric == "expected_abs_error" else f"{value:.3f}"
        ax.annotate(val, (i, value), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="black")
        entries.append({"regime": regime, "metric": metric, "procedure": name, "type": "exact_uniform_expectation",
                        "mean": value, "ci_low": None, "ci_high": None, "n_seeds": None})
    if metric == "expected_abs_error":
        for key, name, style, offset, va in (
            ("restricted_oracle_abs_error", "Anchor best", (0, (4, 2)), 3, "bottom"),
            ("full_oracle_abs_error", "Full best", (0, (1, 2)), -3, "top"),
        ):
            value = refs[key]
            ax.axhline(value, color="#737A7E", lw=.65, linestyle=style, zorder=0)
            ax.annotate(f"{name} {value:.1f}", (4.37, value), xytext=(0, offset), textcoords="offset points",
                        ha="right", va=va, fontsize=8, color="black", bbox={"facecolor":"white", "edgecolor":"none", "pad":0.4})
            entries.append({"regime":regime,"metric":metric,"procedure":key,"type":"task_averaged_attainable_minimum",
                            "mean":value,"ci_low":None,"ci_high":None,"n_seeds":None})
        ax.set_ylim(0, 66 if regime == "single" else 44)
        ax.set_yticks([0, 20, 40, 60] if regime == "single" else [0, 10, 20, 30, 40])
        ax.set_ylabel("Expected absolute yield error\n(percentage points)", fontsize=8.5, labelpad=5)
        title = "Target error"
    else:
        ax.set_ylim(-.025, .88)
        ax.set_yticks([0, .2, .4, .6, .8])
        ax.set_ylabel("Probability of an excluded ligand", fontsize=8.5, labelpad=5)
        title = "Choices outside the training convention"
    ax.axvline(2.5, color="#E1E4E6", lw=.65, zorder=0)
    ax.set_xlim(-.45, 4.45)
    ax.set_xticks(np.arange(5), ["Supervised\nbase", "DPO", "GRPO", "Uniform\nfull", "Uniform\nanchor"])
    ax.tick_params(axis="both", labelsize=8, width=.65, length=2.6, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(.65)
    ax.text(-.17, 1.085, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, color="black")
    ax.text(0, 1.085, title, transform=ax.transAxes, fontweight="bold", fontsize=8.8, color="black")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Read-only source mirror, typically ARCHIVE_ROOT/data/source_tree")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    out = args.output_dir.resolve()
    appeal_output = (source / "NMI_submit" / "1st appeal" / "figure").resolve()
    is_authorized_appeal_output = out == appeal_output or appeal_output in out.parents
    if (out == source or source in out.parents) and not is_authorized_appeal_output:
        raise ValueError("Inside the source tree, output is allowed only under NMI_submit/1st appeal/figure")
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    sources = set()
    derived_tables = []
    all_checks = {}
    for regime in ("single", "multi"):
        seeds, refs, paths, derived, checks = load_regime(source, regime)
        data[regime] = (seeds, refs)
        sources.update(paths)
        derived_tables.append(derived)
        all_checks[regime] = checks
    # Record inspected reference implementations as provenance; they are not imported.
    references = [source / "empirical/plot/plot_empirical_support_expansion.py",
                  source / "final plot/fig7_empirical.py", source / "final plot/_data.py",
                  source / "empirical/empirical_rl/data.py", source / "empirical/empirical_rl/evaluation.py"]
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":8.5,"text.color":"black","axes.labelcolor":"black","xtick.color":"black","ytick.color":"black",
        "axes.edgecolor":"black","axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white",
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35))
    fig.subplots_adjust(left=.095, right=.985, bottom=.105, top=.88, wspace=.29, hspace=.68)
    fig.text(.095, .967, "Requested yield 90", fontsize=9.2, color="black", fontweight="bold")
    fig.text(.095, .496, "Ten requested yields from 10 to 100", fontsize=9.2, color="black", fontweight="bold")
    plotted = []
    for row, regime in enumerate(("single", "multi")):
        seeds, refs = data[regime]
        for col, metric in enumerate(METRICS):
            plotted += draw_panel(axes[row,col], seeds, refs, metric, regime, "abcd"[row*2+col])
    # The small paired shifts in the multi-target panel are quantified by labels.
    # Both probability panels retain the same unbroken scale for honest comparison.
    fig.savefig(out / f"{PREFIX}.pdf", metadata={"Title":"Measured feedback selects productive alternatives beyond training conventions", "Author":"ReactorGen study", "CreationDate":None, "ModDate":None})
    fig.savefig(out / f"{PREFIX}.svg", metadata={"Title":"Measured feedback selects productive alternatives beyond training conventions", "Date":None})
    fig.savefig(out / f"{PREFIX}.png", dpi=300)
    plt.close(fig)
    pd.concat([data[r][0] for r in ("single","multi")], ignore_index=True).to_csv(out/f"{PREFIX}_seed_values.csv", index=False, float_format="%.15g")
    pd.DataFrame(plotted).to_csv(out/f"{PREFIX}_plotted_values.csv", index=False, float_format="%.15g")
    pd.concat(derived_tables, ignore_index=True).to_csv(out/f"{PREFIX}_catalog_reference_values.csv", index=False, float_format="%.15g")
    caption = """Measured feedback selects productive alternatives in a non-language policy. A categorical autoregressive policy is initialized by CPT and SFT on one anchor ligand per aryl-halide task and updated using measured reaction yields. a,b, Expected absolute yield error and probability of a task-specific ligand excluded from supervised training for a requested yield of 90. c,d, The same measurements averaged over ten balanced requested yields from 10 to 100. Small points are five independently trained policies. Thin lines connect each supervised base to its separately aligned DPO and GRPO policies. Large circles and error bars show means and two-sided 95% Student-t intervals across five seeds. Uniform full and uniform anchor sample equally over measured rows in the full and anchor-ligand catalogs for each task. Their open symbols show exact expectations. Horizontal lines in a and c give the task-averaged minimum attainable error in the anchor and full catalogs. All policy measurements are exact catalog expectations averaged equally over tasks and targets. Each aligned policy uses 2,000 measured-yield lookups.\n"""
    (out/f"{PREFIX}_caption.txt").write_text(caption)
    readme = """Figure 4 chemistry reproduction\n\nRun the self-contained script with Python, matplotlib, numpy and pandas.\n\npython paper_figures/plot_fig4_chemistry.py --source-root "$ARCHIVE_ROOT/data/source_tree" --output-dir work/rebuilt_figures\n\nThe script reads completed registered records and the measured reaction CSV. It imports no original plotting code and performs no training or simulations. Original source files are opened for reading only. Output inside the source tree is permitted only in NMI_submit/1st appeal/figure and its descendants; an external staging directory is also allowed. Uniform expectations and attainable errors are recalculated exactly from the catalog and checked against both the stored metrics and summary JSON. Learned policy metrics are aggregated from all task-target rows per seed and checked against the registered summaries. The plotted arm is CPT+SFT.\n\nThe two probability axes use the same unbroken scale. Native PDF text and paths remain vector elements; the SVG preserves text. The seed CSV contains 30 rows, the plotted-values CSV contains all 24 displayed means or reference levels, and the catalog-reference CSV contains the per-condition exact checks.\n"""
    (out/f"{PREFIX}_README.txt").write_text(readme)
    provenance = {
        "figure":PREFIX,"source_root":str(source),"script_sha256":sha256(Path(__file__)),
        "source_files":[{"path":str(p.relative_to(source)),"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(sources)],
        "reference_implementations_inspected_not_imported":[{"path":str(p.relative_to(source)),"sha256":sha256(p)} for p in references],
        "calculation":{"arm":"cpt_sft","seeds":[0,1,2,3,4],"mean":"equal weight over task-target conditions within seed, then equal weight over five seeds",
                       "uncertainty":"two-sided 95% Student t interval, df=4", "t_critical":T95_DF4,
                       "uniform_sampling":"equal probability per measured row within each task; condition means averaged equally",
                       "attainable_error":"minimum absolute error per task-target within full or anchor catalog, averaged equally",
                       "no_training_or_new_reward_evaluations":True,"output_size_inches":[7.2,5.35]},
        "checks":{"metric_means_match_registered_summaries":True,"catalog_recomputation_matches_metrics_and_summaries":True,
                  "all_registered_task_target_rows_present":True,"paired_changes":all_checks},
        "baselines":{r:data[r][1] for r in ("single","multi")},
        "outputs":[{"name":p.name,"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(out.glob(f"{PREFIX}*")) if p.suffix != ".json"],
    }
    (out/f"{PREFIX}_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"output_dir":str(out),"baselines":provenance["baselines"],"checks":provenance["checks"]},indent=2))


if __name__ == "__main__":
    main()
