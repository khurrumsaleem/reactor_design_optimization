#!/usr/bin/env python3
"""Plot the empirical reaction-table validation used in the manuscript.

The figure intentionally mirrors the typography, white-grid background, paired
seed traces, and 95% t intervals used by the ReactorGen figures, while using a
distinct purple/teal palette to identify the empirical validation at a glance.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


T_CRIT_95_DF4 = 2.7764451051977987
METHODS = ("sft_base", "dpo", "grpo")
LABELS = {"sft_base": "SFT base", "dpo": "DPO", "grpo": "GRPO"}
COLORS = {"sft_base": "#6C757D", "dpo": "#7B2CBF", "grpo": "#2A9D8F"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--empirical-root",
        type=Path,
        default=here.parent,
        help="Empirical project root containing results/registered_*.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here,
        help="Directory for the PDF, PNG, and plotted summary CSV.",
    )
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=None,
        help="Optional directory to which the final PDF is copied for LaTeX.",
    )
    return parser.parse_args()


def load_seed_summary(path: Path, regime: str) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    metrics = metrics.loc[
        (metrics["arm"] == "cpt_sft")
        & metrics["algorithm"].isin(METHODS)
        & (metrics["seed"].astype(str) != "pooled")
    ].copy()
    metrics["seed"] = metrics["seed"].astype(int)
    summary = (
        metrics.groupby(["seed", "algorithm"], as_index=False)[
            ["expected_abs_error", "expansion_mass"]
        ]
        .mean()
        .assign(regime=regime)
    )
    expected_rows = 5 * len(METHODS)
    if len(summary) != expected_rows:
        raise ValueError(f"Expected {expected_rows} seed-method rows in {path}, found {len(summary)}")
    return summary


def load_reference_errors(path: Path) -> dict[str, float]:
    """Best attainable expected error over the full and the restricted catalogue."""
    import json

    with open(path) as handle:
        summary = json.load(handle)
    baseline = summary["baselines"]["random_full"]
    return {
        "best in support": float(baseline["restricted_oracle_abs_error"]),
        "best in catalogue": float(baseline["full_oracle_abs_error"]),
    }


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    value_format: str,
    reference_lines: dict[str, float] | None = None,
) -> None:
    x = np.arange(len(METHODS), dtype=float)
    wide = data.pivot(index="seed", columns="algorithm", values=metric).loc[:, METHODS]

    # Thin gray traces retain the paired five-seed structure.
    for _, row in wide.iterrows():
        ax.plot(
            x,
            row.to_numpy(dtype=float),
            color="#A7A9AC",
            linewidth=1.0,
            alpha=0.55,
            marker="o",
            markersize=3.2,
            zorder=1,
        )

    means = wide.mean(axis=0).to_numpy(dtype=float)
    sems = wide.std(axis=0, ddof=1).to_numpy(dtype=float) / np.sqrt(len(wide))
    cis = T_CRIT_95_DF4 * sems
    ax.plot(x, means, color="#343A40", linewidth=1.5, alpha=0.8, zorder=2)

    for index, method in enumerate(METHODS):
        ax.errorbar(
            x[index],
            means[index],
            yerr=cis[index],
            fmt="o",
            markersize=8,
            markerfacecolor=COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=COLORS[method],
            elinewidth=2.0,
            capsize=4,
            capthick=1.6,
            zorder=3,
        )
        # Anchor the value label above the upper end of the CI so it never
        # collides with the error bar; seed traces above the mean are thin
        # and light, so a small offset is sufficient.
        ax.annotate(
            format(means[index], value_format),
            (x[index], means[index] + cis[index]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS[method],
            fontweight="bold",
        )

    top = float(np.max(means + cis))
    top = max(top, float(wide.to_numpy(dtype=float).max()))
    if reference_lines:
        for label, value in reference_lines.items():
            ax.axhline(value, color="#7A7A7A", linestyle=(0, (4, 3)), linewidth=1.0, zorder=0)
            ax.annotate(
                f"{label} {value:{value_format}}",
                (len(METHODS) - 0.68, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8.2,
                color="#5A5A5A",
            )
            top = max(top, value)

    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_xticks(x, [LABELS[m] for m in METHODS])
    ax.set_xlim(-0.35, len(METHODS) - 0.65)
    ax.set_ylim(0, top * 1.16)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    root = args.empirical_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    single = load_seed_summary(root / "results/registered_single/metrics.csv", "Single target")
    multi = load_seed_summary(root / "results/registered_multi/metrics.csv", "Multi target")
    single_ref = load_reference_errors(root / "results/registered_single/summary.json")
    multi_ref = load_reference_errors(root / "results/registered_multi/summary.json")
    plotted = pd.concat([single, multi], ignore_index=True)
    plotted.to_csv(out / "empirical_support_expansion_seed_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#4A4A4A",
            "axes.labelcolor": "#303030",
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    draw_panel(
        axes[0, 0],
        single,
        "expected_abs_error",
        "a  Single-target yield matching",
        "Expected absolute yield error",
        ".1f",
        reference_lines=single_ref,
    )
    draw_panel(
        axes[0, 1],
        single,
        "expansion_mass",
        "b  Single-target support expansion",
        "Probability outside CPT/SFT support",
        ".2f",
    )
    draw_panel(
        axes[1, 0],
        multi,
        "expected_abs_error",
        "c  Multi-target yield matching",
        "Expected absolute yield error",
        ".1f",
        reference_lines=multi_ref,
    )
    draw_panel(
        axes[1, 1],
        multi,
        "expansion_mass",
        "d  Multi-target support expansion",
        "Probability outside CPT/SFT support",
        ".2f",
    )

    pdf_path = out / "empirical_support_expansion.pdf"
    png_path = out / "empirical_support_expansion.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if args.manuscript_dir is not None:
        manuscript_dir = args.manuscript_dir.resolve()
        manuscript_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, manuscript_dir / pdf_path.name)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
