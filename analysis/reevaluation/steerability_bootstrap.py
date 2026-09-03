#!/usr/bin/env python3
"""Recompute prompt-steerability slopes with a documented bootstrap.

Within every (model, target-k) cell, generated samples are independently
resampled with replacement while preserving that cell's sample count. The
seven resampled target means are regressed on target k, and a two-sided 95%
percentile interval is taken from the bootstrap slope distribution.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def analyze_policy(frame, model_name, policy, n_boot, rng):
    original = frame[frame.model_name == model_name].copy()
    if policy == "exclude_truncated" and "is_truncated" in original:
        data = original[original.is_truncated == 0].copy()
    else:
        data = original

    grouped = {
        float(target): group.gd_count.to_numpy(dtype=float)
        for target, group in data.groupby("target_k")
    }
    targets = np.array(sorted(grouped))
    if len(targets) != 7:
        raise ValueError(f"{model_name}/{policy}: found {len(targets)} targets, expected 7")
    means = np.array([grouped[target].mean() for target in targets])
    point_slope, point_intercept = np.polyfit(targets, means, 1)

    slopes = np.empty(n_boot)
    for index in range(n_boot):
        bootstrap_means = np.array([
            rng.choice(values, size=len(values), replace=True).mean()
            for values in (grouped[target] for target in targets)
        ])
        slopes[index] = np.polyfit(targets, bootstrap_means, 1)[0]

    lower, upper = np.percentile(slopes, [2.5, 97.5])
    summary = {
        "model_name": model_name,
        "truncation_policy": policy,
        "n_generated": len(original),
        "n_analyzed": len(data),
        "n_truncated_excluded": len(original) - len(data),
        "n_targets": len(targets),
        "min_samples_per_target": min(map(len, grouped.values())),
        "max_samples_per_target": max(map(len, grouped.values())),
        "point_slope": point_slope,
        "point_intercept": point_intercept,
        "bootstrap_mean_slope": slopes.mean(),
        "bootstrap_sd_slope": slopes.std(ddof=1),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
        "n_bootstrap": n_boot,
        "resampling_unit": "generated sample, independently within each target-k cell",
        "ci_method": "two-sided 95% percentile bootstrap interval",
    }
    curves = []
    for target in targets:
        values = grouped[target]
        n = len(values)
        mean = values.mean()
        sem = stats.sem(values)
        tcrit = stats.t.ppf(0.975, n - 1)
        curves.append({
            "model_name": model_name,
            "truncation_policy": policy,
            "target_k": target,
            "n": n,
            "mean_gd": mean,
            "sd_gd": values.std(ddof=1),
            "sem_gd": sem,
            "ci_lower": mean - tcrit * sem,
            "ci_upper": mean + tcrit * sem,
            "ci_method": "two-sided Student-t interval over generated samples",
        })
    return summary, curves, slopes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data")) / "prompt_sensitivity/prompt_sensitivity_results.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260901)
    parser.add_argument(
        "--truncation-policy", choices=("exclude", "include", "both"), default="both",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)

    policies = {
        "exclude": ["exclude_truncated"],
        "include": ["include_all"],
        "both": ["exclude_truncated", "include_all"],
    }[args.truncation_policy]

    summaries = []
    curves = []
    draws = {}
    master = np.random.SeedSequence(args.random_seed)
    jobs = [(model, policy) for policy in policies for model in sorted(frame.model_name.unique())]
    child_seeds = master.spawn(len(jobs))
    for (model, policy), child_seed in zip(jobs, child_seeds):
        summary, curve_rows, slopes = analyze_policy(
            frame, model, policy, args.n_bootstrap, np.random.default_rng(child_seed)
        )
        summaries.append(summary)
        curves.extend(curve_rows)
        draws[f"{policy}__{model}"] = slopes

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["truncation_policy", "model_name"]
    )
    curve_frame = pd.DataFrame(curves).sort_values(
        ["truncation_policy", "model_name", "target_k"]
    )
    summary_path = args.output_dir / "steerability_bootstrap_summary.csv"
    curve_path = args.output_dir / "steerability_target_mean_95ci.csv"
    draws_path = args.output_dir / "steerability_bootstrap_draws.npz"
    summary_frame.to_csv(summary_path, index=False)
    curve_frame.to_csv(curve_path, index=False)
    np.savez_compressed(draws_path, **draws)

    print(f"Wrote {len(summary_frame)} slope summaries to {summary_path}")
    print(f"Wrote {len(curve_frame)} target-level intervals to {curve_path}")
    print(f"Wrote bootstrap slope draws to {draws_path}")
    print("\nIntervals excluding truncated generations:")
    shown = summary_frame[summary_frame.truncation_policy == "exclude_truncated"]
    print(shown[[
        "model_name", "n_analyzed", "point_slope", "ci_lower", "ci_upper",
        "ci_excludes_zero",
    ]].to_string(index=False, float_format=lambda value: f"{value:.1f}"))


if __name__ == "__main__":
    main()
