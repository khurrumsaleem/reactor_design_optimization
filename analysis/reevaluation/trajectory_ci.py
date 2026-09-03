#!/usr/bin/env python3
"""Recompute every seed-level trajectory band as a Student-t 95% CI.

The script covers the single-target GA/DPO/GRPO trajectories and the
single-/multi-target CPT ablations used in the manuscript figures. It exports
the numerical bands rather than redrawing the figures.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEEDS = range(5)
N_GRID = 400
ROLLING_WINDOW = 25


def cumulative_best(fitness, side_channels):
    fitness = np.asarray(fitness, dtype=float)
    best_so_far = np.inf
    best_index = 0
    owners = np.empty(len(fitness), dtype=int)
    for index, value in enumerate(fitness):
        if value < best_so_far:
            best_so_far = value
            best_index = index
        owners[index] = best_index
    result = {"fitness": fitness[owners]}
    for name, values in side_channels.items():
        result[name] = np.asarray(values)[owners]
    return result


def rolling_mean(values, window=ROLLING_WINDOW):
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy()
    )


def load_ga(path):
    frame = pd.read_csv(path)
    best_rows = frame.loc[frame.groupby("generation")["fitness"].idxmin()]
    best_rows = best_rows.sort_values("generation").reset_index(drop=True)
    x = frame.groupby("generation")["simulation_count"].max().sort_index().to_numpy()
    tracked = cumulative_best(best_rows.fitness, {
        "k": best_rows.k_eff,
        "gd": best_rows.g_count,
    })
    return x, tracked


def load_dpo_single(path):
    frame = pd.read_csv(path).sort_values("simulation_count").reset_index(drop=True)
    tracked = cumulative_best(frame.chosen_fitness, {
        "k": frame.chosen_k_eff,
        "gd": frame.chosen_g_count,
    })
    return frame.simulation_count.to_numpy(), tracked


def load_grpo_single(path):
    frame = pd.read_csv(path)
    best_rows = frame.loc[frame.groupby("step")["fitness"].idxmin()]
    best_rows = best_rows.sort_values("step").reset_index(drop=True)
    x = frame.groupby("step")["sim_count"].max().sort_index().to_numpy()
    tracked = cumulative_best(best_rows.fitness, {
        "k": best_rows.k_eff,
        "gd": best_rows.g_count,
    })
    return x, tracked


def load_dpo_multi(path):
    frame = pd.read_csv(path).sort_values("simulation_count").reset_index(drop=True)
    error_pcm = np.abs(frame.chosen_k_eff - frame.target_k) * 1e5
    return frame.simulation_count.to_numpy(), {
        "step_fitness_roll25": rolling_mean(frame.chosen_fitness),
        "target_error_pcm_roll25": rolling_mean(error_pcm),
    }


def load_grpo_multi(path):
    frame = pd.read_csv(path)
    best_rows = frame.loc[frame.groupby("step")["fitness"].idxmin()]
    best_rows = best_rows.sort_values("step").reset_index(drop=True)
    x = frame.groupby("step")["sim_count"].max().sort_index().to_numpy()
    error_pcm = np.abs(best_rows.k_eff - best_rows.target_k) * 1e5
    return x, {
        "step_fitness_roll25": rolling_mean(best_rows.fitness),
        "target_error_pcm_roll25": rolling_mean(error_pcm),
    }


def interpolate_common(seed_series, n_grid=N_GRID):
    x_min = max(np.nanmin(x) for x, _ in seed_series)
    x_max = min(np.nanmax(x) for x, _ in seed_series)
    common_x = np.linspace(x_min, x_max, n_grid)
    matrix = []
    for x, y in seed_series:
        order = np.argsort(x)
        matrix.append(np.interp(common_x, np.asarray(x)[order], np.asarray(y)[order]))
    return common_x, np.vstack(matrix)


def ci_rows(analysis, method, metric, common_x, matrix):
    n = np.sum(np.isfinite(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0, ddof=1)
    sem = sd / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, n - 1)
    half_width = t_critical * sem
    return pd.DataFrame({
        "analysis": analysis,
        "method": method,
        "metric": metric,
        "simulation_count": common_x,
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "t_critical": t_critical,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "confidence": 0.95,
        "ci_method": "two-sided Student-t interval across independent seeds",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root = args.data_root
    specs = [
        ("main_single", "GA_unconstrained", load_ga,
         root / "baselines/ga_baseline_single_seed{seed}_results.csv"),
        ("main_single", "GA_Gd16", load_ga,
         root / "baselines/ga_baseline_single_gd16_seed{seed}_results.csv"),
        ("main_single", "DPO_CPT_SFT", load_dpo_single,
         root / "alignment/dpo_single_target_seed{seed}_results.csv"),
        ("main_single", "GRPO_CPT_SFT", load_grpo_single,
         root / "alignment/grpo_cpt_sft_single_seed{seed}_results.csv"),
        ("cpt_ablation_single", "DPO_SFT_only", load_dpo_single,
         root / "alignment_sft_only/dpo_single_target_seed{seed}_results.csv"),
        ("cpt_ablation_single", "GRPO_SFT_only", load_grpo_single,
         root / "alignment_sft_only/grpo_sft_only_single_seed{seed}_results.csv"),
        ("multi_target", "DPO_CPT_SFT", load_dpo_multi,
         root / "alignment/dpo_multi_lhs_seed{seed}_results.csv"),
        ("multi_target", "GRPO_CPT_SFT", load_grpo_multi,
         root / "alignment/grpo_cpt_sft_multi_seed{seed}_results.csv"),
        ("cpt_ablation_multi", "DPO_SFT_only", load_dpo_multi,
         root / "alignment_sft_only/dpo_multi_lhs_seed{seed}_results.csv"),
        ("cpt_ablation_multi", "GRPO_SFT_only", load_grpo_multi,
         root / "alignment_sft_only/grpo_sft_only_multi_seed{seed}_results.csv"),
    ]

    output = []
    for analysis, method, loader, template in specs:
        loaded = []
        for seed in SEEDS:
            path = Path(str(template).format(seed=seed))
            if not path.exists():
                raise FileNotFoundError(path)
            loaded.append(loader(path))
        metrics = loaded[0][1].keys()
        for metric in metrics:
            series = [(x, values[metric]) for x, values in loaded]
            common_x, matrix = interpolate_common(series)
            output.append(ci_rows(analysis, method, metric, common_x, matrix))

    result = pd.concat(output, ignore_index=True)
    out_path = args.output_dir / "trajectory_student_t_95ci.csv"
    result.to_csv(out_path, index=False)

    audit = (
        result.groupby(["analysis", "method", "metric"], as_index=False)
        .agg(
            n_grid_points=("simulation_count", "size"),
            min_n_seeds=("n_seeds", "min"),
            max_n_seeds=("n_seeds", "max"),
            final_x=("simulation_count", "max"),
            final_mean=("mean", "last"),
            final_ci_lower=("ci_lower", "last"),
            final_ci_upper=("ci_upper", "last"),
        )
    )
    audit_path = args.output_dir / "trajectory_student_t_95ci_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Wrote {len(result):,} pointwise intervals to {out_path}")
    print(f"Wrote {len(audit):,} trajectory summaries to {audit_path}")
    print("Method: mean +/- t_(0.975, n-1) * sample_SD / sqrt(n); n=5 seeds.")


if __name__ == "__main__":
    main()
