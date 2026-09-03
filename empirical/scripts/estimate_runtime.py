#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from empirical_rl.data import ReactionCatalog
from empirical_rl.evaluation import evaluate_policy
from empirical_rl.model import ReactionPolicy
from empirical_rl.training import clone_policy, online_dpo, online_grpo, set_seed, train_cpt, train_sft


def timed(call):
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def estimate(config_path: Path, catalog: ReactionCatalog, calibration_steps: int) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    targets = [float(value) for value in config["targets"]]
    seeds = config.get("seeds", [config.get("seed", 0)])
    arms = config["initialization_arms"]
    set_seed(9173)
    policy = ReactionPolicy(catalog, len(targets), int(config["hidden_size"]))
    _, cpt_epoch = timed(
        lambda: train_cpt(
            policy, 1, float(config["supervised_learning_rate"]), int(config["batch_size"])
        )
    )
    _, sft_epoch = timed(
        lambda: train_sft(
            policy,
            targets,
            int(config["examples_per_task_target"]),
            1,
            float(config["supervised_learning_rate"]),
            int(config["batch_size"]),
        )
    )
    dpo_policy = clone_policy(policy)
    _, dpo_time = timed(
        lambda: online_dpo(
            dpo_policy,
            targets,
            calibration_steps,
            float(config["online_learning_rate"]),
            float(config["dpo_beta"]),
            float(config["temperature"]),
            config["objective"],
            101,
        )
    )
    grpo_policy = clone_policy(policy)
    _, grpo_time = timed(
        lambda: online_grpo(
            grpo_policy,
            targets,
            calibration_steps,
            int(config["grpo_group_size"]),
            float(config["online_learning_rate"]),
            float(config["temperature"]),
            config["objective"],
            202,
        )
    )
    _, evaluation_time = timed(lambda: evaluate_policy(policy, targets, config["objective"]))

    n_seeds = len(seeds)
    n_arms = len(arms)
    n_cpt_arms = sum(arm == "cpt_sft" for arm in arms)
    dpo_steps = int(config.get("dpo_steps", config.get("online_steps", 1000)))
    grpo_steps = int(config.get("grpo_steps", config.get("online_steps", 500)))
    phase_seconds = {
        "cpt": n_seeds * n_cpt_arms * int(config["cpt_epochs"]) * cpt_epoch,
        "sft": n_seeds * n_arms * int(config["sft_epochs"]) * sft_epoch,
        "dpo": n_seeds * n_arms * dpo_steps * dpo_time / calibration_steps,
        "grpo": n_seeds * n_arms * grpo_steps * grpo_time / calibration_steps,
        "evaluation_and_comparison": n_seeds * n_arms * 5 * evaluation_time,
    }
    raw_total = sum(phase_seconds.values())
    return {
        "config": str(config_path),
        "run_name": config["run_name"],
        "calibration_steps": calibration_steps,
        "measured_seconds": {
            "one_cpt_epoch": cpt_epoch,
            "one_sft_epoch": sft_epoch,
            "dpo_per_step": dpo_time / calibration_steps,
            "grpo_per_step": grpo_time / calibration_steps,
            "one_exact_evaluation": evaluation_time,
        },
        "estimated_phase_seconds": phase_seconds,
        "estimated_raw_seconds": raw_total,
        "estimated_with_checkpoint_io_seconds": raw_total * 1.15,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=[ROOT / "configs" / "full_single.json", ROOT / "configs" / "full.json"],
    )
    parser.add_argument("--calibration-steps", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "runtime_estimate.json")
    args = parser.parse_args()
    catalog = ReactionCatalog(ROOT / "data" / "buchwald_hartwig.csv")
    estimates = [estimate(path, catalog, args.calibration_steps) for path in args.configs]
    total = sum(float(item["estimated_with_checkpoint_io_seconds"]) for item in estimates)
    report = {
        "host_measurement": "local wall-clock calibration",
        "estimates": estimates,
        "full_pipeline_estimated_seconds": total,
        "full_pipeline_estimated_minutes": total / 60.0,
        "planning_range_minutes": [total / 60.0 * 0.8, total / 60.0 * 1.5],
        "note": "Range covers checkpoint I/O, thermal throttling, and short-calibration error.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
