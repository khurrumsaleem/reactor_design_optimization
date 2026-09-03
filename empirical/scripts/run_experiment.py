#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from empirical_rl.data import ReactionCatalog
from empirical_rl.evaluation import (
    algorithm_ligand_nmi,
    evaluate_policy,
    evaluate_uniform,
    policy_jsd,
)
from empirical_rl.model import ReactionPolicy
from empirical_rl.training import (
    atomic_torch_save,
    clone_policy,
    online_dpo,
    online_grpo,
    set_seed,
    train_cpt,
    train_sft,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = [
        "expected_reward",
        "expected_abs_error",
        "expansion_mass",
        "ligand_entropy",
        "full_oracle_abs_error",
        "restricted_oracle_abs_error",
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def fingerprint(config: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def remove_if_fresh(path: Path, resume: bool) -> None:
    if not resume and path.exists():
        path.unlink()


def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    config_hash = fingerprint(config)
    seeds = config.get("seeds", [config.get("seed", 0)])
    targets = [float(value) for value in config["targets"]]
    dpo_steps = int(config.get("dpo_steps", config.get("online_steps", 1000)))
    grpo_steps = int(config.get("grpo_steps", config.get("online_steps", 500)))
    checkpoint_every = int(config.get("checkpoint_every", 25))
    output = args.output or ROOT / "results" / str(config["run_name"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    catalog = ReactionCatalog(args.data)

    all_metrics: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    baseline_metrics: dict[str, list[dict[str, object]]] = {}
    for baseline, restricted in (("random_full", False), ("random_restricted", True)):
        rows = evaluate_uniform(catalog, targets, config["objective"], restricted)
        for row in rows:
            row.update(seed="pooled", arm="baseline", algorithm=baseline)
        all_metrics.extend(rows)
        baseline_metrics[baseline] = rows

    completed_jobs: list[str] = []
    for seed in seeds:
        for arm in config["initialization_arms"]:
            job = f"seed={seed}, arm={arm}"
            print(f"\n=== {job} ===", flush=True)
            set_seed(int(seed))
            base = ReactionPolicy(catalog, len(targets), int(config["hidden_size"]))
            base_path = checkpoints / f"{arm}_base_seed{seed}.pt"
            cpt_path = checkpoints / f"{arm}_cpt_seed{seed}.resume.pt"
            sft_path = checkpoints / f"{arm}_sft_seed{seed}.resume.pt"
            cpt_losses: list[float] = []
            sft_losses: list[float]

            if args.resume and base_path.exists():
                state = torch.load(base_path, map_location=base.device, weights_only=False)
                if state.get("config_fingerprint") != config_hash:
                    raise ValueError(f"base checkpoint/config mismatch: {base_path}")
                base.load_state_dict(state["model"])
                cpt_losses = list(state["cpt_losses"])
                sft_losses = list(state["sft_losses"])
                print("[base] loaded completed SFT checkpoint", flush=True)
            else:
                remove_if_fresh(cpt_path, args.resume)
                remove_if_fresh(sft_path, args.resume)
                remove_if_fresh(base_path, args.resume)
                if arm == "cpt_sft":
                    cpt_losses = train_cpt(
                        base,
                        int(config["cpt_epochs"]),
                        float(config["supervised_learning_rate"]),
                        int(config["batch_size"]),
                        checkpoint_path=cpt_path,
                        resume=args.resume,
                    )
                sft_losses = train_sft(
                    base,
                    targets,
                    int(config["examples_per_task_target"]),
                    int(config["sft_epochs"]),
                    float(config["supervised_learning_rate"]),
                    int(config["batch_size"]),
                    checkpoint_path=sft_path,
                    resume=args.resume,
                )
                atomic_torch_save(
                    {
                        "config_fingerprint": config_hash,
                        "model": base.state_dict(),
                        "cpt_losses": cpt_losses,
                        "sft_losses": sft_losses,
                    },
                    base_path,
                )
                print("[base] SFT checkpoint complete", flush=True)

            base_metrics = evaluate_policy(base, targets, config["objective"])
            for row in base_metrics:
                row.update(seed=seed, arm=arm, algorithm="sft_base")
            all_metrics.extend(base_metrics)
            aligned: dict[str, ReactionPolicy] = {}
            for algorithm in ("dpo", "grpo"):
                policy = clone_policy(base)
                online_path = checkpoints / f"{arm}_{algorithm}_seed{seed}.resume.pt"
                remove_if_fresh(online_path, args.resume)
                if algorithm == "dpo":
                    history = online_dpo(
                        policy,
                        targets,
                        dpo_steps,
                        float(config["online_learning_rate"]),
                        float(config["dpo_beta"]),
                        float(config["temperature"]),
                        config["objective"],
                        int(seed) + 1000,
                        checkpoint_path=online_path,
                        checkpoint_every=checkpoint_every,
                        resume=args.resume,
                    )
                else:
                    history = online_grpo(
                        policy,
                        targets,
                        grpo_steps,
                        int(config["grpo_group_size"]),
                        float(config["online_learning_rate"]),
                        float(config["temperature"]),
                        config["objective"],
                        int(seed) + 2000,
                        checkpoint_path=online_path,
                        checkpoint_every=checkpoint_every,
                        resume=args.resume,
                    )
                for row in history:
                    row.update(seed=seed, arm=arm, algorithm=algorithm)
                all_history.extend(history)
                metrics = evaluate_policy(policy, targets, config["objective"])
                for row in metrics:
                    row.update(seed=seed, arm=arm, algorithm=algorithm)
                all_metrics.extend(metrics)
                aligned[algorithm] = policy
                atomic_torch_save(
                    policy.state_dict(), checkpoints / f"{arm}_{algorithm}_seed{seed}.pt"
                )
                completed_jobs.append(f"{job}, algorithm={algorithm}")
                write_json(
                    output / "progress.json",
                    {
                        "config": str(args.config),
                        "resume_enabled": args.resume,
                        "completed_jobs": completed_jobs,
                    },
                )
            comparisons.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "dpo_grpo_jsd": policy_jsd(aligned["dpo"], aligned["grpo"], targets),
                    "algorithm_ligand_nmi": algorithm_ligand_nmi(
                        aligned["dpo"], aligned["grpo"], targets
                    ),
                    "cpt_final_loss": cpt_losses[-1] if cpt_losses else None,
                    "sft_final_loss": sft_losses[-1],
                }
            )

    write_csv(output / "metrics.csv", all_metrics)
    write_csv(output / "trajectories.csv", all_history)
    write_csv(output / "comparisons.csv", comparisons)
    grouped: dict[str, list[dict[str, object]]] = {}
    for arm in config["initialization_arms"]:
        for algorithm in ("sft_base", "dpo", "grpo"):
            key = f"{arm}/{algorithm}"
            grouped[key] = [
                row for row in all_metrics if row["arm"] == arm and row["algorithm"] == algorithm
            ]
    aggregates = {key: aggregate(rows) for key, rows in grouped.items()}
    deltas: dict[str, dict[str, float]] = {}
    for arm in config["initialization_arms"]:
        base_metrics_for_arm = aggregates[f"{arm}/sft_base"]
        for algorithm in ("dpo", "grpo"):
            aligned_metrics = aggregates[f"{arm}/{algorithm}"]
            deltas[f"{arm}/{algorithm}"] = {
                "expected_abs_error_change": aligned_metrics["expected_abs_error"]
                - base_metrics_for_arm["expected_abs_error"],
                "expansion_mass_change": aligned_metrics["expansion_mass"]
                - base_metrics_for_arm["expansion_mass"],
            }
    summary = {
        "config": config,
        "dataset": {
            "measured_rows": len(catalog.reactions),
            "restricted_rows": len(catalog.restricted_row_ids),
            "ligands": catalog.n_ligands,
            "additives": catalog.n_additives,
            "bases": catalog.n_bases,
            "aryl_halides": catalog.n_aryls,
        },
        "baselines": {key: aggregate(rows) for key, rows in baseline_metrics.items()},
        "aggregates": aggregates,
        "deltas_from_sft": deltas,
        "comparisons": comparisons,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "progress.json",
        {
            "config": str(args.config),
            "resume_enabled": args.resume,
            "complete": True,
            "completed_jobs": completed_jobs,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "buchwald_hartwig.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume CPT/SFT epochs and DPO/GRPO steps from atomic checkpoints",
    )
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted safely. Re-run the same command with --resume.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
