#!/usr/bin/env python3
"""High-statistics confirmation for selected ReactorGen best designs.

By default this evaluates only the globally best DPO and GRPO layouts (two
layouts total), using 20 million active histories per layout.  It does not run
training and does not alter the archived trajectory CSVs.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path

import pandas as pd

from openmc_recheck import FIELDNAMES, evaluate_task, load_best_designs, write_summary


def select_designs(designs, selection):
    if selection == "all-per-seed":
        return designs
    selected = []
    for method in ("DPO", "GRPO"):
        candidates = [design for design in designs if design["method"] == method]
        selected.append(min(candidates, key=lambda design: design["original_fitness"]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument(
        "--selection", choices=("global-best", "all-per-seed"),
        default="global-best",
        help="global-best evaluates 2 layouts; all-per-seed evaluates 10 layouts",
    )
    parser.add_argument(
        "--particles", type=int, default=200_000,
        help="particles per batch (original experiment: 20,000)",
    )
    parser.add_argument("--batches", type=int, default=110)
    parser.add_argument("--inactive", type=int, default=10)
    parser.add_argument(
        "--replicates", type=int, default=1,
        help="independent high-statistics transport seeds per layout",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--openmc-exec",
        default="openmc",
    )
    args = parser.parse_args()

    if not 0 < args.inactive < args.batches:
        parser.error("inactive must be positive and smaller than batches")
    if args.replicates < 1:
        parser.error("replicates must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / f"openmc_high_stat_{args.selection}_raw.csv"
    summary_csv = args.output_dir / f"openmc_high_stat_{args.selection}_summary.csv"

    designs = select_designs(load_best_designs(args.data_root), args.selection)
    active_histories = args.particles * (args.batches - args.inactive)
    print(
        f"Prepared {len(designs)} layouts, {args.replicates} replicate(s) each; "
        f"{active_histories:,} active histories per run."
    )
    for design in designs:
        print(
            f"  {design['method']} training seed {design['training_seed']}: "
            f"original fitness={design['original_fitness']:.4f}, "
            f"k={design['original_k']:.5f}, Gd={design['g_count']}"
        )

    completed = set()
    if raw_csv.exists():
        existing = pd.read_csv(raw_csv)
        completed = set(zip(existing.method, existing.training_seed, existing.replicate))

    tasks = []
    for design in designs:
        method_offset = 0 if design["method"] == "DPO" else 5_000_000
        for replicate in range(args.replicates):
            key = (design["method"], design["training_seed"], replicate)
            if key in completed:
                continue
            openmc_seed = (
                9_000_001 + method_offset + design["training_seed"] * 10_000 + replicate
            )
            tasks.append((
                design, replicate, openmc_seed,
                args.particles, args.batches, args.inactive,
                args.threads, args.openmc_exec,
            ))

    print(f"Running {len(tasks)} remaining high-statistics evaluation(s).")
    write_header = not raw_csv.exists()
    with raw_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            handle.flush()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_task, task) for task in tasks]
            for finished, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                writer.writerow({name: result[name] for name in FIELDNAMES})
                handle.flush()
                print(
                    f"[{finished}/{len(tasks)}] {result['method']} seed "
                    f"{result['training_seed']}: k={result['k_mean']:.7f} +/- "
                    f"{result['k_std']:.7f}, |dk|={result['dk_pcm']:.1f} pcm, "
                    f"elapsed={result['elapsed_sec']:.1f} s",
                    flush=True,
                )

    if raw_csv.exists() and raw_csv.stat().st_size > 0:
        write_summary(raw_csv, summary_csv)


if __name__ == "__main__":
    main()
