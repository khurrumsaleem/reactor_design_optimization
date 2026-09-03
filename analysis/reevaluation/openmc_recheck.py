#!/usr/bin/env python3
"""Independent OpenMC re-evaluation of the 10 per-seed ReactorGen best designs.

The geometry, materials, source, tally, and default particle settings match the
single-target DPO/GRPO alignment scripts.  Unlike the training scripts, this
script records OpenMC's reported k standard deviation and supports independent
replicates with explicit transport seeds.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openmc
import pandas as pd


PITCH = 1.26
FUEL_R = 0.4096
CLAD_ID = 0.835
CLAD_OD = 0.950
ASSEMBLY_SIZE = 17 * PITCH
TARGET_K = 1.05

FIELDNAMES = [
    "method", "training_seed", "replicate", "openmc_seed",
    "particles", "batches", "inactive", "active_histories",
    "original_fitness", "original_k", "original_fq", "original_fdh",
    "g_count", "k_mean", "k_std", "dk_pcm", "fq", "fdh",
    "peaking_only", "fitness", "elapsed_sec", "grid",
]


def create_materials():
    fuel = openmc.Material(name="UO2 3.1wt%")
    fuel.add_nuclide("U235", 0.031)
    fuel.add_nuclide("U238", 0.969)
    fuel.add_nuclide("O16", 2.0)
    fuel.set_density("g/cm3", 10.29769)

    gd = openmc.Material(name="Gd Poison 8wt%")
    gd.add_nuclide("U235", 0.031 * 0.92)
    gd.add_nuclide("U238", 0.969 * 0.92)
    gd.add_nuclide("O16", 2.0 * 0.92 + 3.0 * 0.08)
    gd.add_nuclide("Gd155", 0.08 * 0.148)
    gd.add_nuclide("Gd156", 0.08 * 0.199)
    gd.add_nuclide("Gd157", 0.08 * 0.156)
    gd.add_nuclide("Gd158", 0.08 * 0.249)
    gd.add_nuclide("Gd160", 0.08 * 0.218)
    gd.set_density("g/cm3", 10.5)

    zirc = openmc.Material(name="Zircaloy-4")
    for element, fraction in [
        ("Sn", 0.014), ("Fe", 0.0121), ("Cr", 0.0107),
        ("Ni", 0.0050), ("Zr", 0.9582),
    ]:
        zirc.add_element(element, fraction)
    zirc.set_density("g/cm3", 6.55)

    water = openmc.Material(name="PWR Water")
    water.add_nuclide("H1", 2.0)
    water.add_nuclide("O16", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.set_density("g/cm3", 0.701)

    openmc.Materials([fuel, gd, zirc, water]).export_to_xml()
    return fuel, gd, zirc, water


def create_assembly(grid, fuel, gd, zirc, water):
    def pin_universe(rod_type):
        fuel_cyl = openmc.ZCylinder(r=FUEL_R)
        clad_inner = openmc.ZCylinder(r=CLAD_ID / 2)
        clad_outer = openmc.ZCylinder(r=CLAD_OD / 2)
        box = openmc.model.RectangularPrism(
            width=PITCH, height=PITCH, boundary_type="reflective"
        )
        fill = {0: fuel, 1: gd, 2: water}[rod_type]
        return openmc.Universe(cells=[
            openmc.Cell(fill=fill, region=-fuel_cyl),
            openmc.Cell(fill=None, region=+fuel_cyl & -clad_inner),
            openmc.Cell(fill=zirc, region=+clad_inner & -clad_outer),
            openmc.Cell(fill=water, region=-box & +clad_outer),
        ])

    lattice = openmc.RectLattice()
    lattice.pitch = (PITCH, PITCH)
    lattice.lower_left = (-ASSEMBLY_SIZE / 2, -ASSEMBLY_SIZE / 2)
    lattice.dimension = (17, 17)
    lattice.universes = np.array([
        [pin_universe(grid[i, j]) for j in range(17)] for i in range(17)
    ])
    outer_box = openmc.model.RectangularPrism(
        ASSEMBLY_SIZE, ASSEMBLY_SIZE, boundary_type="reflective"
    )
    return openmc.Geometry([openmc.Cell(fill=lattice, region=-outer_box)])


def load_best_designs(data_root: Path):
    alignment = data_root / "alignment"
    designs = []
    for training_seed in range(5):
        dpo = pd.read_csv(alignment / f"dpo_single_target_seed{training_seed}_results.csv")
        candidates = []
        for prefix in ("chosen", "rejected"):
            row = dpo.loc[dpo[f"{prefix}_fitness"].idxmin()]
            candidates.append((float(row[f"{prefix}_fitness"]), prefix, row))
        _, prefix, row = min(candidates, key=lambda item: item[0])
        designs.append({
            "method": "DPO", "training_seed": training_seed,
            "original_fitness": float(row[f"{prefix}_fitness"]),
            "original_k": float(row[f"{prefix}_k_eff"]),
            "original_fq": float(row[f"{prefix}_fq"]),
            "original_fdh": float(row[f"{prefix}_fdh"]),
            "g_count": int(row[f"{prefix}_g_count"]),
            "grid": str(row[f"{prefix}_grid"]),
        })

        grpo = pd.read_csv(
            alignment / f"grpo_cpt_sft_single_seed{training_seed}_results.csv"
        )
        row = grpo.loc[grpo["fitness"].idxmin()]
        designs.append({
            "method": "GRPO", "training_seed": training_seed,
            "original_fitness": float(row["fitness"]),
            "original_k": float(row["k_eff"]),
            "original_fq": float(row["fq"]),
            "original_fdh": float(row["fdh"]),
            "g_count": int(row["g_count"]),
            "grid": str(row["grid"]),
        })

    for design in designs:
        if len(design["grid"]) != 289:
            raise ValueError(
                f"{design['method']} seed {design['training_seed']} grid has "
                f"length {len(design['grid'])}, expected 289"
            )
    return designs


def evaluate_task(task):
    (
        design, replicate, openmc_seed, particles, batches, inactive,
        threads, openmc_exec,
    ) = task
    started = time.time()
    work_dir = tempfile.mkdtemp(prefix="reactorgen_recheck_")
    original_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        fuel, gd, zirc, water = create_materials()
        grid = np.zeros((17, 17), dtype=int)
        for index, char in enumerate(design["grid"]):
            if char == "g":
                grid[divmod(index, 17)] = 1
            elif char == "c":
                grid[divmod(index, 17)] = 2

        create_assembly(grid, fuel, gd, zirc, water).export_to_xml()
        settings = openmc.Settings()
        settings.batches = batches
        settings.inactive = inactive
        settings.particles = particles
        settings.seed = openmc_seed
        settings.output = {"summary": False}
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box(
                [-ASSEMBLY_SIZE / 2, -ASSEMBLY_SIZE / 2, 0],
                [ASSEMBLY_SIZE / 2, ASSEMBLY_SIZE / 2, 1],
            )
        )
        settings.export_to_xml()

        mesh = openmc.RegularMesh()
        mesh.dimension = (17, 17, 1)
        mesh.lower_left = (-ASSEMBLY_SIZE / 2, -ASSEMBLY_SIZE / 2, 0)
        mesh.upper_right = (ASSEMBLY_SIZE / 2, ASSEMBLY_SIZE / 2, 1)
        power_tally = openmc.Tally(name="power")
        power_tally.filters = [openmc.MeshFilter(mesh)]
        power_tally.scores = ["fission"]
        openmc.Tallies([power_tally]).export_to_xml()

        openmc.run(output=False, threads=threads, openmc_exec=openmc_exec)
        with openmc.StatePoint(f"statepoint.{batches}.h5") as statepoint:
            k_mean = float(statepoint.keff.nominal_value)
            k_std = float(statepoint.keff.std_dev)
            power = statepoint.get_tally(name="power").mean.ravel()
            fq = float(power.max() / power.mean())
            channel = power.reshape(17, 17).mean(axis=0)
            fdh = float(channel.max() / channel.mean())

        peaking_only = 0.6 * fq + 0.4 * fdh
        fitness = peaking_only + 100.0 * abs(k_mean - TARGET_K)
        return {
            **design,
            "replicate": replicate,
            "openmc_seed": openmc_seed,
            "particles": particles,
            "batches": batches,
            "inactive": inactive,
            "active_histories": particles * (batches - inactive),
            "k_mean": k_mean,
            "k_std": k_std,
            "dk_pcm": abs(k_mean - TARGET_K) * 1e5,
            "fq": fq,
            "fdh": fdh,
            "peaking_only": peaking_only,
            "fitness": fitness,
            "elapsed_sec": time.time() - started,
        }
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


def write_summary(results_csv: Path, summary_csv: Path):
    data = pd.read_csv(results_csv)
    per_design = (
        data.groupby(["method", "training_seed"], as_index=False)
        .agg(
            original_k=("original_k", "first"),
            original_dk_pcm=("original_k", lambda x: abs(x.iloc[0] - TARGET_K) * 1e5),
            g_count=("g_count", "first"),
            replicates=("replicate", "count"),
            recheck_k_mean=("k_mean", "mean"),
            between_run_k_sd=("k_mean", "std"),
            mean_openmc_k_std=("k_std", "mean"),
            recheck_fq_mean=("fq", "mean"),
            recheck_fdh_mean=("fdh", "mean"),
            recheck_fitness_mean=("fitness", "mean"),
        )
    )
    per_design["recheck_dk_pcm"] = (
        (per_design["recheck_k_mean"] - TARGET_K).abs() * 1e5
    )
    per_design["between_run_k_sd_pcm"] = per_design["between_run_k_sd"] * 1e5
    per_design["mean_openmc_k_std_pcm"] = per_design["mean_openmc_k_std"] * 1e5
    per_design["mean_k_se_pcm"] = (
        per_design["between_run_k_sd_pcm"] / np.sqrt(per_design["replicates"])
    )
    per_design.to_csv(summary_csv, index=False)

    print("\nPer-design independent re-evaluation")
    print(per_design[[
        "method", "training_seed", "g_count", "replicates",
        "original_dk_pcm", "recheck_dk_pcm", "between_run_k_sd_pcm",
        "mean_openmc_k_std_pcm", "mean_k_se_pcm",
    ]].to_string(index=False, float_format=lambda value: f"{value:.1f}"))

    print("\nMethod-level summary across five selected designs")
    for method, group in per_design.groupby("method"):
        print(
            f"{method}: original |dk|={group.original_dk_pcm.mean():.1f} +/- "
            f"{group.original_dk_pcm.std():.1f} pcm; independently re-evaluated "
            f"|mean k-target|={group.recheck_dk_pcm.mean():.1f} +/- "
            f"{group.recheck_dk_pcm.std():.1f} pcm; typical per-run OpenMC "
            f"sigma={group.mean_openmc_k_std_pcm.mean():.1f} pcm"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--particles", type=int, default=20_000)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--inactive", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--openmc-exec",
        default="openmc",
    )
    args = parser.parse_args()

    if not 0 < args.inactive < args.batches:
        parser.error("inactive must be positive and smaller than batches")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "openmc_recheck_raw.csv"
    summary_csv = args.output_dir / "openmc_recheck_per_design.csv"
    designs = load_best_designs(args.data_root)

    completed = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        completed = set(
            zip(existing.method, existing.training_seed, existing.replicate)
        )

    tasks = []
    for design in designs:
        method_offset = 0 if design["method"] == "DPO" else 5_000_000
        for replicate in range(args.replicates):
            key = (design["method"], design["training_seed"], replicate)
            if key in completed:
                continue
            openmc_seed = (
                1_000_003 + method_offset + design["training_seed"] * 10_000 + replicate
            )
            tasks.append((
                design, replicate, openmc_seed,
                args.particles, args.batches, args.inactive,
                args.threads, args.openmc_exec,
            ))

    print(
        f"Loaded {len(designs)} designs; running {len(tasks)} remaining "
        f"evaluations with {args.workers} workers."
    )
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            handle.flush()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate_task, task): task for task in tasks}
            for finished, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                writer.writerow({name: result[name] for name in FIELDNAMES})
                handle.flush()
                print(
                    f"[{finished}/{len(tasks)}] {result['method']} training seed "
                    f"{result['training_seed']} replicate {result['replicate']}: "
                    f"k={result['k_mean']:.6f} +/- {result['k_std']:.6f}, "
                    f"|dk|={result['dk_pcm']:.1f} pcm, "
                    f"{result['elapsed_sec']:.1f} s",
                    flush=True,
                )

    write_summary(results_csv, summary_csv)


if __name__ == "__main__":
    main()
