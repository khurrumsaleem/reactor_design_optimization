#!/usr/bin/env python3
"""Independently aggregate archived records for the NMI appeal manuscript.

Only Python's standard library is required. No training, OpenMC evaluation,
network access, weight deserialization, or changes to archived data occur.
Chemistry expectations are aggregated from the archived exact-enumeration
metrics; this script does not re-enumerate a neural policy from its weights.

Usage:
  python code/validation/verify_manuscript_results.py --package-root .
  python code/validation/verify_manuscript_results.py --package-root . \
      --output ../manuscript_validation.json

Exit status is 0 only if every executed assertion passes. Missing required
records are failures. --output must be outside the package so running this
check does not change the archive or invalidate its integrity manifest.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys


TARGET_K = 1.05
WINDOW = 500 / 100000
SEEDS = range(5)
CHEMISTRY_EXPECTATIONS = {
    "registered_single": {
        "sft_base": (50.039876475928004, 0.28680705661216055),
        "dpo": (29.886111496509862, 0.7007543555163248),
        "grpo": (29.77065949826603, 0.7142549902543787),
    },
    "registered_multi": {
        "sft_base": (28.97135064880494, 0.036053219483282835),
        "dpo": (24.28286525571156, 0.07980186896250167),
        "grpo": (24.110205789960936, 0.07430508243687081),
    },
}


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def hit(row):
    return abs(float(row["k_eff"]) - TARGET_K) <= WINDOW + 1e-12


def close(observed, expected, tolerance=1e-10):
    return math.isfinite(observed) and abs(observed - expected) <= tolerance


class Verification:
    def __init__(self, root):
        self.root = root
        self.checks = []
        self.notes = []
        self.derived = {}

    def equal(self, name, observed, expected):
        self.checks.append({"check": name, "passed": observed == expected,
                            "expected": expected, "observed": observed})

    def near(self, name, observed, expected, tolerance=1e-10):
        self.checks.append({"check": name,
                            "passed": close(observed, expected, tolerance),
                            "expected": expected, "observed": observed,
                            "absolute_tolerance": tolerance})

    def run(self, name, operation):
        try:
            operation()
        except (OSError, ValueError, KeyError, IndexError,
                ZeroDivisionError, statistics.StatisticsError) as error:
            self.checks.append({"check": name + ": required records readable",
                                "passed": False,
                                "expected": "complete, readable source records",
                                "observed": type(error).__name__ + ": " + str(error)})

    def corpus(self):
        source = self.root / "data/corpus/reactor_10k_final.csv"
        data = rows(source)
        self.equal("corpus rows", len(data), 10000)
        self.equal("corpus layout lengths", sorted({len(x["grid_str"]) for x in data}), [289])
        self.equal("corpus absorber counts", sorted({x["grid_str"].count("g") for x in data}), [16])
        self.equal("corpus guide counts", sorted({x["grid_str"].count("c") for x in data}), [25])
        guide_positions = {tuple(i for i, char in enumerate(x["grid_str"]) if char == "c")
                           for x in data}
        self.equal("corpus fixed guide positions", len(guide_positions), 1)
        self.equal("corpus alphabet", sorted(set("".join(x["grid_str"] for x in data))), ["c", "f", "g"])
        self.equal("corpus target-window candidates", sum(hit(x) for x in data), 0)
        self.near("corpus nearest absolute target error (pcm)",
                  min(abs(float(x["k_eff"]) - TARGET_K) * 1e5 for x in data), 11192, 1e-6)

    def alignment(self):
        all_paths = []
        selected = {}
        for folder, grpo_prefix in (("alignment", "grpo_cpt_sft"),
                                    ("alignment_sft_only", "grpo_sft_only")):
            expected_names = {
                f"{prefix}_seed{seed}_results.csv"
                for prefix in ("dpo_single_target", "dpo_multi_lhs",
                               grpo_prefix + "_single", grpo_prefix + "_multi")
                for seed in SEEDS
            }
            paths = sorted((self.root / "data" / folder).glob("*.csv"))
            self.equal(folder + " file set", sorted(p.name for p in paths), sorted(expected_names))
            all_paths.extend(paths)
            for path in paths:
                data = rows(path)
                dpo = path.name.startswith("dpo_")
                label = folder + "/" + path.name
                self.equal(label + " rows", len(data), 1000 if dpo else 2000)
                self.equal(label + " final step", int(data[-1]["step"]), 1000 if dpo else 500)
                self.equal(label + " evaluations", int(data[-1]["simulation_count" if dpo else "sim_count"]), 2000)
                if folder == "alignment" and "single" in path.name:
                    seed = int(path.stem.split("seed")[1].split("_")[0])
                    if dpo:
                        selected[("DPO", seed)] = {
                            "inventory": int(data[-1]["best_g_count"]),
                            "objective": float(data[-1]["best_fitness"]),
                            "k": float(data[-1]["best_k_eff"]),
                        }
                    else:
                        best = min(data, key=lambda x: float(x["fitness"]))
                        selected[("GRPO", seed)] = {
                            "inventory": int(best["g_count"]),
                            "objective": float(best["fitness"]),
                            "k": float(best["k_eff"]),
                        }
        self.equal("alignment total files", len(all_paths), 40)
        self.equal("principal selected policies", len(selected), 10)
        for method, expected_range in (("DPO", [28, 35]), ("GRPO", [28, 29])):
            values = [selected[(method, s)] for s in SEEDS]
            inventories = [x["inventory"] for x in values]
            self.equal(method + " selected inventory range", [min(inventories), max(inventories)], expected_range)
            self.near(method + " mean best objective (rounded manuscript)",
                      statistics.mean(x["objective"] for x in values), 1.65 if method == "DPO" else 1.66, .005)
            self.near(method + " best-objective sample SD (rounded manuscript)",
                      statistics.stdev(x["objective"] for x in values), .04 if method == "DPO" else .05, .005)
        self.derived["selected_policies"] = [dict(method=m, seed=s, **v)
                                              for (m, s), v in sorted(selected.items())]

    def generation(self):
        base = self.root / "data/generation_yield"
        samples = rows(base / "generation_yield_samples.csv")
        evaluated = rows(base / "generation_yield_openmc_subsample.csv")
        expected_keys = [(m, str(s)) for m in ("DPO", "GRPO") for s in SEEDS]
        self.equal("unselected generation pool", len(samples), 10000)
        self.equal("unselected generation per checkpoint", dict(Counter((x["algo"], x["seed"]) for x in samples)),
                   {key: 1000 for key in expected_keys})
        self.equal("evaluated unselected generations", len(evaluated), 500)
        self.equal("evaluated generations per checkpoint", dict(Counter((x["algo"], x["seed"]) for x in evaluated)),
                   {key: 50 for key in expected_keys})
        self.equal("corrected generation length and count consistency",
                   all(len(x["grid"]) == 289 and x["grid"].count("g") == int(x["gd_count"]) for x in samples), True)
        self.near("corrected inventory mean (rounded manuscript)", statistics.mean(int(x["gd_count"]) for x in samples), 35.8, .05)
        self.near("corrected inventory sample SD (rounded manuscript)", statistics.stdev(int(x["gd_count"]) for x in samples), 14.3, .05)
        expected = {"DPO": (33, 250), "GRPO": (68, 250), "Pooled": (101, 500)}
        for method, (hits, denominator) in expected.items():
            subset = evaluated if method == "Pooled" else [x for x in evaluated if x["algo"] == method]
            observed_hits = sum(hit(x) for x in subset)
            self.equal(method + " unselected target hits", observed_hits, hits)
            self.equal(method + " unselected denominator", len(subset), denominator)
            self.near(method + " unselected target incidence (%)", 100 * observed_hits / len(subset), 100 * hits / denominator)
        conditioned = [x for x in evaluated if 28 <= int(x["gd"]) <= 35]
        self.equal("aligned inventory 28--35 evaluated count", len(conditioned), 282)
        self.equal("aligned inventory 28--35 target hits", sum(hit(x) for x in conditioned), 83)
        self.derived["incidence"] = {method: {"hits": values[0], "n": values[1]}
                                     for method, values in expected.items()}

    def random_search(self):
        directory = self.root / "data/oracle"
        for sampler, prefix, expected_total, expected_interval in (
                ("Informed", "informed_random_search", 521, [20, 40]),
                ("Uniform", "uninformed_random", 36, [0, 264])):
            combined = []
            per_seed_hits = []
            best_objectives = []
            for seed in SEEDS:
                data = rows(directory / f"{prefix}_seed{seed}_results.csv")
                self.equal(f"{sampler} seed{seed} evaluations", len(data), 2000)
                counts = [int(x["g_count"]) for x in data]
                self.equal(f"{sampler} seed{seed} inventories in supplied interval",
                           all(expected_interval[0] <= c <= expected_interval[1] for c in counts), True)
                per_seed_hits.append(sum(hit(x) for x in data))
                best_objectives.append(min(float(x["fitness"]) for x in data))
                combined.extend(data)
            self.equal(sampler + " total evaluations", len(combined), 10000)
            self.equal(sampler + " target hits", sum(per_seed_hits), expected_total)
            self.near(sampler + " target incidence (%)", 100 * sum(per_seed_hits) / len(combined), expected_total / 100)
            conditioned = [x for x in combined if 28 <= int(x["g_count"]) <= 35]
            self.equal(sampler + " inventory 28--35 count", len(conditioned), 3830 if sampler == "Informed" else 281)
            self.equal(sampler + " target hits outside inventory 28--35",
                       sum(hit(x) for x in combined if not 28 <= int(x["g_count"]) <= 35), 0)
            if sampler == "Uniform":
                self.equal("Uniform target hits per seed", per_seed_hits, [6, 11, 10, 5, 4])
            else:
                self.near("Informed mean best objective (rounded manuscript)", statistics.mean(best_objectives), 1.65, .005)
                self.near("Informed best-objective sample SD (rounded manuscript)", statistics.stdev(best_objectives), .04, .005)
            self.derived.setdefault("incidence", {})[sampler] = {"hits": sum(per_seed_hits), "n": len(combined)}
            self.derived[sampler.lower() + "_search"] = {
                "per_seed_target_hits": per_seed_hits,
                "per_seed_best_objective": best_objectives,
                "inventory_28_35_count": len(conditioned),
            }

    def objective_removal(self):
        removed = []
        full = []
        for seed, expected in enumerate([0, 0, 0, 0, 39.29]):
            path = self.root / "data/no_penalty_control" / f"no_penalty_dpo_seed{seed}_results.csv"
            data = rows(path)
            self.equal(f"no-penalty seed{seed} steps", len(data), 1000)
            self.equal(f"no-penalty seed{seed} evaluator calls", int(data[-1]["simulation_count"]), 2000)
            values = [float(x["chosen_g_count"]) for x in data[-100:]]
            endpoint = statistics.mean(values)
            self.near(f"no-penalty seed{seed} final-100 selected inventory", endpoint, expected)
            if seed < 4:
                self.equal(f"no-penalty seed{seed} all final-100 selected inventories zero", all(x == 0 for x in values), True)
            removed.append(endpoint)
            matched = rows(self.root / "data/alignment" / f"dpo_single_target_seed{seed}_results.csv")
            full.append(statistics.mean(float(x["chosen_g_count"]) for x in matched[-100:]))
        self.equal("objective removal zero-inventory seeds", sum(x == 0 for x in removed), 4)
        for name, values, expected_mean, expected_sd in (("removed", removed, 7.86, 17.57), ("full", full, 35.73, 12.02)):
            self.near(name + " objective final-100 inventory mean (rounded manuscript)", statistics.mean(values), expected_mean, .005)
            self.near(name + " objective final-100 inventory sample SD (rounded manuscript)", statistics.stdev(values), expected_sd, .005)
        self.derived["objective_removal"] = {"full_objective_seed_means": full,
                                              "removed_penalty_seed_means": removed}

    def high_statistics(self):
        source = self.root / "data/reevaluation/openmc_high_stat_all-per-seed_raw.csv"
        data = rows(source)
        self.equal("high-statistics layout count", len(data), 10)
        self.equal("high-statistics method/seed pairs", sorted((x["method"], int(x["training_seed"])) for x in data),
                   [(m, s) for m in ("DPO", "GRPO") for s in SEEDS])
        self.equal("high-statistics active histories", sorted({int(x["active_histories"]) for x in data}), [20000000])
        self.equal("high-statistics target attainment", sum(abs(float(x["k_mean"]) - TARGET_K) <= WINDOW + 1e-12 for x in data), 10)
        for x in data:
            name = x["method"] + " seed" + x["training_seed"]
            self.near(name + " stored high-statistics target error (pcm)",
                      float(x["dk_pcm"]), abs(float(x["k_mean"]) - TARGET_K) * 1e5, 1e-6)
            self.near(name + " high-statistics objective decomposition",
                      float(x["fitness"]), .6 * float(x["fq"]) + .4 * float(x["fdh"]) + 100 * abs(float(x["k_mean"]) - TARGET_K), 1e-10)
        self.derived["high_statistics"] = {}
        for method, error_mean, error_sd, objective_mean, objective_sd in (
                ("DPO", 102, 71, 1.515, .062), ("GRPO", 139, 130, 1.551, .134)):
            subset = [x for x in data if x["method"] == method]
            errors = [abs(float(x["k_mean"]) - TARGET_K) * 1e5 for x in subset]
            objectives = [float(x["fitness"]) for x in subset]
            self.near(method + " high-statistics error mean (rounded manuscript)", statistics.mean(errors), error_mean, .5)
            self.near(method + " high-statistics error sample SD (rounded manuscript)", statistics.stdev(errors), error_sd, .5)
            self.near(method + " high-statistics objective mean (rounded manuscript)", statistics.mean(objectives), objective_mean, .0005)
            self.near(method + " high-statistics objective sample SD (rounded manuscript)", statistics.stdev(objectives), objective_sd, .0005)
            self.derived["high_statistics"][method] = {"mean_abs_error_pcm": statistics.mean(errors),
                                                       "sd_abs_error_pcm": statistics.stdev(errors),
                                                       "mean_objective": statistics.mean(objectives),
                                                       "sd_objective": statistics.stdev(objectives)}

    def chemistry(self, experiment):
        directory = self.root / "data/empirical" / experiment
        data = rows(directory / "metrics.csv")
        trajectories = rows(directory / "trajectories.csv")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        task_targets = 15 if experiment == "registered_single" else 150
        self.equal(experiment + " exact-enumeration metric rows", len(data), 480 if task_targets == 15 else 4800)
        self.equal(experiment + " trajectory rows", len(trajectories), 40000)
        expected_groups = {(arm, alg, str(seed)): 2000
                           for arm in ("cpt_sft", "sft_only")
                           for alg in ("dpo", "grpo") for seed in SEEDS}
        self.equal(experiment + " reward lookups per run",
                   dict(Counter((x["arm"], x["algorithm"], x["seed"]) for x in trajectories)), expected_groups)
        self.equal(experiment + " measured catalog count", summary["dataset"]["measured_rows"], 3955)
        self.equal(experiment + " restricted catalog count", summary["dataset"]["restricted_rows"], 990)
        self.equal(experiment + " expected reward/error consistency",
                   all(close(float(x["expected_reward"]), -float(x["expected_abs_error"]) / 100) for x in data), True)
        self.equal(experiment + " off-corpus probability bounds",
                   all(0 <= float(x["expansion_mass"]) <= 1 for x in data), True)
        expected_metric_groups = {(arm, alg, str(seed)): task_targets
                                  for arm in ("cpt_sft", "sft_only")
                                  for alg in ("sft_base", "dpo", "grpo") for seed in SEEDS}
        expected_metric_groups.update({("baseline", alg, "pooled"): task_targets
                                       for alg in ("random_full", "random_restricted")})
        self.equal(experiment + " exact-enumeration grouping",
                   dict(Counter((x["arm"], x["algorithm"], x["seed"]) for x in data)), expected_metric_groups)
        aggregates = {}
        for arm in ("cpt_sft", "sft_only"):
            for alg in ("sft_base", "dpo", "grpo"):
                subset = [x for x in data if x["arm"] == arm and x["algorithm"] == alg]
                values = {key: statistics.mean(float(x[key]) for x in subset)
                          for key in ("expected_abs_error", "expansion_mass")}
                group = arm + "/" + alg
                aggregates[group] = values
                for key, observed in values.items():
                    self.near(experiment + " " + group + " " + key + " vs archived summary",
                              observed, float(summary["aggregates"][group][key]))
                if arm == "cpt_sft":
                    expected_error, expected_mass = CHEMISTRY_EXPECTATIONS[experiment][alg]
                    self.near(experiment + " CPT+SFT " + alg + " expected yield error (pp)", values["expected_abs_error"], expected_error)
                    self.near(experiment + " CPT+SFT " + alg + " off-corpus ligand mass", values["expansion_mass"], expected_mass)
            for alg in ("dpo", "grpo"):
                for seed in SEEDS:
                    paired = {}
                    for policy in ("sft_base", alg):
                        subset = [x for x in data if x["arm"] == arm and x["algorithm"] == policy and x["seed"] == str(seed)]
                        paired[policy] = {key: statistics.mean(float(x[key]) for x in subset)
                                          for key in ("expected_abs_error", "expansion_mass")}
                    self.equal(f"{experiment} {arm}/{alg} seed{seed} paired error decreases",
                               paired[alg]["expected_abs_error"] < paired["sft_base"]["expected_abs_error"], True)
                    self.equal(f"{experiment} {arm}/{alg} seed{seed} paired off-corpus mass increases",
                               paired[alg]["expansion_mass"] > paired["sft_base"]["expansion_mass"], True)
        checkpoints = sorted((directory / "checkpoints").glob("*.pt"))
        self.equal(experiment + " final policy checkpoint files", len(checkpoints), 30)
        self.equal(experiment + " final checkpoints nonempty", all(p.stat().st_size > 0 for p in checkpoints), True)
        self.derived[experiment] = aggregates

    def figure_crosschecks(self):
        directory = self.root / "figures/source_data/reactor_data"
        incidence = directory / "fig2d_target_window_incidence.csv"
        if incidence.exists():
            data = rows(incidence)
            self.equal("figure target-incidence series", len(data), 5)
            for row in data:
                expected = self.derived["incidence"][row["method"]]
                self.equal("figure " + row["method"] + " target hits vs raw records", int(row["hits"]), expected["hits"])
                self.equal("figure " + row["method"] + " sample count vs raw records", int(row["n"]), expected["n"])
                self.near("figure " + row["method"] + " target incidence (%) vs raw records", float(row["percent"]), 100 * expected["hits"] / expected["n"])
        else:
            self.notes.append("Optional figure incidence crosscheck not executed: " + str(incidence.relative_to(self.root)))
        endpoints = directory / "fig3b_objective_removal_endpoint.csv"
        if endpoints.exists():
            data = rows(endpoints)
            self.equal("figure objective-removal paired endpoints", len(data), 10)
            for row in data:
                values = self.derived["objective_removal"]["full_objective_seed_means" if row["condition"] == "Full objective" else "removed_penalty_seed_means"]
                self.near("figure " + row["condition"] + " seed" + row["seed"] + " endpoint vs raw records",
                          float(row["final100_inventory"]), values[int(row["seed"])])
        else:
            self.notes.append("Optional figure objective-removal crosscheck not executed: " + str(endpoints.relative_to(self.root)))

    def report(self):
        passed = sum(x["passed"] for x in self.checks)
        # JSON object keys must be strings, whereas grouping checks use tuples.
        def serializable(value):
            if isinstance(value, dict):
                return {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): serializable(v)
                        for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [serializable(x) for x in value]
            return value
        return serializable({
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_root": str(self.root),
            "status": "passed" if passed == len(self.checks) else "failed",
            "checks_passed": passed,
            "checks_total": len(self.checks),
            "limitations": [
                "This is an independent aggregation of archived numerical records, not a rerun of training or OpenMC.",
                "Chemistry expectations are checked from archived exact-enumeration metrics, not recomputed from neural policy weights.",
                "Checkpoint counts and nonempty files are checked; weights are not deserialized or executed.",
                "Artifact integrity, external DOI publication and upstream dataset licensing require separate checks.",
            ],
            "notes": self.notes,
            "derived_results": self.derived,
            "checks": self.checks,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, help="Optional JSON report path outside the package")
    args = parser.parse_args()
    root = args.package_root.resolve()
    if args.output:
        destination = args.output.resolve()
        if destination == root or root in destination.parents:
            parser.error("--output must be outside --package-root; archive contents are read-only")
    check = Verification(root)
    for name, operation in (
            ("corpus", check.corpus), ("alignment", check.alignment),
            ("generation", check.generation), ("random search", check.random_search),
            ("objective removal", check.objective_removal), ("high-statistics", check.high_statistics),
            ("chemistry single", lambda: check.chemistry("registered_single")),
            ("chemistry multi", lambda: check.chemistry("registered_multi")),
            ("figure crosschecks", check.figure_crosschecks)):
        check.run(name, operation)
    report = check.report()
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        print(f"{report['status'].upper()}: {report['checks_passed']}/{report['checks_total']} checks; report: {destination}")
        for item in report["checks"]:
            if not item["passed"]:
                print("FAILED: " + item["check"] + "; expected " + str(item["expected"]) + "; observed " + str(item["observed"]), file=sys.stderr)
    else:
        print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
