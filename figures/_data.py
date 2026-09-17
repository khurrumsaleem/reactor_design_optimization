"""Data loaders for the ReactorGen figures.

All inputs are read from the archival data package layout (Zenodo record
10.5281/zenodo.22230862, directory ``data/``).  Set ``DATA_ROOT`` to that
directory; by default the loader assumes the archive layout in which
``code/`` and ``data/`` are siblings (``<repo>/../data``).

data/
  corpus/                 reactor_10k_final.csv
  alignment/              CPT+SFT alignment trajectories (DPO/GRPO, single/multi, 5 seeds)
  alignment_sft_only/     SFT-only alignment trajectories (CPT ablation)
  baselines/              GA trajectories and NuScale-type references
  oracle/                 informed random search (inventory uniform on [20, 40]) and
                          uninformed full-space random search (inventory uniform on 0..264),
                          5 sampling seeds each
  no_penalty_control/     no-penalty DPO control trajectories, 5 seeds
  depletion/              per-layout depletion outputs (trajectory.csv, complete.json) for
                          DPO, GRPO, GA, REF16, REF24, plus selected_layouts.json
  analysis/               first_passage_per_seed.csv (analysis/first_passage.py) and the
                          control summaries (analysis/summarize_controls.py)
  prompt_sensitivity/     steerability sweep (100 samples x 7 targets x 10 checkpoints)
  reevaluation/           2e7-history OpenMC re-evaluation, bootstrap and CI audits
  empirical/              Buchwald-Hartwig replication outputs
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
SEEDS = (0, 1, 2, 3, 4)


def _seeds(tmpl: str) -> list[pd.DataFrame]:
    out = []
    for s in SEEDS:
        p = Path(tmpl.format(seed=s))
        if not p.exists():
            print(f"  [warn] missing: {p}")
            continue
        out.append(pd.read_csv(p))
    return out


# ---------------------------------------------------------------- raw loaders
def ga(constrained: bool):
    sub = "ga_baseline_single_gd16" if constrained else "ga_baseline_single"
    return _seeds(str(DATA_ROOT / f"baselines/{sub}_seed{{seed}}_results.csv"))


def _align_dir(base):
    return DATA_ROOT / ("alignment" if base == "cpt_sft" else "alignment_sft_only")


def dpo_single(base="cpt_sft"):
    return _seeds(str(_align_dir(base) / "dpo_single_target_seed{seed}_results.csv"))


def dpo_multi(base="cpt_sft"):
    return _seeds(str(_align_dir(base) / "dpo_multi_lhs_seed{seed}_results.csv"))


def grpo_single(base="cpt_sft"):
    name = "grpo_cpt_sft_single" if base == "cpt_sft" else "grpo_sft_only_single"
    return _seeds(str(_align_dir(base) / f"{name}_seed{{seed}}_results.csv"))


def grpo_multi(base="cpt_sft"):
    name = "grpo_cpt_sft_multi" if base == "cpt_sft" else "grpo_sft_only_multi"
    return _seeds(str(_align_dir(base) / f"{name}_seed{{seed}}_results.csv"))


def nuscale() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "baselines/nuscale_baseline_results.csv")


def no_penalty() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "no_penalty_control/no_penalty_dpo_seed0_results.csv")


def no_penalty_all() -> list[pd.DataFrame]:
    """Five matched-seed no-penalty control runs."""
    return _seeds(str(DATA_ROOT / "no_penalty_control/no_penalty_dpo_seed{seed}_results.csv"))


def uninformed_search() -> list[pd.DataFrame]:
    """Uninformed full-space random search, inventory uniform on 0..264."""
    return _seeds(str(DATA_ROOT / "oracle/uninformed_random_seed{seed}_results.csv"))


def first_passage() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "analysis/first_passage_per_seed.csv")


DEPLETION_CASES = ("DPO", "GRPO", "GA", "REF16", "REF24")


def depletion_trajectory(case: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / f"depletion/{case}/trajectory.csv")


def depletion_summary(case: str) -> dict:
    import json
    with open(DATA_ROOT / f"depletion/{case}/complete.json") as fh:
        return json.load(fh)


def informed_search() -> list[pd.DataFrame]:
    return _seeds(str(DATA_ROOT / "oracle/informed_random_search_seed{seed}_results.csv"))


def high_stat() -> pd.DataFrame:
    """Post-hoc 2e7-history re-evaluation of the ten selected layouts."""
    return pd.read_csv(DATA_ROOT / "reevaluation/openmc_high_stat_all-per-seed_summary.csv")


def prompt_sensitivity_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / "prompt_sensitivity/prompt_sensitivity_results.csv")
    if "is_truncated" in df.columns:
        df = df[df["is_truncated"] == 0].copy()
    return df


def steerability_means() -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / "reevaluation/steerability_target_mean_95ci.csv")
    return df[df["truncation_policy"] == "exclude_truncated"].copy()


def steerability_slopes() -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / "reevaluation/steerability_bootstrap_summary.csv")
    return df[df["truncation_policy"] == "exclude_truncated"].copy()


def empirical_seed_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "empirical/empirical_support_expansion_seed_summary.csv")


def empirical_reference(regime: str) -> dict:
    import json
    name = "registered_single" if regime.startswith("Single") else "registered_multi"
    with open(DATA_ROOT / f"empirical/{name}/summary.json") as fh:
        b = json.load(fh)["baselines"]["random_full"]
    return {"in support": float(b["restricted_oracle_abs_error"]),
            "full catalogue": float(b["full_oracle_abs_error"])}


# ------------------------------------------------------------ trajectory prep
def _cumbest(fit: pd.Series, side: dict[str, pd.Series]) -> dict[str, np.ndarray]:
    v = fit.reset_index(drop=True)
    idx = v.expanding().apply(lambda s: s.idxmin(), raw=False).astype(int).values
    out = {"cum_best_fit": v.cummin().values}
    for k, s in side.items():
        out[k] = s.reset_index(drop=True).iloc[idx].values
    return out


def ga_steps(df: pd.DataFrame) -> pd.DataFrame:
    ib = df.groupby("generation")["fitness"].idxmin()
    sb = df.loc[ib].sort_values("generation").reset_index(drop=True)
    x = df.groupby("generation")["simulation_count"].max().sort_index().values
    cum = _cumbest(sb["fitness"], {"cum_best_k": sb["k_eff"], "cum_best_gd": sb["g_count"],
                                   "cum_best_fq": sb["fq"], "cum_best_fdh": sb["fdh"],
                                   "cum_best_grid": sb["grid"]})
    return pd.DataFrame({"x": x, "step_fit": sb["fitness"].values, "step_gd": sb["g_count"].values, **cum})


def grpo_steps(df: pd.DataFrame) -> pd.DataFrame:
    ib = df.groupby("step")["fitness"].idxmin()
    sb = df.loc[ib].sort_values("step").reset_index(drop=True)
    x = df.groupby("step")["sim_count"].max().sort_index().values
    cum = _cumbest(sb["fitness"], {"cum_best_k": sb["k_eff"], "cum_best_gd": sb["g_count"],
                                   "cum_best_fq": sb["fq"], "cum_best_fdh": sb["fdh"],
                                   "cum_best_grid": sb["grid"]})
    out = pd.DataFrame({"x": x, "step_fit": sb["fitness"].values, "step_k": sb["k_eff"].values,
                        "step_gd": sb["g_count"].values, **cum})
    if "target_k" in sb:
        out["target_k"] = sb["target_k"].values
    return out


def dpo_steps(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    cum = _cumbest(d["chosen_fitness"], {"cum_best_k": d["chosen_k_eff"], "cum_best_gd": d["chosen_g_count"],
                                         "cum_best_fq": d["chosen_fq"], "cum_best_fdh": d["chosen_fdh"],
                                         "cum_best_grid": d["chosen_grid"]})
    out = pd.DataFrame({"x": d["simulation_count"].values, "step_fit": d["chosen_fitness"].values,
                        "step_k": d["chosen_k_eff"].values, "step_gd": d["chosen_g_count"].values, **cum})
    if "target_k" in d:
        out["target_k"] = d["target_k"].values
    return out


def align(xs, ys, n=400):
    if not xs:
        return np.array([]), np.empty((0, 0))
    lo = max(np.nanmin(x) for x in xs)
    hi = min(np.nanmax(x) for x in xs)
    grid = np.linspace(lo, hi, n)
    rows = []
    for x, y in zip(xs, ys):
        o = np.argsort(x)
        rows.append(np.interp(grid, np.asarray(x)[o], np.asarray(y)[o]))
    return grid, np.vstack(rows)


def matrix(step_dfs: list[pd.DataFrame], col: str, n=400):
    return align([d["x"].to_numpy() for d in step_dfs], [d[col].to_numpy() for d in step_dfs], n)


# ------------------------------------------------------------- best per seed
def best_records(step_dfs: list[pd.DataFrame]) -> list[dict]:
    recs = []
    for d in step_dfs:
        i = int(d["cum_best_fit"].idxmin())
        recs.append(dict(fit=float(d.loc[i, "cum_best_fit"]), k=float(d.loc[i, "cum_best_k"]),
                         gd=int(d.loc[i, "cum_best_gd"]), fq=float(d.loc[i, "cum_best_fq"]),
                         fdh=float(d.loc[i, "cum_best_fdh"]), grid=str(d.loc[i, "cum_best_grid"])))
    return recs


def informed_best_records() -> list[dict]:
    recs = []
    for d in informed_search():
        i = int(d["fitness"].idxmin())
        r = d.loc[i]
        recs.append(dict(fit=float(r["fitness"]), k=float(r["k_eff"]), gd=int(r["g_count"]),
                         fq=float(r["fq"]), fdh=float(r["fdh"]), grid=str(r["grid"])))
    return recs


def uninformed_best_records() -> list[dict]:
    recs = []
    for d in uninformed_search():
        i = int(d["fitness"].idxmin())
        r = d.loc[i]
        recs.append(dict(fit=float(r["fitness"]), k=float(r["k_eff"]), gd=int(r["g_count"]),
                         fq=float(r["fq"]), fdh=float(r["fdh"]), grid=str(r["grid"])))
    return recs


def nuscale_records() -> dict[str, dict]:
    nu = nuscale()
    out = {}
    for key, row in (("NU16", nu.iloc[0]), ("NU24", nu.iloc[1])):
        out[key] = dict(fit=float(row["fitness"]), k=float(row["k_eff"]), gd=int(row["gd_count"]),
                        fq=float(row["fq"]), fdh=float(row["fdh"]), grid=str(row["grid"]))
    return out


def parse_grid(s: str) -> np.ndarray:
    m = {"f": 0, "g": 1, "c": 2}
    v = [m.get(ch, 0) for ch in s][:289]
    v += [0] * (289 - len(v))
    return np.array(v).reshape(17, 17)


def dk_pcm(k: float, target: float = 1.05) -> float:
    return abs(k - target) * 1e5


def peaking_only(fq: float, fdh: float) -> float:
    return 0.6 * fq + 0.4 * fdh
