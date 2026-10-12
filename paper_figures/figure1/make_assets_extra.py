"""Extra numbers for the three data schematics added on 8 October 2026.
Appends to data/fig1_assets.json (system python3 with pandas is enough).
  corpus_k        histogram of corpus k_eff (reactor_10k_final.csv), target 1.05, gap in pcm
  anchor_grid     task x ligand grid of the chemistry supervised corpus (anchor = t mod 4 after sorting)
  best_objective  best-of-budget objective per run for uniform, informed, DPO, GRPO (2,000 evaluations)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
J = HERE / "data/fig1_assets.json"
out = json.load(open(J))

# corpus k_eff distribution versus the requested target
corp = pd.read_csv(ROOT / "after/data_generation/reactor_10k_final.csv")
k = corp.k_eff.values
edges = np.round(np.arange(1.155, 1.2201, 0.005), 4)
cnt, _ = np.histogram(k, bins=edges)
out["corpus_k"] = {"n": int(len(k)), "min": float(k.min()), "max": float(k.max()), "target": 1.05,
                   "gap_pcm": int(round((k.min() - 1.05) * 1e5)), "edges": edges.tolist(), "counts": cnt.tolist()}

# chemistry supervised corpus: anchor ligand per task by t mod 4 after sorting identifiers
bh = pd.read_csv(ROOT / "empirical/data/buchwald_hartwig.csv")
tasks = sorted(bh["Aryl halide"].unique())
ligs = sorted(bh["Ligand"].unique())
out["anchor_grid"] = {"n_tasks": len(tasks), "n_ligands": len(ligs), "anchor_index": [t % 4 for t in range(len(tasks))],
                      "restricted_rows": int(sum(((bh["Aryl halide"] == a) & (bh["Ligand"] == ligs[t % 4])).sum() for t, a in enumerate(tasks)))}

# best-of-budget objective per run (lower is better), 2,000 online evaluations each
def best_min(frames, col):
    return [float(f[col].min()) for f in frames]
uni = [pd.read_csv(ROOT / f"alam_additional_expriment/results/uninformed_random_seed{s}_results.csv") for s in tqdm(range(5), desc="uniform", leave=False)]
inf = [pd.read_csv(ROOT / "professor/Test/t3_random_search/random_search_results.csv")] + \
      [pd.read_csv(ROOT / f"professor/Test/t3_random_search/oracle_seed{s}_results.csv") for s in range(1, 5)]
dpo = [pd.read_csv(ROOT / f"after/training/dpo/single_target/dpo_single_target_seed{s}_results.csv") for s in range(5)]
grpo = [pd.read_csv(ROOT / f"after/training/grpo/single/grpo_cpt_sft_single_seed{s}_results.csv") for s in range(5)]
fit_u = [c for c in uni[0].columns if "fit" in c.lower()][0]
best = {"uniform": best_min(uni, fit_u), "informed": best_min(inf, "fitness"),
        "dpo": [float(f.best_fitness.iloc[-1]) for f in dpo], "grpo": best_min(grpo, "fitness")}
out["best_objective"] = {k: {"runs": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in best.items()}
json.dump(out, open(J, "w"), indent=1)
print({k: (round(v["mean"], 3), round(v["sd"], 3)) for k, v in out["best_objective"].items()})
print("corpus k", out["corpus_k"]["min"], out["corpus_k"]["max"], "gap pcm", out["corpus_k"]["gap_pcm"], "counts", out["corpus_k"]["counts"])
print("anchor", out["anchor_grid"])
