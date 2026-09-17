"""Seed-level summaries and exact tests for the control experiments
(Methods, "No-penalty control run", "Uninformed full-space random search",
"Depletion assessment of selected layouts", "Statistical analysis").

Only fully completed runs enter the summaries (no-penalty: 1,000 steps /
2,000 evaluations; uninformed search: 2,000 distinct sample indices).

Inputs (archival layout under DATA_ROOT):
  no_penalty_control/no_penalty_dpo_seed{K}_results.csv     K = 0..4
  alignment/dpo_single_target_seed{K}_results.csv            matched full-objective runs
  oracle/uninformed_random_seed{K}_results.csv
  analysis/first_passage_per_seed.csv                        (analysis/first_passage.py)
  depletion/{DPO,GRPO,GA,REF16,REF24}/complete.json
Outputs (--out, default $DATA_ROOT/analysis):
  controls_summary.json, CONTROLS_SUMMARY.md,
  no_penalty_seed_summary.csv, uninformed_random_seed_summary.csv
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as student_t
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[2] / 'data'))
TARGET_K = 1.05


def peaking(fq, fdh):
    return 0.6 * fq + 0.4 * fdh


def exact_rank_sum_p(x, y):
    """Exact two-sided permutation p-value of the rank-sum statistic (n = 5 vs 5)."""
    ranks = rankdata(np.r_[x, y])
    n, N = len(x), len(ranks)
    expected = n * (N + 1) / 2
    observed = abs(ranks[:n].sum() - expected)
    values = [abs(ranks[list(c)].sum() - expected) for c in itertools.combinations(range(N), n)]
    return float(np.mean(np.array(values) >= observed - 1e-12)), float(ranks[:n].sum() - n * (n + 1) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DATA_ROOT / 'analysis')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {'no_penalty': [], 'uninformed_random': [], 'depletion': []}

    for seed in tqdm(range(5), desc='seeds'):
        p = DATA_ROOT / f'no_penalty_control/no_penalty_dpo_seed{seed}_results.csv'
        if p.exists():
            d = pd.read_csv(p)
            if len(d) == 1000 and int(d.simulation_count.iloc[-1]) == 2000:
                t = d.tail(100)
                full = pd.read_csv(DATA_ROOT / f'alignment/dpo_single_target_seed{seed}_results.csv')
                summary['no_penalty'].append({
                    'seed': seed,
                    'final100_inventory_mean': float(t.chosen_g_count.mean()),
                    'final100_inventory_within_run_sd': float(t.chosen_g_count.std()),
                    'final100_peaking_mean': float(peaking(t.chosen_fq, t.chosen_fdh).mean()),
                    'final_selected_peaking': float(peaking(d.chosen_fq.iloc[-1], d.chosen_fdh.iloc[-1])),
                    'run_min_peaking': float(d.chosen_fitness.min()),
                    'full_objective_final100_inventory_mean': float(full.chosen_g_count.tail(100).mean())})
        p = DATA_ROOT / f'oracle/uninformed_random_seed{seed}_results.csv'
        if p.exists():
            d = pd.read_csv(p)
            if len(d) == 2000 and set(d.sample_idx) == set(range(2000)):
                b = d.loc[d.fitness.idxmin()]
                near = (d.k_eff - TARGET_K).abs() <= 0.005
                inside = d.g_count.between(28, 35)
                summary['uninformed_random'].append({
                    'seed': seed, 'fitness': float(b.fitness),
                    'criticality_error_pcm': float(abs(b.k_eff - TARGET_K) * 1e5),
                    'peaking': float(peaking(b.fq, b.fdh)), 'inventory': int(b.g_count),
                    'candidates_gd_28_35': int(inside.sum()),
                    'candidates_within_500pcm': int(near.sum()),
                    'candidates_both': int((inside & near).sum())})
    for case in ['DPO', 'GRPO', 'GA', 'REF16', 'REF24']:
        p = DATA_ROOT / f'depletion/{case}/complete.json'
        if p.exists():
            summary['depletion'].append(json.loads(p.read_text()))

    lines = ['# Control experiment summaries', '',
             'Only completed runs enter the summaries below; in-progress runs are never reported.', '']
    for key in ['no_penalty', 'uninformed_random']:
        records = summary[key]
        lines.append(f'{key}: {len(records)}/5 completed seeds.')
        if records:
            pd.DataFrame(records).to_csv(a.out / f'{key}_seed_summary.csv', index=False)
        if len(records) == 5:
            df = pd.DataFrame(records)
            stats = {k: {'mean': float(df[k].mean()), 'sd': float(df[k].std(ddof=1))}
                     for k in df.columns if k != 'seed'}
            for k, v in stats.items():
                margin = float(student_t.ppf(.975, 4) * v['sd'] / np.sqrt(5))
                v['mean_95ci'] = [v['mean'] - margin, v['mean'] + margin]
                lines.append(f"- {k}: {v['mean']:.6g} ± {v['sd']:.6g} (between-seed SD)")
            summary[key + '_statistics'] = stats
            if key == 'uninformed_random':
                fp = DATA_ROOT / 'analysis/first_passage_per_seed.csv'
                if fp.exists():
                    original = pd.read_csv(fp)
                    comparisons = {}
                    for method in ['DPO', 'GRPO']:
                        pval, U = exact_rank_sum_p(df.fitness.to_numpy(),
                                                   original.loc[original.method == method, 'best_fitness'].to_numpy())
                        comparisons[method] = {'U': U, 'exact_two_sided_permutation_p': pval}
                        lines.append(f'- Best-of-budget fitness versus {method}: exact two-sided rank-sum p={pval:.6g}; '
                                     'a nonsignificant result is not proof of equivalence.')
                    summary['uninformed_comparisons'] = comparisons
                lines.append(f"- Candidates with 28-35 Gd rods: {int(df.candidates_gd_28_35.sum())} / 10,000; "
                             f"within ±500 pcm of the target: {int(df.candidates_within_500pcm.sum())} / 10,000 "
                             f"(all evaluated candidates, no best-of-budget selection).")
            if key == 'no_penalty':
                delta = (df.full_objective_final100_inventory_mean - df.final100_inventory_mean).to_numpy()
                flips = [abs(np.mean(delta * np.array(s))) for s in itertools.product([-1, 1], repeat=5)]
                pval = float(np.mean(np.array(flips) >= abs(delta.mean()) - 1e-12))
                summary['no_penalty_paired_exact_sign_flip_p'] = pval
                summary['no_penalty_reversed_seeds'] = int((df.final100_inventory_mean < df.full_objective_final100_inventory_mean).sum())
                lines.append(f'- Paired exact two-sided sign-flip test on seed-level final-100 inventory difference: '
                             f'p={pval:.6g} (five pairs permit a minimum two-sided p of 0.0625).')
                lines.append(f"- Seeds whose final-100 inventory fell below the matched full-objective run: "
                             f"{summary['no_penalty_reversed_seeds']}/5.")
    lines.append(f"depletion: {len(summary['depletion'])}/5 completed layouts. Values are infinite-lattice "
                 "unity-crossing proxies and endpoint residual-Gd effects, not operating-reactor cycle lengths.")
    for r in summary['depletion']:
        lines.append(f"- {r['case']}: BOL k={r['BOL_k']:.4f}, endpoint k={r['endpoint_k']:.4f}, unity crossing "
                     f"{r['last_downward_unity_crossing_MWd_per_kg_initial_HM']:.2f} MWd/kgHM, "
                     f"dk_Gd={r['residual_Gd_penalty_delta_k']:.5f} ± {r['residual_penalty_sd_independent']:.5f}")
    (a.out / 'controls_summary.json').write_text(json.dumps(summary, indent=2))
    (a.out / 'CONTROLS_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
