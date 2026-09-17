"""Inventory trajectories, first passage, and layout selection from the
recorded alignment / GA logs (Methods, "Inventory trajectories and first
passage"; Extended Data Fig. 3 a, c-e).

For each of the five DPO, GRPO and unconstrained-GA runs:
  - first passage: the first generated candidate with >= 20 Gd rods, reported
    as the completion count of its evaluation batch (DPO 2, GRPO 4, GA 40
    candidates) and the possible within-batch interval [lower, upper];
  - the cumulative-best inventory sequence (DPO: recorded best_g_count; GRPO
    and GA: reconstructed with strict fitness improvements in step / candidate
    order), its maximum and final value, number of decreases, monotonicity;
  - the best-of-run composite fitness and layout.
It also selects the layouts used by depletion/run_depletion.py: the best-of-
five-seeds DPO, GRPO and GA designs plus the symmetric 16-Gd and 24-Gd
references.

Inputs (archival layout under DATA_ROOT): alignment/, baselines/.
Outputs (--out, default $DATA_ROOT/analysis):
  first_passage_per_seed.csv, first_passage_summary.json,
  first_passage_report.md, selected_layouts.json, log_provenance.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[2] / 'data'))
THRESHOLD = 20
SOURCES = [('DPO', 'alignment/dpo_single_target_seed{}_results.csv'),
           ('GRPO', 'alignment/grpo_cpt_sft_single_seed{}_results.csv'),
           ('GA', 'baselines/ga_baseline_single_seed{}_results.csv')]


def cumulative_best_inventory(fitness, g_count):
    """Running-best inventory updated only at strict improvements (ties keep the earlier candidate)."""
    f = np.asarray(fitness)
    idx = np.empty(len(f), int)
    b = 0
    for i in range(len(f)):
        if f[i] < f[b]:
            b = i
        idx[i] = b
    return np.asarray(g_count)[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DATA_ROOT / 'analysis')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, provenance, layouts = [], {}, {}
    for method, template in tqdm(SOURCES, desc='methods'):
        candidates = []
        for seed in range(5):
            p = DATA_ROOT / template.format(seed)
            provenance[str(p.relative_to(DATA_ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
            df = pd.read_csv(p)
            if method == 'DPO':
                assert len(df) == 1000 and df.simulation_count.iloc[-1] == 2000
                hits = df[(df.chosen_g_count >= THRESHOLD) | (df.rejected_g_count >= THRESHOLD)]
                upper = int(hits.simulation_count.iloc[0]) if len(hits) else None
                lower = upper - 1 if upper else None
                inv = df.best_g_count.to_numpy()
                best = df.loc[df.chosen_fitness.idxmin()]
                record = {'fitness': float(best.chosen_fitness), 'grid': best.chosen_grid, 'seed': seed}
                final_gd = float(df.chosen_g_count.tail(100).mean())
            else:
                df = df.sort_values(['step', 'group_idx'] if method == 'GRPO' else ['generation', 'individual_idx'])
                assert len(df) == 2000
                hits = df[df.g_count >= THRESHOLD]
                h = hits.iloc[0] if len(hits) else None
                upper = int(h.sim_count if method == 'GRPO' else h.simulation_count) if h is not None else None
                group_size = 4 if method == 'GRPO' else 40
                lower = upper - group_size + 1 if upper else None      # within-batch order is not logged
                inv = cumulative_best_inventory(df.fitness, df.g_count)
                best = df.loc[df.fitness.idxmin()]
                record = {'fitness': float(best.fitness), 'grid': best.grid, 'seed': seed}
                final_gd = None
            candidates.append(record)
            rows.append(dict(method=method, seed=seed,
                             first_ge20_eval_lower=lower, first_ge20_eval_upper=upper,
                             first_ge20_budget_pct_upper=upper / 20 if upper else None,
                             maximum_cumulative_best_gd=int(inv.max()), final_cumulative_best_gd=int(inv[-1]),
                             downward_changes=int(np.sum(np.diff(inv) < 0)),
                             nondecreasing=bool(np.all(np.diff(inv) >= 0)),
                             overshot_final=bool(inv.max() > inv[-1]),
                             final100_selected_mean_gd=final_gd, best_fitness=record['fitness']))
        layouts[method] = min(candidates, key=lambda x: x['fitness'])
    nu = pd.read_csv(DATA_ROOT / 'baselines/nuscale_baseline_results.csv')
    for key, i in (('REF16', 0), ('REF24', 1)):
        layouts[key] = {'grid': nu.iloc[i]['grid'], 'fitness': float(nu.iloc[i]['fitness'])}
    for x in layouts.values():
        assert len(x['grid']) == 289 and x['grid'].count('c') == 25

    out = pd.DataFrame(rows)
    out.to_csv(a.out / 'first_passage_per_seed.csv', index=False)
    summary = {}
    for method, g in out.groupby('method'):
        summary[method] = {c: {'mean': float(g[c].mean()), 'sd': float(g[c].std(ddof=1))}
                           for c in ['first_ge20_eval_lower', 'first_ge20_eval_upper', 'first_ge20_budget_pct_upper',
                                     'maximum_cumulative_best_gd', 'final_cumulative_best_gd', 'best_fitness']}
        summary[method]['nonmonotone_runs'] = int((~g.nondecreasing).sum())
        summary[method]['overshoot_runs'] = int(g.overshot_final.sum())
    (a.out / 'first_passage_summary.json').write_text(json.dumps(summary, indent=2))
    (a.out / 'selected_layouts.json').write_text(json.dumps(layouts, indent=2))
    (a.out / 'log_provenance.json').write_text(json.dumps(provenance, indent=2))
    text = ['# First-passage and inventory-trajectory analysis', '',
            'First-entry results are bounded by the recorded evaluation batch (DPO 2, GRPO 4, GA 40 candidates). '
            'These are first visits to the inventory threshold, not evidence of a stable useful window; '
            f'Gd >= {THRESHOLD} has no upper bound.', '']
    for m, s in summary.items():
        lo, hi = s['first_ge20_eval_lower'], s['first_ge20_eval_upper']
        text.append(f"{m}: mean first-entry evaluation in [{lo['mean']:.1f}, {hi['mean']:.1f}], "
                    f"upper-endpoint SD {hi['sd']:.2f}; nonmonotone cumulative-best trajectories "
                    f"{s['nonmonotone_runs']}/5; overshoot {s['overshoot_runs']}/5.")
    text += ['', 'Selected layouts for depletion: ' + ', '.join(f"{k} (fitness {v['fitness']:.4f})" for k, v in layouts.items())]
    (a.out / 'first_passage_report.md').write_text('\n'.join(text) + '\n')
    print('\n'.join(text))


if __name__ == '__main__':
    main()
