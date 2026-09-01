"""Aggregate the headline numbers reported in the manuscript from raw run CSVs.

Computes, from the trajectory and oracle CSVs:
  - per-seed best designs and the fitness decomposition table
    (criticality error in pcm, peaking-only fitness, composite fitness)
  - the matched comparison against the unconstrained GA (12.7%, MWU p)
  - the constrained-GA constraint-cost ratio (7.5x)
  - informed random-search oracle best-of-budget across seeds (mean +/- s.d.)
    and its Mann-Whitney U comparison against the aligned policies
  - no-penalty control convergence statistics

Expects the data layout of the archival package (see the archive README), or
set DATA_ROOT to a directory containing the CSVs in the documented layout.
Missing components are skipped with a notice.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
TARGET_K = 1.05


def pcm(k):
    return abs(k - TARGET_K) * 1e5


def peaking_only(fq, fdh):
    return 0.6 * fq + 0.4 * fdh


def load_stream(path, kind):
    df = pd.read_csv(path)
    if kind == 'dpo':
        rows = []
        for _, r in df.iterrows():
            for pref in ('chosen', 'rejected'):
                rows.append({'fitness': r[f'{pref}_fitness'], 'k_eff': r[f'{pref}_k_eff'],
                             'fq': r[f'{pref}_fq'], 'fdh': r[f'{pref}_fdh'],
                             'g_count': r[f'{pref}_g_count']})
        return pd.DataFrame(rows)
    return df[['fitness', 'k_eff', 'fq', 'fdh', 'g_count']]


def best_design(stream):
    valid = stream[stream.fitness < 99.0]
    return valid.loc[valid.fitness.idxmin()]


def main():
    specs = []
    for s in range(5):
        specs += [
            ('GA (Gd=16)', s, DATA_ROOT / f'baselines/ga_baseline_single_gd16_seed{s}_results.csv', 'ga'),
            ('GA (no constraint)', s, DATA_ROOT / f'baselines/ga_baseline_single_seed{s}_results.csv', 'ga'),
            ('DPO (CPT+SFT)', s, DATA_ROOT / f'alignment/dpo_single_target_seed{s}_results.csv', 'dpo'),
            ('GRPO (CPT+SFT)', s, DATA_ROOT / f'alignment/grpo_cpt_sft_single_seed{s}_results.csv', 'ga'),
        ]

    rows = []
    for method, seed, path, kind in tqdm(specs, desc='loading trajectories'):
        if not path.exists():
            print(f'skip (missing): {path}')
            continue
        bd = best_design(load_stream(path, kind))
        rows.append({'method': method, 'seed': seed, 'best_fitness': bd.fitness,
                     'dk_pcm': pcm(bd.k_eff), 'peaking_only': peaking_only(bd.fq, bd.fdh),
                     'gd': bd.g_count})
    ps = pd.DataFrame(rows)

    print('\n== Decomposition (per-seed best, mean +/- s.d.) ==')
    for m, g in ps.groupby('method', sort=False):
        print(f'{m:20s} dk={g.dk_pcm.mean():7.0f}+/-{g.dk_pcm.std():5.0f} pcm  '
              f'peaking={g.peaking_only.mean():.3f}+/-{g.peaking_only.std():.3f}  '
              f'composite={g.best_fitness.mean():.3f}+/-{g.best_fitness.std():.3f}')

    ga = ps[ps.method == 'GA (no constraint)'].best_fitness.values
    ga16 = ps[ps.method == 'GA (Gd=16)'].best_fitness.values
    for name in ('DPO (CPT+SFT)', 'GRPO (CPT+SFT)'):
        x = ps[ps.method == name].best_fitness.values
        if len(x) == 5 and len(ga) == 5:
            u, p = mannwhitneyu(x, ga, alternative='two-sided')
            gain = (ga.mean() - x.mean()) / ga.mean() * 100
            print(f'{name} vs unconstrained GA: {gain:.1f}% improvement, MWU p={p:.4f}')
        if len(x) == 5 and len(ga16) == 5:
            print(f'{name} constraint-cost ratio vs GA(Gd=16): {ga16.mean() / x.mean():.1f}x')

    oracle_bests = []
    for s in range(5):
        path = DATA_ROOT / f'oracle/informed_random_search_seed{s}_results.csv'
        if path.exists():
            d = pd.read_csv(path)
            oracle_bests.append(d[d.fitness < 99].fitness.min())
    if oracle_bests:
        ob = np.array(oracle_bests)
        print(f'\n== Oracle ({len(ob)} seeds) == best-of-budget {ob.mean():.2f} +/- {ob.std(ddof=1):.2f}')
        for name in ('DPO (CPT+SFT)', 'GRPO (CPT+SFT)'):
            x = ps[ps.method == name].best_fitness.values
            if len(x) == 5 and len(ob) == 5:
                u, p = mannwhitneyu(ob, x, alternative='two-sided')
                print(f'oracle vs {name}: MWU p={p:.2f}')

    ctrl = DATA_ROOT / 'no_penalty_control/no_penalty_dpo_seed0_results.csv'
    if ctrl.exists():
        n = pd.read_csv(ctrl)
        print(f'\n== No-penalty control == steps={len(n)}, '
              f'inventory mean over last 100 steps={n.chosen_g_count.tail(100).mean():.2f}, '
              f'best peaking-only fitness={n.best_fitness.min():.3f} '
              f'(Gd={int(n.best_g_count.iloc[-1])})')


if __name__ == '__main__':
    main()
