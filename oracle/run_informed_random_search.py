"""Informed random-search oracle (Results section "The discovered inventory
window, not the arrangement within it, is what the policy contributes").

Samples lattice configurations uniformly at random with Gd inventory drawn
uniformly from the integer interval [20, 40], the region occupied by the
aligned policies, and Gd positions drawn uniformly over the 264 non-guide-tube
locations. Each configuration is evaluated under the identical high-fidelity
OpenMC settings used for the GA and alignment campaigns, with an identical
budget of 2,000 evaluations per seed. Five independent sampling seeds are run;
best-of-budget composite fitness across seeds gives the oracle value reported
in the manuscript (mean +/- s.d.).

Usage:
    python run_informed_random_search.py            # seeds 0..4
    python run_informed_random_search.py --seeds 0  # single seed
Resumable: each seed writes informed_random_search_seed{K}_results.csv
incrementally and skips already-evaluated samples on restart.
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import eval_lib as E

N_EVAL = 2000
MASTER_SEED = 20260830
N_WORKERS = int(os.environ.get('ORACLE_WORKERS', max(1, (os.cpu_count() or 4) - 2)))


def worker(args):
    idx, grid_str, sim_seed = args
    t0 = time.time()
    fit, k, fq, fdh, gcount, gs = E.evaluate_grid(grid_str, sim_seed=sim_seed)
    return (idx, fit, k, fq, fdh, gcount, gs, time.time() - t0)


def run_seed(seed, pool):
    out = f'informed_random_search_seed{seed}_results.csv'
    rng = np.random.default_rng(MASTER_SEED + seed)
    layouts = [(i, E.random_layout(rng), seed * 100000 + i + 1) for i in range(N_EVAL)]

    done = set()
    if os.path.exists(out):
        import pandas as pd
        done = {int(i) for i in pd.read_csv(out)['sample_idx']}
        print(f'seed {seed}: resuming, {len(done)} evaluations already done')
    else:
        with open(out, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['sample_idx', 'fitness', 'k_eff', 'fq', 'fdh', 'g_count', 'grid', 'eval_sec'])
    todo = [x for x in layouts if x[0] not in done]

    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        for res in tqdm(pool.imap_unordered(worker, todo), total=len(todo),
                        desc=f'oracle seed {seed}', unit='eval'):
            idx, fit, k, fq, fdh, gcount, gs, dt = res
            w.writerow([idx, round(fit, 4), round(k, 5), round(fq, 4),
                        round(fdh, 4), gcount, gs, round(dt, 1)])
            f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for s in args.seeds:
            run_seed(s, pool)
    print(f'done in {(time.time() - t0) / 3600:.1f} h')


if __name__ == '__main__':
    main()
