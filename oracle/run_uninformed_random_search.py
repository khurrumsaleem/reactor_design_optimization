"""Uninformed full-space random search (Methods, "Uninformed full-space random
search"; Extended Data Fig. 3; Fig. 4 c-e, g).

Samples lattice configurations with the Gd inventory drawn uniformly from the
integers 0..264 (every non-guide-tube position may carry an absorber) and the
absorber positions drawn uniformly without replacement over the 264
non-guide-tube locations. The 25 guide-tube positions are fixed. Unlike the
informed oracle (run_informed_random_search.py), no productive inventory
window is supplied. Each candidate is evaluated under the identical online
OpenMC settings and composite objective used by the alignment and GA
campaigns, with an identical budget of 2,000 evaluations per sampling seed.

This is an inventory-uniform baseline, not a uniform distribution over all
possible lattices.

Every evaluation carries a deterministic OpenMC seed so the run is exactly
reproducible and resumable: on restart the existing CSV is validated against
the regenerated layouts and only the missing sample indices are evaluated.
An OpenMC failure raises instead of being recorded as a sentinel fitness.

Usage:
    python run_uninformed_random_search.py                 # seeds 0..4
    python run_uninformed_random_search.py --seeds 0 --workers 8
Outputs: uninformed_random_seed{K}_results.csv (+ uninformed_seed{K}_complete.json)
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

import eval_lib as E

N_EVAL = 2000
MASTER_SEED = 20260912          # layout RNG: default_rng(MASTER_SEED + seed)
SIM_SEED_BASE = 3_000_000       # OpenMC seed: SIM_SEED_BASE + seed*100000 + idx + 1
GD_MIN, GD_MAX = 0, 265         # rng.integers(0, 265) -> 0..264 inclusive
HEADER = ['sample_idx', 'fitness', 'k_eff', 'fq', 'fdh', 'g_count', 'grid', 'eval_sec', 'sim_seed']


def evaluate(job):
    seed, idx, grid, sim_seed = job
    t0 = time.time()
    fit, k, fq, fdh, ng, gs = E.evaluate_grid(grid, sim_seed=sim_seed)
    if not np.all(np.isfinite([fit, k, fq, fdh])) or k <= 0 or fit >= 99.0:
        raise RuntimeError(f'Invalid OpenMC evaluation for seed {seed}, sample {idx}')
    return [idx, fit, k, fq, fdh, ng, gs, time.time() - t0, sim_seed]


def run_seed(seed, n_eval, out_dir, pool):
    path = out_dir / f'uninformed_random_seed{seed}_results.csv'
    rng = np.random.default_rng(MASTER_SEED + seed)
    jobs = [(seed, i, E.random_layout(rng, GD_MIN, GD_MAX),
             SIM_SEED_BASE + seed * 100_000 + i + 1) for i in range(n_eval)]

    done = set()
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                i = int(row['sample_idx'])
                if i in done or row['grid'] != jobs[i][2] or int(row['sim_seed']) != jobs[i][3]:
                    raise RuntimeError(f'{path} does not match the regenerated layouts; refuse to resume')
                done.add(i)
        print(f'seed {seed}: resuming, {len(done)}/{n_eval} already evaluated')
    else:
        with path.open('w', newline='') as f:
            csv.writer(f).writerow(HEADER)

    todo = [j for j in jobs if j[1] not in done]
    with path.open('a', newline='') as f:
        w = csv.writer(f)
        for row in tqdm(pool.imap_unordered(evaluate, todo), total=len(todo),
                        desc=f'uninformed seed {seed}', unit='eval'):
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            done.add(row[0])
    (out_dir / f'uninformed_seed{seed}_complete.json').write_text(
        json.dumps({'seed': seed, 'evaluations': len(done)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--evaluations', type=int, default=N_EVAL)
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('ORACLE_WORKERS', max(1, (os.cpu_count() or 4) - 2))))
    ap.add_argument('--output-dir', type=Path, default=Path('.'))
    a = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for seed in a.seeds:
            run_seed(seed, a.evaluations, a.output_dir, pool)
    print(f'done in {(time.time() - t0) / 3600:.1f} h')


if __name__ == '__main__':
    main()
