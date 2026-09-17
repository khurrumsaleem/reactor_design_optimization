"""No-penalty control, seeds 0-4 (Methods, "No-penalty control run").

Runs the identical online single-target DPO procedure defined in
no_penalty_dpo.py (same prompt, SFT initialization, learning rate, sampling
temperature, guide-tube correction and OpenMC settings, penalty weight 0) but
with three operational changes used for the five-seed campaign reported in the
manuscript:

  1. Atomic periodic checkpoints (every 10 steps) that store the model,
     optimizer, and Python / NumPy / Torch / CUDA RNG states, so an interrupted
     run resumes exactly; any CSV rows written after the last checkpoint are
     discarded and recomputed.
  2. Strict failures: an OpenMC error raises instead of being logged as the
     sentinel fitness 99.9.
  3. The two candidates of each step are generated in the original order on
     the GPU and evaluated concurrently by two isolated OpenMC worker processes
     (oracle_worker.py); GPU computation and sampling order are unchanged.
     Both raw candidates are also logged (first_grid / second_grid columns).

Usage:
    python run_no_penalty_resumable.py --seed 1
    python run_no_penalty_resumable.py --seed 1 --output-root /path/to/results
Outputs (under --output-root, default: this directory):
    no_penalty_dpo_seed{K}_results.csv, checkpoints/no_penalty_seed{K}.pt,
    no_penalty_seed{K}_complete.json
"""
import argparse
import atexit
import csv
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--checkpoint-every', type=int, default=10)
ap.add_argument('--output-root', type=Path, default=HERE)
args = ap.parse_args()
args.output_root.mkdir(parents=True, exist_ok=True)

try:  # POSIX advisory lock against a duplicate run of the same seed
    import fcntl
    _lock = (args.output_root / f'seed{args.seed}.lock').open('w')
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except ImportError:
    pass

# Import the original control script as a module (it parses --seed itself).
sys.argv = [sys.argv[0], '--seed', str(args.seed)]
_spec = importlib.util.spec_from_file_location('no_penalty_dpo', HERE / 'no_penalty_dpo.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

root = args.output_root.resolve()
(root / 'checkpoints').mkdir(exist_ok=True)
out = root / f'no_penalty_dpo_seed{args.seed}_results.csv'
cp = root / 'checkpoints' / f'no_penalty_seed{args.seed}.pt'
done_marker = root / f'no_penalty_seed{args.seed}_complete.json'
if done_marker.exists():
    print('Already complete')
    sys.exit(0)

FIELDS = ['step', 'simulation_count', 'elapsed_time_sec',
          'chosen_grid', 'chosen_fitness', 'chosen_k_eff', 'chosen_fq', 'chosen_fdh', 'chosen_g_count',
          'rejected_grid', 'rejected_fitness', 'rejected_k_eff', 'rejected_fq', 'rejected_fdh', 'rejected_g_count',
          'best_fitness', 'best_k_eff', 'best_fq', 'best_fdh', 'best_g_count', 'loss',
          'first_grid', 'first_g_count', 'second_grid', 'second_g_count']

model, tokenizer = M.load_model(M.MODEL_PATH)
opt = torch.optim.AdamW(model.parameters(), lr=M.LEARNING_RATE)
use_cuda = M.DEVICE.type == 'cuda'

start, elapsed0, best = 1, 0.0, (99.9, 0.0, 99.9, 99.9, 0)
if cp.exists():
    c = torch.load(cp, map_location='cpu', weights_only=False)
    model.load_state_dict(c['model'])
    opt.load_state_dict(c['optimizer'])
    start, best, elapsed0 = c['step'] + 1, tuple(c['best']), c['elapsed']
    random.setstate(c['python_rng'])
    np.random.set_state(c['numpy_rng'])
    torch.set_rng_state(c['torch_rng'])
    if use_cuda and c.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(c['cuda_rng'])
    # The CSV may hold rows written after the last checkpoint; drop and recompute them.
    with out.open() as f:
        records = [r for r in csv.DictReader(f) if int(r['step']) < start]
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f'Resuming from step {start}', flush=True)
elif out.exists():
    raise RuntimeError(f'{out} exists without a checkpoint; refusing to overwrite')
else:
    with out.open('w', newline='') as f:
        csv.writer(f).writerow(FIELDS)

workers = [subprocess.Popen([sys.executable, str(HERE / 'oracle_worker.py')],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
           for _ in range(2)]


def close_workers():
    for w in workers:
        if w.poll() is None:
            w.terminate()


atexit.register(close_workers)


def evaluate_pair(g1, g2):
    for worker, grid in zip(workers, (g1, g2)):
        worker.stdin.write(json.dumps({'grid': grid}) + '\n')
        worker.stdin.flush()
    results = []
    for worker in workers:
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError('OpenMC worker exited; see its stderr')
        results.append(tuple(json.loads(line)))
    return results


t0 = time.time()
for step in tqdm(range(start, args.steps + 1), initial=start - 1, total=args.steps,
                 desc=f'no-penalty DPO seed {args.seed}', unit='step'):
    prompt = M.create_prompt(M.TARGET_K_EFF, M.TARGET_FQ, M.TARGET_FDH)
    g1 = M.generate_grid(model, tokenizer, prompt, temperature=1.0)
    g2 = M.generate_grid(model, tokenizer, prompt, temperature=1.0)
    m1, m2 = evaluate_pair(g1, g2)
    if not np.all(np.isfinite(m1 + m2)):
        raise RuntimeError('Non-finite oracle output')
    cg, cm, rg, rm = (g1, m1, g2, m2) if m1[0] <= m2[0] else (g2, m2, g1, m1)
    if cm[0] < best[0]:
        best = cm
    model.train()
    opt.zero_grad()
    loss = M.dpo_loss(model, tokenizer, prompt, cg, rg, beta=M.BETA)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    elapsed = elapsed0 + time.time() - t0
    with out.open('a', newline='') as f:
        csv.writer(f).writerow([step, step * 2, elapsed, cg, *cm, rg, *rm, *best, float(loss.item()),
                                g1, g1.count('g'), g2, g2.count('g')])
        f.flush()
        os.fsync(f.fileno())
    if step % args.checkpoint_every == 0 or step == args.steps:
        state = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                 'best': best, 'elapsed': elapsed, 'python_rng': random.getstate(),
                 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state() if use_cuda else None}
        tmp = cp.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, cp)

done_marker.write_text(json.dumps({'seed': args.seed, 'steps': args.steps,
                                   'evaluations': args.steps * 2, 'best': best,
                                   'elapsed_sec': elapsed}, indent=2))
print('complete', flush=True)
