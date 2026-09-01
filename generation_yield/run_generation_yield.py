"""Generation yield and constraint satisfaction (Methods, "Generation yield
and constraint satisfaction").

Samples N_GEN generations from each aligned single-target CPT+SFT checkpoint
(DPO seeds 0-4, GRPO seeds 0-4) at T = 1.0 under the single-target prompt and
measures, per sample and prior to any correction, whether the generation
supplies a complete 289-token lattice stream free of extraneous characters,
whether it places a non-guide-tube token at one of the 25 fixed guide-tube
coordinates, and the generated Gd inventory after the standard correction
pipeline. A fixed-size subsample per checkpoint is then evaluated with OpenMC
for the feasible fraction within BAND of the criticality target.

Outputs generation_yield_samples.csv, generation_yield_openmc_subsample.csv,
and generation_yield_summary.md. Resumable per checkpoint.
"""
import csv
import gc
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
import eval_lib as E

PROJECT_ROOT = os.environ.get('REACTORGEN_ROOT', str(Path(__file__).resolve().parents[1]))

N_GEN = 1000
BATCH = 50
BAND = 0.005          # 500 pcm
SUBSAMPLE_PER_CKPT = 50
OPENMC_WORKERS = int(os.environ.get('ORACLE_WORKERS', 4))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROMPT = 'Reactor Core Design (k=1.05000, fq=1.0000, fdh=1.0000):\n'

BASE = os.path.join(PROJECT_ROOT, 'training')
CKPTS = ([('DPO', s, f'{BASE}/dpo/single_target/dpo_optimized_single_seed{s}_model') for s in range(5)]
         + [('GRPO', s, f'{BASE}/grpo/single/grpo_cpt_sft_single_seed{s}_model') for s in range(5)])

GT_INDICES = set(E.GT_INDICES)


def parse_generation(text):
    """Replicate the deployment parsing, without any correction.

    The pipeline truncates the generated stream at 289 lattice tokens; a
    generation is complete if it supplies at least 289 valid tokens without
    padding and contains no non-lattice characters."""
    i = text.find(':\n')
    body = text[i + 2:] if i >= 0 else ''
    cleaned = body.replace(' ', '').replace('\n', '')
    chars = [c for c in cleaned if c in 'fgc']
    n_nonfgc = len(cleaned) - len(chars)
    raw_valid = (len(chars) >= 289 and n_nonfgc == 0)
    padded = (chars + ['f'] * 289)[:289]
    gt_viol = any(padded[i] != 'c' for i in GT_INDICES)
    fixed = list(padded)
    for i in range(289):
        if i in GT_INDICES:
            fixed[i] = 'c'
        elif fixed[i] == 'c':
            fixed[i] = 'f'
    fixed = ''.join(fixed)
    return raw_valid, gt_viol, fixed.count('g'), fixed, len(chars), n_nonfgc


def generate_for_ckpt(algo, seed, path, writer, fout):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True).to(DEVICE)
    model.eval()
    torch.manual_seed(1000 + seed)
    inputs = tokenizer(PROMPT, return_tensors='pt').to(DEVICE)

    n = 0
    with torch.no_grad():
        pbar = tqdm(total=N_GEN, desc=f'{algo} seed{seed}', unit='gen')
        while n < N_GEN:
            b = min(BATCH, N_GEN - n)
            out = model.generate(**inputs, max_new_tokens=400, temperature=1.0,
                                 do_sample=True, num_return_sequences=b,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
            for row in out:
                text = tokenizer.decode(row, skip_special_tokens=True)
                raw_valid, gt_viol, gd, fixed, nfgc, njunk = parse_generation(text)
                writer.writerow([algo, seed, n, int(raw_valid), int(gt_viol), gd, fixed, nfgc, njunk])
                n += 1
            fout.flush()
            pbar.update(b)
        pbar.close()

    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def openmc_worker(args):
    idx, grid = args
    fit, k, fq, fdh, g, _ = E.evaluate_grid(grid, sim_seed=90000 + idx)
    return (idx, fit, k, fq, fdh, g)


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    t0 = time.time()

    out_csv = 'generation_yield_samples.csv'
    if not os.path.exists(out_csv):
        with open(out_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['algo', 'seed', 'idx', 'raw_valid', 'gt_viol', 'gd_count',
                                    'grid', 'n_lattice_tokens', 'n_nonlattice'])
    import pandas as pd
    done = pd.read_csv(out_csv)
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        for algo, seed, path in CKPTS:
            if len(done[(done.algo == algo) & (done.seed == seed)]) >= N_GEN:
                print(f'{algo} seed{seed}: already sampled, skipping')
                continue
            generate_for_ckpt(algo, seed, path, w, f)

    df = pd.read_csv(out_csv)
    print(f'generation done, {len(df)} samples, {(time.time() - t0) / 60:.0f} min')

    sub = df.groupby(['algo', 'seed'], group_keys=False).sample(
        n=SUBSAMPLE_PER_CKPT, random_state=7).reset_index(drop=True)
    ev_csv = 'generation_yield_openmc_subsample.csv'
    done_ev = set()
    if os.path.exists(ev_csv):
        done_ev = set(pd.read_csv(ev_csv)['idx'])
    else:
        with open(ev_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['idx', 'algo', 'seed', 'fitness', 'k_eff', 'fq', 'fdh', 'gd'])
    jobs = [(i, sub.grid.iloc[i]) for i in range(len(sub)) if i not in done_ev]
    with Pool(OPENMC_WORKERS) as pool, open(ev_csv, 'a', newline='') as f:
        w = csv.writer(f)
        for idx, fit, k, fq, fdh, g in tqdm(pool.imap_unordered(openmc_worker, jobs),
                                            total=len(jobs), desc='OpenMC subsample'):
            w.writerow([idx, sub.algo.iloc[idx], sub.seed.iloc[idx],
                        round(fit, 4), round(k, 5), round(fq, 4), round(fdh, 4), g])
            f.flush()

    ev = pd.read_csv(ev_csv)
    feas = ((ev.k_eff - 1.05).abs() < BAND) & (ev.k_eff > 0.5)
    lines = [
        '# Generation yield summary',
        f'- N_GEN per checkpoint = {N_GEN}, checkpoints = {len(CKPTS)} (DPO x5, GRPO x5), total = {len(df)}',
        f'- complete-lattice fraction (>=289 valid tokens, no extraneous characters): {100 * df.raw_valid.mean():.1f}%',
        f'- guide-tube violation pre-correction: {100 * df.gt_viol.mean():.1f}%',
        f'- generated inventory: {df.gd_count.mean():.1f} +/- {df.gd_count.std():.1f} rods',
        f'- feasible within 500 pcm (OpenMC, n={len(ev)}): {100 * feas.mean():.1f}%',
        f'- lattice tokens per generation: {df.n_lattice_tokens.mean():.0f} +/- {df.n_lattice_tokens.std():.0f}',
        f'- per-algo validity: DPO {100 * df[df.algo == "DPO"].raw_valid.mean():.1f}%, '
        f'GRPO {100 * df[df.algo == "GRPO"].raw_valid.mean():.1f}%',
        f'- per-algo feasible: DPO {100 * feas[ev.algo == "DPO"].mean():.1f}%, '
        f'GRPO {100 * feas[ev.algo == "GRPO"].mean():.1f}%',
    ]
    open('generation_yield_summary.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
