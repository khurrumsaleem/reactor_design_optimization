"""Inventory distribution of the supervised (CPT+SFT) policies before alignment.

Samples N_GEN generations at temperature 1.0 under the single-target prompt
from each of the five seed-specific SFT checkpoints that initialise the
five-seed DPO and GRPO runs, and applies the same parsing and correction as
the aligned-policy audit (professor/Test/t6_generation_yield/run_t6.py).
No OpenMC. Read-only with respect to after/.

Outputs (claude_figure/data/):
  supervised_generation.csv           one row per generation
  supervised_generation_summary.json  counts per inventory and per-seed shares
"""
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/Users/yoon/project/UIUC/reactor_genome/professor/Test/t3_random_search")
import eval_lib as E  # GT_INDICES only; no transport is run here

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
N_GEN = 1000
BATCH = 50
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
PROMPT = "Reactor Core Design (k=1.05000, fq=1.0000, fdh=1.0000):\n"
BASE = "/Users/yoon/project/UIUC/reactor_genome/after/training/base/sft"
CKPTS = [(s, f"{BASE}/final_sft_model_seed{s}") for s in range(5)]
GT_INDICES = set(E.GT_INDICES)


def parse_generation(text):
    """Identical to run_t6.parse_generation: alphabet filter, pad or truncate
    to 289 symbols, guide-coordinate correction, then count g."""
    i = text.find(":\n")
    body = text[i + 2:] if i >= 0 else ""
    cleaned = body.replace(" ", "").replace("\n", "")
    chars = [c for c in cleaned if c in "fgc"]
    n_nonfgc = len(cleaned) - len(chars)
    raw_valid = (len(chars) >= 289 and n_nonfgc == 0)
    padded = (chars + ["f"] * 289)[:289]
    gt_viol = any(padded[j] != "c" for j in GT_INDICES)
    fixed = list(padded)
    for j in range(289):
        if j in GT_INDICES:
            fixed[j] = "c"
        elif fixed[j] == "c":
            fixed[j] = "f"
    fixed = "".join(fixed)
    return raw_valid, gt_viol, fixed.count("g"), fixed, len(chars), n_nonfgc


def generate_for_ckpt(seed, path, writer, fout):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True).to(DEVICE)
    model.eval()
    torch.manual_seed(2000 + seed)
    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    n = 0
    with torch.no_grad():
        pbar = tqdm(total=N_GEN, desc=f"SFT seed{seed}", unit="gen")
        while n < N_GEN:
            b = min(BATCH, N_GEN - n)
            out = model.generate(**inputs, max_new_tokens=400, temperature=1.0, do_sample=True,
                                 num_return_sequences=b, pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
            for row in out:
                text = tokenizer.decode(row, skip_special_tokens=True)
                raw_valid, gt_viol, gd, fixed, nfgc, njunk = parse_generation(text)
                writer.writerow(["SFT", seed, n, int(raw_valid), int(gt_viol), gd, fixed, nfgc, njunk])
                n += 1
            fout.flush()
            pbar.update(b)
        pbar.close()
    del model
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


def main():
    t0 = time.time()
    DATA.mkdir(exist_ok=True)
    out_csv = DATA / "supervised_generation.csv"
    if not out_csv.exists():
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["algo", "seed", "idx", "raw_valid", "gt_viol", "gd_count", "grid", "n_lattice_tokens", "n_nonlattice"])
    done = pd.read_csv(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        for seed, path in CKPTS:
            if len(done[done.seed == seed]) >= N_GEN:
                print(f"seed{seed}: already sampled, skipping")
                continue
            generate_for_ckpt(seed, path, w, f)

    df = pd.read_csv(out_csv)
    assert df.grid.str.len().eq(289).all() and df.grid.str.count("g").eq(df.gd_count).all()
    cnt = df.gd_count.value_counts().sort_index()
    per_seed = {}
    for s, g in df.groupby("seed"):
        per_seed[int(s)] = {"n": int(len(g)), "exactly16": float((g.gd_count == 16).mean()),
                            "ge20": float((g.gd_count >= 20).mean()), "in28_35": float(g.gd_count.between(28, 35).mean()),
                            "mean": float(g.gd_count.mean())}
    summary = {"n": int(len(df)), "n_per_seed": N_GEN, "checkpoints": [p for _, p in CKPTS], "prompt": PROMPT,
               "decoding": "temperature 1.0, 400-token cap, EOS, top-k 64 and top-p 0.95 from generation_config.json",
               "hist": {int(k): int(v) for k, v in cnt.items()}, "mean": float(df.gd_count.mean()), "sd": float(df.gd_count.std(ddof=1)),
               "exactly16": float((df.gd_count == 16).mean()), "ge20": float((df.gd_count >= 20).mean()),
               "in28_35": float(df.gd_count.between(28, 35).mean()), "raw_valid": float(df.raw_valid.mean()),
               "gt_viol": float(df.gt_viol.mean()), "per_seed": per_seed, "minutes": (time.time() - t0) / 60}
    json.dump(summary, open(DATA / "supervised_generation_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "hist"}, indent=1))


if __name__ == "__main__":
    main()
