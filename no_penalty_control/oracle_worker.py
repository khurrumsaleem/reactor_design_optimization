"""Persistent OpenMC evaluator used by run_no_penalty_resumable.py.

One worker per pipe: reads {"grid": "<289 symbols>"} lines on stdin and writes
[peaking_only_fitness, k_eff, fq, fdh, g_count] JSON lines on stdout. Torch is
never imported here, so physics evaluation runs in an isolated process with
its own temporary OpenMC working directory (see oracle/eval_lib.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
import eval_lib as E  # noqa: E402

for line in sys.stdin:
    grid = json.loads(line)['grid']
    fit, k, fq, fdh, count, _ = E.evaluate_grid(grid, sim_seed=None)
    if fit >= 99.0 or k <= 0:
        raise RuntimeError('OpenMC evaluation failed inside worker')
    print(json.dumps([0.6 * fq + 0.4 * fdh, k, fq, fdh, count]), flush=True)
