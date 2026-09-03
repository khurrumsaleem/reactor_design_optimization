# Empirical ReactorGen: measured-reaction alignment

This project transfers the ReactorGen experimental logic to a real, finite
high-throughput chemistry data set. It does **not** call OpenMC, a chemistry
simulator, or a learned yield surrogate. Every reward is a lookup of a yield
that was measured in the Buchwald–Hartwig experiment.

## Research question

Does online preference/reinforcement alignment move a generative policy beyond
the support imposed by CPT/SFT, and do DPO and GRPO develop measurably different
policies, when feedback comes from real experimental measurements?

The task prompt is an aryl halide and a requested yield. The generated reaction
sequence is:

```text
aryl-halide | target-yield -> ligand | additive | base
```

The CPT/SFT corpus contains one deterministic `anchor ligand` per aryl-halide
task. The mapping is balanced by sorted task index, so all four ligand tokens
are learned globally while 75% of task–ligand pairings remain outside the
training support. Alignment samples from all measured task–ligand pairings.
This tests combination-support expansion rather than acquisition of an unseen
vocabulary token.

## What makes the reward empirical

- The catalog contains 3,955 measured reactions.
- A prefix-trie mask permits only catalogued reaction combinations.
- The reward is computed from the `Output` cell for the selected row.
- Five missing combinations in the nominal 4 × 22 × 3 × 15 factorial grid are
  never generated or imputed.
- No interpolation, regression, neural predictor, or simulator is used.

This is therefore **empirical-data-mediated alignment**, not online wet-lab RL.
An online wet-lab claim would require synthesizing newly proposed reactions.

## Environment

Create the requested local virtual environment as `venv` (not `.venv`). The
`openmc` environment already supplies the compatible PyTorch build, so the venv
can inherit those packages without downloading another large wheel:

```bash
conda run -n openmc python -m venv --system-site-packages venv
venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

## Runtime estimate first

Measure phase throughput on the current host before launching the study:

```bash
venv/bin/python -u scripts/estimate_runtime.py
```

This calibrates CPT, SFT, DPO, GRPO, and exact evaluation with the registered
model size and writes `results/runtime_estimate.json`.

## Resumable registered experiment

The full pipeline contains both the single-target and 10-bin multi-target arms,
five seeds, CPT+SFT versus SFT-only, and DPO versus GRPO:

```bash
venv/bin/python -u scripts/run_full_pipeline.py --resume
```

`Ctrl-C` is safe. CPT/SFT are checkpointed after every epoch. DPO/GRPO are
checkpointed atomically every 25 steps and once more when interrupted. Re-run
the identical command to continue. Checkpoints include the model, optimizer,
Python sampling state, PyTorch generator state, next step, and prior trajectory.
Changing a run's step count or seed causes an explicit checkpoint mismatch
instead of silently mixing experiments.

For a quick validation:

```bash
venv/bin/python scripts/run_experiment.py --config configs/smoke.json --resume
venv/bin/python -m unittest discover -s tests -v
```

Results are written below `results/<run_name>/`:

- `metrics.csv`: exact policy expectations and support mass;
- `summary.json`: aggregate results, DPO–GRPO JSD, and policy/ligand NMI;
- `trajectories.csv`: online samples and measured rewards;
- `checkpoints/*.pt`: final policy states.
- `progress.json`: completed seed/arm/algorithm jobs.

## Primary endpoints

1. Expected absolute target error under the exact finite policy distribution.
2. Probability mass outside the CPT/SFT task–ligand support.
3. DPO–GRPO Jensen–Shannon divergence over measured reactions.
4. Ligand entropy and algorithm–ligand normalized mutual information (secondary).

The empirical full-catalog and restricted-catalog oracles are always reported,
so support expansion can be distinguished from genuine reward improvement.

## Correspondence to ReactorGen

The registered budgets match the source pipeline: 5 CPT epochs, 10 SFT epochs,
1,000 online DPO steps with `beta=0.01`, and 500 GRPO steps with group size 4.
Both initialization histories, five seeds, and single/multi-target conditions are
retained. AdamW, pairwise online DPO, and standardized within-group GRPO losses
also follow the source scripts.

The backbone is deliberately not identical: ReactorGen uses Gemma 3 270M to
generate 17×17 cores, whereas this finite empirical study uses a small masked
autoregressive categorical policy. The mask is necessary to guarantee that
every reward corresponds to an actually measured reaction. The multi-target
analogue balances 10 yield bins rather than sampling continuous `k` values.

See [METHODS.md](METHODS.md) for the estimand, controls, and interpretation, and
[data/SOURCE.md](data/SOURCE.md) for provenance and checksums. A completed
five-seed single- and multi-target study is summarized in
[FULL_RESULTS.md](FULL_RESULTS.md). The earlier implementation pilot remains in
[PILOT_RESULTS.md](PILOT_RESULTS.md) for provenance but is superseded by the
registered result.
