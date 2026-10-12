# Measured-reaction policy adaptation beyond training conventions

This project implements the measured-reaction setting in *Scientific feedback
reveals productive regimes beyond training conventions*, using a finite
high-throughput chemistry data set. It does **not** call OpenMC, a chemistry
simulator, or a learned yield surrogate. Every reward is a lookup of a yield
that was measured in the Buchwald–Hartwig experiment.

## Research question

Does measured scientific feedback redistribute a generative policy toward
productive task–ligand combinations omitted from its task-specific CPT/SFT
corpus, and do DPO and GRPO develop measurably different policies?

The task prompt is an aryl halide and a requested yield. The generated reaction
sequence is:

```text
aryl-halide | target-yield -> ligand | additive | base
```

The CPT/SFT corpus contains one deterministic `anchor ligand` per aryl-halide
task. The mapping is balanced by sorted task index, so all four ligand tokens
are learned globally while 75% of task–ligand pairings remain outside the
training support. Alignment samples from all measured task–ligand pairings.
The allowed catalog remains fixed throughout adaptation. This tests movement
of probability beyond task-specific training combinations; all ligand
identities are already known globally.

## What makes the reward empirical

- The catalog contains 3,955 measured reactions.
- A prefix-trie mask permits only catalogued reaction combinations.
- The reward is computed from the `Output` cell for the selected row.
- Five missing combinations in the nominal 4 × 22 × 3 × 15 factorial grid are
  never generated or imputed.
- No interpolation, regression, neural predictor, or simulator is used.

This is therefore **empirical-data-mediated alignment**, not online wet-lab RL.
An online wet-lab claim would require synthesizing newly proposed reactions.

## Restore the measured input table

The measured table is not redistributed in this package. From this `empirical/`
directory, install the small input-restoration dependency and fetch the source:

```bash
python -m pip install -r requirements-inputs.txt
python scripts/fetch_buchwald_hartwig.py
```

The helper downloads the upstream workbook into memory, verifies its SHA-256,
exports `Plates1-3` with the original values, columns and row order, and verifies
the exact CSV SHA-256 before writing `data/buchwald_hartwig.csv`. An existing
matching CSV is verified and reused; a different file is never overwritten.
Use `--workbook /path/to/original.xlsx` for offline restoration, or
`--output /path/to/buchwald_hartwig.csv` for a custom destination. To regenerate
current Figure 4, set `ARCHIVE_ROOT` to the separately extracted data archive
and restore the file to its source mirror instead:

```bash
python scripts/fetch_buchwald_hartwig.py \
  --output "$ARCHIVE_ROOT/data/source_tree/empirical/data/buchwald_hartwig.csv"
```

See `data/SOURCE.md` for the upstream publication, URL and registered hashes.
The preparation and archived-result figure commands perform no model training.

## Environment for optional experiment reruns

Use Python 3.10+ with PyTorch and NumPy. For example:

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements.txt -r requirements-inputs.txt
```

The separately distributed archive already contains completed results. The commands below are for
rerunning the registered experiments, rather than necessary setup for viewing
or regenerating the published results.

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
venv/bin/python -u scripts/run_full_pipeline.py --data data/buchwald_hartwig.csv --resume
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

The historical plotting entry point is
`plot/plot_empirical_support_expansion.py`; its filename is retained for
compatibility. In this study, the corresponding quantity is probability mass
on task–ligand combinations excluded from the training corpus, within the
fixed measured catalog. Current manuscript Figure 4 is reproduced with
[`paper_figures/plot_fig4_chemistry.py`](../paper_figures/plot_fig4_chemistry.py).

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
so movement beyond task-specific training combinations can be assessed
alongside reward improvement.

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
