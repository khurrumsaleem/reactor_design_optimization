# Independent OpenMC re-evaluation and statistical audits

Inputs are read from the archival data package (`DATA_ROOT`, default `<repo>/../data`); outputs are written to `./results`. The recorded outputs are archived under `data/reevaluation/`.

This directory re-evaluates the five per-seed best DPO designs and five
per-seed best GRPO designs reported in the ReactorGen manuscript. It preserves
the original OpenMC geometry and settings while recording the `k` standard
deviation that the training scripts discarded.

Default experiment:

```bash
OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml DATA_ROOT=/path/to/data python \
  openmc_recheck.py --replicates 5 --workers 8
```

The script is resumable. Completed combinations of method, training seed, and
transport replicate are skipped when the same output CSV already exists.

## High-statistics confirmation (prepared, not automatically run)

The minimum high-statistics check evaluates only the globally best DPO and
GRPO layouts. Its default is 200,000 particles per batch, 110 batches, and 10
inactive batches: 20 million active histories per layout, 50 times the active
histories of one original evaluation.

```bash
OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml DATA_ROOT=/path/to/data python \
  openmc_high_stat.py --selection global-best
```

To apply the same high-statistics check to all five per-seed best layouts from
both methods (10 layouts total), use `--selection all-per-seed`. Particle,
batch, replicate, worker, and thread counts are command-line options so that
the compute budget can be reduced before execution if needed.

## Statistical-method audit (no new simulations)

`trajectory_ci.py` numerically reconstructs all trajectory bands from the five
seed CSVs. It uses the procedure already present in the figure code: a
pointwise two-sided Student-t interval,
`mean +/- t(0.975, n-1) * sample_sd / sqrt(n)`, with `n=5` independent seeds.

`steerability_bootstrap.py` recomputes every prompt-steerability slope using
10,000 bootstrap resamples by default. Samples are independently resampled
with replacement within each target value, the seven resampled target means
are fit by ordinary least squares, and the 2.5th and 97.5th percentiles form
the interval. Both exclusion and inclusion of flagged truncated generations
are reported so that this data-cleaning choice is auditable.

```bash
python trajectory_ci.py
python steerability_bootstrap.py
```
