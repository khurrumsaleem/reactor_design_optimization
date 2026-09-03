# Registered five-seed results

Both registered experiments completed successfully on the local host. Each run
contains five seeds, CPT+SFT and SFT-only initialization arms, online DPO and
GRPO, exact finite-catalog evaluation, and 40,000 measured-reward trajectories.

## Runtime

- Calibrated estimate before launch: 6.96 minutes
- Planning range: 5.56–10.43 minutes
- Observed wall time from first checkpoint through final summary: 7.66 minutes

The estimate is saved in `results/runtime_estimate.json`.

## Single target: requested yield 90

| Initialization | Policy | Expected absolute error | Outside-support mass |
|---|---|---:|---:|
| CPT+SFT | SFT base | 50.040 | 0.287 |
| CPT+SFT | DPO | 29.886 | 0.701 |
| CPT+SFT | GRPO | 29.771 | 0.714 |
| SFT-only | SFT base | 50.929 | 0.669 |
| SFT-only | DPO | 30.530 | 0.728 |
| SFT-only | GRPO | 29.878 | 0.731 |

Paired per-seed changes relative to the corresponding SFT base were:

| Initialization / algorithm | Error change, mean ± SD | Support-mass change, mean ± SD |
|---|---:|---:|
| CPT+SFT / DPO | -20.154 ± 0.299 | +0.414 ± 0.057 |
| CPT+SFT / GRPO | -20.269 ± 0.444 | +0.427 ± 0.025 |
| SFT-only / DPO | -20.399 ± 0.454 | +0.059 ± 0.013 |
| SFT-only / GRPO | -21.052 ± 0.746 | +0.062 ± 0.010 |

All five seeds in all four comparisons reduced error and increased off-support
probability. The full empirical oracle error was 20.577, versus 24.628 for the
restricted-support oracle.

## Multi target: ten balanced yield bins

| Initialization | Policy | Expected absolute error | Outside-support mass |
|---|---|---:|---:|
| CPT+SFT | SFT base | 28.971 | 0.036 |
| CPT+SFT | DPO | 24.283 | 0.080 |
| CPT+SFT | GRPO | 24.110 | 0.074 |
| SFT-only | SFT base | 29.352 | 0.073 |
| SFT-only | DPO | 24.402 | 0.152 |
| SFT-only | GRPO | 24.252 | 0.127 |

Paired per-seed changes were:

| Initialization / algorithm | Error change, mean ± SD | Support-mass change, mean ± SD |
|---|---:|---:|
| CPT+SFT / DPO | -4.688 ± 0.444 | +0.044 ± 0.023 |
| CPT+SFT / GRPO | -4.861 ± 0.289 | +0.038 ± 0.015 |
| SFT-only / DPO | -4.950 ± 0.612 | +0.079 ± 0.023 |
| SFT-only / GRPO | -5.100 ± 0.476 | +0.055 ± 0.033 |

Again, all five seeds in all four comparisons improved error and expanded
support. The full empirical oracle error was 8.627, versus 10.830 under the
restricted catalog.

## Policy differentiation and the NMI issue

In the multi-target study, exact DPO–GRPO JSD was `0.0436 ± 0.0090` for CPT+SFT
and `0.0492 ± 0.0093` for SFT-only. Algorithm–ligand NMI was only
`0.00020 ± 0.00014` and `0.00068 ± 0.00030`, respectively.

This is the anticipated distinction: DPO and GRPO assign different probability
to complete measured reactions, but their four-category ligand marginals remain
similar. Ligand-only NMI therefore misses differentiation expressed primarily
through ligand–additive–base combinations. NMI should be reported as a secondary
projection, with full-action JSD as the direct policy-level endpoint.

## Completion checks

- `registered_single`: 480 metric rows, 40,000 trajectories, 10 comparisons
- `registered_multi`: 4,800 metric rows, 40,000 trajectories, 10 comparisons
- 20/20 online checkpoints complete in each experiment
- 20/20 seed/arm/algorithm jobs complete in each experiment
- All rewards identify measured catalog rows; no missing combination is imputed

Machine-readable results are under `results/registered_single/` and
`results/registered_multi/`.

