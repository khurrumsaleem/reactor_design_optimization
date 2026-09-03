# Historical pilot result (superseded)

The registered five-seed single- and multi-target experiments have now been
completed. Use `FULL_RESULTS.md` and `results/registered_*` for analysis. This
file is retained only to document the earlier implementation pilot and must not
be mixed with the registered results after the training-budget alignment.

This pilot checks that the registered experiment is capable of identifying the
proposed phenomenon before spending the full five-seed budget. It used target
yields 30, 60, and 90; 12 SFT epochs; and 200 online updates.

## CPT + SFT initialization

| Policy | Expected absolute target error | Outside-support probability |
|---|---:|---:|
| SFT base | 32.05 | 0.107 |
| DPO | 30.90 | 0.140 |
| GRPO | 28.56 | 0.137 |

Relative to SFT, DPO reduced expected error by 1.15 yield points and increased
outside-support mass by 0.033. GRPO reduced error by 3.50 points and increased
outside-support mass by 0.030. Thus the observed expansion was accompanied by
better measured-reward performance rather than diffusion alone.

## SFT-only initialization

| Policy | Expected absolute target error | Outside-support probability |
|---|---:|---:|
| SFT base | 32.49 | 0.449 |
| DPO | 30.71 | 0.477 |
| GRPO | 27.86 | 0.483 |

SFT-only retained substantially more off-support mass before alignment. This is
itself a useful control: CPT makes the imposed corpus support much sharper, so
the meaning of subsequent expansion depends on initialization history.

## Differentiation metrics

The exact DPO–GRPO Jensen–Shannon divergence was 0.0331 in the CPT+SFT arm and
0.0417 in the SFT-only arm. In contrast, algorithm–ligand NMI was only 0.00063
and 0.0117. This is not contradictory: NMI uses only the four-category ligand
marginal, whereas JSD sees differences among all measured ligand–additive–base
reactions. The pilot illustrates why NMI should be secondary rather than the
sole test of policy differentiation.

The mean full-catalog oracle target error was 9.33, compared with 11.76 for the
restricted-support oracle. The physical measurements therefore contain a real,
nonzero opportunity for support expansion to improve attainable performance.

## Status

These numbers are a single-seed implementation pilot and must not be presented
as a confirmatory result. `configs/full.json` predefines five seeds and 1,000
online steps for the actual comparison. The complete machine-readable pilot
output is in `results/pilot_seed0/`.
