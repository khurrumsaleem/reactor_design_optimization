# Values encoded in the figure panels (best-of-budget, mean ± s.d. over n = 5 seeds)

Former Table 'decomposition' + 'oracle' + 'per-design metrics' -> Fig. 3 g-i, Fig. 4 b-d, Fig. 5 e-f.

| Method | Gd | Composite | |Δk| (pcm) | Peaking-only | Fq | FΔH | Figure panels |
|---|---|---|---|---|---|---|---|
| 16-Gd symmetric reference | 16.0 | 13.512 | 11996 | 1.516 | 1.72 | 1.21 | 3a, 3g-i (dotted) |
| 24-Gd symmetric reference | 24.0 | 5.126 | 3475 | 1.651 | 1.91 | 1.26 | 3b, 3g-i (dotted) |
| GA (Gd = 16) | 16.0 ± 0.0 | 12.377 ± 0.055 | 10806 ± 88 | 1.572 ± 0.043 | 1.81 ± 0.09 | 1.21 ± 0.05 | 3c, 3g-i, 4b |
| GA (unconstrained) | 30.0 ± 0.7 | 1.895 ± 0.129 | 181 ± 138 | 1.714 ± 0.043 | 2.01 ± 0.09 | 1.26 ± 0.05 | 3d, 3g-i, 4b |
| Informed random search [20, 40] | 29.4 ± 0.5 | 1.653 ± 0.037 | 35 ± 13 | 1.618 ± 0.041 | 1.87 ± 0.07 | 1.24 ± 0.05 | 4b-d |
| DPO (CPT + SFT) | 31.0 ± 2.9 | 1.654 ± 0.042 | 44 ± 36 | 1.610 ± 0.051 | 1.88 ± 0.11 | 1.20 ± 0.06 | 3e, 3g-i, 4b-d, 5e-f |
| GRPO (CPT + SFT) | 28.8 ± 0.4 | 1.655 ± 0.051 | 40 ± 28 | 1.616 ± 0.037 | 1.87 ± 0.05 | 1.24 ± 0.03 | 3f, 3g-i, 4b-d, 5e-f |
| DPO (SFT only) | 31.2 ± 1.9 | 1.715 ± 0.108 | 37 ± 62 | 1.678 ± 0.047 | 1.95 ± 0.08 | 1.28 ± 0.05 | 5e-f |
| GRPO (SFT only) | 36.0 ± 8.4 | 1.694 ± 0.102 | 25 ± 14 | 1.669 ± 0.108 | 1.98 ± 0.19 | 1.21 ± 0.04 | 5e-f |

## Post-hoc re-evaluation, 2e7 active histories per layout (Fig. 3h hollow markers)

| Method | |Δk| (pcm) online | |Δk| (pcm) re-evaluated | OpenMC σ_k re-evaluated (pcm) |
|---|---|---|---|
| DPO | 44 ± 36 | 102 ± 71 | 20 ± 1 |
| GRPO | 40 ± 28 | 139 ± 130 | 20 ± 1 |

## No-penalty control (Fig. 4a)

- mean chosen-design Gd over final 100 steps: 0.0 ± 0.0
- minimum peaking-only fitness in run: 1.4561

## Steerability slopes, bootstrap 95% CI, 10,000 resamples (Fig. 6e)

| Checkpoint | slope | CI | excludes 0 |
|---|---|---|---|
| CPT_SFT_Base | -2 | [-12, 9] | False |
| DPO_Multi_CPT_SFT | -10 | [-39, 19] | False |
| DPO_Multi_SFT | -1 | [-39, 41] | False |
| DPO_Single_CPT_SFT | 18 | [-5, 45] | False |
| DPO_Single_SFT | -9 | [-47, 29] | False |
| GRPO_Multi_CPT_SFT | -163 | [-194, -134] | True |
| GRPO_Multi_SFT | 18 | [-14, 50] | False |
| GRPO_Single_CPT_SFT | -11 | [-43, 20] | False |
| GRPO_Single_SFT | -6 | [-41, 29] | False |
| SFT_Only_Base | 23 | [-13, 71] | False |
