"""Write panel_values.md: the exact numbers that the former tables carried,
now encoded in figure panels, for use in captions / Supplementary Tables."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D

def ms(v, f=".3f"):
    v = np.asarray(v, float)
    return f"{v.mean():{f}} ± {v.std(ddof=1):{f}}" if len(v) > 1 else f"{v[0]:{f}}"

nu = D.nuscale_records()
recs = {
    "16-Gd symmetric reference": [nu["NU16"]], "24-Gd symmetric reference": [nu["NU24"]],
    "GA (Gd = 16)": D.best_records([D.ga_steps(d) for d in D.ga(True)]),
    "GA (unconstrained)": D.best_records([D.ga_steps(d) for d in D.ga(False)]),
    "Informed random search [20, 40]": D.informed_best_records(),
    "DPO (CPT + SFT)": D.best_records([D.dpo_steps(d) for d in D.dpo_single()]),
    "GRPO (CPT + SFT)": D.best_records([D.grpo_steps(d) for d in D.grpo_single()]),
    "DPO (SFT only)": D.best_records([D.dpo_steps(d) for d in D.dpo_single("sft_only")]),
    "GRPO (SFT only)": D.best_records([D.grpo_steps(d) for d in D.grpo_single("sft_only")]),
}
hs = D.high_stat()
lines = ["# Values encoded in the figure panels (best-of-budget, mean ± s.d. over n = 5 seeds)\n",
         "Former Table 'decomposition' + 'oracle' + 'per-design metrics' -> Fig. 3 g-i, Fig. 4 b-d, Fig. 5 e-f.\n",
         "| Method | Gd | Composite | |Δk| (pcm) | Peaking-only | Fq | FΔH | Figure panels |", "|---|---|---|---|---|---|---|---|"]
where = {"16-Gd symmetric reference": "3a, 3g-i (dotted)", "24-Gd symmetric reference": "3b, 3g-i (dotted)",
         "GA (Gd = 16)": "3c, 3g-i, 4b", "GA (unconstrained)": "3d, 3g-i, 4b", "Informed random search [20, 40]": "4b-d",
         "DPO (CPT + SFT)": "3e, 3g-i, 4b-d, 5e-f", "GRPO (CPT + SFT)": "3f, 3g-i, 4b-d, 5e-f",
         "DPO (SFT only)": "5e-f", "GRPO (SFT only)": "5e-f"}
for name, r in recs.items():
    lines.append(f"| {name} | {ms([x['gd'] for x in r], '.1f')} | {ms([x['fit'] for x in r])} | "
                 f"{ms([D.dk_pcm(x['k']) for x in r], '.0f')} | {ms([D.peaking_only(x['fq'], x['fdh']) for x in r])} | "
                 f"{ms([x['fq'] for x in r], '.2f')} | {ms([x['fdh'] for x in r], '.2f')} | {where[name]} |")
lines += ["", "## Post-hoc re-evaluation, 2e7 active histories per layout (Fig. 3h hollow markers)", "",
          "| Method | |Δk| (pcm) online | |Δk| (pcm) re-evaluated | OpenMC σ_k re-evaluated (pcm) |", "|---|---|---|---|"]
for m in ("DPO", "GRPO"):
    sub = hs[hs["method"] == m]
    lines.append(f"| {m} | {ms(sub['original_dk_pcm'], '.0f')} | {ms(sub['recheck_dk_pcm'], '.0f')} | {ms(sub['mean_openmc_k_std_pcm'], '.0f')} |")
npn = D.no_penalty()
lines += ["", "## No-penalty control (Fig. 4a)", "",
          f"- mean chosen-design Gd over final 100 steps: {npn['chosen_g_count'].tail(100).mean():.1f} ± {npn['chosen_g_count'].tail(100).std(ddof=1):.1f}",
          f"- minimum peaking-only fitness in run: {(0.6*npn['chosen_fq']+0.4*npn['chosen_fdh']).min():.4f}"]
sl = D.steerability_slopes().set_index("model_name")
lines += ["", "## Steerability slopes, bootstrap 95% CI, 10,000 resamples (Fig. 6e)", "", "| Checkpoint | slope | CI | excludes 0 |", "|---|---|---|---|"]
for n, r in sl.iterrows():
    lines.append(f"| {n} | {r['point_slope']:.0f} | [{r['ci_lower']:.0f}, {r['ci_upper']:.0f}] | {r['ci_excludes_zero']} |")
Path(__file__).with_name("panel_values.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
