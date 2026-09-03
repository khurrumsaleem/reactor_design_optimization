"""Extended Data Fig. 2 | Supplementary steerability panels.

a  generated Gd inventory vs prompted target for the SFT-only checkpoints
b  k_eff of the cumulative-best design, single-target DPO and GRPO
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from fig6_steerability import STYLE
from _style import C, WIDTH_DOUBLE, apply_style, band, panel_label, refline, save

OUT = Path(__file__).resolve().parent / "ed2_steerability_supp"
SFT_MODELS = ["SFT_Only_Base", "DPO_Single_SFT", "DPO_Multi_SFT", "GRPO_Single_SFT", "GRPO_Multi_SFT"]


def main():
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 2.7), gridspec_kw=dict(wspace=0.35))
    sm = D.steerability_means()
    for name in SFT_MODELS:
        sub = sm[sm["model_name"] == name].sort_values("target_k"); col, ls, lab = STYLE[name]
        a.fill_between(sub["target_k"], sub["ci_lower"], sub["ci_upper"], color=col, alpha=0.15, linewidth=0)
        a.plot(sub["target_k"], sub["mean_gd"], color=col, linestyle=ls, marker="o", markersize=3, linewidth=1.2, label=lab)
    a.set_xlabel(r"Prompted target $k_{\mathrm{eff}}$"); a.set_ylabel("Generated Gd inventory (rods)")
    a.set_xticks(np.round(np.arange(1.02, 1.081, 0.01), 2)); a.set_title("SFT-only checkpoints", loc="left", fontsize=7.5)
    a.legend(fontsize=6, loc="upper right")
    for key, col in (("DPO", C["DPO"]), ("GRPO", C["GRPO"])):
        steps = [D.dpo_steps(d) for d in D.dpo_single()] if key == "DPO" else [D.grpo_steps(d) for d in D.grpo_single()]
        x, m = D.matrix(steps, "cum_best_k"); band(b, x, m, color=col, label=f"{key} (CPT + SFT)")
    refline(b, 1.05, r"target $k_{\mathrm{eff}}$ = 1.05")
    b.set_xlabel("OpenMC evaluations"); b.set_ylabel(r"$k_{\mathrm{eff}}$ of cumulative-best design"); b.set_xlim(0, 2000)
    b.yaxis.set_major_formatter(FormatStrFormatter("%.2f")); b.legend(fontsize=6.3, loc="upper right")
    panel_label(a, "a", x=-0.18); panel_label(b, "b", x=-0.22)
    fig.tight_layout()
    save(fig, OUT)


if __name__ == "__main__":
    main()
