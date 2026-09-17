"""Fig. 5 (label fig:depletion) | Depletion characteristics of selected
configurations (Methods, "Depletion assessment of selected layouts").

a  k_inf trajectories to 50 MWd/kgHM for the best DPO, GRPO and unconstrained
   GA layouts and the symmetric 16-Gd / 24-Gd references (±1 Monte Carlo SD).
b  endpoint residual Gd effect: remaining Gd isotopes removed at fixed
   endpoint densities of all other nuclides (propagated MC SD).
c  BOL / endpoint k_inf, interpolated unity-crossing burnup, residual effect.
Inputs: $DATA_ROOT/depletion/{case}/trajectory.csv, complete.json
(depletion/run_depletion.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D
from _style import C, apply_style, panel_title, save, table_panel

OUT = Path(__file__).resolve().parent / "fig_depletion"
NAMES = {"DPO": "DPO", "GRPO": "GRPO", "GA": "GA", "REF16": "16-Gd reference", "REF24": "24-Gd reference"}


def main():
    apply_style()
    plt.rcParams.update({"savefig.bbox": None, "font.size": 7.5})
    fig = plt.figure(figsize=(7.2, 6.7))
    g = fig.add_gridspec(2, 2, height_ratios=[2.7, 1.85], left=.09, right=.98, top=.92, bottom=.13, hspace=.6, wspace=.42)
    a, b = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1])
    rows = []
    for i, k in enumerate(D.DEPLETION_CASES):
        r, d = D.depletion_summary(k), D.depletion_trajectory(k)
        assert len(d) == 27 and d.iloc[-1, 0] == 50, f"{k}: expected 27 burnup nodes ending at 50 MWd/kgHM"
        x, y, sd = (d[c].to_numpy() for c in ("burnup_MWd_per_kg_initial_HM", "k_infinity", "k_sd"))
        a.plot(x, y, label=NAMES[k], color=C[k], lw=1.2, ls="--" if k.startswith("REF") else "-")
        a.fill_between(x, y - sd, y + sd, color=C[k], alpha=.10, lw=0)
        assert np.isclose(y[0], r["BOL_k"]) and np.isclose(y[-1], r["endpoint_k"])
        cross = [x[j - 1] + (1 - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])
                 for j in range(1, len(x)) if y[j - 1] >= 1 and y[j] < 1]
        assert np.isclose(cross[-1], r["last_downward_unity_crossing_MWd_per_kg_initial_HM"])
        b.errorbar(r["residual_Gd_penalty_delta_k"] * 1e3, i, xerr=r["residual_penalty_sd_independent"] * 1e3,
                   fmt="o", ms=4, color=C[k], capsize=3, lw=1)
        rows.append([NAMES[k], f"{r['BOL_k']:.4f}", f"{r['endpoint_k']:.4f}",
                     f"{r['last_downward_unity_crossing_MWd_per_kg_initial_HM']:.2f}",
                     f"{r['residual_Gd_penalty_delta_k'] * 1e3:.2f} ± {r['residual_penalty_sd_independent'] * 1e3:.2f}"])
    a.axhline(1, color="#999", lw=.8, ls=":")
    a.set(xlim=(0, 50), xlabel="Burnup (MWd/kg initial HM)", ylabel=r"$k_\infty$")
    a.legend(fontsize=6.6, loc="upper right")
    panel_title(a, "a", "Depletion trajectories")
    b.axvline(0, color="#999", lw=.8, ls=":")
    b.set(yticks=range(5), yticklabels=list(NAMES.values()),
          xlabel=r"Residual Gd effect, $\Delta k_{\mathrm{Gd}}$ ($10^{-3}$)", xlim=(-1.2, 11.5))
    b.invert_yaxis()
    panel_title(b, "b", "Gd removal at 50 MWd/kg")
    table_panel(fig.add_subplot(g[1, :]), "c", "Selected-layout depletion results", rows,
                ["Layout", r"BOL $k_\infty$", r"50 MWd/kg $k_\infty$", "Unity-crossing burnup\n(MWd/kg initial HM)",
                 r"$\Delta k_{\mathrm{Gd}}$ ($10^{-3}$)" + "\n(estimate ± MC SD)"],
                widths=[.22, .13, .18, .25, .22], fontsize=7.2)
    fig.text(.09, .965, "35 W/g initial HM · 26 depletion intervals · CE/CM · 2D reflective assembly", fontsize=8)
    fig.text(.09, .045, "Shading and error bars: ±1 Monte Carlo SD. One selected layout per method/reference.\n"
             "Residual effect: remove remaining Gd at fixed endpoint densities of all other nuclides.\n"
             "Unity crossings are interpolated lattice burnups; depletion quantities were not included in alignment rewards.",
             fontsize=6.8, linespacing=1.6)
    save(fig, OUT)


if __name__ == "__main__":
    main()
