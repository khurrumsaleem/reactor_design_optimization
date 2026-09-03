# Manuscript figures

Regenerates Figs. 2-7 and Extended Data Figs. 1-2 from the archival data package.

    export DATA_ROOT=/path/to/zenodo/data     # default: <repo>/../data
    python run_all.py
    python panel_values.py                    # numbers encoded in each panel

| Script | Manuscript role |
|---|---|
| fig2_dynamics | Fig. 2, alignment and inventory dynamics (a-c) |
| fig3_designs | Fig. 3, best designs (a-f), composite fitness, criticality error with 2e7-history re-evaluation, peaking-only fitness (g-i) |
| fig4_controls | Fig. 4, no-penalty control trajectory (a) and attribution against informed random search (b-d) |
| fig5_cpt | Fig. 5, CPT ablation (a-d) and per-seed fitness / inventory by initialization arm (e-f) |
| fig6_steerability | Fig. 6, fitness cost of multi-target alignment (a-c) and prompt steerability (d-e) |
| fig7_empirical | Fig. 7, Buchwald-Hartwig replication |
| ed1_correlation | Extended Data Fig. 1, correlation structure of trajectories |
| ed2_steerability_supp | Extended Data Fig. 2, SFT-only prompt curves and single-target k_eff trajectories |

Requires matplotlib, pandas, scipy. Style conventions are in `_style.py`, loaders in `_data.py`.
