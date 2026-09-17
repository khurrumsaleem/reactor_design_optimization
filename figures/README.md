# Manuscript figures

Regenerates the main-text figures and Extended Data Figs. 1-3 from the archival data package.

    export DATA_ROOT=/path/to/zenodo/data     # default: <repo>/../data
    python ../analysis/first_passage.py       # writes data/analysis/first_passage_per_seed.csv (needed by ed3)
    python run_all.py
    python panel_values.py                    # numbers encoded in each panel

| Script | Manuscript role |
|---|---|
| fig2_discovery | Fig. 2, alignment dynamics (a-c), best designs (d-i) and fitness decomposition (j-l); merges the former fig2_dynamics + fig3_designs |
| fig4_controls | Fig. 3 (label `fig:controls`), objective-dependent adaptation with five no-penalty seeds (a-b), fixed-search controls including uninformed random search (c-e), and the numeric tables (f-g) |
| fig_depletion | Fig. 4 (label `fig:depletion`), depletion trajectories, residual Gd effect and numeric table |
| fig5_cpt | CPT ablation (a-d) and per-seed fitness / inventory by initialization arm (e-f) |
| fig6_steerability | fitness cost of multi-target alignment (a-c) and prompt steerability (d-e) |
| fig7_empirical | Buchwald-Hartwig replication |
| ed1_correlation | Extended Data Fig. 1, correlation structure of trajectories |
| ed2_steerability_supp | Extended Data Fig. 2, SFT-only prompt curves and single-target k_eff trajectories |
| ed3_inventory_sampling | Extended Data Fig. 3, first passage, cumulative-best inventory trajectories and full-space sampling counts |
| fig2_dynamics, fig3_designs | earlier separate versions of Fig. 2, kept for reference |

Figure numbers in the file names are historical; the manuscript numbers its display items automatically.
Requires matplotlib, pandas, scipy. Style conventions are in `_style.py`, loaders in `_data.py`.
