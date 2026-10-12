# Manuscript figure reproduction

The revised manuscript is *Scientific feedback reveals productive regimes
beyond training conventions*. These builders cover the two scientific
implementations and their attribution controls. Earlier `figures/` filenames
are historical; the current mapping is:

| Current display item | Source | Archived input |
|---|---|---|
| Figure 1: two implementations and inventory-axis attribution | `figure1/make_fig1.js` | Packaged `figure1/assets/` and `figure1/data/` |
| Figure 2: feedback-directed redistribution and useful generation | `build_reactor_figures.py` | `data/source_tree/` |
| Figure 3: objective removal and search controls | `build_reactor_figures.py` | `data/source_tree/` |
| Figure 4: measured-reaction policy redistribution | `plot_fig4_chemistry.py` | `data/source_tree/` plus the restored measured CSV |
| SI convergence traces and selected layouts | `build_deduplicated_si_figures.py` | `figures/source_data/supplementary/` vector PDFs |
| SI first-passage and seedwise sampling counts | `build_deduplicated_si_figures.py` | Same vector PDF inputs |
| SI initialization ablation (`fig5_cpt.pdf`) | `../figures/fig5_cpt.py` | `data/` via `DATA_ROOT` |
| SI prompt sweep (`si_prompt_sweep.pdf`) | `build_supplementary_prompt_sweep.py` | `data/source_tree/` |
| SI depletion (`fig_depletion.pdf`) | `../figures/fig_depletion.py` | `data/` via `DATA_ROOT` |

## Build from a GitHub checkout and the data archive

Run from the repository root. Extract `ScientificFeedback_code_data_v3.zip`
separately and set `ARCHIVE_ROOT` to its top-level directory, which contains
`data/`, `figures/`, and `code/`. The code repository does not include the
experiment logs, final checkpoints, or source PDF collection.

```bash
export ARCHIVE_ROOT=/path/to/ScientificFeedback_code_data_v3
python -m pip install -r paper_figures/requirements.txt
python paper_figures/build_reactor_figures.py \
  --source-root "$ARCHIVE_ROOT/data/source_tree" --output-dir work/rebuilt_figures
python paper_figures/build_supplementary_prompt_sweep.py \
  --source-root "$ARCHIVE_ROOT/data/source_tree" --output-dir work/rebuilt_figures
python paper_figures/build_deduplicated_si_figures.py \
  --source-dir "$ARCHIVE_ROOT/figures/source_data/supplementary" \
  --output-dir work/rebuilt_figures
```

The Python builders read completed records; they do not load policy weights,
perform alignment, run transport or depletion, or redo bootstrap resampling.
Outputs must be outside the input source directory.

Figure 4 recomputes catalog-uniform reference expectations exactly. Restore the
registered upstream measured table before running it:

```bash
python -m pip install -r empirical/requirements-inputs.txt
python empirical/scripts/fetch_buchwald_hartwig.py \
  --output "$ARCHIVE_ROOT/data/source_tree/empirical/data/buchwald_hartwig.csv"
python paper_figures/plot_fig4_chemistry.py \
  --source-root "$ARCHIVE_ROOT/data/source_tree" --output-dir work/rebuilt_figures
```

Both the workbook and exported CSV must match the SHA-256 values in
[`empirical/data/SOURCE.md`](../empirical/data/SOURCE.md). The measured input is
obtained separately. The source mirror also preserves reference implementation
files that the builders checksum for provenance; they are not imported as
plotting code.

For the two SI figures still built by the earlier plotting modules:

```bash
DATA_ROOT="$ARCHIVE_ROOT/data" python figures/fig5_cpt.py
DATA_ROOT="$ARCHIVE_ROOT/data" python figures/fig_depletion.py
```

Those historical entry points write their PDFs and PNGs beside the scripts.
Final manuscript PDFs are supplied in the archive's `figures/` tree. PDF
metadata, installed fonts and library versions can affect rebuilt file bytes;
the archived PDFs and their checksums identify the submitted artwork.

To use the copy of this code inside the extracted archive, change into
`$ARCHIVE_ROOT/code` and run the same commands with `ARCHIVE_ROOT` set as above.

`build_prompt_control.py` is preserved as an intermediate manuscript figure
builder. Its `fig5_prompt_control` output is not a current main-text figure.
The current SI uses `build_supplementary_prompt_sweep.py` for that analysis.
Figure 1 has separate Node/PPTX instructions in [`figure1/README.md`](figure1/README.md).
