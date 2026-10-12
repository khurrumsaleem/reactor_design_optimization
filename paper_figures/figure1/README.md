# Figure 1: editable build from packaged assets

This is the creator used for the current `fig1_concept.pdf`. It comes from the
manuscript's `claude_figure/` authoring directory. The later `codex_v2` variant is not
the submitted artwork.

`make_fig1.js` assembles the editable single-slide PPTX from the supplied PNG
assets and JSON values. It uses the existing Blender renders, scientific field
charts, reaction drawings and measured-result charts; this build performs no
scientific simulation, model loading, generation or Blender rendering.

Install Node.js and `pptxgenjs`, then run from the repository root (or from
the extracted archive's `code/` directory):

```bash
npm install --prefix work/figure1-node pptxgenjs
PPTXGENJS="$PWD/work/figure1-node/node_modules/pptxgenjs" \
  node paper_figures/figure1/make_fig1.js \
  --output work/rebuilt_figures/fig1_concept.pptx
```

A regular local `pptxgenjs` installation on Node's module search path also
works without `PPTXGENJS`. With no `--output`, the script writes
`Figure1_NMI_v2.pptx` beside itself. To export PDF, open the PPTX in PowerPoint
or LibreOffice and preserve the slide size and fonts. The final submitted PDF
and editable PPTX are already included in the separately extracted archive's `figures/main/` tree.
PDF bytes and text placement can vary with the exporting application; use the
archived PDF when an exact match is required.

## Asset provenance and optional authoring scripts

`assets/` and `data/` are copied from the original figure creator. In particular,
`data/supervised_generation.csv` contains the 5,000 archived supervised-policy
outputs used by the attribution panel. The PNG assets directly used by `make_fig1.js` are retained. Unused raw
Blender intermediates and PyVista fallback renders, logs, and generated
PDF/preview files are omitted from this public code directory. The archive
retains the complete source asset set.

The following original authoring scripts are retained for provenance:

- `make_assets.py`, `make_assets_extra.py`: assemble source data, molecule
  drawings and rendered assets; require the original project tree, its
  original figure intermediates, the measured reaction CSV, and RDKit.
- `make_charts.py`: constructs the chart PNGs using the original source tree,
  including `claude_figure/revised_nature_3/fields_dpo_best_hi.npz` and archived
  data. Requires NumPy, pandas, matplotlib, SciPy and tqdm.
- `render_blender.py`: creates 3D assembly renders with Blender's Python
  (`bpy`), and reads `claude_figure/revised_fable1/fig1_data.json` from the
  original project tree.
- `gen_supervised_policy.py`: generates the supervised-policy response sample;
  requires the original model checkpoints, PyTorch/Transformers, and original
  local project paths. It is not needed to rebuild the packaged figure.

These optional scripts retain their original project-relative or local source
paths. They are not claimed to be a self-contained fresh asset-generation
pipeline. The packaged assets/data plus `make_fig1.js` provide the portable
figure assembly. `README_KO.txt` is the original authoring note and retains
historical machine setup; use this README for the archive build instructions.
