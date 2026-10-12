# Scientific feedback reveals productive regimes beyond training conventions

Software accompanying the manuscript by Yoon Pyo Lee, Samrendra Roy, Souvik
Chakraborty and Syed Bahauddin Alam, submitted to Nature Machine Intelligence.
The reactor implementation retains the name ReactorGen.

The study asks whether scientific performance feedback can redirect a
generative policy trained on a restricted design convention toward productive
regions beyond that convention. It evaluates two implementations: a Gemma 3
270M sequence policy with OpenMC reactor-lattice feedback, and a small
non-language categorical policy with measured Buchwald–Hartwig reaction yields.
The allowed design space stays fixed; adaptation redistributes probability
within it. The objective rewards scientific performance without an explicit
novelty reward or a reward for changing the convention-defining feature.

We distinguish candidate availability, learned redistribution and reusable
search information. The reactor experiments provide the full attribution
controls, including informed random search and objective removal. The chemistry
experiments test redistribution with measured feedback and exact policy
summaries, without a learned yield predictor.

This repository contains training, evaluation and control implementations,
current figure builders, and an archived-result verifier. Completed trajectories,
figure source data and final chemistry checkpoints are supplied separately in
`ScientificFeedback_code_data_v3.zip`. The archive has been prepared for Zenodo;
a public archive DOI will be added after publication. Large reactor checkpoints
are not included. The supervised reactor corpus is also available from the
[ReactorGen dataset on Hugging Face](https://huggingface.co/datasets/Sixticket/ReactorGen-Dataset).

## Repository layout

```
data_generation/              OpenMC corpus generation (10K OpenMC-evaluated lattices)
training/base/cpt/            Stage 1: Continued Pre-Training (full fine-tuning)
training/base/sft/            Stage 2: Supervised Fine-Tuning (full fine-tuning)
training/dpo/single_target/   Stage 3a: DPO with fixed k_eff target
training/dpo/multi_target/    Stage 3b: DPO with LHS-sampled k_eff prompts
training/grpo/single/         Stage 3c: GRPO with fixed k_eff target
training/grpo/multi/          Stage 3d: GRPO with LHS-sampled k_eff prompts
only_sft/                     Ablation: same alignment without CPT pre-training
ga/                           Genetic-algorithm baselines (constrained / unconstrained)
nuscale/                      NuScale standard-design references
eval_prompt_sensitivity/      Prompt-controllability evaluation across all checkpoints
plot/                         Legacy single-run figure scripts
figures/                      Historical figure and supplementary-analysis implementation
paper_figures/                Current main-text Figures 1-4 and prompt-sweep SI builders
oracle/                       Informed (Gd 20-40) and uninformed (Gd 0-264) random-search controls
no_penalty_control/           DPO with the criticality penalty removed (five matched seeds, resumable)
depletion/                    Pin-wise depletion of the selected layouts to 50 MWd/kgHM
generation_yield/             Raw generation validity and constraint satisfaction
analysis/aggregate_results.py Historical aggregation of selected experiment endpoints
analysis/first_passage.py     Inventory first passage, cumulative-best trajectories, layout selection
analysis/summarize_controls.py Seed-level control summaries and exact tests
analysis/reevaluation/        2e7-history OpenMC re-evaluation, bootstrap and CI audits
empirical/                    Buchwald-Hartwig replication on measured reaction yields
run_all_script/               Five-seed reactor campaign launcher
validation/                   Standard-library checks of archived manuscript results
```

## Hardware

Reactor policy training was run with an NVIDIA RTX 3070 (8 GB VRAM), and
OpenMC evaluation with an Intel Core i5-12400F and 32 GB RAM. Additional controls
were run on an Apple-silicon workstation. Device selection and interpreter
configuration should be checked in each training entry point before a new run.
Plan for approximately 50 GB of working space for full reactor campaigns.

Regenerating the current plots and verifying archived numerical results do not
require a GPU, OpenMC, policy training or neutron cross-section data. Figure 1
assembly uses Node.js and `pptxgenjs`. Chemistry reruns use NumPy and PyTorch;
the registered measured input is restored separately as described below.

## Setup

1. **Python environment** (Python 3.10+ recommended).  Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenMC** must be installed separately; see
   <https://docs.openmc.org/en/stable/quickinstall.html>.  Download the
   ENDF/B-VII.1 HDF5 cross-section library and unpack it.  Scripts default
   to:
   ```
   ~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml
   ```
   Override with the standard OpenMC environment variable if your data
   lives elsewhere:
   ```bash
   export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
   ```

3. **Project root** (optional). The campaign launcher accepts an explicit
   repository location:
   ```bash
   export REACTORGEN_ROOT=/path/to/this/repo
   ```

## Reproduce current figures from archived results

Run these commands from the GitHub repository root. Extract
`ScientificFeedback_code_data_v3.zip` separately and set `ARCHIVE_ROOT` to its
extracted top-level directory. The archive's `data/source_tree/` mirrors the
relative input paths used by the current figure builders. Rebuilt outputs go to
`work/`, outside the archived source tree.

```bash
export ARCHIVE_ROOT=/absolute/path/to/ScientificFeedback_code_data_v3
python -m pip install -r paper_figures/requirements.txt
python -m pip install -r empirical/requirements-inputs.txt
python empirical/scripts/fetch_buchwald_hartwig.py \
  --output "$ARCHIVE_ROOT/data/source_tree/empirical/data/buchwald_hartwig.csv"
python paper_figures/build_reactor_figures.py \
  --source-root "$ARCHIVE_ROOT/data/source_tree" --output-dir work/rebuilt_figures
python paper_figures/plot_fig4_chemistry.py \
  --source-root "$ARCHIVE_ROOT/data/source_tree" --output-dir work/rebuilt_figures
python paper_figures/build_supplementary_prompt_sweep.py \
  --source-root "$ARCHIVE_ROOT/data/source_tree" --output-dir work/rebuilt_figures
python paper_figures/build_deduplicated_si_figures.py \
  --source-dir "$ARCHIVE_ROOT/figures/source_data/supplementary" --output-dir work/rebuilt_figures
```

The measured chemistry table is not redistributed. The fetch helper checks both
the upstream workbook and exported CSV against registered SHA-256 values.
It restores that external input without changing the completed result logs.

| Current manuscript figure | Repository entry point | Archived input |
|---|---|---|
| Fig. 1: two scientific settings and attribution overview | `paper_figures/figure1/make_fig1.js` | Packaged assets and chart data beside the builder |
| Fig. 2: reactor redistribution and useful generation | `paper_figures/build_reactor_figures.py` | `data/source_tree/` |
| Fig. 3: reactor objective intervention and search controls | `paper_figures/build_reactor_figures.py` | `data/source_tree/` |
| Fig. 4: chemistry redistribution and yield error | `paper_figures/plot_fig4_chemistry.py` | `data/source_tree/` and restored measured input |
| SI convergence/layouts and first-passage/count panels | `paper_figures/build_deduplicated_si_figures.py` | `figures/source_data/supplementary/` |
| SI initialization ablation | `figures/fig5_cpt.py` | `data/` via `DATA_ROOT` |
| SI prompt sweep | `paper_figures/build_supplementary_prompt_sweep.py` | `data/source_tree/` |
| SI depletion | `figures/fig_depletion.py` | `data/` via `DATA_ROOT` |

Figure 1 can be assembled from its packaged assets without running Blender or
scientific simulations:

```bash
npm install --prefix work/figure1-node pptxgenjs
PPTXGENJS="$PWD/work/figure1-node/node_modules/pptxgenjs" \
  node paper_figures/figure1/make_fig1.js \
  --output work/rebuilt_figures/fig1_concept.pptx
```

Export the PPTX to PDF with PowerPoint or LibreOffice. The archive includes the
final manuscript PDFs in `figures/main/` and `figures/supplementary/` and the
editable Fig. 1 PPTX in `figures/main/`. Fonts, library versions and PDF export
applications can affect rebuilt file bytes.

The two remaining SI plots use the earlier plotting modules:

```bash
DATA_ROOT="$ARCHIVE_ROOT/data" python figures/fig5_cpt.py
DATA_ROOT="$ARCHIVE_ROOT/data" python figures/fig_depletion.py
```

Those entry points write PDFs and PNGs beside their scripts. The other filenames
in `figures/` retain historical numbering; the table above gives the current
manuscript mapping.

## Verify the archived numerical results

The verifier requires only Python's standard library. Run it from the repository
root, with the same `ARCHIVE_ROOT` as above:

```bash
mkdir -p work
python validation/verify_manuscript_results.py \
  --package-root "$ARCHIVE_ROOT" --output "$PWD/work/manuscript_validation.json"
```

It checks the fixed-convention corpus, all 40 reactor alignment traces,
unselected-generation and random-search target incidence, all five
objective-removal runs, independent high-statistics evaluations, and both
chemistry schedules. It aggregates archived exact-enumeration metrics and checks
checkpoint file inclusion; it does not deserialize weights or independently
re-enumerate the neural policies. A failed assertion or missing required input
returns a nonzero exit status.

## Re-running original campaigns

Full reactor reproduction requires a training environment, OpenMC and the
ENDF/B-VII.1 library. Generate the corpus with
`data_generation/data_generation_10K.py`, or use the archived corpus. Run base
training before alignment for each initialization and seed; run search controls
and supplementary evaluations separately from the training campaigns.

Representative entry points are:

```bash
python training/base/cpt/cpt.py --seed 0
python training/base/sft/sft.py --seed 0
python training/dpo/single_target/single_dpo.py --seed 0
python training/grpo/single/single.py --seed 0
```

`only_sft/` contains the matched initialization ablation. The eight alignment
variants span DPO/GRPO, single/multi-target feedback and CPT+SFT/SFT-only
initialization, each with five seeds. Each alignment run uses 2,000 scientific
evaluations. Corpus generation, controls, prompt evaluation and depletion have
separate entry points.

`run_all_script/run_all_script.py` is the legacy campaign launcher for base
training, alignment and two genetic-algorithm baselines. It retains a local
OpenMC interpreter path and submits stages to a shared worker queue without
waiting for prerequisite stages to finish. For a fresh campaign, use the
individual entry points in stage order as shown above; the legacy launcher
requires adapting its environment and scheduling first. Its `.done` markers
skip completed tasks. It does not generate the corpus or run supplementary
controls and evaluations.

For measured chemistry, install `empirical/requirements.txt`, restore the
registered table, then run both schedules with an explicit input path:

```bash
python -m pip install -r empirical/requirements.txt
python empirical/scripts/fetch_buchwald_hartwig.py
python empirical/scripts/run_full_pipeline.py \
  --data "$PWD/empirical/data/buchwald_hartwig.csv" --resume
```

The pipeline executes `full_single.json` and `full.json`, each covering two
initialization arms and five seeds. See `empirical/README.md` for the categorical
policy, registered budgets and exact-enumeration outputs.

## Outputs

Each training script writes a per-seed CSV trace
(`{stage}_seed{N}_results.csv`) recording, at every step, the candidate
designs proposed, their OpenMC-evaluated $k_\text{eff}$ / $F_q$ /
$F_{\Delta H}$, the running cumulative-best individual, and the loss.
Final model checkpoints are written under `final_*` directories and
loaded by the downstream alignment stages.

## Controls and yield analysis

The reactor attribution controls compare availability, redistribution and
reusable search information under matched evaluation budgets. Additional
validation assesses objective direction, transport precision and depletion.

- `oracle/` -- **Informed random-search oracle.** Uniform random sampling with
  the Gd inventory drawn from the interval [20, 40], a bracket informed by
  alignment, evaluated under identical OpenMC settings and budget.
  Five independent sampling seeds
  (`python oracle/run_informed_random_search.py`); best-of-budget composite
  fitness across seeds gives the oracle value in the manuscript.
- `oracle/run_uninformed_random_search.py` -- **Uninformed full-space random
  search.** Gd inventory uniform on the integers 0-264 and positions uniform
  over the 264 non-guide-tube sites, five seeds x 2,000 evaluations, no
  productive window supplied. Deterministic per-sample OpenMC seeds; resumable.
- `no_penalty_control/` -- **No-penalty control run, five matched seeds.** The
  single-target online DPO procedure with the criticality penalty removed, so
  the reward reduces to the peaking-only term; everything else is identical to
  `training/dpo/single_target/single_dpo.py`. `no_penalty_dpo.py --seed K` is
  the original single-process script; `run_no_penalty_resumable.py --seed K`
  runs the same algorithm with atomic checkpoints (model, optimizer and all
  RNG states every 10 steps), strict OpenMC failure handling and two isolated
  evaluator processes (`oracle_worker.py`), and was used for seeds 1-4.
- `depletion/` -- **Depletion assessment of selected layouts.** Pin-wise
  CE/CM depletion of the best DPO, GRPO and unconstrained-GA layouts and the
  symmetric 16-Gd / 24-Gd references to 50 MWd/kgHM at 35 W/gHM with the
  228-nuclide simplified CASL PWR chain, followed by an endpoint transport with
  the remaining Gd removed (residual Gd effect). `python depletion/fetch_chain.py`
  downloads and checksums the chain; `python depletion/run_depletion.py --case DPO`
  runs one layout (resumable at the last completed burnup node). Layouts come
  from `selected_layouts.json` written by `analysis/first_passage.py`.
- `analysis/first_passage.py` -- first candidate with >= 20 Gd rods per run
  (batch-bounded), cumulative-best inventory monotonicity, and the
  best-of-five-seeds layout selection used by the depletion study.
- `analysis/summarize_controls.py` -- seed-level statistics for the five
  no-penalty runs (paired exact sign-flip test against the matched
  full-objective runs), the uninformed search (exact rank-sum tests against
  DPO / GRPO, target-region candidate counts) and the depletion outputs.
- `generation_yield/` -- **Generation yield and constraint satisfaction.**
  Samples 1,000 generations per aligned checkpoint, measures complete-lattice
  fraction and pre-correction guide-tube violations, and evaluates a fixed
  subsample with OpenMC for the feasible fraction
  (`python generation_yield/run_generation_yield.py`).
- `analysis/aggregate_results.py` -- preserves earlier endpoint aggregation
  (fitness decomposition, matched-GA comparison and Mann-Whitney tests,
  informed-search statistics and control convergence). Use
  `validation/verify_manuscript_results.py` for the current manuscript checks.

## Re-evaluation, empirical replication, and figures

- `analysis/reevaluation/` -- re-evaluates the ten selected layouts with
  2 x 10^7 active histories per layout, recomputes every trajectory confidence
  band, and recomputes the prompt-steerability slopes with 10,000 bootstrap
  resamples (`python analysis/reevaluation/openmc_high_stat.py --selection all-per-seed`,
  `python analysis/reevaluation/trajectory_ci.py`,
  `python analysis/reevaluation/steerability_bootstrap.py`).
- `empirical/` -- repeats the CPT/SFT/DPO/GRPO protocol with a small
  categorical policy on the measured Buchwald-Hartwig reaction-yield table of
  Ahneman et al. (Science 2018); the reward is a lookup of measured yields.
  The data table is restored separately; `empirical/data/SOURCE.md` records
  its origin and checksum. Preparation and rerun commands are given above.
- `paper_figures/` -- reproduces current Figures 1--4 and the supplementary
  prompt sweep from completed records.
- `figures/` -- preserves historical artwork and supplementary analyses;
  filenames retain their earlier numbering and are not the current main-text
  figure mapping. See `figures/README.md`.

## Companion archive data layout

```
ScientificFeedback_code_data_v3/data/
  corpus/               reactor_10k_final.csv
  alignment/            DPO / GRPO single and multi-target trajectories, 5 seeds
  alignment_sft_only/   same, SFT-only initialization
  baselines/            ga_baseline_single[_gd16]_seed{K}_results.csv, nuscale_baseline_results.csv
  oracle/               informed_random_search_seed{K}_results.csv, uninformed_random_seed{K}_results.csv
  no_penalty_control/   no_penalty_dpo_seed{K}_results.csv (K = 0..4)
  depletion/            {DPO,GRPO,GA,REF16,REF24}/trajectory.csv + complete.json, selected_layouts.json
  analysis/             first_passage_per_seed.csv and the control summaries (regenerable)
  prompt_sensitivity/   prompt_sensitivity_results.csv
  reevaluation/         2e7-history re-evaluation, trajectory CI, steerability bootstrap
  empirical/            Buchwald-Hartwig outputs and final policy checkpoints
  generation_yield/     Corrected unselected samples and OpenMC subsample
  source_tree/          Input mirror used by current paper_figures/ builders
```

## Citation

Lee, Y. P., Roy, S., Chakraborty, S. & Alam, S. B. *Scientific feedback reveals
productive regimes beyond training conventions* (revised manuscript submitted
to Nature Machine Intelligence, 2026).

Development repository: https://github.com/sixticket/reactor_design_optimization.
Software citation metadata is provided in [`CITATION.cff`](CITATION.cff).
The companion archive has separate citation metadata. No archive DOI is claimed
until its Zenodo record is published.


## Contact

For questions about reproduction, extensions, or bug reports, please open
a GitHub issue or contact Yoonpyo Lee at <yoonpyo2@illinois.edu>.

## License

The existing distribution statement is research and academic use. Please cite
the accompanying manuscript and software. This repository does not currently
provide a formal software license; the companion archive's data license does
not assign a software license to this code.
