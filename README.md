# ReactorGen

**Agentic physics-adjudicated constraint discovery in nuclear reactor core
design.**

ReactorGen couples a compact pretrained language model (Gemma 3, 270M
parameters) with the OpenMC Monte Carlo neutron-transport simulator in a
closed generate-evaluate-update loop, learning to generate
17 x 17 PWR fuel-assembly layouts that satisfy multi-objective neutronic
targets ($k_\text{eff}$, $F_q$, $F_{\Delta H}$).  Trained exclusively on
16-rod lattices, and aligned with either DPO or GRPO against live OpenMC
feedback with no novelty, diversity, or inventory term in the reward, the
policy shifts the gadolinium-absorber inventory to 28--35 rods, a region
absent from its training corpus---a measurable, reward-contingent event we
term *physics-adjudicated constraint discovery*.

This repository contains the full training, evaluation, and baseline
pipeline used to produce the results reported in our manuscript.  All
scripts run on a single consumer-grade NVIDIA GPU (e.g.\ RTX 3070, 8 GB
VRAM) with CPU fallback.

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
figures/                      Manuscript figures (Figs. 2-7, Extended Data) from the archival data
oracle/                       Informed (Gd 20-40) and uninformed (Gd 0-264) random-search controls
no_penalty_control/           DPO with the criticality penalty removed (five matched seeds, resumable)
depletion/                    Pin-wise depletion of the selected layouts to 50 MWd/kgHM
generation_yield/             Raw generation validity and constraint satisfaction
analysis/aggregate_results.py Recomputes every headline number from the archived CSVs
analysis/first_passage.py     Inventory first passage, cumulative-best trajectories, layout selection
analysis/summarize_controls.py Seed-level control summaries and exact tests
analysis/reevaluation/        2e7-history OpenMC re-evaluation, bootstrap and CI audits
empirical/                    Buchwald-Hartwig replication on measured reaction yields
run_all_script/               Master orchestrator (5-seed reproducibility sweep)
```

## Hardware

- **GPU**: NVIDIA RTX 3070 (8 GB VRAM) or comparable.  Larger GPUs reduce
  the need for gradient accumulation but the codebase runs as-is on 8 GB.
  Every script selects `cuda` when available and falls back to CPU.  (The
  supplemental controls reported in the manuscript -- no-penalty seeds 1-4,
  uninformed random search, generation yield and depletion -- were run on an
  Apple-silicon workstation with the same code paths; only the device string
  differs.)
- **CPU / RAM**: any modern x86-64 desktop is sufficient for the OpenMC
  simulations that dominate wall-clock time during DPO / GRPO and dataset
  generation.  Tested on Intel Core i5-12400F (6 cores) with 32 GB RAM.
- **Disk**: ~50 GB free for raw OpenMC outputs, model checkpoints, and
  result CSVs across all five seeds.

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

3. **Project root** (optional).  All scripts auto-detect the project root
   from their own location.  To pin it explicitly:
   ```bash
   export REACTORGEN_ROOT=/path/to/this/repo
   ```

## Reproducing the manuscript

The full five-seed sweep is orchestrated by a single command:

```bash
python run_all_script/run_all_script.py
```

This sequentially executes (with two-task parallelism by default):

1. Corpus generation -- 10,000 OpenMC-evaluated lattices
   (~70 hours on the reference hardware; one-time cost).
2. CPT and SFT stages on each of five random seeds.
3. DPO single, DPO multi, GRPO single, GRPO multi alignment for both
   CPT+SFT and SFT-only base checkpoints (twenty 1000-step runs total).
4. Genetic-algorithm baselines (with and without the 16-Gd inventory
   constraint).
5. Prompt-sensitivity evaluation across all ten trained checkpoints.

End-to-end wall time on the reference hardware is approximately seven
days.  Each task is resumable: the runner skips any seed/method
combination whose `.done` marker already exists.

To reproduce a single component manually, see the script-specific entry
points in each subdirectory, e.g.:

```bash
python training/base/cpt/cpt.py --seed 0
python training/base/sft/sft.py --seed 0
python training/dpo/single_target/single_dpo.py --seed 0
python training/grpo/single/single.py --seed 0
```

## Outputs

Each training script writes a per-seed CSV trace
(`{stage}_seed{N}_results.csv`) recording, at every step, the candidate
designs proposed, their OpenMC-evaluated $k_\text{eff}$ / $F_q$ /
$F_{\Delta H}$, the running cumulative-best individual, and the loss.
Final model checkpoints are written under `final_*` directories and
loaded by the downstream alignment stages.

## Controls and yield analysis

Three additional experiment groups support the revised manuscript. All of them
reuse the OpenMC settings, fitness definition, and 2,000-evaluation budget of
the main campaigns.

- `oracle/` -- **Informed random-search oracle.** Uniform random sampling with
  the Gd inventory drawn from the interval [20, 40], the region occupied by
  the aligned policies, evaluated under identical OpenMC settings and budget.
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
- `analysis/aggregate_results.py` -- recomputes the headline numbers reported
  in the manuscript (fitness decomposition, matched-GA comparison and
  Mann-Whitney tests, oracle statistics, control convergence) from the raw
  CSVs in the archival data package.

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
  The data table is not redistributed; `empirical/data/SOURCE.md` records its
  origin and checksum (`cd empirical && python scripts/run_full_pipeline.py`).
- `figures/` -- regenerates all manuscript figures from the archival data
  package (`DATA_ROOT=/path/to/data python figures/run_all.py`); see
  `figures/README.md` for the script-to-figure mapping.

## Archival data layout expected by the analysis and figure scripts

```
data/
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
  empirical/            Buchwald-Hartwig replication outputs
```

## Citation

Lee, Y. P., Roy, S., Chakraborty, S. & Alam, S. B. ReactorGen: Agentic
Physics-Adjudicated Constraint Discovery (manuscript submitted to Nature
Machine Intelligence, 2026). Data and code archive:
https://doi.org/10.5281/zenodo.22230862.

## Contact

For questions about reproduction, extensions, or bug reports, please open
a GitHub issue or contact Yoonpyo Lee at <yoonpyo2@illinois.edu>.

## License

Released for research and academic use.  Please cite the accompanying
manuscript when citation information becomes available.
