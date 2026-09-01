# ReactorGen

**Toward an agentic foundation model for nuclear reactor core design via
emergent physical AI.**

ReactorGen couples a compact pretrained language model (Gemma 3, 270M
parameters) with the OpenMC Monte Carlo neutron-transport simulator in a
closed perception-reasoning-action loop, learning to generate
17 x 17 PWR fuel-assembly layouts that satisfy multi-objective safety
constraints ($k_\text{eff}$, $F_q$, $F_{\Delta H}$).  During preference
alignment with either DPO or GRPO, the model autonomously expands the
gadolinium-absorber inventory beyond the training distribution---a
physics-consistent out-of-distribution behaviour we term *emergent
constraint relaxation*.

This repository contains the full training, evaluation, and baseline
pipeline used to produce the results reported in our manuscript.  All
scripts run on a single consumer-grade NVIDIA GPU (e.g.\ RTX 3070, 8 GB
VRAM) with CPU fallback.

## Repository layout

```
data_generation/              OpenMC dataset generation (100K low-fi + 10K hi-fi)
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
plot/                         Figure-generation scripts
run_all_script/               Master orchestrator (5-seed reproducibility sweep)
```

## Hardware

- **GPU**: NVIDIA RTX 3070 (8 GB VRAM) or comparable.  Larger GPUs reduce
  the need for gradient accumulation but the codebase runs as-is on 8 GB.
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

1. Dataset generation -- 100K low-fidelity + 10K high-fidelity samples
   (~80 hours on the reference hardware; one-time cost).
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
- `no_penalty_control/` -- **No-penalty control run.** The single-target
  online DPO procedure with the criticality penalty removed, so the reward
  reduces to the peaking-only term; everything else is identical to
  `training/dpo/single_target/single_dpo.py`
  (`python no_penalty_control/no_penalty_dpo.py --seed 0`).
- `generation_yield/` -- **Generation yield and constraint satisfaction.**
  Samples 1,000 generations per aligned checkpoint, measures complete-lattice
  fraction and pre-correction guide-tube violations, and evaluates a fixed
  subsample with OpenMC for the feasible fraction
  (`python generation_yield/run_generation_yield.py`).
- `analysis/aggregate_results.py` -- recomputes the headline numbers reported
  in the manuscript (fitness decomposition, matched-GA comparison and
  Mann-Whitney tests, oracle statistics, control convergence) from the raw
  CSVs in the archival data package.

## Citation

A manuscript describing this work is in preparation.  Citation details
(preprint and journal references) will be added here once available.

## Contact

For questions about reproduction, extensions, or bug reports, please open
a GitHub issue or contact Yoonpyo Lee at <yoonpyo2@illinois.edu>.

## License

Released for research and academic use.  Please cite the accompanying
manuscript when citation information becomes available.
