# Methods and interpretation

## Dataset and action space

We use the Buchwald–Hartwig high-throughput reaction-yield table associated with
Ahneman et al., *Science* (2018), DOI `10.1126/science.aar5169`. The canonical
`Plates1-3` worksheet contains 3,955 unique measured combinations of four
ligands, 22 additives, three bases, and 15 aryl halides. The nominal factorial
grid has 3,960 combinations; unmeasured cells are excluded rather than filled.

For each aryl-halide task, the empirical action space is the set of measured
`(ligand, additive, base)` tuples. Autoregressive masks are constructed directly
from that set. Consequently, a sampled trajectory always identifies one and only
one experimental row.

## Restricted support

Let `a(t)` be the anchor ligand for aryl-halide task `t`, defined as the ligand
whose sorted integer ID equals `t mod 4`. CPT and SFT use only rows satisfying
`ligand = a(t)`. This supplies approximately 66 measured reactions per task and
uses every ligand token across tasks. Online DPO/GRPO use the complete measured
catalog. A reaction is outside training support when `ligand != a(t)`.

The anchor rule is outcome-independent: it never examines yield. This prevents
the support split from being chosen to manufacture a favorable result.

## Policy and training stages

The policy is a small categorical autoregressive network. It conditions on the
aryl-halide task and either an unconditional CPT token or a requested yield.
It then generates ligand, additive, and base. Each next-token softmax is masked
by the empirical prefix trie.

- **CPT:** maximum likelihood over all restricted-support reactions using the
  unconditional target token.
- **SFT:** maximum likelihood of the `k` restricted-support reactions closest to
  each requested yield, per task.
- **CPT+SFT arm:** CPT followed by SFT.
- **SFT-only arm:** SFT from the same random initialization scheme.
- **Online DPO:** two samples are ranked by measured reward and optimized with
  `-log sigmoid(beta * (log p(chosen) - log p(rejected)))`.
- **GRPO:** a sample group is assigned standardized within-group measured
  rewards, followed by a policy-gradient update.

Registered training budgets mirror ReactorGen: CPT 5 epochs, SFT 10 epochs,
DPO 1,000 steps with `beta=0.01`, and GRPO 500 steps with four samples per
group. Single-target and 10-bin balanced multi-target experiments are run for
five seeds in both CPT+SFT and SFT-only initialization arms. Learning rates are
scaled for the much smaller categorical policy and are therefore not copied
numerically from the 270M-parameter language model.

The source multi-target run uses continuous Latin-hypercube samples over the
`k` interval. The empirical analogue uses a balanced random permutation of ten
yield bins per cycle. This preserves equal target-bin coverage while keeping
the target representation identifiable in a small categorical model.

## Interruption and exact continuation

All checkpoints are written through a temporary file followed by an atomic
rename. Supervised checkpoints store the epoch, model, optimizer, and global
random-number states. Online checkpoints additionally store the next step,
Python condition sampler, PyTorch categorical generator, and accumulated
trajectory. Re-running with `--resume` therefore continues the same stochastic
trajectory rather than merely warm-starting a new run.

For target matching, reward is `-|measured yield - target yield| / 100`. The
code also supports yield maximization as `measured yield / 100`.

## Evaluation

The catalog is small enough to enumerate. Metrics therefore use exact policy
probabilities rather than Monte Carlo estimates:

- expected reward and absolute target error;
- probability mass outside restricted support;
- ligand entropy;
- best full-catalog and restricted-catalog attainable errors;
- DPO–GRPO Jensen–Shannon divergence for the same seed and initialization arm.

Algorithm–ligand NMI is reported only as a secondary coarse summary. It can be
near zero even when policies differ within a ligand, and it can be high when
ligand marginals differ without better target matching. JSD and support mass are
the more direct differentiation endpoints.

## Claims this experiment can and cannot support

A positive result supports the claim that ReactorGen-style policy
differentiation and support expansion are not unique to a transport simulator:
they also occur under feedback from a table of physical experiments.

It does not establish online laboratory autonomy, extrapolation to unmeasured
chemistry, or discovery of a previously unperformed reaction. The policy is
confined to measured rows, so causal conclusions are limited to this empirical
design space.
