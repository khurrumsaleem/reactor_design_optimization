# Archived manuscript result checks

From the repository root, point the verifier at the separately extracted
`ScientificFeedback_code_data_v3` archive. Python 3 and the standard library
are sufficient:

```sh
export ARCHIVE_ROOT=/path/to/ScientificFeedback_code_data_v3
python3 validation/verify_manuscript_results.py \
  --package-root "$ARCHIVE_ROOT" --output work/manuscript_validation.json
```

To use the code copy inside the archive, change into `$ARCHIVE_ROOT/code` and
omit `--output` to print the report, or explicitly choose an output outside
the archive, such as `--output "$ARCHIVE_ROOT/../manuscript_validation.json"`.
`--package-root` refers to the directory containing `data/` and `figures/`,
rather than the GitHub checkout.

The output report must be outside the archive directory. Without `--output`,
the complete JSON report is printed to standard output. A successful run exits
with status 0; any failed assertion or missing required source record exits
with status 1. Running the verifier never modifies archived records.

The checks cover:

- The 10,000-row corpus, fixed 16-absorber convention, 25 fixed guide positions,
  289-cell representation, and absence of target-window corpus layouts.
- All 40 lattice alignment logs, their five-seed grouping and 2,000-evaluation
  budgets, selected inventory ranges, and best objectives.
- The 10,000 corrected unselected generations and 500 transport evaluations,
  including 101 target-window hits (33 DPO and 68 GRPO).
- Five informed and five inventory-uniform random searches, including 521 and
  36 target-window hits, and conditional counts within inventories 28–35.
- All five objective-removal runs, including the final-100-step inventory
  vector `[0, 0, 0, 0, 39.29]` and the retained exceptional seed.
- The ten independent high-statistics evaluations, their 20-million-history
  budgets, target attainment, and objective decomposition.
- Both measured-reaction schedules: 480 and 4,800 exact-enumeration metric
  records, 40,000 trajectory records per schedule, five-seed changes, aggregate
  error and off-corpus ligand mass, and 30 final checkpoint files per schedule.
- Archived figure incidence and objective-removal endpoints, when present,
  crosschecked against the underlying numerical logs.

The lattice target window is `abs(k_eff - 1.05) <= 500 / 100000`. Chemistry error
is expressed in yield percentage points. Dispersion checks use the sample
standard deviation. Explicit tolerances match manuscript-rounded numbers;
expected and observed values are included in the report.

This verifies archived numerical evidence. It does not train a model, execute
OpenMC, or deserialize model weights. Chemistry checks independently aggregate
the archived exact-enumeration metrics; they do not independently enumerate the
neural policies from checkpoints. Artifact checksums and external publication
state are separate from these numerical checks.
