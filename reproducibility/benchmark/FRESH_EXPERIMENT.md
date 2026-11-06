# Nine-problem benchmark rerun

This folder now contains all nine deterministic minimization test functions:
KNO1, VLMOP2, VLMOP3, OKA1, OKA2, DTLZ1a, DTLZ2a, DTLZ4a and DTLZ7a.
`test_functions.py` is copied byte for byte from an archived benchmark
definition supplied with the earlier q=1 study. It has the problem-specific
bounds, dimensions, and complete objective formulas. Its KNO1 formula uses
`sqrt(x1+x2)` in the first sine term, whereas the root KNO1 *demo* currently
uses `(x1+x2)**2`. Keep their results separate until that discrepancy is
resolved against the final experimental source and saved observations.

`run_full_experiment.py` is a **new runner for a fresh study** built on the
repository's public `methods.py`. It is not a recovered snapshot of the final
runner that produced the manuscript's saved results. It uses 118 method
configurations, nine problems, q=1,2,4, 20 paired repetitions, 150 total
evaluations including 11d-1 Latin-hypercube initial designs, and NSGA-II
population/generations 100/100. The method module implements sequential HVC.
The full grid comprises 63,720 independent method/problem/q/repetition runs.
The runner stores complete normalized designs and original-scale minimization
objectives, with a configuration manifest, seed, method label, versions and code
hashes. It fails instead of quietly substituting a random acquisition step.

Install the repository requirements and inspect the grid:

```bash
python reproducibility/benchmark/run_full_experiment.py --list-tasks
python reproducibility/benchmark/run_full_experiment.py --list-methods
```

A small **new** run (not a manuscript check):

```bash
python reproducibility/benchmark/run_full_experiment.py \
  --problem kno1 --q 1 --method-key qpots --rep 0 \
  --total-evals 23 --population 12 --generations 2 \
  --output-root scratch_check
```

For the full settings, run an indexed job for each `--task-index` from 0
through 63719 with the *same* output root (for example, as a scheduler array).
The program skips compatible completed files and checks the shared manifest.
Do not mix smoke-run output with full-run output. On the complete archive,
derive the problem-wise reporting reference and final HV:

```bash
python reproducibility/benchmark/summarize_full_experiment.py \
  --output-root full_results --archive all118
python reproducibility/benchmark/summarize_full_experiment.py \
  --output-root full_results --archive family108
```

The first analysis uses all 118 configurations; the second selects the
108-method qPOTS/UCB/TSEMO family subset. Each reference is computed from
pooled nondominated observations across q and repetitions using the formula
in the manuscript; no convenient integer reference is substituted. Outputs
are **new-run** HV, never the archived manuscript values.

## Remaining provenance gap

Neither this fresh runner nor the older `recompute_matched_family.py` script
establishes that the historical q=1,2,4 computations used precisely this
runner. The verified final sequential-HVC runner snapshot, exact configurations
and seed mapping, complete/failed-job manifest, raw objective observations,
reference points for both archives, run-level final-HV data, and figure scripts
are still needed to reproduce the reported numerical study. The rerun can
produce new observations, but it cannot retroactively verify the saved results.
