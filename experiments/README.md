# Experiments for the ε-MOBO manuscript

This folder has two purposes: (1) run a **new** nine-problem benchmark grid using the public method implementation, and (2) inspect selected **saved** manuscript summaries when their original result files are supplied locally. It also includes the modified car cab objective evaluator. The small KNO1 demonstration in the repository root uses different settings. This folder is **not a complete release of the historical experiment**, and a fresh rerun must not be presented as reproduction of the saved results.

To run the fresh grid, follow [benchmark/FRESH_EXPERIMENT.md](benchmark/FRESH_EXPERIMENT.md). It covers 118 method configurations, q=1,2,4, 20 paired repetitions, 150 evaluations, and archive-specific HV post-processing. This is computationally expensive; the default command enumerates 63,720 tasks, and the full grid has not been executed in this repository.

## Files

| Path | Purpose |
| --- | --- |
| `car_cab/re91_problem.py` | Preserved application evaluator, reproduced in Supplementary Section S3. It contains the actual `0.00087570001` coefficient in the third objective, stochastic latent inputs and nominal assessment. |
| `car_cab/recompute_table5.py` | Recomputes the eight-method Table 5 average ranks, means and Holm-adjusted W/T/L if the matching per-run final-HV CSV is supplied locally. |
| `benchmark/recompute_matched_family.py` | Recomputes the criterion-matched W/T/L rows of manuscript Table 1 if the matching benchmark family final-HV CSV is supplied locally. |
| `benchmark/test_functions.py` | Archived definitions, domains and dimensions of all nine benchmark functions. |
| `benchmark/run_full_experiment.py` | Runs a new 118-method, nine-problem, q=1,2,4 study with 20 paired repetitions using the public method implementation. |
| `benchmark/summarize_full_experiment.py` | Computes new-run reporting references and final HV from a complete raw-observation archive. |
| `benchmark/FRESH_EXPERIMENT.md` | Commands, protocol, and the differences from reproducing the saved manuscript study. |

**The run-level CSV inputs are not published here yet.** Do not cite this folder as a complete replication package or imply that the demo produces the numerical tables and figures of the manuscript.

## Checking saved manuscript summaries

The following two scripts require their matching saved result files; the fresh runner above does not create those historical inputs.

### Data inputs and commands

The scripts expect the following local files beside them:

- `car_cab/final_hv_runs.csv.gz`: columns `q`, `method_id`, `rep`, `n_points`, `final_hv`; 3,360 saved application rows (56 method configurations × 3 batch sizes × 20 repetitions). Each method/q has repetition indices 0–19 and 326 evaluated designs.
- `benchmark/family_final_hv_20reps.csv.gz`: columns `problem`, `q`, `family`, `structure`, `method`, `epsilon`, `is_baseline`, `rep`, `final_hv`; 58,320 run-level final-HV values from the 108-method family archive across nine problems and three batch sizes.

With the matching saved data present, use Python with NumPy, pandas and SciPy:

```bash
python experiments/benchmark/recompute_matched_family.py
python experiments/car_cab/recompute_table5.py
```

The manuscript records Python 3.12.7, NumPy 1.26.4, SciPy 1.11.3 and pandas 2.3.3 for its analyses. The benchmark script applies Holm correction within each problem/family/q panel of 42 (qPOTS or UCB) or 21 (TSEMO) tests. The application script ranks eight methods within each of 20 paired repetitions at each q, then applies Holm correction over the 28 pairwise tests within that q. Both scripts check the expected record counts and pairings.

## Remaining materials for a full release

The verified **historical** benchmark experiment runner and its mapping to the saved results, full benchmark objective observations and both sets of numerical reporting reference points, complete original configurations and seed/manifest records, representative trajectories, application raw observations, all original final run-level metrics and every analysis/figure script still need to be packaged and released with a stable identifier. The fresh runner above does not close that provenance gap. An earlier runner that uses top-q HVC must not be represented as the final sequential-HVC experiment runner.

Reporting-HV references must be preserved as used for the published results. Substituting convenient integer references would require recomputing HV and all dependent comparisons; it is not a display-only change.
