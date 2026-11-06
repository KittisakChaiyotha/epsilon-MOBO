# Saved-result audit for the ε-MOBO manuscript

This folder makes **part of the saved numerical evidence** independently checkable. It is separate from the small KNO1 demonstration in the repository root. Neither program here reruns Bayesian optimization or regenerates every figure in the manuscript.

## Contents

| Path | Scope |
| --- | --- |
| `benchmark/family_final_hv_20reps.csv.gz` | 58,320 run-level final-HV values: 108 methods × 9 problems × 3 batch sizes × 20 paired repetitions. These are the saved **family archive** values under its reporting reference points. |
| `benchmark/recompute_matched_family.py` | Recomputes the criterion-matched W/T/L rows of manuscript Table 1 with the original problem–family–batch Holm panels. |
| `car_cab/re91_problem.py` | Preserved modified application evaluator (also reproduced in Supplementary Section S3), including the coefficient `0.00087570001` in the third objective and the stochastic and nominal evaluation methods. |
| `car_cab/final_hv_runs.csv.gz` | Final nominal HV for all 3,360 saved application runs: 56 method configurations × 3 batch sizes × 20 paired repetitions. The exported columns are `q`, `method_id`, `rep`, `n_points` and `final_hv`. |
| `car_cab/recompute_table5.py` | Recomputes the eight-method Table 5 average ranks, means and Holm-adjusted W/T/L at each batch size from the saved final HV. |

The application observations were checked in the archived post-processing against the modified evaluator: stochastic outcomes and nominal outcomes were reproduced from the saved designs and latent draws. The CSV here retains final HV rather than all of those raw observations. The archive does **not** establish whether the same run-level observations were used in the separate family and representative benchmark archives.

## Check the manuscript summaries

Use Python with NumPy, pandas and SciPy. The manuscript records Python 3.12.7, NumPy 1.26.4, SciPy 1.11.3 and pandas 2.3.3 for the analyses. From the repository root, run:

```bash
python reproducibility/benchmark/recompute_matched_family.py
python reproducibility/car_cab/recompute_table5.py
```

The first command prints 15 criterion-matched Table 1 rows. The second prints eight methods at each of q = 1, 2 and 4, with average rank computed from 20 paired repetition blocks (lower is better), `1000_mean_final_hv`, W/T/L and Net. The original Table 5 W/T/L panel uses all 28 pairs within each q and a two-sided Wilcoxon signed-rank test with Holm correction. Each script checks the record counts and pairing before computing its summaries.

**Reporting reference points are not selection references.** The benchmark family values above use their original reporting reference points. The representative benchmark archive uses a separately pooled reference, and the full numerical family reference vectors are not recovered from this final-HV-only CSV. Replacing either reference with convenient integer values would change HV and require recalculation of downstream results.

## Remaining release materials

The full experimental reproducibility release still requires the final benchmark runner (including its verified sequential-HVC implementation and mapping to saved results), all benchmark objective observations and both sets of numerical reporting references, complete configurations and seed/manifest records, the representative trajectories, application raw observations and scripts for every published table and figure. Earlier runner copies must not be presented as the final runner without verification. The manuscript's code-availability statement describes this full release as forthcoming.
