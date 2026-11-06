# Experiments for ε-MOBO

This folder contains the nine benchmark objective functions, code for running the 118-method benchmark grid, a script for computing reporting references and final hypervolumes from the **new run's raw observations**, and the car cab objective evaluator. The repository-root KNO1 demonstration uses a smaller budget.

## Benchmark experiment

Follow [benchmark/FRESH_EXPERIMENT.md](benchmark/FRESH_EXPERIMENT.md) to inspect the method grid, run individual jobs, and process the completed archive. The full configuration contains nine problems, 118 methods, q = 1, 2, 4, 20 paired repetitions, and 150 evaluations per run: 63,720 method/problem/q/repetition jobs. Completing the grid requires substantial computational resources.

| Path | Purpose |
| --- | --- |
| `benchmark/test_functions.py` | Objective definitions, dimensions and bounds for all nine benchmark problems. |
| `benchmark/verify_test_functions.py` | Checks the function definitions. |
| `benchmark/run_full_experiment.py` | Enumerates and runs the full benchmark grid; saves designs and minimization objectives with a configuration manifest. |
| `benchmark/summarize_full_experiment.py` | Derives the reporting references and final hypervolumes from the completed outputs of `run_full_experiment.py`. |
| `car_cab/re91_problem.py` | Car cab objective evaluator with the modified third-objective coefficient and nominal assessment. |

The post-processing command reads the raw observations written by this benchmark runner. The scripts in this folder do not require any separately supplied final-HV CSV files. The full grid has not been executed in this repository; running the code generates a new result set rather than the numerical results already reported in the manuscript.
