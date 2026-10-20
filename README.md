# ε-MOBO — Epsilon-greedy acquisition strategies for multi-objective Bayesian optimization

Supplementary code repository for peer-review reproducibility verification.

This repository accompanies the manuscript *ε-MOBO: Epsilon-greedy acquisition
strategies for multi-objective Bayesian optimization*. It provides an
executable demonstration of the proposed ε-greedy methods and their fixed
comparators on the deterministic KNO1 benchmark. All **22 method
configurations** are evaluated from paired initial designs. The reduced
evaluation budget makes the implementation accessible for independent
inspection; the demonstration is distinct from the manuscript's full
experimental study.

---

## Folder Structure

~~~text
epsilon-MOBO/
├── benchmarks.py                 # KNO1 objective functions and bounds
├── methods.py                    # GP models, acquisition and batch-selection methods
├── run_demo.py                   # experiment runner, HV summary and timing
├── requirements.txt              # Python dependencies
├── tests/test_demo.py            # numerical and reporting checks
├── .gitignore                    # excludes locally generated results
└── README.md
~~~

---

## Requirements

Python **3.11** is recommended. Install the packages from this folder:

~~~bash
python -m pip install -r requirements.txt
~~~

| Package | Purpose |
|---|---|
| NumPy, SciPy | Arrays, Latin hypercube sampling and numerical routines |
| PyTorch, GPyTorch, BoTorch | Gaussian process models, posterior sampling and ParEGO |
| pymoo | NSGA-II approximation of the surrogate Pareto set |

Install a PyTorch build suitable for your computer if the standard installation
does not work. No benchmark data download is required.

---

## Quick Start

**Step 1** — Extract the ZIP and open a terminal in the epsilon-MOBO folder.

**Step 2** — Install the requirements using the command above.

**Step 3** — Run the KNO1 demonstration:

~~~bash
python run_demo.py
~~~

The script finds its own folder and writes results there. To use a separate
results folder, run for example:

~~~bash
python run_demo.py --output-root kno1_demo
~~~

For a shorter check with only ε-UCB-HVC-PF and its fixed UCB comparator:

~~~bash
python run_demo.py --methods eps-ucb ucb --criteria HVC --modes PFexp --repetitions 1 --q 1 --total-evals 22 --output-root quick_check
~~~

Run the same command again to resume missing method runs in that output folder.
Use a new output folder when changing settings; --overwrite replaces results
already stored in the chosen folder. Run python run_demo.py --help for all
options.

---

## What the Demo Does

| Problem | Dimension | Domain | Objectives |
|---|---:|---|---|
| KNO1 | 2 | [0, 3]² | Two deterministic objectives, both minimized |

**Default settings** (small enough for a local demonstration):

| Parameter | Value |
|---|---|
| Batch size | q = 2 |
| Total budget | 23 evaluations, including the initial design |
| Initial design | 21 Latin hypercube points, shared by all methods within each repetition |
| Repetitions | 2 paired runs per method |
| ε | 0.1 |
| Pareto approximation | NSGA-II, population 12 and 2 generations |
| HV reporting | Exact two-objective minimization HV, common KNO1 reference (20, 20) |

For each repetition, the script creates the same initial observations for all
methods. It then fits the GP models, selects a batch, evaluates KNO1, records
the HV after each evaluation, and summarizes the final HV across repetitions.
The internal BoTorch calculations use maximization values; the recorded
objectives and HV report use minimization values.

### Methods evaluated

| Method or family | Configurations | Batch selection |
|---|---:|---|
| qPOTS | 1 | Decision-space maximin (DSM) |
| TSEMO | 1 | Sequential hypervolume improvement (HVI) |
| UCB | 1 | Sequential hypervolume contribution (HVC) |
| ParEGO | 1 | BoTorch qLogNParEGO |
| Greedy-Mean, Greedy-Std | 2 | Fixed mean-front and uncertainty-front policies |
| Greedy-MS | 1 | ε switch between the mean and uncertainty policies |
| ε-qPOTS | 6 | HVI, HVC or DSM × PF or DS exploration |
| ε-TSEMO | 3 | HVI, HVC or DSM |
| ε-UCB | 6 | HVI, HVC or DSM × PF or DS exploration |

**Total: 22 configurations** for ε=0.1. In ε-qPOTS and ε-UCB, **PF** draws an
exploratory point from the surrogate Pareto set; **DS** draws one from the
decision space. ε-TSEMO instead switches between a single spectral posterior
path and an average of paths, so it has no PF/DS option. HVC selects batch
members sequentially.

The demo settings do **not** reproduce the full nine-problem, 20-repetition
experiment in the paper. Running more ε values or using the manuscript budget
takes substantially longer. To inspect that grid on KNO1, use:

~~~bash
python run_demo.py --epsilons 0.1 0.2 0.3 0.5 0.7 0.9 0.99 --output-root kno1_full_epsilon_grid
~~~

This option creates **118 configurations**: 16 ε-dependent configurations at
each of seven ε values, plus six fixed comparators.

---

## Expected Output

While running, the terminal reports each completed method and its elapsed
time. After all runs, it prints the final HV table. The KNO1 rows below come
from one local run; values may vary by software version and times by computer:

~~~text
Running 22 methods on 1 problem; 2 paired repetitions.
KNO1    qPOTS                                  run=1 final HV=204.367385 time=00:00:10.7
...
ε-MOBO — FINAL HV SUMMARY (higher is better)
q=2; total evaluations=23 (including 21 LHS); paired repetitions=2; ε=[0.1]

Problem  Method                         Mean ± SD of final HV  Run 1  Run 2
KNO1     qPOTS                      201.562431 ± 3.966803  204.367385  198.757478
KNO1     ε-qPOTS-HVI-PF (ε=0.1)     200.594663 ± 2.598172  202.431847  198.757478
...
Total wall time (all sessions): <depends on the computer>
Full timing breakdown: .../demo_timing.txt
~~~

Each row reports mean ± **sample** standard deviation of final HV and the
individual repetition values. Higher HV is better for this fixed KNO1
reference; a single repetition shows SD as n/a. The run writes a numerical
summary and an elapsed-time report locally. These generated files are excluded
from the repository by .gitignore.

---

## File Descriptions

### benchmarks.py

Defines the KNO1 objective functions and the [0, 3]² bounds. It also converts
normalized designs in [0, 1]² into the benchmark's input domain.

### methods.py

Implements the GP models, qPOTS, TSEMO, UCB, ParEGO, the ε-greedy variants,
Greedy-Mean/Std/MS, and Pareto approximation and batch selection helpers.
The runner calls these methods through the installed dependencies.

### run_demo.py

Provides the command-line options, shared Latin hypercube designs, paired
optimization loops, exact two-objective HV calculation, final HV table and
cumulative runtime report. The KNO1 reporting reference is defined here.

### requirements.txt

Specifies the Python packages needed to run the demonstration.

### tests/test_demo.py

Checks the KNO1 implementation, the method grid, the hypervolume summary and
the timing calculations without requiring a full optimization run.

### .gitignore

Prevents generated results, local environments and Python cache files from
being included in the repository.
