#!/usr/bin/env python3
"""Fresh nine-problem, 118-method benchmark grid using the public methods module.

This is a new runnable experiment, not the archived runner that produced the
manuscript's saved results. Raw minimization observations are saved for later
reporting-HV calculation with a reference derived from the complete archive.
Run one indexed task per scheduler job, or use filters for a local smoke run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from run_demo import GROUPS, MethodSpec, method_specs  # noqa: E402

EPSILONS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99)
Q_VALUES = (1, 2, 4)
CRITERIA = ("HVI", "HVC", "DSM")
MODES = ("PFexp", "DSexp")
PROBLEMS = ("kno1", "vlmop2", "vlmop3", "oka1", "oka2",
            "dtlz1a", "dtlz2a", "dtlz4a", "dtlz7a")


def task_specs():
    specs = method_specs(list(GROUPS), list(CRITERIA), list(MODES), list(EPSILONS))
    if len(specs) != 118 or len({s.key for s in specs}) != 118:
        raise RuntimeError("the benchmark grid must contain 118 unique methods")
    return specs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", choices=PROBLEMS, help="one problem; default: all")
    p.add_argument("--q", type=int, choices=Q_VALUES, help="one batch size; default: all")
    p.add_argument("--method-key", help="one key shown by --list-methods")
    p.add_argument("--rep", type=int, help="zero-based repetition, 0 through 19")
    p.add_argument("--task-index", type=int, help="zero-based index in the filtered grid")
    p.add_argument("--list-methods", action="store_true")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--output-root", type=Path, default=HERE / "fresh_results")
    p.add_argument("--total-evals", type=int, default=150)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--seed0", type=int, default=1234)
    a = p.parse_args(argv)
    if a.rep is not None and not 0 <= a.rep < a.repetitions:
        p.error("--rep must be within the configured repetitions")
    if a.repetitions < 1 or a.total_evals < 1 or a.population < 4 or a.generations < 1:
        p.error("invalid repetitions, budget, population or generations")
    if a.seed0 < 0:
        p.error("--seed0 must be nonnegative")
    return a


def tasks(args, specs):
    problems = (args.problem,) if args.problem else PROBLEMS
    qs = (args.q,) if args.q else Q_VALUES
    reps = (args.rep,) if args.rep is not None else range(args.repetitions)
    chosen = [s for s in specs if args.method_key is None or s.key == args.method_key]
    if not chosen:
        raise ValueError("unknown --method-key; use --list-methods")
    result = [(p, q, s, r) for p in problems for q in qs for s in chosen for r in reps]
    if args.task_index is not None:
        if not 0 <= args.task_index < len(result):
            raise ValueError(f"task index outside [0,{len(result) - 1}]")
        return [result[args.task_index]], len(result)
    return result, len(result)


def environment():
    import torch
    import botorch
    import gpytorch
    import pymoo
    import scipy
    return {k: getattr(v, "__version__", "unknown") for k, v in {
        "numpy": np, "scipy": scipy, "torch": torch, "botorch": botorch,
        "gpytorch": gpytorch, "pymoo": pymoo,
    }.items()}


def configuration(args):
    import methods as core
    return {
        "protocol": "fresh-public-runner-v1; not the archived manuscript runner",
        "problems": list(PROBLEMS), "q": list(Q_VALUES),
        "method_keys": [s.key for s in task_specs()],
        "epsilons": list(EPSILONS), "repetitions": args.repetitions,
        "total_evals": args.total_evals, "population": args.population,
        "generations": args.generations, "seed0": args.seed0,
        "seed_rule": "seed0+1000*zero_based_rep; shared across methods and q",
        "initial_n_rule": "11*d-1 Latin-hypercube designs",
        "objective_sense": "minimization; model sees negated objectives",
        "method_code_version": core.CODE_VERSION,
        "test_function_sha256": hashlib.sha256((HERE / "test_functions.py").read_bytes()).hexdigest(),
        "method_code_sha256": hashlib.sha256((ROOT / "methods.py").read_bytes()).hexdigest(),
        "packages": environment(),
    }


def write_manifest(root: Path, config: dict):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    contents = json.dumps(config, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=root, prefix=".config-",
                                     suffix=".tmp", encoding="utf-8", delete=False) as f:
        f.write(contents)
        temp = Path(f.name)
    try:
        try:
            os.link(temp, path)  # create the shared manifest atomically
        except FileExistsError:
            if path.read_text(encoding="utf-8") != contents:
                raise RuntimeError("different configuration at output root; choose a new root")
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def initial_data(problem, seed, torch):
    x = torch.as_tensor(
        LatinHypercube(d=problem.dim, seed=seed).random(11 * problem.dim - 1),
        dtype=torch.double,
    )
    bounds = problem.get_bounds()
    y = problem.evaluate(bounds[0] + x * (bounds[1] - bounds[0]))
    return x, y


def acquisition_ref(y_max, torch):
    lo, hi = y_max.min(0).values, y_max.max(0).values
    return lo - 0.1 * (hi - lo).clamp(min=1e-6)


def suggest(core, torch, x, y, q, spec: MethodSpec, population, generations):
    d, k = x.shape[1], y.shape[1]
    y_max = -y
    bounds = torch.stack((torch.zeros(d, dtype=torch.double),
                          torch.ones(d, dtype=torch.double)))
    model = core.fit_gp_models(x, y_max, k)
    if spec.key == "qpots":
        result = core._qpots_candidate(model, x, bounds, k, q, generations, population)
    elif spec.key == "tsemo":
        result = core._tsemo_candidate(model, x, y_max, bounds, k, q,
                                       generations, population, n_features=512)
    elif spec.key == "ucb":
        result = core._ucb_candidate(model, x, bounds, k, q,
                                     generations, population, beta=2.0)
    elif spec.key == "parego":
        result = core._parego_candidate(model, x, y_max, bounds, q)
    elif spec.family.startswith("greedy-"):
        policy = {"greedy-mean": "mean", "greedy-std": "std",
                  "greedy-ms": "epsilon"}[spec.family]
        result = core.EpsilonGreedyMeanStdPF(
            model_obj=model, bounds=bounds, nobj=k, q=q,
            epsilon=spec.epsilon if spec.epsilon is not None else 0.0,
            ngen=generations, npop=population, policy=policy,
            observed_x=x,
        ).suggest()
    elif spec.family == "tsemo":
        result = core.EpsilonTSEMO(
            train_x=x, train_y=y_max, gp_list=list(model.models),
            bounds=bounds, nobj=k, q=q, epsilon=spec.epsilon,
            Ns=50, Np=512, ngen=generations, npop=population,
            selection_criteria={"DSM": core.SelectionCriteria.DS_MAXIMIN,
                                "HVI": core.SelectionCriteria.HVI,
                                "HVC": core.SelectionCriteria.HVC}[spec.criterion],
        ).suggest()
    else:
        common = dict(model_obj=model, bounds=bounds, nobj=k, q=q,
                      epsilon=spec.epsilon, ngen=generations, npop=population,
                      selection_criteria={"DSM": core.SelectionCriteria.DS_MAXIMIN,
                                          "HVI": core.SelectionCriteria.HVI,
                                          "HVC": core.SelectionCriteria.HVC}[spec.criterion],
                      observed_x=x, observed_y=y_max)
        if spec.family == "qpots":
            cls = core.EpsilonQPOTSExplorePF if spec.mode == "PFexp" else core.EpsilonQPOTS
        else:
            cls = core.EpsilonUCBExplorePF if spec.mode == "PFexp" else core.EpsilonUCB
            common["beta"] = 2.0
        result = cls(**common).suggest()
    return core.validate_candidate_batch(result, x, q, d)


def run_task(args, task, config_hash):
    import torch
    import methods as core
    from test_functions import get_test_function
    p, q, spec, rep = task
    tf = get_test_function(p)
    n0 = 11 * tf.dim - 1
    if args.total_evals <= n0:
        raise ValueError(f"{p}: total-evals must exceed initial design {n0}")
    path = args.output_root / "runs" / p / f"q{q}" / spec.key / f"rep{rep:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["config_hash"]) != config_hash
                    or tuple(saved["x_unit"].shape) != (args.total_evals, tf.dim)
                    or tuple(saved["y_min"].shape) != (args.total_evals, tf.nobj)):
                raise RuntimeError(f"incompatible saved run: {path}")
        print(f"Already complete: {path}", flush=True)
        return
    seed = args.seed0 + 1000 * rep
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    x, y = initial_data(tf, seed, torch)
    initial_x = x.clone()
    initial_y = y.clone()
    start = time.perf_counter()
    while len(x) < args.total_evals:
        q_now = min(q, args.total_evals - len(x))
        candidate = suggest(core, torch, x, y, q_now, spec,
                            args.population, args.generations)
        bounds = tf.get_bounds()
        new_y = tf.evaluate(bounds[0] + candidate * (bounds[1] - bounds[0]))
        if new_y.shape != (q_now, tf.nobj) or not torch.isfinite(new_y).all():
            raise RuntimeError("invalid objective values; run not saved")
        x = torch.cat((x, candidate.detach()), dim=0)
        y = torch.cat((y, new_y.detach()), dim=0)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, x_unit=x.numpy(), y_min=y.numpy(),
        initial_x=initial_x.numpy(), initial_y=initial_y.numpy(),
        problem=p, q=q, method_key=spec.key, method=json.dumps(asdict(spec)),
        rep=rep, seed=seed, config_hash=config_hash,
        elapsed_seconds=time.perf_counter() - start,
    )
    tmp.replace(path)
    print(f"Saved {path} ({len(x)} evaluations)", flush=True)


def main(argv=None):
    args = parse_args(argv)
    specs = task_specs()
    if args.list_methods:
        for s in specs:
            print(s.key, s.label)
        return 0
    selected, total = tasks(args, specs)
    if args.list_tasks:
        print(f"{total} tasks in filtered grid; index 0..{total-1}")
        for p, q, s, rep in selected[:10]:
            print(p, q, s.key, rep)
        return 0
    config_hash = write_manifest(args.output_root, configuration(args))
    for task in selected:
        run_task(args, task, config_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
