#!/usr/bin/env python3
"""Derive reporting references and final HV from a complete fresh raw archive.

The reference is pooled across q, methods and repetitions on each problem.
It is computed from nondominated observed minimization objective vectors,
with the manuscript's 1% outward margin and 1e-6 range floor. This program
never substitutes the archived manuscript references or their saved HV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_full_experiment import PROBLEMS, Q_VALUES, task_specs


def update_front(front: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Incremental exact nondominated set for two or three objectives."""
    for y in observations:
        if not np.isfinite(y).all():
            raise ValueError("nonfinite observed objective")
        if len(front) and np.any(np.all(front <= y, axis=1)):
            continue
        if len(front):
            front = front[~np.all(y <= front, axis=1)]
        front = np.vstack((front, y))
    return front


def selected_methods(archive):
    specs = task_specs()
    if archive == "family108":
        specs = [s for s in specs if s.key in ("qpots", "tsemo", "ucb")
                 or s.key.startswith(("eps-qpots-", "eps-ucb-", "eps-tsemo-"))]
        if len(specs) != 108:
            raise RuntimeError("family selection must have 108 methods")
    return specs


def result_file(root, problem, q, key, rep):
    return root / "runs" / problem / f"q{q}" / key / f"rep{rep:02d}.npz"


def rows_for_problem(root, problem, config, specs, config_hash):
    count = config["total_evals"]
    for q in Q_VALUES:
        for spec in specs:
            for rep in range(config["repetitions"]):
                path = result_file(root, problem, q, spec.key, rep)
                if not path.exists():
                    raise FileNotFoundError(f"incomplete archive; missing {path}")
                with np.load(path, allow_pickle=False) as z:
                    y = z["y_min"]
                    if (str(z["config_hash"]) != config_hash
                            or str(z["problem"]) != problem
                            or int(z["q"]) != q
                            or str(z["method_key"]) != spec.key
                            or int(z["rep"]) != rep
                            or y.shape[0] != count):
                        raise ValueError(f"incompatible saved run {path}")
                yield q, spec, rep, y


def hv_min(y, ref):
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated
    valid = y[np.all(y < ref, axis=1)]
    if len(valid) == 0:
        return 0.0
    values = -torch.as_tensor(valid, dtype=torch.double)
    return float(Hypervolume(-torch.as_tensor(ref, dtype=torch.double)).compute(
        values[is_non_dominated(values)]))


def summarize(root: Path, archive: str, problems=PROBLEMS):
    config_path = root / "config.json"
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if config["problems"] != list(PROBLEMS) or config["q"] != list(Q_VALUES):
        raise ValueError("unexpected benchmark configuration")
    config_hash = hashlib.sha256(raw).hexdigest()
    specs = selected_methods(archive)
    out = root / "summaries"
    out.mkdir(exist_ok=True)
    reference_path = out / f"{archive}_references.json"
    metric_path = out / f"{archive}_final_hv.csv"
    references = {}
    for problem in problems:
        front = np.empty((0, 2 if problem in ("kno1", "vlmop2", "oka1", "oka2", "dtlz1a") else 3))
        for _, _, _, y in rows_for_problem(root, problem, config, specs, config_hash):
            if y.shape[1] != front.shape[1]:
                raise ValueError(f"objective dimension mismatch: {problem}")
            front = update_front(front, y)
        ideal, nadir = front.min(axis=0), front.max(axis=0)
        ref = nadir + 0.01 * np.maximum(nadir - ideal, 1e-6)
        references[problem] = dict(
            reference=ref.tolist(), ideal=ideal.tolist(), nadir=nadir.tolist(),
            nondominated_observations=len(front), method_count=len(specs),
            archive=archive,
        )
        print(f"{problem}: ref={ref.tolist()}, ND={len(front)}", flush=True)
    # Do not publish partial summaries: every result has been checked above.
    tmp_csv = metric_path.with_suffix(".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("problem", "q", "method_key", "rep", "final_hv"))
        for problem in problems:
            ref = np.asarray(references[problem]["reference"])
            for q, spec, rep, y in rows_for_problem(root, problem, config, specs, config_hash):
                writer.writerow((problem, q, spec.key, rep, hv_min(y, ref)))
    tmp_csv.replace(metric_path)
    tmp_ref = reference_path.with_suffix(".tmp")
    tmp_ref.write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")
    tmp_ref.replace(reference_path)
    print(f"Saved {reference_path} and {metric_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--archive", choices=("all118", "family108"), required=True)
    p.add_argument("--problem", choices=PROBLEMS, help="summarize one complete problem")
    a = p.parse_args(argv)
    summarize(a.output_root, a.archive, (a.problem,) if a.problem else PROBLEMS)


if __name__ == "__main__":
    main()
