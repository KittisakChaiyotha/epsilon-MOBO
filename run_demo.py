#!/usr/bin/env python3
"""Run the ε-MOBO method grid on KNO1 and summarize final HV.

The optimization dependencies are loaded only when a run begins. The
``--summary-only`` option reads saved numerical results without BoTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

from benchmarks import BENCHMARKS, Benchmark


HERE = Path(__file__).resolve().parent
REPORTING_REFERENCES = {"KNO1": (20.0, 20.0)}
CRITERIA = ("HVI", "HVC", "DSM")
MODES = ("PFexp", "DSexp")
GROUPS = ("qpots", "tsemo", "ucb", "parego", "greedy-mean",
          "greedy-std", "greedy-ms", "eps-qpots", "eps-tsemo", "eps-ucb")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    family: str
    criterion: str | None = None
    mode: str | None = None
    epsilon: float | None = None


def eps_tag(epsilon: float) -> str:
    return f"{epsilon:g}".replace(".", "p")


def method_specs(groups: list[str], criteria: list[str], modes: list[str],
                 epsilons: list[float]) -> list[MethodSpec]:
    """Six fixed comparators and 16 structural ε configurations per ε."""
    result = []
    baseline_labels = {
        "qpots": "qPOTS", "tsemo": "TSEMO", "ucb": "UCB",
        "parego": "ParEGO", "greedy-mean": "Greedy-Mean",
        "greedy-std": "Greedy-Std",
    }
    for family, label in baseline_labels.items():
        if family in groups:
            result.append(MethodSpec(family, label, family))
    for epsilon in epsilons:
        if "greedy-ms" in groups:
            result.append(MethodSpec(f"greedy-ms-e{eps_tag(epsilon)}",
                                     f"Greedy-MS (ε={epsilon:g})", "greedy-ms",
                                     epsilon=epsilon))
        for family in ("tsemo", "qpots", "ucb"):
            if f"eps-{family}" not in groups:
                continue
            name = {"tsemo": "TSEMO", "qpots": "qPOTS", "ucb": "UCB"}[family]
            for criterion in criteria:
                if family == "tsemo":
                    result.append(MethodSpec(
                        f"eps-tsemo-{criterion.lower()}-e{eps_tag(epsilon)}",
                        f"ε-TSEMO-{criterion} (ε={epsilon:g})", family,
                        criterion=criterion, epsilon=epsilon))
                else:
                    for mode in modes:
                        suffix = "PF" if mode == "PFexp" else "DS"
                        result.append(MethodSpec(
                            f"eps-{family}-{criterion.lower()}-{suffix.lower()}-e{eps_tag(epsilon)}",
                            f"ε-{name}-{criterion}-{suffix} (ε={epsilon:g})",
                            family, criterion=criterion, mode=mode,
                            epsilon=epsilon))
    return result


def hypervolume_2d(values: np.ndarray, reference: np.ndarray) -> float:
    """Exact dominated area of 2D minimization observations."""
    points = np.asarray(values, dtype=float).reshape(-1, 2)
    ref = np.asarray(reference, dtype=float).reshape(2)
    if not np.isfinite(points).all() or not np.isfinite(ref).all():
        raise ValueError("HV inputs must be finite")
    valid = points[np.all(points < ref, axis=1)]
    previous_second, volume = float(ref[1]), 0.0
    for first, second in valid[np.argsort(valid[:, 0], kind="stable")]:
        if second < previous_second:
            volume += (float(ref[0]) - first) * (previous_second - second)
            previous_second = second
    return float(volume)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=BENCHMARKS,
                        default=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=GROUPS,
                        default=list(GROUPS),
                        help="Method groups to include; default is all 22 configurations.")
    parser.add_argument("--criteria", nargs="+", choices=CRITERIA,
                        default=list(CRITERIA))
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=list(MODES), help="Applies to ε-qPOTS and ε-UCB.")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1],
                        help="One or more ε probabilities; default: 0.1.")
    parser.add_argument("--q", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--total-evals", type=int, default=23,
                        help="Includes 21 initial evaluations; manuscript uses 150.")
    parser.add_argument("--repetitions", type=int, default=2,
                        help="Matched repetitions per method; manuscript uses 20.")
    parser.add_argument("--population", type=int, default=12,
                        help="NSGA-II population; manuscript uses 100.")
    parser.add_argument("--generations", type=int, default=2,
                        help="NSGA-II generations; manuscript uses 100.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace previous results at output-root.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Recreate the table from saved NPZ results.")
    args = parser.parse_args(argv)
    if any(not np.isfinite(e) or not 0 <= e <= 1 for e in args.epsilons):
        parser.error("all --epsilons must be finite numbers in [0,1]")
    if args.total_evals <= 21 or args.repetitions < 1:
        parser.error("need --total-evals > 21 and --repetitions >= 1")
    if args.population < 4 or args.generations < 1 or args.seed < 0:
        parser.error("need --population >= 4, --generations >= 1, --seed >= 0")
    for field in ("problems", "methods", "criteria", "modes", "epsilons"):
        values = getattr(args, field)
        if len(values) != len(set(values)):
            parser.error(f"--{field} contains a duplicate")
    tags = [eps_tag(e) for e in args.epsilons]
    if len(tags) != len(set(tags)):
        parser.error("--epsilons produce duplicate filename labels; use less similar values")
    return args


def initial_design(problem: Benchmark, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unit = LatinHypercube(d=2, seed=seed).random(problem.initial_n)
    return unit, problem.evaluate(problem.from_unit_cube(unit))


def suggest_batch(core, torch, model, x_unit, y_min, q: int,
                  spec: MethodSpec, args):
    criterion = {"DSM": core.SelectionCriteria.DS_MAXIMIN,
                 "HVC": core.SelectionCriteria.HVC,
                 "HVI": core.SelectionCriteria.HVI}
    bounds = torch.stack((torch.zeros(2, dtype=torch.double),
                          torch.ones(2, dtype=torch.double)))
    x_obs = torch.as_tensor(x_unit, dtype=torch.double)
    y_max = -torch.as_tensor(y_min, dtype=torch.double)
    if spec.key == "qpots":
        candidate = core._qpots_candidate(model, x_obs, bounds, 2, q,
                                           args.generations, args.population)
    elif spec.key == "tsemo":
        candidate = core._tsemo_candidate(model, x_obs, y_max, bounds, 2, q,
                                           args.generations, args.population,
                                           n_features=512)
    elif spec.key == "ucb":
        candidate = core._ucb_candidate(model, x_obs, bounds, 2, q,
                                         args.generations, args.population, beta=2.0)
    elif spec.key == "parego":
        candidate = core._parego_candidate(model, x_obs, y_max, bounds, q)
    elif spec.family.startswith("greedy-"):
        policy = {"greedy-mean": "mean", "greedy-std": "std",
                  "greedy-ms": "epsilon"}[spec.family]
        candidate = core.EpsilonGreedyMeanStdPF(
            model_obj=model, bounds=bounds, nobj=2, q=q,
            epsilon=spec.epsilon if spec.epsilon is not None else 0.,
            ngen=args.generations, npop=args.population,
            policy=policy, observed_x=x_obs,
        ).suggest()
    elif spec.family == "tsemo":
        candidate = core.EpsilonTSEMO(
            train_x=x_obs, train_y=y_max, gp_list=list(model.models),
            bounds=bounds, nobj=2, q=q, epsilon=spec.epsilon,
            Ns=50, Np=512, ngen=args.generations, npop=args.population,
            selection_criteria=criterion[spec.criterion],
        ).suggest()
    else:
        common = dict(model_obj=model, bounds=bounds, nobj=2, q=q,
                      epsilon=spec.epsilon, ngen=args.generations,
                      npop=args.population,
                      selection_criteria=criterion[spec.criterion],
                      observed_x=x_obs, observed_y=y_max)
        if spec.family == "qpots":
            cls = core.EpsilonQPOTSExplorePF if spec.mode == "PFexp" else core.EpsilonQPOTS
        else:
            cls = core.EpsilonUCBExplorePF if spec.mode == "PFexp" else core.EpsilonUCB
            common["beta"] = 2.0
        candidate = cls(**common).suggest()
    return core.validate_candidate_batch(candidate, x_obs, q, 2)


def run_one(core, torch, problem: Benchmark, spec: MethodSpec, run: int,
            args, x0: np.ndarray, y0: np.ndarray) -> dict:
    seed = args.seed + run - 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    x_unit, y_min = x0.copy(), y0.copy()
    reference = np.asarray(REPORTING_REFERENCES[problem.name], dtype=float)
    hv_trace = np.array([hypervolume_2d(y_min[:i + 1], reference)
                         for i in range(len(y_min))])
    start = time.monotonic()
    while len(x_unit) < args.total_evals:
        q_now = min(args.q, args.total_evals - len(x_unit))
        model = core.fit_gp_models(
            torch.as_tensor(x_unit, dtype=torch.double),
            -torch.as_tensor(y_min, dtype=torch.double), nobj=2)
        candidate = suggest_batch(core, torch, model, x_unit, y_min, q_now,
                                  spec, args)
        new_x = candidate.detach().cpu().numpy()
        new_y = problem.evaluate(problem.from_unit_cube(new_x))
        for row in new_y:
            y_min = np.vstack((y_min, row))
            hv_trace = np.append(hv_trace, hypervolume_2d(y_min, reference))
        x_unit = np.vstack((x_unit, new_x))
    return dict(x_unit=x_unit, y_min=y_min, hv_trace=hv_trace,
                reference=reference, initial_n=problem.initial_n,
                seed=seed, problem=problem.name, method=spec.key,
                q=args.q, total_evals=args.total_evals,
                elapsed_seconds=time.monotonic() - start)


def result_path(root: Path, problem: str, key: str, run: int, config: dict) -> Path:
    return root / "demo_results" / (
        f"{problem}_{run}_{config['q']}_{config['total_evals']}_{key}.npz")


def configuration(args, specs: list[MethodSpec]) -> dict:
    return dict(paper="ε-MOBO: Epsilon-greedy acquisition strategies for multi-objective Bayesian optimization",
                problems=args.problems, methods=[asdict(spec) for spec in specs],
                q=args.q, total_evals=args.total_evals,
                repetitions=args.repetitions, epsilons=args.epsilons,
                population=args.population, generations=args.generations,
                seed=args.seed, initial_n=21,
                references={k: list(REPORTING_REFERENCES[k]) for k in args.problems},
                objective_sense="minimize", demo=True)


def duration(seconds: float) -> str:
    """Human-readable elapsed time, retaining tenths of a second."""
    hours, remainder = divmod(float(seconds), 3600)
    minutes, remaining = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{remaining:04.1f}"


def timing_checkpoint(path: Path, sessions: list[dict], start: float,
                      new_runs: int, status: str) -> dict:
    """Save measured wall time after each run, so a resumed job retains it."""
    current = sessions[-1]
    current.update(wall_seconds=time.perf_counter() - start,
                   new_runs=new_runs, status=status,
                   last_updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    record = {
        "sessions": sessions,
        "total_wall_seconds": sum(item["wall_seconds"] for item in sessions),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return record


def timing_report(root: Path, config: dict, record: dict) -> None:
    """Write both overall elapsed time and the per-method/run breakdown."""
    per_run = []
    per_problem = {problem: 0.0 for problem in config["problems"]}
    per_method = {entry["key"]: 0.0 for entry in config["methods"]}
    for problem in config["problems"]:
        for entry in config["methods"]:
            for run in range(1, config["repetitions"] + 1):
                with np.load(result_path(root, problem, entry["key"], run, config),
                             allow_pickle=False) as data:
                    seconds = float(data["elapsed_seconds"])
                per_run.append({"problem": problem, "method": entry["label"],
                                "run": run, "seconds": seconds})
                per_problem[problem] += seconds
                per_method[entry["key"]] += seconds
    record["sum_method_run_seconds"] = sum(item["seconds"] for item in per_run)
    record["per_problem_method_run_seconds"] = per_problem
    record["per_method_seconds"] = per_method
    record["per_run"] = per_run
    (root / "demo_results" / "demo_timing.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["ε-MOBO — RUNTIME SUMMARY",
             f"Total wall time (all sessions): {duration(record['total_wall_seconds'])}",
             f"Last command wall time: {duration(record['sessions'][-1]['wall_seconds'])}",
             f"Sum of completed method-run times: {duration(record['sum_method_run_seconds'])}",
             "Method-run time excludes shared initialization; total wall time includes it.",
             "", "Per-problem method-run totals:"]
    lines.extend(f"  {problem}: {duration(seconds)}" for problem, seconds in per_problem.items())
    lines.extend(["", "Per completed run (problem | method | repetition | elapsed):"])
    lines.extend(f"  {item['problem']} | {item['method']} | {item['run']} | "
                 f"{duration(item['seconds'])}" for item in per_run)
    (root / "demo_timing.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def show_total_timing(root: Path) -> None:
    path = root / "demo_results" / "demo_timing.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        message = (f"Total wall time (all sessions): "
                   f"{duration(record['total_wall_seconds'])}\n")
        with (root / "demo_summary.txt").open("a", encoding="utf-8") as stream:
            stream.write("\n" + message)
        print(message, end="")


def write_summary(root: Path, config: dict) -> None:
    rows = []
    for problem_name in config["problems"]:
        for entry in config["methods"]:
            spec = MethodSpec(**entry)
            run_hv = []
            for run in range(1, config["repetitions"] + 1):
                path = result_path(root, problem_name, spec.key, run, config)
                with np.load(path, allow_pickle=False) as data:
                    if (str(data["problem"]) != problem_name
                            or str(data["method"]) != spec.key
                            or int(data["q"]) != config["q"]
                            or int(data["total_evals"]) != config["total_evals"]
                            or int(data["seed"]) != config["seed"] + run - 1
                            or len(data["hv_trace"]) != config["total_evals"]
                            or not np.array_equal(data["reference"],
                                                  config["references"][problem_name])
                            or not np.isclose(
                                hypervolume_2d(data["y_min"], data["reference"]),
                                float(data["hv_trace"][-1]), rtol=1e-10, atol=1e-10)):
                        raise ValueError(f"incompatible saved result: {path}")
                    run_hv.append(float(data["hv_trace"][-1]))
            mean = float(np.mean(run_hv))
            sd = float(np.std(run_hv, ddof=1)) if len(run_hv) > 1 else None
            rows.append((problem_name, spec, mean, sd, run_hv))

    width = max(25, *(len(row[1].label) for row in rows))
    header = f"{'Problem':<8} {'Method':<{width}} {'Mean ± SD of final HV':>28}  "
    header += "  ".join(f"Run {r}" for r in range(1, config["repetitions"] + 1))
    lines = ["ε-MOBO — FINAL HV SUMMARY (higher is better)",
             f"q={config['q']}; total evaluations={config['total_evals']} (including 21 LHS); "
             f"paired repetitions={config['repetitions']}; ε={config['epsilons']}",
             "HV reference (minimization): " + ", ".join(
                 f"{k}={v}" for k, v in config["references"].items()),
             "Demo results, separate from the paper's complete nine-problem study.",
             "", header, "-" * len(header)]
    for problem, spec, mean, sd, values in rows:
        mean_sd = f"{mean:.6f} ± {sd:.6f}" if sd is not None else f"{mean:.6f} ± n/a"
        lines.append(f"{problem:<8} {spec.label:<{width}} {mean_sd:>28}  " +
                     "  ".join(f"{v:.6f}" for v in values))
    table = "\n".join(lines) + "\n"
    (root / "demo_summary.txt").write_text(table, encoding="utf-8")
    with (root / "demo_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["problem", "method", "family", "criterion", "mode", "epsilon",
                         "mean_final_hv", "sd_final_hv",
                         *[f"run_{r}_final_hv" for r in range(1, config["repetitions"] + 1)]])
        for problem, spec, mean, sd, values in rows:
            writer.writerow([problem, spec.label, spec.family, spec.criterion or "",
                             spec.mode or "", "" if spec.epsilon is None else spec.epsilon,
                             mean, "" if sd is None else sd, *values])
    print(table)
    print(f"Saved: {root / 'demo_summary.txt'} and {root / 'demo_summary.csv'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_root.resolve()
    results = root / "demo_results"
    training = root / "demo_training_data"
    manifest = results / "demo_config.json"
    if args.summary_only:
        if not manifest.is_file():
            raise SystemExit(f"Missing results manifest: {manifest}")
        write_summary(root, json.loads(manifest.read_text(encoding="utf-8")))
        show_total_timing(root)
        return 0
    specs = method_specs(args.methods, args.criteria, args.modes, args.epsilons)
    if not specs:
        raise SystemExit("No methods selected")
    config = configuration(args, specs)
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != config and not args.overwrite:
            raise SystemExit("Existing results use different settings; change --output-root or add --overwrite.")
    elif results.exists() and list(results.glob("*.npz")) and not args.overwrite:
        raise SystemExit("Results exist without manifest; change --output-root or add --overwrite.")
    timing_path = results / "demo_timing.json"
    sessions = [] if args.overwrite or not timing_path.exists() else json.loads(
        timing_path.read_text(encoding="utf-8"))["sessions"]
    if sessions and sessions[-1]["status"] == "running":
        sessions[-1]["status"] = "interrupted"
    session_start = time.perf_counter()
    try:
        import torch
        import methods as core
    except ImportError as exc:
        raise SystemExit(f"Missing package: {exc}; install: python -m pip install -r requirements.txt") from exc
    results.mkdir(parents=True, exist_ok=True)
    training.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for file in results.glob("*.npz"):
            file.unlink()
    manifest.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    sessions.append({
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": 0.0, "new_runs": 0, "status": "running",
    })
    new_runs = 0
    timing_checkpoint(timing_path, sessions, session_start, new_runs, "running")
    print(f"Running {len(specs)} methods on {len(args.problems)} "
          f"problem{'s' if len(args.problems) != 1 else ''}; "
          f"{args.repetitions} paired repetitions.", flush=True)
    try:
        for problem_name in args.problems:
            problem = BENCHMARKS[problem_name]
            for run in range(1, args.repetitions + 1):
                seed = args.seed + run - 1
                x0, y0 = initial_design(problem, seed)
                np.savez(training / f"{problem_name}_{run}.npz", x_unit=x0,
                         y_min=y0, seed=seed)
                for spec in specs:
                    output = result_path(root, problem_name, spec.key, run, config)
                    if output.exists() and not args.overwrite:
                        print(f"Already exists: {output.name}")
                        continue
                    result = run_one(core, torch, problem, spec, run, args, x0, y0)
                    np.savez(output, **result)
                    new_runs += 1
                    timing_checkpoint(timing_path, sessions, session_start,
                                      new_runs, "running")
                    print(f"{problem_name:<7} {spec.label:<38} run={run} "
                          f"final HV={result['hv_trace'][-1]:.6f} "
                          f"time={duration(result['elapsed_seconds'])}", flush=True)
        write_summary(root, config)
        record = timing_checkpoint(timing_path, sessions, session_start,
                                   new_runs, "completed")
        timing_report(root, config, record)
        show_total_timing(root)
        print(f"Full timing breakdown: {root / 'demo_timing.txt'}")
    except BaseException:
        timing_checkpoint(timing_path, sessions, session_start,
                          new_runs, "interrupted")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
