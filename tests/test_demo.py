"""Numerical and reporting checks; no GP installation required."""

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from benchmarks import BENCHMARKS
from run_demo import (CRITERIA, GROUPS, MODES, REPORTING_REFERENCES,
                      hypervolume_2d, initial_design, method_specs,
                      result_path, show_total_timing, timing_checkpoint,
                      timing_report, write_summary)


class DemoTests(unittest.TestCase):
    def test_paper_benchmarks(self):
        self.assertEqual(set(BENCHMARKS), {"KNO1"})
        kno = BENCHMARKS["KNO1"]
        self.assertEqual(kno.initial_n, 21)
        self.assertEqual(kno.lower, (0., 0.))
        r = 9 - 5 * np.sin(2)
        np.testing.assert_allclose(kno.evaluate([[0., 0.]])[0],
                                   [20 - r / np.sqrt(2), 20 - r / np.sqrt(2)])
        x1, y1 = initial_design(kno, 1234)
        x2, y2 = initial_design(kno, 1234)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_hv_union_and_filter(self):
        points = np.array([[0., 1.], [1., 0.], [1.5, 1.5], [3., -10.]])
        self.assertAlmostEqual(hypervolume_2d(points, [2., 2.]), 3.)
        self.assertAlmostEqual(hypervolume_2d(points[::-1], [2., 2.]), 3.)
        self.assertAlmostEqual(hypervolume_2d(np.empty((0, 2)), [2., 2.]), 0.)

    def test_complete_method_grid(self):
        specs = method_specs(list(GROUPS), list(CRITERIA), list(MODES), [0.1])
        self.assertEqual(len(specs), 22)
        self.assertEqual(len({spec.key for spec in specs}), 22)
        self.assertEqual(len(method_specs(list(GROUPS), list(CRITERIA),
                                          list(MODES), [0.1, 0.2, 0.3, 0.5,
                                                        0.7, 0.9, 0.99])), 118)
        self.assertEqual(sum(spec.family == "tsemo" and spec.epsilon is not None
                             for spec in specs), 3)
        self.assertTrue(all(spec.mode is None for spec in specs
                            if spec.family == "tsemo"))
        self.assertEqual({spec.mode for spec in specs
                          if spec.family == "qpots" and spec.epsilon is not None},
                         {"DSexp", "PFexp"})
        self.assertEqual(REPORTING_REFERENCES["KNO1"], (20., 20.))

    def test_summary_uses_final_hv_and_sample_sd(self):
        config = dict(problems=["KNO1"],
                      methods=[vars(s) for s in method_specs(
                          ["qpots", "eps-qpots"], ["DSM"], ["PFexp"], [0.1])],
                      repetitions=2, q=2, total_evals=23, epsilons=[0.1],
                      seed=1234, references={"KNO1": [20., 20.]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "demo_results").mkdir()
            for problem in config["problems"]:
                for spec in config["methods"]:
                    for run in (1, 2):
                        ref = np.array(config["references"][problem])
                        outcomes = np.tile(ref, (23, 1))
                        outcomes[-1] = [19., 19.] if run == 1 else [17., 19.]
                        final = hypervolume_2d(outcomes, ref)
                        path = result_path(root, problem, spec["key"], run, config)
                        np.savez(path, problem=problem, method=spec["key"], q=2,
                                 total_evals=23, reference=config["references"][problem],
                                 seed=1234 + run - 1, y_min=outcomes,
                                 hv_trace=np.r_[np.zeros(22), final],
                                 elapsed_seconds=run * 1.25)
            write_summary(root, config)
            with (root / "demo_summary.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["mean_final_hv"]), 2.)
            self.assertAlmostEqual(float(rows[0]["sd_final_hv"]), np.sqrt(2.))
            self.assertIn("ε-qPOTS-DSM-PF", rows[1]["method"])
            sessions = [dict(wall_seconds=7.5, status="completed"),
                        dict(wall_seconds=0., status="running")]
            timing_path = root / "demo_results" / "demo_timing.json"
            record = timing_checkpoint(timing_path, sessions,
                                       time.perf_counter() - 2., 4, "completed")
            self.assertAlmostEqual(record["total_wall_seconds"], 9.5, delta=0.5)
            timing_report(root, config, record)
            show_total_timing(root)
            saved = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["sum_method_run_seconds"], 7.5)
            self.assertEqual(saved["per_problem_method_run_seconds"]["KNO1"], 7.5)
            self.assertEqual(len(saved["per_run"]), 4)
            self.assertIn("Total wall time (all sessions):",
                          (root / "demo_summary.txt").read_text(encoding="utf-8"))
            self.assertIn("Per completed run (problem | method | repetition | elapsed)",
                          (root / "demo_timing.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
