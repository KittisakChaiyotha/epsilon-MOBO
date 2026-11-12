"""Small numerical checks for the publicly linked RE91 evaluator."""

import unittest

import numpy as np
import torch

from experiments.car_cab.re91_problem import RE91, RE91Function, make_noise_table


class RE91EvaluatorTests(unittest.TestCase):
    def test_third_objective_coefficient(self):
        # z3 changes x10 from 0 to 10; with z4=0, no other f2 term changes.
        x = torch.tensor([[1., 1., 1., 1., 1.75, .8, .8]], dtype=torch.double)
        draws = np.array([[0., 0., 0., 0.], [0., 0., 1., 0.]])
        evaluator = RE91Function(draws)
        y = evaluator.evaluate(x.repeat(2, 1))
        self.assertEqual(tuple(y.shape), (2, RE91.nobj))
        self.assertAlmostEqual((y[1, 2] - y[0, 2]).item(),
                               0.00087570001 * 1.75 * 10 / 0.32,
                               places=12)
        self.assertEqual(evaluator.cursor, 2)

    def test_nominal_does_not_consume_noise(self):
        x = torch.tensor([1., 1., 1., 1., 1.75, .8, .8], dtype=torch.double)
        evaluator = RE91Function(np.zeros((1, 4)))
        nominal = evaluator.evaluate_nominal(x)
        self.assertEqual(tuple(nominal.shape), (1, RE91.nobj))
        self.assertEqual(evaluator.cursor, 0)
        torch.testing.assert_close(evaluator.evaluate(x), nominal)
        self.assertEqual(evaluator.cursor, 1)

    def test_seeded_common_random_numbers(self):
        draws = make_noise_table(17, 3)
        np.testing.assert_array_equal(draws, make_noise_table(17, 3))
        self.assertEqual(draws.shape, (3, 4))
        x = torch.tensor([[1., 1., 1., 1., 1.75, .8, .8]], dtype=torch.double)
        first = RE91Function(draws)
        second = RE91Function(draws.copy())
        torch.testing.assert_close(first.evaluate(x), second.evaluate(x))
        self.assertEqual(first.cursor, second.cursor)


if __name__ == "__main__":
    unittest.main()
