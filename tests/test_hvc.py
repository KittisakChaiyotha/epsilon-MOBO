"""Checks for HVC scores computed on the fixed initial candidate front."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from scipy.stats.qmc import Sobol

import methods as core


class HVCTests(unittest.TestCase):
    def setUp(self):
        # Maximization front with distinct initial exclusive contributions:
        # [0.015, 0.0651, 0.0143, 0.1178, 0.0252, 0.007].
        self.front = torch.tensor(
            [[0.0, 1.0], [0.21, 0.85], [0.34, 0.54],
             [0.72, 0.43], [0.93, 0.12], [1.0, 0.0]],
            dtype=torch.double,
        )

    def test_batch_uses_initial_contributions_at_q2_and_q4(self):
        # Recomputing contributions after removing the first point would
        # select index 4 next. The initial scores instead select index 1.
        for q, expected in ((1, [3]), (2, [3, 1]), (4, [3, 1, 4, 0])):
            with self.subTest(q=q):
                self.assertEqual(core._hvc_selection(self.front, q).tolist(), expected)

    def test_candidate_normalization_handles_affine_scales(self):
        transformed = self.front * torch.tensor([17.0, 0.03]) + torch.tensor([4.0, -9.0])
        self.assertEqual(core._hvc_selection(transformed, 4).tolist(), [3, 1, 4, 0])

    def test_constant_objective_keeps_the_selection(self):
        front = torch.cat((self.front, torch.ones(6, 1, dtype=torch.double)), dim=1)
        self.assertEqual(core._hvc_selection(front, 4).tolist(), [3, 1, 4, 0])

    def test_duplicate_objective_vectors_keep_original_indices(self):
        front = torch.cat((self.front[:3], self.front[1:2], self.front[3:]), dim=0)
        self.assertEqual(core._hvc_selection(front, 4).tolist(), [4, 1, 5, 0])

    def test_empty_zero_and_complete_candidate_sets(self):
        self.assertEqual(core._hvc_selection(self.front[:0], 2).numel(), 0)
        self.assertEqual(core._hvc_selection(self.front, 0).numel(), 0)
        self.assertEqual(core._hvc_selection(self.front, 10).tolist(), list(range(6)))

    def test_qmc_uses_one_scrambled_base_of_4096_points(self):
        front = torch.cat((self.front, torch.ones(6, 7, dtype=torch.double)), dim=1)
        draws = []

        def make_engine(**kwargs):
            engine = Sobol(**kwargs)
            draw = Mock(wraps=engine.random)
            draws.append(draw)
            return SimpleNamespace(random=draw)

        torch.manual_seed(20261120)
        expected_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        with patch.object(core, "Sobol", side_effect=make_engine) as factory:
            torch.manual_seed(20261120)
            selected = core._hvc_selection(front, 2)
            factory.assert_called_once_with(d=9, scramble=True, seed=expected_seed)
        self.assertEqual(len(draws), 1)
        draws[0].assert_called_once_with(4096)
        self.assertEqual(selected.tolist(), [3, 1])
        torch.manual_seed(20261120)
        self.assertTrue(torch.equal(core._hvc_selection(front, 2), selected))


if __name__ == "__main__":
    unittest.main()
