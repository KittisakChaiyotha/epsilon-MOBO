#!/usr/bin/env python3
"""RE91 engineering design benchmark with paired common random numbers.

The analytical definition and bounds follow Tanabe and Ishibuchi's RE
benchmark implementation:
https://github.com/ryojitanabe/reproblems

RE91 has seven decision variables and nine minimization objectives.  Four
latent normal variates make the original benchmark stochastic.  For a fair
paired MOBO experiment, this module receives a pre-generated noise table so
that every method and q uses the same latent draw at the same evaluation
index within a repetition (common random numbers, CRN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DTYPE = torch.double


@dataclass(frozen=True)
class RE91Info:
    name: str = "re91"
    dim: int = 7
    nobj: int = 9
    ntrain: int = 76
    acq_eval_budget: int = 250
    source_reference: str = "Tanabe-Ishibuchi RE91; github.com/ryojitanabe/reproblems"
    budget_reference: str = "BOCoDe arXiv:2608.15073 Appendix E.1 Table 5; ninit + 250 proposals/evaluated points"

    @property
    def total_evals(self) -> int:
        return self.ntrain + self.acq_eval_budget

    def batch_schedule(self, q: int) -> tuple[int, ...]:
        if q not in (1, 2, 4):
            raise ValueError("q must be one of {1,2,4}")
        full, rem = divmod(self.acq_eval_budget, q)
        return (q,) * full + ((rem,) if rem else ())


RE91 = RE91Info()


def make_noise_table(seed: int, total_evals: int = RE91.total_evals) -> np.ndarray:
    """Return standard-normal CRN draws with shape ``(total_evals, 4)``."""
    table = np.random.default_rng(int(seed)).standard_normal((total_evals, 4))
    if table.shape != (total_evals, 4) or not np.isfinite(table).all():
        raise RuntimeError("invalid RE91 noise table")
    return table


class RE91Function:
    """Stateful objective evaluator consuming CRN rows in evaluation order."""

    def __init__(self, noise_table: np.ndarray):
        self.noise_table = np.asarray(noise_table, dtype=float)
        if self.noise_table.ndim != 2 or self.noise_table.shape[1] != 4:
            raise ValueError("noise_table must have shape (N,4)")
        self.cursor = 0

    def get_bounds(self) -> torch.Tensor:
        lower = [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]
        upper = [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]
        return torch.tensor([lower, upper], dtype=DTYPE)

    @staticmethod
    def _evaluate_with_latents(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=DTYPE, device="cpu")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != RE91.dim:
            raise ValueError(f"RE91 input must have shape (N,{RE91.dim})")
        n = int(x.shape[0])
        z = torch.as_tensor(z, dtype=DTYPE, device="cpu")
        if z.shape != (n, 4):
            raise ValueError(f"RE91 latent input must have shape ({n},4)")

        x1, x2, x3, x4, x5, x6, x7 = x.unbind(dim=1)
        x8 = 0.006 * z[:, 0] + 0.345
        x9 = 0.006 * z[:, 1] + 0.192
        x10 = 10.0 * z[:, 2]
        x11 = 10.0 * z[:, 3]

        f0 = 1.98 + 4.9*x1 + 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.75*x5 + 1e-5*x6 + 2.73*x7
        f1 = torch.clamp(1.16 - 0.3717*x2*x4 - 0.00931*x2*x10 - 0.484*x3*x9 + 0.01343*x6*x10, min=0.0)
        f2 = torch.clamp(
            (0.261 - 0.0159*x1*x2 - 0.188*x1*x8 - 0.019*x2*x7 + 0.0144*x3*x5
             + 0.00087570001*x5*x10 + 0.08045*x6*x9 + 0.00139*x8*x11
             + 0.00001575*x10*x11) / 0.32,
            min=0.0,
        )
        f3 = torch.clamp(
            (0.214 + 0.00817*x5 - 0.131*x1*x8 - 0.0704*x1*x9 + 0.03099*x2*x6
             - 0.018*x2*x7 + 0.0208*x3*x8 + 0.121*x3*x9 - 0.00364*x5*x6
             + 0.0007715*x5*x10 - 0.0005354*x6*x10 + 0.00121*x8*x11
             + 0.00184*x9*x10 - 0.018*x2*x2) / 0.32,
            min=0.0,
        )
        f4 = torch.clamp(
            (0.74 - 0.61*x2 - 0.163*x3*x8 + 0.001232*x3*x10 - 0.166*x7*x9
             + 0.227*x2*x2) / 0.32,
            min=0.0,
        )
        f5_tmp = (
            (28.98 + 3.818*x3 - 4.2*x1*x2 + 0.0207*x5*x10 + 6.63*x6*x9 - 7.77*x7*x8 + 0.32*x9*x10)
            + (33.86 + 2.95*x3 + 0.1792*x10 - 5.057*x1*x2 - 11*x2*x8 - 0.0215*x5*x10 - 9.98*x7*x8 + 22*x8*x9)
            + (46.36 - 9.9*x2 - 12.9*x1*x8 + 0.1107*x3*x10)
        ) / 3.0
        f5 = torch.clamp(f5_tmp / 32.0, min=0.0)
        f6 = torch.clamp((4.72 - 0.5*x4 - 0.19*x2*x3 - 0.0122*x4*x10 + 0.009325*x6*x10 + 0.000191*x11*x11) / 4.0, min=0.0)
        f7 = torch.clamp((10.58 - 0.674*x1*x2 - 1.95*x2*x8 + 0.02054*x3*x10 - 0.0198*x4*x10 + 0.028*x6*x10) / 9.9, min=0.0)
        f8 = torch.clamp((16.45 - 0.489*x3*x7 - 0.843*x5*x6 + 0.0432*x9*x10 - 0.0556*x9*x11 - 0.000786*x11*x11) / 15.7, min=0.0)
        y = torch.stack((f0, f1, f2, f3, f4, f5, f6, f7, f8), dim=1)
        if y.shape != (n, RE91.nobj) or not torch.isfinite(y).all():
            raise RuntimeError("RE91 returned invalid objectives")
        return y

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=DTYPE, device="cpu")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != RE91.dim:
            raise ValueError(f"RE91 input must have shape (N,{RE91.dim})")
        n = int(x.shape[0])
        stop = self.cursor + n
        if stop > self.noise_table.shape[0]:
            raise RuntimeError("RE91 noise table exhausted")
        z = torch.tensor(self.noise_table[self.cursor:stop], dtype=DTYPE)
        self.cursor = stop
        return self._evaluate_with_latents(x, z)

    def evaluate_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the noise-free nominal response at latent means z=(0,0,0,0).

        This does not advance the stochastic evaluator cursor and is intentionally
        distinct from an expectation over the nonlinear, clipped RE91 response.
        Saving X keeps alternative posterior/Monte-Carlo definitions possible.
        """
        x = torch.as_tensor(x, dtype=DTYPE, device="cpu")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self._evaluate_with_latents(x, torch.zeros((x.shape[0], 4), dtype=DTYPE))
