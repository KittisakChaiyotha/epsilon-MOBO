"""Deterministic KNO1 benchmark in minimization form.

The formula follows the TS-EMO test functions (originally from ParEGO).
The bounds are [0, 3]^2.
Reporting HV references are specified in ``run_demo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Benchmark:
    name: str
    lower: tuple[float, float]
    upper: tuple[float, float]

    @property
    def initial_n(self) -> int:
        return 11 * len(self.lower) - 1

    def from_unit_cube(self, x_unit: np.ndarray) -> np.ndarray:
        points = np.asarray(x_unit, dtype=float)
        if points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ValueError("expected finite 2D normalized designs")
        if np.any((points < 0) | (points > 1)):
            raise ValueError("normalized designs must lie in [0,1]^2")
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return lo + points * (hi - lo)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        points = np.asarray(x, dtype=float)
        if points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ValueError("expected finite 2D designs")
        if np.any(points < self.lower) or np.any(points > self.upper):
            raise ValueError(f"{self.name} designs outside its domain")
        x1, x2 = points[..., 0], points[..., 1]
        if self.name == "KNO1":
            summation = x1 + x2
            radius = 9 - (3 * np.sin(2.5 * summation**2)
                          + 3 * np.sin(4 * summation)
                          + 5 * np.sin(2 * summation + 2))
            phi = np.pi / 12 * (x1 - x2 + 3)
            f1, f2 = 20 - radius * np.cos(phi), 20 - radius * np.sin(phi)
        else:
            raise ValueError(f"unknown benchmark: {self.name}")
        return np.stack((f1, f2), axis=-1)


BENCHMARKS = {
    "KNO1": Benchmark("KNO1", (0., 0.), (3., 3.)),
}
