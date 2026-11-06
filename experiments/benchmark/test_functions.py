"""
epsMOBO_test_functions.py
(formerly tsemo_test_functions_oka1.py)

Custom test functions for TSEMO paper (Bradford et al., JOGO 2018) Table 4.
All functions from Knowles' ParEGO-TR3 (2005), originally defined in:
  - KNO1: Knowles & Corne (2002)
  - VLMOP2, VLMOP3: Van Veldhuizen (1999)
  - OKA1, OKA2: Okabe et al. (2004)
  - DTLZ1, DTLZ2, DTLZ4, DTLZ7: Deb et al. (2005)

The "a" suffix on DTLZ problems indicates Knowles' specific
dimensionality configuration from ParEGO-TR3 Table 4.

All functions are MINIMIZATION problems.
Bounds: all in [0,1]^d unless otherwise noted.

References:
  Knowles (2005) ParEGO-TR3: https://www.cs.bham.ac.uk/~jdk/parego/ParEGO-TR3.pdf
  Deb et al. (2005) DTLZ test problems for EMO
  Bradford et al. (2018) TSEMO - JOGO
"""

import math
import torch
from typing import List

DTYPE = torch.double


class TestFunction:
    """Base class for TSEMO test functions (minimization)."""

    def __init__(self, name: str, dim: int, nobj: int,
                 bounds_lower: List[float], bounds_upper: List[float]):
        self.name = name
        self.dim = dim
        self.nobj = nobj
        self._bounds = torch.tensor(
            [bounds_lower, bounds_upper], dtype=DTYPE
        )

    def get_bounds(self) -> torch.Tensor:
        return self._bounds.clone()

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate at real-scale X. Returns (n, nobj) tensor."""
        raise NotImplementedError

    def get_cons(self):
        return []


# ============================================================
# KNO1: d=2, K=2
# Bounds: x in [0, 3]^2
# Source: Knowles & Corne (2002), Knowles (2005) ParEGO-TR3
# ============================================================
class KNO1(TestFunction):
    """KNO1: True PF lies beyond a locally optimum PF with larger basin."""

    def __init__(self):
        super().__init__("kno1", dim=2, nobj=2,
                         bounds_lower=[0.0, 0.0],
                         bounds_upper=[3.0, 3.0])

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2 = X[:, 0], X[:, 1]
        r = x1 + x2
        phi = math.pi / 12.0
        c = 9.0 - (3.0 * torch.sin(2.5 * r ** 0.5) +
                    3.0 * torch.sin(4.0 * r) +
                    5.0 * torch.sin(2.0 * r + 2.0))
        theta = phi * (x1 - x2 + 3.0)
        f1 = 20.0 - c * torch.cos(theta)
        f2 = 20.0 - c * torch.sin(theta)
        return torch.stack([f1, f2], dim=-1)


# ============================================================
# VLMOP2: d=2, K=2
# Bounds: x in [-2, 2]^2
# Source: Van Veldhuizen (1999), Knowles (2005)
# ============================================================
class VLMOP2(TestFunction):
    """VLMOP2: Concave Pareto front."""

    def __init__(self):
        super().__init__("vlmop2", dim=2, nobj=2,
                         bounds_lower=[-2.0, -2.0],
                         bounds_upper=[2.0, 2.0])

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n = self.dim
        inv_sqrt_n = 1.0 / math.sqrt(n)
        f1 = 1.0 - torch.exp(-((X - inv_sqrt_n) ** 2).sum(dim=-1))
        f2 = 1.0 - torch.exp(-((X + inv_sqrt_n) ** 2).sum(dim=-1))
        return torch.stack([f1, f2], dim=-1)


# ============================================================
# VLMOP3: d=2, K=3
# Bounds: x in [-3, 3]^2
# Source: Van Veldhuizen (1999), Knowles (2005)
# ============================================================
class VLMOP3(TestFunction):
    """VLMOP3: Disconnected Pareto set, convoluted PF curve in 3D."""

    def __init__(self):
        super().__init__("vlmop3", dim=2, nobj=3,
                         bounds_lower=[-3.0, -3.0],
                         bounds_upper=[3.0, 3.0])

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2 = X[:, 0], X[:, 1]
        f1 = 0.5 * (x1 ** 2 + x2 ** 2) + torch.sin(x1 ** 2 + x2 ** 2)
        f2 = ((3.0 * x1 - 2.0 * x2 + 4.0) ** 2) / 8.0 + \
             ((x1 - x2 + 1.0) ** 2) / 27.0 + 15.0
        f3 = 1.0 / (x1 ** 2 + x2 ** 2 + 1.0) - \
             1.1 * torch.exp(-(x1 ** 2 + x2 ** 2))
        return torch.stack([f1, f2, f3], dim=-1)


# ============================================================
# OKA1: d=2, K=2
# Bounds: x1 in [6*sin(pi/12), 6*sin(pi/12) + 2*pi*cos(pi/12)]
#         x2 in [-2*pi*sin(pi/12), 6*cos(pi/12)]
# Source: Okabe et al. (2004), Knowles (2005)
# ============================================================

class OKA1(TestFunction):
    """OKA1: Density of solutions falls away near Pareto front."""

    def __init__(self):
        s = math.sin(math.pi / 12.0)
        c = math.cos(math.pi / 12.0)
        x1_lo = 6.0 * s
        x1_hi = x1_lo + 2.0 * math.pi * c
        x2_lo = -2.0 * math.pi * s
        x2_hi = 6.0 * c
        super().__init__("oka1", dim=2, nobj=2,
                         bounds_lower=[x1_lo, x2_lo],
                         bounds_upper=[x1_hi, x2_hi])
        self._s = s
        self._c = c

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2 = X[:, 0], X[:, 1]
        
        # Rotation
        x1p = self._c * x1 - self._s * x2
        x2p = self._s * x1 + self._c * x2
        
        f1 = x1p
        
        # FIXED: Use math.sqrt for the float constant instead of torch.sqrt
        constant_term = math.sqrt(2.0 * math.pi)
        
        f2 = constant_term - torch.sqrt(torch.abs(x1p)) + \
             2.0 * (x2p - 3.0 * torch.cos(x1p) - 3.0).abs() ** (1.0 / 3.0)
             
        return torch.stack([f1, f2], dim=-1)


# ============================================================
# OKA2: d=3, K=2
# Bounds: x1 in [-pi, pi], x2 in [-5, 5], x3 in [-5, 5]
# Source: Okabe et al. (2004), Knowles (2005)
# ============================================================
class OKA2(TestFunction):
    """OKA2: Spiral-shaped Pareto set, density falls away steeply."""

    def __init__(self):
        super().__init__("oka2", dim=3, nobj=2,
                         bounds_lower=[-math.pi, -5.0, -5.0],
                         bounds_upper=[math.pi, 5.0, 5.0])

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        f1 = x1
        f2 = 1.0 - (x1 + math.pi) ** 2 / (4.0 * math.pi ** 2) + \
             torch.abs(x2 - 5.0 * torch.cos(x1)) ** (1.0 / 3.0) + \
             torch.abs(x3 - 5.0 * torch.sin(x1)) ** (1.0 / 3.0)
        return torch.stack([f1, f2], dim=-1)


# ============================================================
# DTLZ1a: d=6, K=2  (k = d - K + 1 = 5)
# Bounds: x in [0, 1]^6
# Source: Deb et al. (2005), Knowles (2005) config
#
# g(x_M) = 100 * [|x_M| + sum_{x_i in x_M} (x_i-0.5)^2 - cos(20*pi*(x_i-0.5))]
# f_1(x) = 0.5 * x_1 * (1 + g(x_M))
# f_2(x) = 0.5 * (1 - x_1) * (1 + g(x_M))
# PF: convex, f1 + f2 = 0.5
# ============================================================
class DTLZ1a(TestFunction):
    """DTLZ1a: d=6, K=2. Increased function ruggedness (many local PFs)."""

    def __init__(self):
        super().__init__("dtlz1a", dim=6, nobj=2,
                         bounds_lower=[0.0] * 6,
                         bounds_upper=[1.0] * 6)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        M = self.nobj  # 2
        k = self.dim - M + 1  # 5
        x_M = X[:, M - 1:]  # last k variables

        g = 100.0 * (k + ((x_M - 0.5) ** 2 -
                          torch.cos(20.0 * math.pi * (x_M - 0.5))).sum(dim=-1))

        f1 = 0.5 * X[:, 0] * (1.0 + g)
        f2 = 0.5 * (1.0 - X[:, 0]) * (1.0 + g)
        return torch.stack([f1, f2], dim=-1)


# ============================================================
# DTLZ2a: d=8, K=3  (k = d - K + 1 = 6)
# Bounds: x in [0, 1]^8
# Source: Deb et al. (2005), Knowles (2005) config
#
# g(x_M) = sum_{x_i in x_M} (x_i - 0.5)^2
# f_1 = (1+g) * cos(x_1*pi/2) * cos(x_2*pi/2)
# f_2 = (1+g) * cos(x_1*pi/2) * sin(x_2*pi/2)
# f_3 = (1+g) * sin(x_1*pi/2)
# PF: unit sphere in first octant
# ============================================================
class DTLZ2a(TestFunction):
    """DTLZ2a: d=8, K=3. Pareto front is a sphere."""

    def __init__(self):
        super().__init__("dtlz2a", dim=8, nobj=3,
                         bounds_lower=[0.0] * 8,
                         bounds_upper=[1.0] * 8)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        M = self.nobj  # 3
        x_M = X[:, M - 1:]  # last k variables

        g = ((x_M - 0.5) ** 2).sum(dim=-1)

        f1 = (1.0 + g) * torch.cos(X[:, 0] * math.pi / 2.0) * \
             torch.cos(X[:, 1] * math.pi / 2.0)
        f2 = (1.0 + g) * torch.cos(X[:, 0] * math.pi / 2.0) * \
             torch.sin(X[:, 1] * math.pi / 2.0)
        f3 = (1.0 + g) * torch.sin(X[:, 0] * math.pi / 2.0)
        return torch.stack([f1, f2, f3], dim=-1)


# ============================================================
# DTLZ4a: d=8, K=3  (k = d - K + 1 = 6)
# Bounds: x in [0, 1]^8
# Source: Deb et al. (2005), Knowles (2005) config
#
# Same as DTLZ2 but with x_i -> x_i^alpha (alpha=100)
# in the position parameters (first M-1 variables).
# This biases the density of solutions.
# PF: sphere (same as DTLZ2), but biased density
# ============================================================
class DTLZ4a(TestFunction):
    """DTLZ4a: d=8, K=3. Sphere PF with biased solution density (alpha=100)."""

    def __init__(self, alpha: float = 100.0):
        super().__init__("dtlz4a", dim=8, nobj=3,
                         bounds_lower=[0.0] * 8,
                         bounds_upper=[1.0] * 8)
        self.alpha = alpha

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        M = self.nobj  # 3
        x_M = X[:, M - 1:]  # last k variables

        g = ((x_M - 0.5) ** 2).sum(dim=-1)

        # Apply alpha power to position parameters
        x1a = X[:, 0] ** self.alpha
        x2a = X[:, 1] ** self.alpha

        f1 = (1.0 + g) * torch.cos(x1a * math.pi / 2.0) * \
             torch.cos(x2a * math.pi / 2.0)
        f2 = (1.0 + g) * torch.cos(x1a * math.pi / 2.0) * \
             torch.sin(x2a * math.pi / 2.0)
        f3 = (1.0 + g) * torch.sin(x1a * math.pi / 2.0)
        return torch.stack([f1, f2, f3], dim=-1)


# ============================================================
# DTLZ7a: d=8, K=3  (k = d - K + 1 = 6)
# Bounds: x in [0, 1]^8
# Source: Deb et al. (2005), Knowles (2005) config
#
# f_i = x_i  for i = 1, ..., M-1
# g(x_M) = 1 + (9 / |x_M|) * sum_{x_i in x_M} x_i
# h(f_1,...,f_{M-1}, g) = M - sum_{i=1}^{M-1} [f_i/(1+g) * (1 + sin(3*pi*f_i))]
# f_M = (1 + g) * h
# PF: 4 disconnected regions (for K=3)
# ============================================================
class DTLZ7a(TestFunction):
    """DTLZ7a: d=8, K=3. PF consists of 4 disconnected regions."""

    def __init__(self):
        super().__init__("dtlz7a", dim=8, nobj=3,
                         bounds_lower=[0.0] * 8,
                         bounds_upper=[1.0] * 8)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        M = self.nobj  # 3
        k = self.dim - M + 1  # 6
        x_M = X[:, M - 1:]  # last k variables

        g = 1.0 + (9.0 / k) * x_M.sum(dim=-1)

        # f_1 = x_1, f_2 = x_2
        f = [X[:, i] for i in range(M - 1)]

        # h function
        h = float(M)
        for fi in f:
            h = h - (fi / (1.0 + g)) * (1.0 + torch.sin(3.0 * math.pi * fi))

        f_M = (1.0 + g) * h

        f.append(f_M)
        return torch.stack(f, dim=-1)


# ============================================================
# FACTORY
# ============================================================
FUNCTION_MAP = {
    "kno1":   KNO1,
    "vlmop2": VLMOP2,
    "vlmop3": VLMOP3,
    "oka1":   OKA1,
    "oka2":   OKA2,
    "dtlz1a": DTLZ1a,
    "dtlz2a": DTLZ2a,
    "dtlz4a": DTLZ4a,
    "dtlz7a": DTLZ7a,
}


def get_test_function(name: str) -> TestFunction:
    """Factory: return test function by name."""
    name_lower = name.lower()
    if name_lower not in FUNCTION_MAP:
        raise ValueError(
            f"Unknown test function: {name}. "
            f"Available: {list(FUNCTION_MAP.keys())}"
        )
    return FUNCTION_MAP[name_lower]()


# ============================================================
# QUICK VERIFICATION
# ============================================================
if __name__ == "__main__":
    """Quick smoke test: evaluate each function at a random point."""
    print("=" * 60)
    print("TSEMO Test Functions - Verification")
    print("=" * 60)

    for name, cls in FUNCTION_MAP.items():
        tf = cls()
        bounds = tf.get_bounds()
        # Random point in bounds
        X = bounds[0] + torch.rand(5, tf.dim, dtype=DTYPE) * (bounds[1] - bounds[0])
        Y = tf.evaluate(X)

        print(f"\n{name}: d={tf.dim}, K={tf.nobj}")
        print(f"  bounds: [{bounds[0].tolist()}, {bounds[1].tolist()}]")
        print(f"  Y shape: {Y.shape}")
        print(f"  Y[0]: {Y[0].tolist()}")
        print(f"  Y min:  {Y.min(dim=0).values.tolist()}")
        print(f"  Y max:  {Y.max(dim=0).values.tolist()}")