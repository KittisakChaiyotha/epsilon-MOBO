#!/usr/bin/env python3
"""Numerical smoke checks for the archived nine-function definitions."""

import math

import torch

from test_functions import FUNCTION_MAP, get_test_function


def main():
    expected = {
        "kno1": (2, 2), "vlmop2": (2, 2), "vlmop3": (2, 3),
        "oka1": (2, 2), "oka2": (3, 2), "dtlz1a": (6, 2),
        "dtlz2a": (8, 3), "dtlz4a": (8, 3), "dtlz7a": (8, 3),
    }
    assert set(FUNCTION_MAP) == set(expected)
    for name, (d, k) in expected.items():
        f = get_test_function(name)
        bounds = f.get_bounds()
        x = torch.stack((bounds[0], (bounds[0] + bounds[1]) / 2, bounds[1]))
        y = f.evaluate(x)
        assert (f.dim, f.nobj) == (d, k)
        assert tuple(bounds.shape) == (2, d)
        assert tuple(y.shape) == (3, k) and torch.isfinite(y).all()

    # Distinguishes the archived KNO1 definition from the root demo's square.
    f = get_test_function("kno1")
    y = f.evaluate(torch.tensor([[1.0, 1.0]], dtype=torch.double))[0]
    r = 2.0
    c = 9 - (3 * math.sin(2.5 * math.sqrt(r)) + 3 * math.sin(4 * r)
             + 5 * math.sin(2 * r + 2))
    theta = math.pi / 4
    target = torch.tensor([20 - c * math.cos(theta), 20 - c * math.sin(theta)],
                          dtype=torch.double)
    assert torch.allclose(y, target, atol=1e-6)

    half6 = torch.full((1, 6), 0.5, dtype=torch.double)
    assert torch.allclose(get_test_function("dtlz1a").evaluate(half6),
                          torch.tensor([[0.25, 0.25]], dtype=torch.double))
    x8 = torch.cat((torch.zeros((1, 2), dtype=torch.double),
                    torch.full((1, 6), 0.5, dtype=torch.double)), dim=1)
    for name in ("dtlz2a", "dtlz4a"):
        assert torch.allclose(get_test_function(name).evaluate(x8),
                              torch.tensor([[1., 0., 0.]], dtype=torch.double), atol=1e-12)
    assert torch.allclose(get_test_function("dtlz7a").evaluate(
                              torch.zeros((1, 8), dtype=torch.double)),
                          torch.tensor([[0., 0., 6.]], dtype=torch.double))
    print("PASS: all nine functions and analytical anchors")


if __name__ == "__main__":
    main()
