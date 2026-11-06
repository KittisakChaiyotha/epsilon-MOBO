"""Recompute manuscript Table 5 from saved final nominal HV for 20 paired runs.

This post-processing program does not run Bayesian optimization. It keeps the
same representative selections as the manuscript, then ranks the eight
methods within each paired repetition at each batch size (1 is best).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV = Path(__file__).with_name("final_hv_runs.csv.gz")
SELECTED = {
    1: ["QPHVI_PFexp_eps10", "UCBHVI_PFexp_eps10", "UCBHVC_PFexp_eps20",
        "QPHVC_PFexp_eps20", "BASELINE_UCB_HVC", "QPDSM_PFexp_eps99",
        "UCBDSM_PFexp_eps50", "BASELINE_qPOTS"],
    2: ["QPHVI_PFexp_eps10", "UCBHVI_PFexp_eps10", "QPHVC_PFexp_eps20",
        "BASELINE_UCB_HVC", "UCBHVC_PFexp_eps10", "QPDSM_PFexp_eps99",
        "UCBDSM_PFexp_eps50", "BASELINE_qPOTS"],
    4: ["QPHVI_PFexp_eps10", "UCBHVI_PFexp_eps10", "QPHVC_PFexp_eps10",
        "BASELINE_UCB_HVC", "UCBHVC_PFexp_eps10", "QPDSM_PFexp_eps90",
        "UCBDSM_PFexp_eps20", "BASELINE_qPOTS"],
}


def holm_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
        p[order] * (len(p) - np.arange(len(p)))))
    return adjusted


def main() -> None:
    data = pd.read_csv(CSV)
    if len(data) != 3360 or data.duplicated(["q", "method_id", "rep"]).any():
        raise ValueError("Expected 3,360 unique method/q/repetition records")
    for q, methods in SELECTED.items():
        subset = data.loc[(data.q == q) & data.method_id.isin(methods)]
        hv = subset.pivot(index="rep", columns="method_id", values="final_hv")
        if hv.shape != (20, 8) or hv.isna().any().any() or not (subset.n_points == 326).all():
            raise ValueError(f"Incomplete or invalid paired records at q={q}")
        hv = hv[methods]
        avg = hv.rank(axis=1, ascending=False, method="average").mean()
        pairs = list(combinations(methods, 2))
        p = [wilcoxon(hv[a].to_numpy() - hv[b].to_numpy(),
                      alternative="two-sided", zero_method="wilcox",
                      method="auto").pvalue for a, b in pairs]
        adjusted = holm_adjust(p)
        counts = {method: [0, 0, 0] for method in methods}
        for (a, b), p_holm in zip(pairs, adjusted):
            if p_holm >= .05:
                counts[a][1] += 1
                counts[b][1] += 1
            elif hv[a].mean() > hv[b].mean():
                counts[a][0] += 1
                counts[b][2] += 1
            else:
                counts[a][2] += 1
                counts[b][0] += 1
        rows = pd.DataFrame([
            (method, avg[method], hv[method].mean() * 1_000,
             "/".join(map(str, counts[method])), counts[method][0] - counts[method][2])
            for method in methods
        ], columns=["method_id", "avg_rank", "1000_mean_final_hv", "W/T/L", "net"])
        rows = rows.sort_values(["avg_rank", "1000_mean_final_hv"],
                                ascending=[True, False])
        print(f"q={q}; 20 paired repetitions; Holm over 28 two-sided pairs")
        print(rows.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()


if __name__ == "__main__":
    main()
