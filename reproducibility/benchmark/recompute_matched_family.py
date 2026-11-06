"""Recompute criterion-matched Table 1 W/T/L counts from the family archive.

The archive contains final hypervolume, not raw objectives or full trajectories.
This script recomputes two-sided paired Wilcoxon tests, Holm-corrected within
problem/family/q panels of 42, 42, or 21 hypotheses, respectively.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATA = Path(__file__).with_name("family_final_hv_20reps.csv.gz")
MATCHED = {"qPOTS": ("qPOTS-DSM-PF", "qPOTS-DSM-DS"),
           "UCB": ("UCB-HVC-PF", "UCB-HVC-DS"),
           "TSEMO": ("TSEMO-HVI",)}


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p_values[order] * (len(p_values) - np.arange(len(p_values)))))
    return adjusted


def main() -> None:
    d = pd.read_csv(DATA)
    if len(d) != 58_320 or d.duplicated(["problem", "q", "family", "method", "rep"]).any():
        raise ValueError("Expected 58,320 unique run-level final-HV values")
    totals: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    for (problem, q, family), panel in d.groupby(["problem", "q", "family"], sort=True):
        baseline = panel[panel.is_baseline]
        variants = panel[~panel.is_baseline]
        if baseline.method.nunique() != 1 or baseline.rep.nunique() != 20:
            raise ValueError(f"Incomplete baseline: {problem}, {q}, {family}")
        base = baseline.sort_values("rep").final_hv.to_numpy()
        comparisons = []
        for (structure, eps), row in variants.groupby(["structure", "epsilon"]):
            row = row.sort_values("rep")
            if len(row) != 20 or not np.array_equal(row.rep.to_numpy(), np.arange(20)):
                raise ValueError(f"Incomplete pairing: {problem}, {q}, {structure}, {eps}")
            differences = row.final_hv.to_numpy() - base
            p = wilcoxon(differences, alternative="two-sided",
                         zero_method="wilcox", method="auto").pvalue
            comparisons.append((structure, differences.mean(), p))
        if len(comparisons) != {"qPOTS": 42, "UCB": 42, "TSEMO": 21}[family]:
            raise ValueError(f"Wrong Holm family: {problem}, {q}, {family}")
        adjusted = holm_adjust(np.array([x[2] for x in comparisons]))
        for (structure, mean, _), p_adj in zip(comparisons, adjusted):
            if structure in MATCHED[family]:
                totals[family, structure, q]["T" if p_adj >= .05 else
                                             ("W" if mean > 0 else "L")] += 1
    for family, structures in MATCHED.items():
        for structure in structures:
            for q in (1, 2, 4):
                c = totals[family, structure, q]
                if sum(c.values()) != 63:
                    raise ValueError(f"Expected 63 outcomes: {family}, {structure}, {q}")
                print(f"{structure:18} q={q}: {c['W']}/{c['T']}/{c['L']}")


if __name__ == "__main__":
    main()
