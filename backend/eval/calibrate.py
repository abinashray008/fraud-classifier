"""Threshold sweep: recommend t_low / t_high for a target step-up rate and miss rate.

Objective: among (t_low, t_high) pairs satisfying
    step_up_rate <= max_step_up_rate   and   fraud_approved_rate <= max_fnr
choose the pair that maximizes fraud declined outright while keeping legit false
declines under `max_fpr`. Falls back to the closest feasible pair.

Usage:
    python -m eval.calibrate --answers eval/reports/answers_jev-1.13.0.parquet \
        --max-step-up 0.15 --max-fnr 0.05 --max-fpr 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import decisions


def sweep(
    df: pd.DataFrame,
    c_min: float,
    grid: np.ndarray,
) -> pd.DataFrame:
    y = df["label"].astype(int).to_numpy()
    rows = []
    for t_low in grid:
        for t_high in grid:
            if t_high < t_low:
                continue
            dec = decisions(df, float(t_low), float(t_high), c_min).to_numpy()
            fraud = y == 1
            legit = y == 0
            rows.append(
                {
                    "t_low": round(float(t_low), 3),
                    "t_high": round(float(t_high), 3),
                    "step_up_rate": float((dec == "STEP_UP").mean()),
                    "fraud_declined": float(((dec == "DECLINE") & fraud).sum() / max(fraud.sum(), 1)),
                    "fraud_approved": float(((dec == "APPROVE") & fraud).sum() / max(fraud.sum(), 1)),
                    "legit_declined": float(((dec == "DECLINE") & legit).sum() / max(legit.sum(), 1)),
                    "legit_stepped_up": float(((dec == "STEP_UP") & legit).sum() / max(legit.sum(), 1)),
                }
            )
    return pd.DataFrame(rows)


def recommend(table: pd.DataFrame, max_step_up: float, max_fnr: float, max_fpr: float) -> tuple[dict, bool]:
    feasible = table[
        (table["step_up_rate"] <= max_step_up)
        & (table["fraud_approved"] <= max_fnr)
        & (table["legit_declined"] <= max_fpr)
    ]
    if len(feasible):
        best = feasible.sort_values(["fraud_declined", "step_up_rate"], ascending=[False, True]).iloc[0]
        return best.to_dict(), True
    # Closest by normalized constraint violation
    viol = (
        np.maximum(table["step_up_rate"] - max_step_up, 0) / max(max_step_up, 1e-9)
        + np.maximum(table["fraud_approved"] - max_fnr, 0) / max(max_fnr, 1e-9)
        + np.maximum(table["legit_declined"] - max_fpr, 0) / max(max_fpr, 1e-9)
    )
    best = table.iloc[int(viol.idxmin())]
    return best.to_dict(), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--c-min", type=float, default=0.6)
    ap.add_argument("--max-step-up", type=float, default=0.15)
    ap.add_argument("--max-fnr", type=float, default=0.05, help="max share of fraud auto-approved")
    ap.add_argument("--max-fpr", type=float, default=0.01, help="max share of legit auto-declined")
    ap.add_argument("--grid-steps", type=int, default=39)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.answers)
    grid = np.linspace(0.025, 0.975, args.grid_steps)
    table = sweep(df, args.c_min, grid)
    best, feasible = recommend(table, args.max_step_up, args.max_fnr, args.max_fpr)

    result = {
        "model": sorted(df["model"].unique().tolist()),
        "constraints": {
            "max_step_up": args.max_step_up,
            "max_fnr": args.max_fnr,
            "max_fpr": args.max_fpr,
            "c_min": args.c_min,
        },
        "feasible": feasible,
        "recommended": best,
        "env": (f"POLICY_T_LOW={best['t_low']}\nPOLICY_T_HIGH={best['t_high']}\nPOLICY_C_MIN={args.c_min}"),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        table.to_csv(args.out.with_suffix(".sweep.csv"), index=False)


if __name__ == "__main__":
    main()
