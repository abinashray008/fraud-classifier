"""Select action thresholds t_low / t_high. This does not calibrate probabilities.

Whether `is_fraud.noul` matches observed fraud rates is `python -m eval.calibration`
(Brier score and a reliability diagram).

Objective: among (t_low, t_high) pairs satisfying
    step_up_rate <= max_step_up_rate   and   fraud_approved_rate <= max_fnr
choose the pair that maximizes fraud declined outright while keeping legit false
declines under `max_fpr`. Falls back to the closest feasible pair.

Decisions use the live policy (contradictory or thin answers go to review).
The report also measures the historical extra gate that required
`risk.confidence >= c_min` before a decline. That confidence is how concentrated
the risk-level distribution is. It does not establish that the fraud answer is
correct, and the gate can only turn declines into step-ups.

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

from app.policy.decision import PolicyThresholds
from eval.metrics import confidence_gate_effect, decisions, frame_for, sample_weights


def sweep(
    df: pd.DataFrame,
    grid: np.ndarray,
    th: PolicyThresholds | None = None,
) -> pd.DataFrame:
    y = df["label"].astype(int).to_numpy()
    w = sample_weights(df)
    fraud = y == 1
    legit = y == 0

    def rate(mask: np.ndarray, group: np.ndarray | None = None) -> float:
        if group is not None:
            mask = mask & group
            base = group
        else:
            base = np.ones(len(mask), dtype=bool)
        if w is None:
            denom = float(base.sum())
            return float(mask.sum() / denom) if denom else 0.0
        denom = float(w[base].sum())
        return float(w[mask].sum() / denom) if denom else 0.0

    rows = []
    for t_low in grid:
        for t_high in grid:
            if t_high < t_low:
                continue
            dec = decisions(df, float(t_low), float(t_high), th).to_numpy()
            rows.append(
                {
                    "t_low": round(float(t_low), 3),
                    "t_high": round(float(t_high), 3),
                    "step_up_rate": rate(dec == "STEP_UP"),
                    "fraud_declined": rate(dec == "DECLINE", fraud),
                    "fraud_approved": rate(dec == "APPROVE", fraud),
                    "legit_declined": rate(dec == "DECLINE", legit),
                    "legit_stepped_up": rate(dec == "STEP_UP", legit),
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
    ap.add_argument(
        "--c-min",
        type=float,
        default=0.6,
        help="historical risk.confidence gate to measure; not written into the recommended policy",
    )
    ap.add_argument("--evidence-min", type=float, default=0.5)
    ap.add_argument("--max-step-up", type=float, default=0.15)
    ap.add_argument("--max-fnr", type=float, default=0.05, help="max share of fraud auto-approved")
    ap.add_argument("--max-fpr", type=float, default=0.01, help="max share of legit auto-declined")
    ap.add_argument("--grid-steps", type=int, default=39)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = frame_for(pd.read_parquet(args.answers), "calibration")
    grid = np.linspace(0.025, 0.975, args.grid_steps)
    th = PolicyThresholds(evidence_min=args.evidence_min)
    table = sweep(df, grid, th)
    best, feasible = recommend(table, args.max_step_up, args.max_fnr, args.max_fpr)
    gate = confidence_gate_effect(df, best["t_low"], best["t_high"], args.c_min)

    result = {
        "note": (
            "Selects action thresholds. It does not calibrate is_fraud.noul; "
            "run python -m eval.calibration for a Brier score and reliability diagram."
        ),
        "model": sorted(df["model"].unique().tolist()),
        "constraints": {
            "max_step_up": args.max_step_up,
            "max_fnr": args.max_fnr,
            "max_fpr": args.max_fpr,
            "evidence_min": args.evidence_min,
        },
        "feasible": feasible,
        "recommended": best,
        "confidence_gate": gate,
        "env": (
            f"POLICY_T_LOW={best['t_low']}\nPOLICY_T_HIGH={best['t_high']}\nPOLICY_EVIDENCE_MIN={args.evidence_min}"
        ),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        table.to_csv(args.out.with_suffix(".sweep.csv"), index=False)


if __name__ == "__main__":
    main()
