"""Metrics over cached Jev answers.

Usage:
    python -m eval.metrics --answers eval/reports/answers_jev-1.13.0.parquet --t-low 0.2 --t-high 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def decisions(df: pd.DataFrame, t_low: float, t_high: float, c_min: float) -> pd.Series:
    p = df["p_fraud"]
    conf = df["risk_confidence"]
    out = pd.Series("STEP_UP", index=df.index)
    out[p < t_low] = "APPROVE"
    out[(p > t_high) & (conf >= c_min)] = "DECLINE"
    return out


def compute_metrics(df: pd.DataFrame, t_low: float, t_high: float, c_min: float) -> dict:
    y = df["label"].astype(int).to_numpy()
    p = df["p_fraud"].to_numpy()
    dec = decisions(df, t_low, t_high, c_min)

    metrics: dict = {
        "n": int(len(df)),
        "fraud_rate": float(y.mean()),
        "model": sorted(df["model"].unique().tolist()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "risk_score_roc_auc": float(roc_auc_score(y, df["risk_score"])) if len(np.unique(y)) > 1 else None,
        "thresholds": {"t_low": t_low, "t_high": t_high, "c_min": c_min},
    }

    # Binary view at t_high: DECLINE = positive
    pred_decline = (dec == "DECLINE").astype(int).to_numpy()
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred_decline, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred_decline, labels=[0, 1]).ravel()
    metrics["decline"] = {
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Three-way operating point
    counts = dec.value_counts()
    metrics["decision_mix"] = {k: int(counts.get(k, 0)) for k in ("APPROVE", "STEP_UP", "DECLINE")}
    metrics["step_up_rate"] = float((dec == "STEP_UP").mean())
    fraud_mask = y == 1
    metrics["fraud_caught"] = {
        "declined": float(((dec == "DECLINE") & fraud_mask).sum() / max(fraud_mask.sum(), 1)),
        "stepped_up": float(((dec == "STEP_UP") & fraud_mask).sum() / max(fraud_mask.sum(), 1)),
        "approved_missed": float(((dec == "APPROVE") & fraud_mask).sum() / max(fraud_mask.sum(), 1)),
    }
    legit_mask = y == 0
    metrics["legit_friction"] = {
        "approved": float(((dec == "APPROVE") & legit_mask).sum() / max(legit_mask.sum(), 1)),
        "stepped_up": float(((dec == "STEP_UP") & legit_mask).sum() / max(legit_mask.sum(), 1)),
        "declined_false_positive": float(((dec == "DECLINE") & legit_mask).sum() / max(legit_mask.sum(), 1)),
    }

    # Per-pattern breakdown
    per_pattern = (
        df.assign(label=y)
        .groupby("pattern")
        .agg(n=("label", "size"), fraud_rate=("label", "mean"), mean_p=("p_fraud", "mean"))
        .reset_index()
    )
    metrics["per_pattern"] = per_pattern.to_dict(orient="records")

    # Cost / latency
    if "input_tokens" in df:
        metrics["usage"] = {
            "mean_input_tokens": float(df["input_tokens"].dropna().mean())
            if df["input_tokens"].notna().any()
            else None,
            "mean_output_tokens": float(df["output_tokens"].dropna().mean())
            if df["output_tokens"].notna().any()
            else None,
            "p50_latency_ms": float(df["latency_ms"].dropna().quantile(0.5))
            if df["latency_ms"].notna().any()
            else None,
            "p95_latency_ms": float(df["latency_ms"].dropna().quantile(0.95))
            if df["latency_ms"].notna().any()
            else None,
        }
    return metrics


def decline_vs_caught_curve(df: pd.DataFrame, steps: int = 20) -> list[dict]:
    """Trade-off curve: for each t_high, decline rate on legit vs fraud caught."""
    y = df["label"].astype(int)
    rows = []
    for t in np.linspace(0.05, 0.95, steps):
        declined = df["p_fraud"] > t
        rows.append(
            {
                "t_high": round(float(t), 3),
                "legit_declined": float((declined & (y == 0)).sum() / max((y == 0).sum(), 1)),
                "fraud_declined": float((declined & (y == 1)).sum() / max((y == 1).sum(), 1)),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--t-low", type=float, default=0.2)
    ap.add_argument("--t-high", type=float, default=0.8)
    ap.add_argument("--c-min", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.answers)
    report = compute_metrics(df, args.t_low, args.t_high, args.c_min)
    report["curve"] = decline_vs_caught_curve(df)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
