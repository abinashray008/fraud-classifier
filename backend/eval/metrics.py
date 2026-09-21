"""Metrics over cached Jev answers.

Decisions follow the live policy in `app.policy.decision`: fraud probability
against action thresholds, with contradictory or thin answers sent to review.
`risk.confidence` is not a decline authorizer. `confidence_gate_effect` measures
the historical gate that required it.

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

from app.policy.decision import FRAUD_PATTERNS, PolicyThresholds
from eval.calibration import probability_calibration


def decisions(df: pd.DataFrame, t_low: float, t_high: float, th: PolicyThresholds | None = None) -> pd.Series:
    """Live policy: probability thresholds, then review on contradiction or thin evidence.

    `t_low` / `t_high` set the action thresholds. Other fields (evidence floor,
    risk-score bands) come from `th` when given, otherwise the policy defaults.
    Mirrors `app.policy.decision.decide`. A parity test keeps the two in step.
    """
    base = th or PolicyThresholds()
    th = PolicyThresholds(
        t_low=t_low,
        t_high=t_high,
        evidence_min=base.evidence_min,
        low_risk_max=base.low_risk_max,
        high_risk_min=base.high_risk_min,
        signal_support=base.signal_support,
    )
    p = df["p_fraud"]
    score = df["risk_score"]
    conf = df["risk_confidence"].fillna(0)
    pattern = df["pattern"]
    contradict = (
        ((p > th.t_high) & (score < th.low_risk_max))
        | ((p < th.t_low) & (score >= th.high_risk_min))
        | ((p > th.t_high) & (pattern == "legitimate"))
        | ((p < th.t_low) & pattern.isin(FRAUD_PATTERNS))
    )
    auto = (p < th.t_low) | (p > th.t_high)
    thin = conf < th.evidence_min
    if "pattern_confidence" in df.columns:
        thin = thin | (df["pattern_confidence"].fillna(1.0) < th.evidence_min)
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    if signal_cols:
        signal_max = df[signal_cols].max(axis=1, skipna=True)
        thin = thin | ((p > th.t_high) & signal_max.notna() & (signal_max < th.signal_support))
    review = contradict | (auto & thin & ~contradict)

    out = pd.Series("STEP_UP", index=df.index)
    out[p < th.t_low] = "APPROVE"
    out[p > th.t_high] = "DECLINE"
    out[review] = "STEP_UP"
    return out


def decisions_probability_only(df: pd.DataFrame, t_low: float, t_high: float) -> pd.Series:
    """Decline exactly when the fraud probability clears t_high. No other gate."""
    p = df["p_fraud"]
    out = pd.Series("STEP_UP", index=df.index)
    out[p < t_low] = "APPROVE"
    out[p > t_high] = "DECLINE"
    return out


def decisions_confidence_gated(df: pd.DataFrame, t_low: float, t_high: float, c_min: float) -> pd.Series:
    """Historical rule: a decline also required risk.confidence >= c_min.

    That confidence is concentration of the risk-level distribution. Kept only
    so `confidence_gate_effect` can measure whether the extra gate helps.
    """
    p = df["p_fraud"]
    conf = df["risk_confidence"]
    out = pd.Series("STEP_UP", index=df.index)
    out[p < t_low] = "APPROVE"
    out[(p > t_high) & (conf >= c_min)] = "DECLINE"
    return out


def sample_weights(df: pd.DataFrame) -> np.ndarray | None:
    """Per-row weights that restore the source period's class mix. None if absent."""
    if "sample_weight" not in df.columns:
        return None
    w = df["sample_weight"].to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("sample_weight must be finite and positive")
    return w


def frame_for(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Keep one chronological split. Frames with no split column are returned as-is."""
    if "split" not in df.columns:
        return df
    out = df[df["split"] == split]
    if out.empty:
        raise ValueError(f"answers have a split column but no {split!r} rows")
    return out


def _share(event: np.ndarray, group: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        denom = float(group.sum())
        return float(event[group.astype(bool)].sum() / denom) if denom else 0.0
    denom = float(w[group.astype(bool)].sum())
    return float(w[event.astype(bool) & group.astype(bool)].sum() / denom) if denom else 0.0


def _operating_point(df: pd.DataFrame, dec: pd.Series) -> dict:
    y = df["label"].astype(int).to_numpy()
    w = sample_weights(df)
    d = dec.to_numpy()
    fraud = y == 1
    legit = y == 0
    declined = d == "DECLINE"
    step = d == "STEP_UP"
    precision = _share(fraud, declined, w) if declined.any() else None
    return {
        "step_up_rate": float(np.average(step, weights=w)) if w is not None else float(step.mean()),
        "fraud_declined": _share(declined, fraud, w),
        "legit_declined": _share(declined, legit, w),
        "decline_precision": precision,
    }


def confidence_gate_effect(df: pd.DataFrame, t_low: float, t_high: float, c_min: float) -> dict:
    """Does requiring risk.confidence >= c_min before a decline improve outcomes?

    The gate only moves rows with p > t_high and confidence < c_min from DECLINE
    to STEP_UP. It cannot catch more fraud. It improves the decline set only
    when the rows it removes are less often fraud than the declines it keeps.
    """
    y = df["label"].astype(int).to_numpy()
    w = sample_weights(df)
    only = decisions_probability_only(df, t_low, t_high)
    gated = decisions_confidence_gated(df, t_low, t_high, c_min)
    only_pt = _operating_point(df, only)
    gated_pt = _operating_point(df, gated)
    moved = (only.to_numpy() == "DECLINE") & (gated.to_numpy() != "DECLINE")
    kept = gated.to_numpy() == "DECLINE"
    moved_fraud_rate = _share(y == 1, moved, w) if moved.any() else None
    kept_precision = _share(y == 1, kept, w) if kept.any() else None
    if moved_fraud_rate is None:
        removes_more_legitimate = None
    elif kept_precision is None:
        removes_more_legitimate = False
    else:
        removes_more_legitimate = moved_fraud_rate < kept_precision

    def _delta(key: str) -> float | None:
        a, b = only_pt[key], gated_pt[key]
        if a is None or b is None:
            return None
        return float(b - a)

    return {
        "c_min": c_min,
        "note": (
            "risk.confidence is how concentrated the risk-level distribution is. "
            "It does not establish that the fraud answer is correct. "
            "The gate improves the decline set only when removes_disproportionately_legitimate is true."
        ),
        "rows_moved": int(moved.sum()),
        "moved_fraud_rate": moved_fraud_rate,
        "kept_decline_precision": kept_precision,
        "removes_disproportionately_legitimate": removes_more_legitimate,
        "probability_only": only_pt,
        "confidence_gated": gated_pt,
        "delta_gated_minus_probability_only": {
            "step_up_rate": _delta("step_up_rate"),
            "fraud_declined": _delta("fraud_declined"),
            "legit_declined": _delta("legit_declined"),
            "decline_precision": _delta("decline_precision"),
        },
    }


def compute_metrics(
    df: pd.DataFrame,
    t_low: float,
    t_high: float,
    *,
    confidence_gate_c_min: float = 0.6,
) -> dict:
    y = df["label"].astype(int).to_numpy()
    p = df["p_fraud"].to_numpy()
    w = sample_weights(df)
    dec = decisions(df, t_low, t_high)
    kw = {"sample_weight": w} if w is not None else {}

    metrics: dict = {
        "n": int(len(df)),
        "fraud_rate": float(np.average(y, weights=w)) if w is not None else float(y.mean()),
        "prevalence_weighted": w is not None,
        "model": sorted(df["model"].unique().tolist()),
        "roc_auc": float(roc_auc_score(y, p, **kw)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p, **kw)) if len(np.unique(y)) > 1 else None,
        "risk_score_roc_auc": float(roc_auc_score(y, df["risk_score"], **kw)) if len(np.unique(y)) > 1 else None,
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "probability_calibration": probability_calibration(df),
        "confidence_gate": confidence_gate_effect(df, t_low, t_high, confidence_gate_c_min),
    }

    # Binary view at t_high: DECLINE = positive. Weights restore dataset prevalence.
    pred_decline = (dec == "DECLINE").astype(int).to_numpy()
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred_decline, average="binary", zero_division=0, **kw)
    tn, fp, fn, tp = confusion_matrix(y, pred_decline, labels=[0, 1], **kw).ravel()
    metrics["decline"] = {
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "confusion": {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)},
    }

    # Three-way operating point. decision_mix is the scored rows; rates are weighted.
    counts = dec.value_counts()
    metrics["decision_mix"] = {k: int(counts.get(k, 0)) for k in ("APPROVE", "STEP_UP", "DECLINE")}
    step = (dec == "STEP_UP").to_numpy()
    metrics["step_up_rate"] = float(np.average(step, weights=w)) if w is not None else float(step.mean())
    fraud_mask = y == 1
    legit_mask = y == 0
    metrics["fraud_caught"] = {
        "declined": _share((dec == "DECLINE").to_numpy(), fraud_mask, w),
        "stepped_up": _share((dec == "STEP_UP").to_numpy(), fraud_mask, w),
        "approved_missed": _share((dec == "APPROVE").to_numpy(), fraud_mask, w),
    }
    metrics["legit_friction"] = {
        "approved": _share((dec == "APPROVE").to_numpy(), legit_mask, w),
        "stepped_up": _share((dec == "STEP_UP").to_numpy(), legit_mask, w),
        "declined_false_positive": _share((dec == "DECLINE").to_numpy(), legit_mask, w),
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
    ap.add_argument(
        "--c-min",
        type=float,
        default=0.6,
        help="historical risk.confidence gate to measure; not used to authorize declines",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from eval.calibration import reliability_svg

    df = frame_for(pd.read_parquet(args.answers), "test")
    report = compute_metrics(df, args.t_low, args.t_high, confidence_gate_c_min=args.c_min)
    report["curve"] = decline_vs_caught_curve(df)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        svg_path = args.out.with_suffix(".reliability.svg")
        svg_path.write_text(reliability_svg(report["probability_calibration"]["reliability"]))


if __name__ == "__main__":
    main()
