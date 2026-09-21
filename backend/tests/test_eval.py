"""Eval harness tests on synthetic IEEE-CIS-shaped data (no network, no dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.calibrate import recommend, sweep
from eval.load_ieee import row_to_transaction, stratified_sample
from eval.metrics import compute_metrics, decisions
from eval.regression import compare, flatten


def _ieee_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "isFraud": rng.integers(0, 2, n),
            "TransactionDT": rng.integers(86400, 86400 * 30, n),
            "TransactionAmt": rng.uniform(1, 500, n).round(2),
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "card4": rng.choice(["visa", "mastercard", None], n),
            "card6": rng.choice(["debit", "credit"], n),
            "addr1": rng.choice([315.0, 204.0, np.nan], n),
            "addr2": 87.0,
            "dist1": rng.choice([3.0, 120.0, np.nan], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "R_emaildomain": None,
            "C1": rng.integers(1, 10, n).astype(float),
            "C2": 1.0,
            "C13": 2.0,
            "C14": 1.0,
            "D1": rng.choice([0.0, 14.0, np.nan], n),
            "D2": np.nan,
            "D4": np.nan,
            "D10": np.nan,
            "D15": rng.choice([0.0, 300.0, np.nan], n),
            "DeviceType": rng.choice(["mobile", "desktop", None], n),
            "DeviceInfo": rng.choice(["iOS Device", "Windows", None], n),
            "card_id": [f"card_{i % 20}" for i in range(n)],
            "card_avg_amount": 120.0,
        }
    )


def _answers(df: pd.DataFrame, seed: int = 0, quality: float = 0.8) -> pd.DataFrame:
    """Synthetic Jev answers correlated with the label."""
    rng = np.random.default_rng(seed)
    y = df["isFraud"].to_numpy()
    noise = rng.uniform(0, 1, len(df))
    p = np.clip(quality * y + (1 - quality) * noise + rng.normal(0, 0.05, len(df)), 0, 1)
    return pd.DataFrame(
        {
            "transaction_id": df["TransactionID"].astype(str),
            "label": y,
            "model": "jev-test",
            "p_fraud": p,
            "risk_score": p * 4,
            "risk_confidence": rng.uniform(0.5, 1.0, len(df)),
            "pattern": np.where(y == 1, "stolen_card", "legitimate"),
            "pattern_confidence": 0.9,
            "input_tokens": 150,
            "output_tokens": 40,
            "latency_ms": rng.uniform(30, 90, len(df)),
        }
    )


def test_row_to_transaction_handles_nan_and_aliases():
    df = _ieee_frame(5)
    df.loc[0, "card4"] = None
    tx = row_to_transaction(df.iloc[0])
    assert tx.transaction_id == "1"
    assert tx.amount == df.loc[0, "TransactionAmt"]
    assert tx.card_avg_amount == 120.0
    assert tx.recipient_email_domain is None


def test_stratified_sample_respects_fraud_frac():
    df = _ieee_frame(400)
    s = stratified_sample(df, n=100, fraud_frac=0.3, seed=1)
    assert len(s) == 100
    assert s["isFraud"].sum() == 30


def test_decisions_three_way():
    df = pd.DataFrame({"p_fraud": [0.1, 0.5, 0.9, 0.9], "risk_confidence": [0.9, 0.9, 0.9, 0.2]})
    assert decisions(df, 0.2, 0.8, 0.6).tolist() == ["APPROVE", "STEP_UP", "DECLINE", "STEP_UP"]


def test_compute_metrics_shape():
    df = _ieee_frame(300)
    ans = _answers(df)
    m = compute_metrics(ans, 0.2, 0.8, 0.6)
    assert 0.5 < m["roc_auc"] <= 1.0
    assert sum(m["decision_mix"].values()) == 300
    caught = m["fraud_caught"]
    assert pytest.approx(caught["declined"] + caught["stepped_up"] + caught["approved_missed"], abs=1e-9) == 1.0
    assert {r["pattern"] for r in m["per_pattern"]} == {"stolen_card", "legitimate"}
    assert m["usage"]["p95_latency_ms"] is not None


def test_calibrate_finds_feasible_thresholds_for_good_model():
    df = _ieee_frame(500)
    ans = _answers(df, quality=0.9)
    table = sweep(ans, c_min=0.0, grid=np.linspace(0.05, 0.95, 19))
    best, feasible = recommend(table, max_step_up=0.5, max_fnr=0.1, max_fpr=0.1)
    assert feasible
    assert best["t_low"] <= best["t_high"]
    assert best["fraud_approved"] <= 0.1


def test_calibrate_falls_back_when_infeasible():
    df = _ieee_frame(200)
    ans = _answers(df, quality=0.2)
    table = sweep(ans, c_min=0.0, grid=np.linspace(0.05, 0.95, 10))
    best, feasible = recommend(table, max_step_up=0.0, max_fnr=0.0, max_fpr=0.0)
    assert not feasible
    assert "t_low" in best


def test_regression_detects_drift():
    df = _ieee_frame(400)
    good = flatten(compute_metrics(_answers(df, quality=0.9), 0.2, 0.8, 0.6))
    bad = flatten(compute_metrics(_answers(df, seed=3, quality=0.3), 0.2, 0.8, 0.6))
    rows, ok = compare(
        good,
        bad,
        {
            "roc_auc": 0.02,
            "pr_auc": 0.03,
            "step_up_rate": 0.05,
            "fraud_approved_missed": 0.02,
            "legit_declined_false_positive": 0.01,
        },
    )
    assert not ok
    assert any(r["metric"] == "roc_auc" and r["status"] == "FAIL" for r in rows)

    rows, ok = compare(
        good,
        good,
        {
            "roc_auc": 0.02,
            "pr_auc": 0.03,
            "step_up_rate": 0.05,
            "fraud_approved_missed": 0.02,
            "legit_declined_false_positive": 0.01,
        },
    )
    assert ok
