"""Eval harness tests on synthetic IEEE-CIS-shaped data (no network, no dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.policy.decision import PolicyThresholds, decide
from eval.calibrate import recommend, sweep
from eval.calibration import (
    brier_score,
    expected_calibration_error,
    probability_calibration,
    reliability_svg,
    reliability_table,
)
from eval.load_ieee import (
    SPLIT_CALIBRATION,
    SPLIT_DEVELOPMENT,
    SPLIT_TEST,
    add_card_history_features,
    build_eval_samples,
    row_history,
    row_to_transaction,
    sample_period,
)
from eval.metrics import (
    compute_metrics,
    confidence_gate_effect,
    decisions,
    decisions_confidence_gated,
    frame_for,
)
from eval.regression import compare, flatten
from tests.conftest import make_answers


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
            "risk_score": np.where(y == 1, 3.6, 0.4),
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
    assert tx.c14 == 1.0
    d15 = df.loc[0, "D15"]
    if pd.isna(d15):
        assert tx.d15 is None
    else:
        assert tx.d15 == d15
    assert not hasattr(tx, "device_txn_count")


def test_card_history_features_are_time_respecting():
    df = pd.DataFrame(
        {
            "card_id": ["a", "a", "a", "b"],
            "TransactionDT": [0, 3600, 86400, 0],
            "TransactionAmt": [10.0, 10.0, 50.0, 5.0],
            "DeviceInfo": ["SM-A Build/1", "SM-A Build/1", "SM-B Build/2", None],
            "P_emaildomain": ["gmail.com", "gmail.com", "gmail.com", "yahoo.com"],
        }
    )
    out = add_card_history_features(df)
    first, second, third, other = (out.iloc[i] for i in range(4))
    assert first["prior_transaction_count"] == 0
    assert pd.isna(first["days_since_previous_transaction"])
    assert second["prior_transaction_count"] == 1
    assert second["transactions_last_1h"] == 1
    assert second["matching_amount_count_last_24h"] == 1
    assert bool(second["device_seen_before_on_this_card"]) is True
    assert third["prior_transaction_count"] == 2
    assert third["transactions_last_24h"] == 2
    assert bool(third["device_seen_before_on_this_card"]) is False
    assert other["prior_transaction_count"] == 0
    hist = row_history(second)
    assert hist["prior_transaction_count"] == 1
    assert "C14" not in hist


def test_future_amount_does_not_change_earlier_card_average():
    """A full-card mean leaks: (10+1000)/2 = 505, and editing the later row makes it 5005."""
    base = pd.DataFrame(
        {
            "card_id": ["card_a", "card_a"],
            "TransactionDT": [0, 10_000],
            "TransactionAmt": [10.0, 1000.0],
        }
    )
    leaked = base.groupby("card_id")["TransactionAmt"].transform("mean")
    assert leaked.iloc[0] == 505.0
    changed = base.copy()
    changed.loc[1, "TransactionAmt"] = 10_000.0
    leaked_after = changed.groupby("card_id")["TransactionAmt"].transform("mean")
    assert leaked_after.iloc[0] == 5005.0

    before = add_card_history_features(base)
    after = add_card_history_features(changed)
    assert pd.isna(before.iloc[0]["mean_amount_usd_prior"])
    assert pd.isna(after.iloc[0]["mean_amount_usd_prior"])
    assert before.iloc[0]["prior_transaction_count"] == 0
    assert before.iloc[1]["mean_amount_usd_prior"] == 10.0
    assert after.iloc[1]["mean_amount_usd_prior"] == 10.0


def test_history_keeps_events_that_sampling_would_drop():
    df = pd.DataFrame(
        {
            "card_id": ["card_a", "card_a", "card_a"],
            "TransactionDT": [0, 100, 200],
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    )
    featured = add_card_history_features(df)
    last = featured.iloc[-1]
    assert last["prior_transaction_count"] == 2
    assert last["mean_amount_usd_prior"] == 15.0


def test_chronological_splits_do_not_overlap():
    df = _ieee_frame(300, seed=2)
    sample = build_eval_samples(df, n=len(df), fraud_frac=None, seed=2)
    dev = sample.loc[sample["split"] == SPLIT_DEVELOPMENT, "TransactionDT"]
    cal = sample.loc[sample["split"] == SPLIT_CALIBRATION, "TransactionDT"]
    test = sample.loc[sample["split"] == SPLIT_TEST, "TransactionDT"]
    assert len(dev) and len(cal) and len(test)
    assert dev.max() < cal.min()
    assert cal.max() < test.min()
    assert set(sample["split"]) <= {SPLIT_DEVELOPMENT, SPLIT_CALIBRATION, SPLIT_TEST}


def test_representative_sample_keeps_prevalence_and_weights_undo_oversample():
    # 35 / 1000 = 3.5%, the IEEE-CIS train prevalence. A 30% fraud draw would
    # report a different precision and step-up rate unless sample_weight is applied.
    df = _ieee_frame(1000, seed=1)
    df["isFraud"] = 0
    df.loc[:34, "isFraud"] = 1
    assert df["isFraud"].mean() == pytest.approx(0.035)

    natural = sample_period(df, n=200, fraud_frac=None, seed=1)
    assert natural["isFraud"].mean() < 0.12
    assert natural["sample_weight"].nunique() == 1

    over = sample_period(df, n=100, fraud_frac=0.30, seed=1)
    assert over["isFraud"].mean() == pytest.approx(0.30)
    weighted = np.average(over["isFraud"].astype(float), weights=over["sample_weight"])
    assert weighted == pytest.approx(df["isFraud"].mean())

    split_sample = build_eval_samples(df, n=300, fraud_frac=0.30, seed=1)
    split_weighted = np.average(split_sample["isFraud"].astype(float), weights=split_sample["sample_weight"])
    assert split_weighted == pytest.approx(df["isFraud"].mean())


def test_metrics_and_calibration_use_separate_periods():
    rows = []
    for split, label, p in (
        (SPLIT_CALIBRATION, 1, 0.95),
        (SPLIT_CALIBRATION, 0, 0.05),
        (SPLIT_TEST, 1, 0.05),
        (SPLIT_TEST, 0, 0.95),
    ):
        rows.append(
            {
                "label": label,
                "p_fraud": p,
                "risk_score": p * 4,
                "risk_confidence": 0.9,
                "pattern": "stolen_card" if label else "legitimate",
                "pattern_confidence": 0.9,
                "model": "jev-test",
                "split": split,
            }
        )
    df = pd.DataFrame(rows)
    cal = frame_for(df, SPLIT_CALIBRATION)
    test = frame_for(df, SPLIT_TEST)
    assert set(cal["split"]) == {SPLIT_CALIBRATION}
    assert set(test["split"]) == {SPLIT_TEST}
    # Calibration separates perfectly; the untouched test is inverted.
    assert compute_metrics(cal, 0.2, 0.8)["roc_auc"] == 1.0
    assert compute_metrics(test, 0.2, 0.8)["roc_auc"] < 0.5
    table = sweep(cal, grid=np.linspace(0.2, 0.8, 4))
    best, feasible = recommend(table, max_step_up=0.5, max_fnr=0.01, max_fpr=0.01)
    assert feasible
    assert best["fraud_approved"] == 0.0


def test_weights_restore_prevalence_in_reported_rates():
    # One fraud and one legit, both declined. Unweighted precision is 0.5;
    # weights at the dataset rate make precision and the fraud rate 3.5%.
    df = pd.DataFrame(
        {
            "label": [1, 0],
            "p_fraud": [0.95, 0.95],
            "risk_score": [3.8, 3.8],
            "risk_confidence": [0.9, 0.9],
            "pattern": ["stolen_card", "stolen_card"],
            "pattern_confidence": [0.9, 0.9],
            "model": ["jev-test", "jev-test"],
            "sample_weight": [3.5, 96.5],
            "input_tokens": [1, 1],
            "output_tokens": [1, 1],
            "latency_ms": [10.0, 10.0],
        }
    )
    m = compute_metrics(df, 0.2, 0.8)
    assert m["fraud_rate"] == pytest.approx(0.035)
    assert m["decline"]["precision"] == pytest.approx(0.035)
    assert m["prevalence_weighted"] is True


def test_decisions_three_way():
    df = pd.DataFrame(
        {
            "p_fraud": [0.1, 0.5, 0.9, 0.9, 0.9],
            "risk_score": [0.4, 2.0, 3.6, 3.6, 3.6],
            "risk_confidence": [0.9, 0.9, 0.9, 0.2, 0.95],
            "pattern": ["legitimate", "other", "stolen_card", "stolen_card", "legitimate"],
            "pattern_confidence": [0.9, 0.9, 0.9, 0.9, 0.95],
        }
    )
    assert decisions(df, 0.2, 0.8).tolist() == [
        "APPROVE",
        "STEP_UP",
        "DECLINE",
        "STEP_UP",  # spread risk distribution
        "STEP_UP",  # pattern contradicts the fraud probability
    ]


def test_eval_decisions_match_policy():
    cases = [
        (0.05, 0.9, "legitimate", 0.2, None),
        (0.5, 0.9, "other", 2.0, None),
        (0.95, 0.9, "stolen_card", 3.8, 0.9),
        (0.95, 0.2, "stolen_card", 3.8, 0.1),
        (0.95, 0.95, "legitimate", 3.8, None),
        (0.05, 0.95, "stolen_card", 0.3, None),
    ]
    th = PolicyThresholds(t_low=0.2, t_high=0.8)
    for p, conf, pattern, score, signal in cases:
        signals = None if signal is None else {"amount_anomalous": signal}
        answers = make_answers(p, risk_conf=conf, pattern=pattern, risk_score=score, signals=signals)
        outcome, _ = decide(answers, th)
        row = {
            "p_fraud": p,
            "risk_score": score,
            "risk_confidence": conf,
            "pattern": pattern,
            "pattern_confidence": answers.pattern.confidence,
        }
        if signal is not None:
            row["signal_amount_anomalous"] = signal
        assert decisions(pd.DataFrame([row]), 0.2, 0.8).iloc[0] == outcome.value


def test_compute_metrics_shape():
    df = _ieee_frame(300)
    ans = _answers(df)
    m = compute_metrics(ans, 0.2, 0.8)
    assert 0.5 < m["roc_auc"] <= 1.0
    assert sum(m["decision_mix"].values()) == 300
    caught = m["fraud_caught"]
    assert pytest.approx(caught["declined"] + caught["stepped_up"] + caught["approved_missed"], abs=1e-9) == 1.0
    assert {r["pattern"] for r in m["per_pattern"]} == {"stolen_card", "legitimate"}
    assert m["usage"]["p95_latency_ms"] is not None
    assert "brier" in m["probability_calibration"]
    assert m["probability_calibration"]["reliability"]
    assert "removes_disproportionately_legitimate" in m["confidence_gate"]


def test_calibrate_finds_feasible_thresholds_for_good_model():
    df = _ieee_frame(500)
    ans = _answers(df, quality=0.9)
    table = sweep(ans, grid=np.linspace(0.05, 0.95, 19))
    best, feasible = recommend(table, max_step_up=0.5, max_fnr=0.1, max_fpr=0.1)
    assert feasible
    assert best["t_low"] <= best["t_high"]
    assert best["fraud_approved"] <= 0.1


def test_calibrate_falls_back_when_infeasible():
    df = _ieee_frame(200)
    ans = _answers(df, quality=0.2)
    table = sweep(ans, grid=np.linspace(0.05, 0.95, 10))
    best, feasible = recommend(table, max_step_up=0.0, max_fnr=0.0, max_fpr=0.0)
    assert not feasible
    assert "t_low" in best


def test_regression_detects_drift():
    df = _ieee_frame(400)
    good = flatten(compute_metrics(_answers(df, quality=0.9), 0.2, 0.8))
    bad = flatten(compute_metrics(_answers(df, seed=3, quality=0.3), 0.2, 0.8))
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


def test_brier_score_and_reliability_diagram():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(y, perfect) == 0.0
    assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)

    rows = reliability_table(y, np.array([0.1, 0.2, 0.8, 0.9]), n_bins=5)
    assert rows[0]["n"] == 1
    assert rows[0]["fraction_positive"] == 0.0
    perfect_rows = reliability_table(y, perfect, n_bins=2)
    assert expected_calibration_error(perfect_rows) == pytest.approx(0.0)

    svg = reliability_svg(rows)
    assert svg.startswith("<svg") and "<circle" in svg

    report = probability_calibration(pd.DataFrame({"label": y.astype(int), "p_fraud": perfect}), n_bins=2)
    assert report["brier"] == 0.0
    assert "eval.calibrate" in report["note"]


def test_confidence_gate_helps_only_by_dropping_legitimate_declines():
    helpful = pd.DataFrame({"label": [0, 1], "p_fraud": [0.95, 0.95], "risk_confidence": [0.2, 0.9]})
    effect = confidence_gate_effect(helpful, 0.2, 0.8, c_min=0.6)
    assert effect["rows_moved"] == 1
    assert effect["moved_fraud_rate"] == 0.0
    assert effect["removes_disproportionately_legitimate"] is True
    delta = effect["delta_gated_minus_probability_only"]
    assert delta["fraud_declined"] == 0.0
    assert delta["legit_declined"] < 0

    # High concentration on the legitimate row does not mean that decline is correct.
    harmful = pd.DataFrame({"label": [1, 0], "p_fraud": [0.95, 0.95], "risk_confidence": [0.2, 0.9]})
    effect = confidence_gate_effect(harmful, 0.2, 0.8, c_min=0.6)
    assert effect["removes_disproportionately_legitimate"] is False
    assert effect["delta_gated_minus_probability_only"]["fraud_declined"] < 0
    assert decisions_confidence_gated(harmful, 0.2, 0.8, 0.6).tolist() == ["STEP_UP", "DECLINE"]


def test_confidence_gate_cannot_decline_more_than_the_probability():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, 40),
            "p_fraud": rng.uniform(0, 1, 40),
            "risk_confidence": rng.uniform(0, 1, 40),
        }
    )
    delta = confidence_gate_effect(df, 0.2, 0.8, 0.6)["delta_gated_minus_probability_only"]
    assert delta["fraud_declined"] <= 0
    assert delta["legit_declined"] <= 0
    assert delta["step_up_rate"] >= -1e-12
