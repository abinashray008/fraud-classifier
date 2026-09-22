"""Pending authorizations count once, independently of approvals and fraud labels."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.agent.feature_store import FeatureStore
from app.schemas.transaction import Transaction
from tests.conftest import make_answers
from tests.test_investigation_async import _service


async def test_three_unresolved_stepups_are_visible_to_next_score_and_tools(client, fake_classifier):
    responses = []
    for _ in range(3):
        response = await client.post("/transactions/score", json={"amount": 500, "card_id": "pending_card"})
        assert response.status_code == 200
        assert response.json()["decision"] == "STEP_UP"
        responses.append(response.json())
    store = client.app.state.container.feature_store
    history = store.history_features(Transaction(amount=500, card_id="pending_card"))
    velocity = store.velocity_stats("pending_card", window_hours=1)
    for window in ("1h", "24h", "7d"):
        assert history[f"transactions_last_{window}"] == 3
        assert history[f"attempts_last_{window}"] == 3
        assert history[f"approvals_last_{window}"] == 0
        assert history[f"confirmed_fraud_last_{window}"] == 0
    assert velocity["transactions_in_window"] == velocity["attempts_in_window"] == 3
    assert velocity["approvals_in_window"] == velocity["confirmed_fraud_in_window"] == 0
    assert history["matching_amount_count_last_24h"] == 3
    assert history["amount_usd_last_1h"] == 1500
    assert history["authorization_decisions"]["pending"] == 3

    await client.post("/transactions/score", json={"amount": 500, "card_id": "pending_card"})
    assert [s["card_history"]["attempts_last_1h"] for s in fake_classifier.calls] == [0, 1, 2, 3]
    assert fake_classifier.calls[-1]["card_history"]["approvals_last_1h"] == 0

    # Resolving and retrying an OTP updates the original row; it never adds attempts.
    approved = responses[0]
    for _ in range(2):
        result = await client.post(
            f"/challenges/{approved['challenge_id']}/verify", json={"code": approved["dev_otp_code"]}
        )
        assert result.json()["final_decision"] == "APPROVE"
    for _ in range(3):
        result = await client.post(f"/challenges/{responses[1]['challenge_id']}/verify", json={"code": "wrong"})
    assert result.json()["final_decision"] == "DECLINE"
    store.record_verified_outcome(
        "pending_card",
        responses[2]["decision_id"],
        "fraud",
        source="issuer_adjudication",
        evidence_ref="case:1",
    )
    velocity = store.velocity_stats("pending_card", window_hours=1)
    assert velocity["attempts_in_window"] == 4
    assert velocity["approvals_in_window"] == 1
    assert velocity["confirmed_fraud_in_window"] == 1
    assert velocity["total_amount_in_window_usd"] == 2000
    assert len(store.cards["pending_card"].history) == 4


async def test_unresolved_small_attempts_trigger_repeated_amount_probe():
    svc = _service(None)
    states = []

    async def ambiguous(state):
        states.append(state)
        return make_answers(0.5)

    svc.classifier.classify = ambiguous
    for _ in range(3):
        result = await svc.score(Transaction(amount=2, card_id="probes", channel="card_present"))
        assert result.decision == "STEP_UP"
    assert [s["card_present"]["rules"]["probing_amount"] for s in states] == [False, False, True]
    assert states[-1]["card_history"]["matching_amount_count_last_24h"] == 2
    assert states[-1]["card_history"]["amount_usd_last_1h"] == 4
    assert svc.feature_store.velocity_stats("probes", 1)["attempts_in_window"] == 3


async def test_inflight_model_calls_are_visible_without_counting_themselves():
    svc = _service(None)
    states = []
    first_entered, both_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked(state):
        states.append(state)
        first_entered.set()
        if len(states) == 2:
            both_entered.set()
        await release.wait()
        return make_answers(0.5)

    svc.classifier.classify = blocked
    first = asyncio.create_task(svc.score(Transaction(amount=500, card_id="inflight")))
    await asyncio.wait_for(first_entered.wait(), 1)
    assert svc.feature_store.velocity_stats("inflight", 1)["attempts_in_window"] == 1
    second = asyncio.create_task(svc.score(Transaction(amount=500, card_id="inflight")))
    await asyncio.wait_for(both_entered.wait(), 1)
    assert svc.feature_store.velocity_stats("inflight", 1)["attempts_in_window"] == 2
    assert [s["card_history"]["attempts_last_1h"] for s in states] == [0, 1]
    release.set()
    await asyncio.gather(first, second)
    assert len(svc.feature_store.cards["inflight"].history) == 2


@pytest.mark.parametrize("failure", ["model", "otp_delivery"])
async def test_processing_failures_leave_an_observed_attempt(failure):
    svc = _service(None)

    async def fail(*args):
        raise RuntimeError("dependency unavailable")

    if failure == "model":
        svc.classifier.classify = fail
    else:
        svc.otp.provider.send = fail
    with pytest.raises(RuntimeError, match="dependency unavailable"):
        await svc.score(Transaction(amount=500, card_id="failed_processing"))
    rows = svc.feature_store.cards["failed_processing"].history
    assert len(rows) == 1
    assert rows[0].authorization_decision == "pending"
    stats = svc.feature_store.velocity_stats("failed_processing", 1)
    assert stats["attempts_in_window"] == 1
    assert stats["approvals_in_window"] == stats["confirmed_fraud_in_window"] == 0


def test_window_counters_use_original_attempt_time_and_independent_fraud_labels():
    store = FeatureStore()
    at = datetime(2026, 1, 10, tzinfo=UTC)
    for key, age in (
        ("boundary", timedelta(hours=1)),
        ("old", timedelta(hours=1, seconds=1)),
        ("future", timedelta(seconds=-1)),
    ):
        store.record(Transaction(amount=2, card_id="card", transaction_id=key, transaction_time=at - age), "pending")
    # A later approval changes only the decision, not the original attempt time.
    store.record(Transaction(amount=2, card_id="card", transaction_id="old", transaction_time=at), "approved")
    store.record_verified_outcome(
        "card", "boundary", "chargeback", source="issuer", evidence_ref="cb:1", verified_at=at
    )
    history = store.history_features(Transaction(amount=2, card_id="card", transaction_time=at))
    assert history["attempts_last_1h"] == 1
    assert history["approvals_last_1h"] == 0
    assert history["confirmed_fraud_last_1h"] == 0  # Chargeback alone is not confirmed fraud.
    assert history["attempts_last_24h"] == 2
    assert history["approvals_last_24h"] == 1
    store.record_verified_outcome("card", "boundary", "fraud", source="issuer", evidence_ref="fraud:1", verified_at=at)
    history = store.history_features(Transaction(amount=2, card_id="card", transaction_time=at))
    assert history["confirmed_fraud_last_1h"] == 1
    assert history["attempts_last_1h"] == 1


def test_velocity_tool_excludes_future_attempts_and_accepts_naive_timestamps():
    store = FeatureStore()
    now = datetime.now(UTC)
    store.record(
        Transaction(amount=2, card_id="card", transaction_time=(now - timedelta(minutes=1)).replace(tzinfo=None))
    )
    store.record(Transaction(amount=2, card_id="card", transaction_time=now + timedelta(days=1)))
    assert store.velocity_stats("card", 1)["attempts_in_window"] == 1
