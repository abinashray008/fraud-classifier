from datetime import UTC, datetime, timedelta

import pytest

from app.agent.feature_store import FeatureStore
from app.schemas.transaction import Transaction


def _tx(**overrides):
    return Transaction(
        **{
            "amount": 20,
            "card_id": "card_a",
            "device_id": "issuer:device:a",
            "transaction_id": "txn_a",
            **overrides,
        }
    )


def test_authorization_reversal_updates_one_attempt_without_verifying_anything():
    store = FeatureStore()
    row = store.record(_tx(), "approved")
    when = row.timestamp
    assert store.history_features(_tx())["current_device_is_trusted"] is False
    assert store.history_features(_tx())["confirmed_outcomes"] is None
    store.record(_tx(), "declined")
    store.record(_tx(), "declined")
    assert len(store.cards["card_a"].history) == 1
    assert row.timestamp == when
    assert row.authorization_decision == "declined"
    assert row.fraud_outcome is None
    assert store.cards["card_a"].known_devices == set()
    history = store.history_features(_tx())
    assert history["prior_transaction_count"] == 1
    assert history["transactions_last_1h"] == 1
    assert history["amount_usd_last_1h"] == 20
    assert history["authorization_decisions"] == {"approved": 0, "declined": 1, "pending": 0}
    assert history["current_device_is_trusted"] is False


def test_independent_ownership_is_card_scoped_time_respecting_and_revocable():
    store = FeatureStore()
    at = datetime(2026, 1, 10, tzinfo=UTC)
    store.verify_device_ownership(
        "card_a",
        "issuer:device:a",
        source="issuer_enrollment",
        evidence_ref="enrollment:1",
        verified_at=at,
    )
    before = store.history_features(_tx(transaction_time=at - timedelta(seconds=1)))
    assert before["current_device_is_trusted"] is False
    assert store.history_features(_tx(transaction_time=at))["current_device_is_trusted"] is True
    assert store.history_features(_tx(card_id="card_b", transaction_time=at))["current_device_is_trusted"] is False
    store.revoke_device_ownership(
        "card_a",
        "issuer:device:a",
        source="issuer_enrollment",
        evidence_ref="revocation:1",
        revoked_at=at + timedelta(days=1),
    )
    assert store.history_features(_tx(transaction_time=at))["current_device_is_trusted"] is True
    later = at + timedelta(days=2)
    assert store.history_features(_tx(transaction_time=later))["current_device_is_trusted"] is False
    store.record(_tx(transaction_time=later), "approved")
    # Replayed enrollment evidence cannot restore trust after revocation.
    store.verify_device_ownership(
        "card_a",
        "issuer:device:a",
        source="issuer_enrollment",
        evidence_ref="enrollment:1",
        verified_at=later,
    )
    assert store.history_features(_tx(transaction_time=later))["trusted_device_ids"] == []
    store.verify_device_ownership(
        "card_a",
        "issuer:device:a",
        source="issuer_authentication",
        evidence_ref="authentication:2",
        verified_at=later,
    )
    assert store.history_features(_tx(transaction_time=later))["current_device_is_trusted"] is True
    events = store.card_history("card_a")["device_ownership_events"]
    assert [e["verified"] for e in events] == [True, False, True]
    assert events[-1]["evidence_ref"] == "authentication:2"


def test_outcome_feed_is_separate_idempotent_and_correctable():
    store = FeatureStore()
    at = datetime(2026, 1, 10, tzinfo=UTC)
    store.record(_tx(transaction_time=at), "approved")
    for _ in range(2):
        store.record_verified_outcome(
            "card_a",
            "txn_a",
            "chargeback",
            source="issuer_disputes",
            evidence_ref="dispute:1",
            verified_at=at + timedelta(days=1),
        )
    assert store.history_features(_tx(transaction_time=at + timedelta(hours=1)))["confirmed_outcomes"] is None
    history = store.history_features(_tx(transaction_time=at + timedelta(days=2)))
    assert history["confirmed_outcomes"] == {"legitimate": 0, "fraud": 0, "chargeback": 1}
    assert history["chargebacks_on_card"] == 1
    assert history["current_device_is_trusted"] is False
    assert store.device_history("issuer:device:a")["chargebacks_on_device"] == 1
    store.record(_tx(), "declined")
    assert store.cards["card_a"].history[0].fraud_outcome == "chargeback"
    store.record_verified_outcome(
        "card_a",
        "txn_a",
        "legitimate",
        source="issuer_disputes",
        evidence_ref="dispute:1:correction",
    )
    assert store.device_history("issuer:device:a")["chargebacks_on_device"] == 0
    assert store.history_features(_tx())["confirmed_outcomes"] == {"legitimate": 1, "fraud": 0, "chargeback": 0}
    assert store.history_features(_tx())["current_device_is_trusted"] is False


def test_verification_requires_provenance_and_is_not_accepted_from_transaction_input():
    store = FeatureStore()
    tx = _tx(current_device_is_trusted=True, trusted_device_ids=["issuer:device:a"], fraud_outcome="legitimate")
    store.record(tx, "approved")
    assert store.history_features(tx)["current_device_is_trusted"] is False
    assert store.history_features(tx)["confirmed_outcomes"] is None
    with pytest.raises(ValueError, match="source and evidence"):
        store.verify_device_ownership("card_a", "issuer:device:a", source="issuer", evidence_ref="")
    with pytest.raises(ValueError, match="source and evidence"):
        store.record_verified_outcome("card_a", "txn_a", "legitimate", source="", evidence_ref="label:1")
    with pytest.raises(ValueError, match="record_verified_outcome"):
        store.record(tx, "chargeback")
