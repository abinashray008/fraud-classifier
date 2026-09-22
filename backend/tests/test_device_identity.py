"""Device descriptions never establish identity, regardless of their shape."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.agent.feature_store import FeatureStore
from app.agent.tools import build_tools
from app.features.history import PriorTxn, compute_card_history_features
from app.features.state_builder import build_state
from app.schemas.transaction import Transaction


@pytest.mark.parametrize(
    "description",
    [
        "SM-G9650 Build/R16NW",
        "acme/product/device:12/LMYXX/3359:userdebug/test-keys",
        "Windows",
        "Mozilla/5.0 (Linux; Android 12)",
        "f029dc7a-a366-4ac2-849a-6024ca6b9039",
    ],
)
def test_four_unrelated_cards_with_same_description_are_not_a_shared_device(description):
    store = FeatureStore()
    for i in range(4):
        store.record(Transaction(amount=20, card_id=f"card_{i}", DeviceInfo=description), "declined")

    assert not store.device_index
    assert not store.device_chargebacks
    assert store.device_history(description)["shared_across_many_cards"] is None
    tx = Transaction(amount=20, card_id="card_0", DeviceInfo=description)
    history = store.history_features(tx)
    assert history["trusted_device_ids"] == []
    state = build_state(tx, history=history)
    assert state["device"]["device_identifier_kind"] == "description"
    for key in (
        "device_seen_before_on_this_card",
        "device_is_new_for_card",
        "current_device_is_trusted",
        "device_distinct_cards_seen",
        "device_chargebacks",
    ):
        assert state["device"][key] == "unknown"
    assert store.velocity_stats("card_0")["distinct_devices_in_window"] is None
    assert state["card_history"]["distinct_devices_last_24h"] == "unknown"


def test_distinct_enrolled_ids_do_not_merge_when_descriptions_match():
    store = FeatureStore()
    for i in range(4):
        tx = Transaction(
            amount=20, card_id=f"card_{i}", DeviceInfo="SM-G9650 Build/R16NW", device_id=f"issuer:device:{i}"
        )
        store.record(tx)
    assert len(store.device_index) == 4
    for i in range(4):
        history = store.device_history(f"issuer:device:{i}")
        assert history["distinct_cards_seen"] == 1
        assert history["shared_across_many_cards"] is False


def test_same_enrolled_id_survives_description_changes_and_tracks_chargebacks():
    store = FeatureStore()
    for i in range(4):
        store.record(
            Transaction(
                amount=20,
                card_id=f"card_{i}",
                transaction_id=f"txn_{i}",
                DeviceInfo=f"Build/{i}",
                device_id="issuer:device:shared",
            ),
            "declined" if i == 3 else "approved",
        )
    store.record_verified_outcome("card_3", "txn_3", "chargeback", source="issuer", evidence_ref="cb:3")
    history = store.device_history("issuer:device:shared")
    assert history["distinct_cards_seen"] == 4
    assert history["chargebacks_on_device"] == 1
    assert history["shared_across_many_cards"] is True
    tx = Transaction(amount=20, card_id="card_0", device_id="issuer:device:shared")
    features = store.history_features(tx)
    assert features["device_seen_before_on_this_card"] is True
    assert features["current_device_is_trusted"] is False
    assert features["trusted_device_ids"] == []
    assert store.velocity_stats("card_0")["distinct_devices_in_window"] == 1
    assert store.cards["card_3"].known_devices == set()


def test_incomplete_identity_history_is_unknown_but_positive_matches_are_known():
    at = datetime(2026, 1, 10, tzinfo=UTC)
    rows = [
        PriorTxn(amount=10, timestamp=at - timedelta(hours=1), device_id="issuer:device:a"),
        PriorTxn(amount=10, timestamp=at - timedelta(hours=2)),
        PriorTxn(amount=10, timestamp=at + timedelta(hours=1), device_id="issuer:device:b"),
    ]
    hist = compute_card_history_features(prior=rows, amount=10, at=at, device_id="issuer:device:b")
    assert hist["device_seen_before_on_this_card"] is None
    assert hist["distinct_devices_last_24h"] is None
    assert hist["distinct_devices_last_7d"] is None
    hist = compute_card_history_features(prior=rows, amount=10, at=at, device_id="issuer:device:a")
    assert hist["device_seen_before_on_this_card"] is True
    hist = compute_card_history_features(prior=[], amount=10, at=at, device_id="issuer:device:a")
    assert hist["device_seen_before_on_this_card"] is False
    assert hist["distinct_devices_last_24h"] == 0


def test_legacy_offline_descriptions_cannot_establish_identity():
    description = "SM-G9650 Build/R16NW"
    tx = Transaction(amount=10, DeviceInfo=description, card_known_devices=[description])
    assert "device_seen_before_on_this_card" not in build_state(tx)["device"]
    state = build_state(tx, history={"device_seen_before_on_this_card": True, "device_distinct_cards_seen": 4})
    assert state["device"]["device_seen_before_on_this_card"] == "unknown"
    assert state["device"]["device_distinct_cards_seen"] == "unknown"


def test_investigation_tool_uses_explicit_id_only():
    store = FeatureStore()
    store.record(Transaction(amount=10, card_id="card_a", device_id="issuer:device:a"))
    tool = next(t for t in build_tools(store) if t.name == "get_device_history")
    assert set(tool.args) == {"device_id"}
    assert json.loads(tool.invoke({"device_id": "issuer:device:a"}))["distinct_cards_seen"] == 1
    assert json.loads(tool.invoke({"device_id": "SM-G9650 Build/R16NW"}))["shared_across_many_cards"] is None


@pytest.mark.parametrize("device_id", ["", "  ", " issuer:device:a", "issuer:device:a "])
def test_device_id_rejects_blank_or_ambiguous_whitespace(device_id):
    with pytest.raises(ValidationError, match="device_id"):
        Transaction(amount=10, device_id=device_id)


async def test_api_build_description_does_not_inherit_seeded_device_risk(client, fake_classifier):
    response = await client.post(
        "/transactions/score",
        json={
            "amount": 20,
            "card_id": "card_new",
            "DeviceInfo": "SM-G9650 Build/R16NW",
        },
    )
    assert response.status_code == 200
    device = fake_classifier.calls[0]["device"]
    assert device["device_identifier_kind"] == "description"
    assert device["device_distinct_cards_seen"] == "unknown"
    assert device["device_chargebacks"] == "unknown"
