from datetime import UTC, datetime

from app.schemas.transaction import InvestigationRecord, Verdict


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["policy"] == {"t_low": 0.2, "t_high": 0.8, "evidence_min": 0.5}


async def test_health_api_prefix(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_score_uses_store_history(client, fake_classifier):
    """An approval never investigates, so the first state must already carry store history."""
    r = await client.post(
        "/transactions/score",
        json={
            "amount": 50,
            "card_id": "card_good_001",
            "DeviceInfo": "SM-G950F Build/R16NW",
            "card_avg_amount": 1,
            "card_known_devices": ["other Build/1"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "APPROVE"
    state = fake_classifier.calls[0]
    assert state["card_history"]["prior_transaction_count"] == 12
    assert state["card_history"]["mean_amount_usd_prior"] == 62.5
    assert state["card_history"]["trusted_device_ids"] == ["SM-G950F Build/R16NW"]
    assert state["card_history"]["confirmed_outcomes"]["approved"] == 12
    assert len(state["card_history"]["recent_attempts"]) == 5
    assert state["device"]["device_seen_before_on_this_card"] is True
    assert state["device"]["current_device_is_trusted"] is True
    assert "transactions_from_this_device" not in state.get("device", {})

    d = await client.get(f"/decisions/{body['decision_id']}")
    assert d.json()["investigation"]["status"] == "skipped"
    assert d.json()["final_decision"] == "APPROVE"


async def test_score_generic_device_stays_unknown(client, fake_classifier):
    r = await client.post(
        "/transactions/score",
        json={"amount": 50, "card_id": "card_ato_002", "DeviceInfo": "Windows"},
    )
    assert r.status_code == 200
    state = fake_classifier.calls[0]
    assert state["card_history"]["history_found"] is True
    assert state["card_history"]["mean_amount_usd_prior"] is not None
    assert state["card_history"]["trusted_device_ids"] == []
    assert state["device"]["device_identifier_kind"] == "description"
    assert state["device"]["device_seen_before_on_this_card"] == "unknown"
    assert state["device"]["current_device_is_trusted"] == "unknown"


async def test_score_without_card_id_keeps_history_unknown(client, fake_classifier):
    r = await client.post(
        "/transactions/score",
        json={"amount": 50, "DeviceInfo": "Windows", "card_avg_amount": 10, "card_known_devices": ["Windows"]},
    )
    assert r.status_code == 200
    state = fake_classifier.calls[0]
    assert state["card_history"]["mean_amount_usd_prior"] == "unknown"
    assert state["card_history"]["recent_attempts"] == "unknown"
    assert state["device"]["device_seen_before_on_this_card"] == "unknown"


async def test_card_present_state_is_what_jev_scores(client, fake_classifier):
    r = await client.post(
        "/transactions/score",
        json={
            "amount": 52,
            "card_id": "card_swipe_001",
            "channel": "card_present",
            "entry_mode": "swipe",
            "card_status": "open",
            "card4": "visa",
            "cvm_result": "pin_verified",
            "track_cvv": "match",
            "merchant_id": "merch_grocery_88",
            "mcc": "5411",
            "merchant_country": "US",
            "cardholder_country": "US",
            "available_credit_usd": 5000,
            "single_purchase_limit_usd": 2000,
        },
    )
    assert r.status_code == 200
    assert r.json()["decision"] == "APPROVE"
    cp = fake_classifier.calls[0]["card_present"]
    assert cp["entry_mode"] == "swipe"
    assert cp["rules"]["card_unusable"] is False
    assert cp["rules"]["over_limit"] is False
    assert cp["merchant_seen_before_on_this_card"] is True
    assert cp["merchant_country_changed_within_2h"] is False
    assert cp["rules"]["merchant_burst_1h"] is False
    assert cp["amount_usd"] == 52.0
    assert cp["amount_vs_mean_prior_ratio"] < 4
    assert cp["rules"]["amount_far_above_history"] is False
    assert cp["rules"]["probing_amount"] is False
    assert cp["amount_vs_this_merchant_ratio"] < 4


async def test_fallback_swipe_is_left_to_jev(client, fake_classifier):
    """No hard control: the fraud score (here amount/1000) still decides."""
    r = await client.post(
        "/transactions/score",
        json={
            "amount": 420,
            "card_id": "card_swipe_001",
            "channel": "card_present",
            "entry_mode": "fallback_swipe",
            "card_status": "open",
            "cvm_result": "signature",
            "track_cvv": "match",
            "merchant_id": "merch_elec_19",
            "merchant_country": "BR",
            "cardholder_country": "US",
            "mcc": "5732",
            "available_credit_usd": 5000,
            "single_purchase_limit_usd": 2000,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "STEP_UP"
    rules = fake_classifier.calls[0]["card_present"]["rules"]
    assert rules["magstripe_fallback"] is True
    assert rules["cross_border"] is True
    assert rules["merchant_country_changed_within_2h"] is True
    assert rules["swipe_without_pin"] is True
    assert rules["amount_far_above_history"] is True
    assert fake_classifier.calls[0]["card_present"]["amount_usd"] == 420.0
    assert "real-time authorization decline" not in body["explanation"]["rule"]


async def test_stolen_presentment_declines_before_a_low_fraud_score(client, fake_classifier):
    r = await client.post(
        "/transactions/score",
        json={
            "amount": 48,
            "channel": "card_present",
            "entry_mode": "swipe",
            "card_status": "stolen",
            "track_cvv": "mismatch",
            "available_credit_usd": 5000,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "DECLINE"
    assert body["challenge_id"] is None
    assert body["jev"]["is_fraud"]["noul"] == 0.048
    assert "card_unusable" in body["explanation"]["rule"]
    assert "track_cvv_mismatch" in body["explanation"]["rule"]
    assert fake_classifier.calls[0]["card_present"]["card_status"] == "stolen"
    record = (await client.get(f"/decisions/{body['decision_id']}")).json()
    assert record["final_decision"] == "DECLINE"
    assert "card_unusable" in record["final_reason"]


async def test_over_limit_declines_a_genuine_looking_swipe(client):
    r = await client.post(
        "/transactions/score",
        json={
            "amount": 80,
            "channel": "card_present",
            "entry_mode": "chip",
            "card_status": "open",
            "cvm_result": "pin_verified",
            "track_cvv": "match",
            "available_credit_usd": 50,
        },
    )
    body = r.json()
    assert body["decision"] == "DECLINE"
    assert "over_limit" in body["explanation"]["rule"]
    assert body["jev"]["is_fraud"]["noul"] == 0.08


async def test_score_approve(client, fake_classifier):
    r = await client.post("/transactions/score", json={"TransactionAmt": 50, "card4": "visa"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "APPROVE"
    assert body["challenge_id"] is None
    assert body["jev"]["is_fraud"]["noul"] == 0.05
    assert fake_classifier.calls[0]["transaction"]["card_network"] == "visa"

    d = await client.get(f"/decisions/{body['decision_id']}")
    assert d.status_code == 200
    assert d.json()["final_decision"] == "APPROVE"
    assert d.json()["investigation"]["status"] == "skipped"


async def test_score_decline(client):
    r = await client.post("/transactions/score", json={"amount": 950})
    assert r.json()["decision"] == "DECLINE"


async def test_step_up_flow_verified(client):
    r = await client.post("/transactions/score", json={"amount": 500, "card_id": "card_good_001"})
    body = r.json()
    assert body["decision"] == "STEP_UP"
    assert body["challenge_id"]
    assert body["dev_otp_code"]

    v = await client.post(f"/challenges/{body['challenge_id']}/verify", json={"code": body["dev_otp_code"]})
    assert v.status_code == 200
    vb = v.json()
    assert vb["challenge_status"] == "VERIFIED"
    assert vb["final_decision"] == "APPROVE"
    assert vb["attempts_remaining"] == 2


async def test_step_up_flow_failed(client):
    r = await client.post("/transactions/score", json={"amount": 500})
    body = r.json()
    for _ in range(3):
        v = await client.post(f"/challenges/{body['challenge_id']}/verify", json={"code": "000000"})
    vb = v.json()
    assert vb["challenge_status"] == "FAILED"
    assert vb["final_decision"] == "DECLINE"
    assert vb["attempts_remaining"] == 0


async def test_strong_verdict_overrides_verified_otp(client):
    container = client.app.state.container
    r = await client.post("/transactions/score", json={"amount": 500})
    body = r.json()
    record = await container.store.get_decision(body["decision_id"])
    record.investigation = InvestigationRecord(
        status="completed",
        verdict=Verdict(fraud_prob=0.97, pattern="account_takeover", rationale="x", evidence=[]),
        finished_at=datetime.now(UTC),
    )
    v = await client.post(f"/challenges/{body['challenge_id']}/verify", json={"code": body["dev_otp_code"]})
    vb = v.json()
    assert vb["challenge_status"] == "VERIFIED"
    assert vb["final_decision"] == "DECLINE"
    assert "0.97" in vb["final_reason"]


async def test_not_found(client):
    assert (await client.get("/decisions/nope")).status_code == 404
    assert (await client.post("/challenges/nope/verify", json={"code": "1"})).status_code == 404


async def test_validation_error(client):
    r = await client.post("/transactions/score", json={"amount": -1})
    assert r.status_code == 422
