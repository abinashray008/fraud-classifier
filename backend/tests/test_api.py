from datetime import UTC, datetime

from app.schemas.transaction import InvestigationRecord, Verdict


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["policy"] == {"t_low": 0.2, "t_high": 0.8, "c_min": 0.6}


async def test_health_api_prefix(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


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
