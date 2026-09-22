"""Async investigation orchestration with a fake investigator (no LLM calls)."""

from __future__ import annotations

import asyncio

import pytest

from app.agent.feature_store import FeatureStore
from app.policy.decision import PolicyThresholds
from app.schemas.transaction import (
    DecisionOutcome,
    InvestigationRecord,
    Transaction,
    Verdict,
)
from app.services.decision_service import DecisionService
from app.stepup.otp import MockSmsProvider, OtpService
from app.store.memory import MemoryStore
from tests.conftest import FakeClassifier


class SlowFakeInvestigator:
    def __init__(self, fraud_prob: float, delay: float = 0.01) -> None:
        self.fraud_prob = fraud_prob
        self.delay = delay
        self.calls = 0

    async def investigate(self, tx, state, jev) -> InvestigationRecord:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return InvestigationRecord(
            status="completed",
            verdict=Verdict(fraud_prob=self.fraud_prob, pattern="stolen_card", rationale="fake", evidence=["e1"]),
            model_route="fast",
            tool_calls=[{"name": "get_card_history", "args": {"card_id": tx.card_id}}],
        )


def _service(investigator) -> DecisionService:
    store = MemoryStore()
    return DecisionService(
        classifier=FakeClassifier(),
        store=store,
        otp=OtpService(store, MockSmsProvider(), dev_mode=True),
        thresholds=PolicyThresholds(0.2, 0.8, 0.6),
        feature_store=FeatureStore(),
        investigator=investigator,
        agent_decline_prob=0.9,
    )


async def test_investigation_runs_in_background_for_step_up():
    inv = SlowFakeInvestigator(fraud_prob=0.3)
    svc = _service(inv)
    resp = await svc.score(Transaction(amount=500, card_id="c1"))
    assert resp.decision == DecisionOutcome.STEP_UP
    record = await svc.store.get_decision(resp.decision_id)
    assert record.investigation.status == "running"

    await svc.wait_for_background()
    record = await svc.store.get_decision(resp.decision_id)
    assert record.investigation.status == "completed"
    assert record.investigation.verdict.fraud_prob == 0.3
    assert record.investigation.finished_at is not None
    assert inv.calls == 1


async def test_investigation_not_run_for_clear_cases():
    inv = SlowFakeInvestigator(fraud_prob=0.3)
    svc = _service(inv)
    await svc.score(Transaction(amount=10))
    await svc.score(Transaction(amount=990))
    await svc.wait_for_background()
    assert inv.calls == 0


async def test_late_strong_verdict_resolves_pending_to_decline():
    inv = SlowFakeInvestigator(fraud_prob=0.99, delay=0.05)
    svc = _service(inv)
    resp = await svc.score(Transaction(amount=500, card_id="c1", device_id="issuer:device:a"))
    # OTP verified before the agent finishes: authorization remains pending.
    v = await svc.verify(resp.challenge_id, (await svc.store.get_challenge(resp.challenge_id)).dev_code)
    assert v.final_decision is None
    assert "awaiting" in v.final_reason
    assert svc.feature_store.cards["c1"].history[0].authorization_decision == "pending"

    await svc.wait_for_background()
    record = await svc.store.get_decision(resp.decision_id)
    assert record.final_decision == DecisionOutcome.DECLINE
    assert record.investigation.verdict.fraud_prob == 0.99
    history = svc.feature_store.cards["c1"].history
    assert len(history) == 1
    assert history[0].authorization_decision == "declined"
    assert svc.feature_store.cards["c1"].known_devices == set()
    features = svc.feature_store.history_features(Transaction(amount=20, card_id="c1", device_id="issuer:device:a"))
    assert features["current_device_is_trusted"] is False
    assert features["confirmed_outcomes"] is None


async def test_investigator_exception_is_recorded():
    class Boom:
        async def investigate(self, tx, state, jev):
            raise RuntimeError("llm down")

    svc = _service(Boom())
    resp = await svc.score(Transaction(amount=500))
    await svc.wait_for_background()
    record = await svc.store.get_decision(resp.decision_id)
    assert record.investigation.status == "failed"
    assert "llm down" in record.investigation.error


class ControlledInvestigator:
    def __init__(self, status="completed", fraud_prob=0.1):
        self.release = asyncio.Event()
        self.status = status
        self.fraud_prob = fraud_prob

    async def investigate(self, tx, state, jev):
        await self.release.wait()
        if self.status == "failed":
            raise RuntimeError("investigator unavailable")
        return InvestigationRecord(
            status=self.status,
            verdict=Verdict(fraud_prob=self.fraud_prob, pattern="other", rationale="test")
            if self.status == "completed" and self.fraud_prob is not None
            else None,
        )


@pytest.mark.parametrize("otp_first", [True, False])
@pytest.mark.parametrize("fraud_prob, expected", [(0.1, DecisionOutcome.APPROVE), (0.99, DecisionOutcome.DECLINE)])
async def test_both_completion_orders_write_one_final_authorization(otp_first, fraud_prob, expected):
    inv = ControlledInvestigator(fraud_prob=fraud_prob)
    svc = _service(inv)
    tx = Transaction(amount=500, card_id="c1", device_id="issuer:device:a")
    response = await svc.score(tx)
    code = response.dev_otp_code
    if otp_first:
        first = await svc.verify(response.challenge_id, code)
        assert first.final_decision is None
    inv.release.set()
    await svc.wait_for_background()
    if not otp_first:
        assert (await svc.store.get_decision(response.decision_id)).final_decision is None
    results = await asyncio.gather(*(svc.verify(response.challenge_id, code) for _ in range(3)))
    assert all(r.final_decision == expected for r in results)
    rows = svc.feature_store.cards["c1"].history
    assert len(rows) == 1
    assert rows[0].authorization_decision == ("approved" if expected == DecisionOutcome.APPROVE else "declined")
    assert rows[0].fraud_outcome is None
    assert svc.feature_store.history_features(tx)["current_device_is_trusted"] is False


@pytest.mark.parametrize("status", ["failed", "completed", "skipped"])
async def test_missing_required_verdict_never_approves_after_otp(status):
    inv = ControlledInvestigator(status=status, fraud_prob=None)
    svc = _service(inv)
    response = await svc.score(Transaction(amount=500, card_id="c1", device_id="issuer:device:a"))
    assert (await svc.verify(response.challenge_id, response.dev_otp_code)).final_decision is None
    inv.release.set()
    await svc.wait_for_background()
    result = await svc.verify(response.challenge_id, response.dev_otp_code)
    assert result.final_decision is None
    assert "investigation" in result.final_reason
    rows = svc.feature_store.cards["c1"].history
    assert len(rows) == 1
    assert rows[0].authorization_decision == "pending"
    assert svc.feature_store.cards["c1"].known_devices == set()


async def test_disabled_investigation_and_otp_approval_do_not_enroll_device():
    svc = _service(None)
    tx = Transaction(amount=500, card_id="c1", device_id="issuer:device:a")
    response = await svc.score(tx)
    result = await svc.verify(response.challenge_id, response.dev_otp_code)
    assert result.final_decision == DecisionOutcome.APPROVE
    assert len(svc.feature_store.cards["c1"].history) == 1
    assert svc.feature_store.history_features(tx)["current_device_is_trusted"] is False
    assert svc.feature_store.history_features(tx)["confirmed_outcomes"] is None


async def test_model_approval_does_not_enroll_device_or_confirm_legitimacy():
    svc = _service(None)
    tx = Transaction(amount=20, card_id="c1", device_id="issuer:device:a")
    response = await svc.score(tx)
    assert response.decision == DecisionOutcome.APPROVE
    assert svc.feature_store.history_features(tx)["current_device_is_trusted"] is False
    assert svc.feature_store.history_features(tx)["confirmed_outcomes"] is None


async def test_http_otp_waits_for_required_verdict_and_history_stays_single(client):
    svc = client.app.state.container.decision_service
    inv = ControlledInvestigator(fraud_prob=0.99)
    svc.investigator = inv
    scored = (
        await client.post(
            "/transactions/score",
            json={
                "amount": 500,
                "card_id": "api_card",
                "device_id": "issuer:device:api",
            },
        )
    ).json()
    verified = await client.post(f"/challenges/{scored['challenge_id']}/verify", json={"code": scored["dev_otp_code"]})
    assert verified.status_code == 200
    assert verified.json()["challenge_status"] == "VERIFIED"
    assert verified.json()["final_decision"] is None
    pending = (await client.get(f"/decisions/{scored['decision_id']}")).json()
    assert pending["investigation_required"] is True
    assert pending["final_decision"] is None
    inv.release.set()
    await svc.wait_for_background()
    final = (await client.get(f"/decisions/{scored['decision_id']}")).json()
    assert final["final_decision"] == "DECLINE"
    assert len(svc.feature_store.cards["api_card"].history) == 1
    assert svc.feature_store.cards["api_card"].known_devices == set()
