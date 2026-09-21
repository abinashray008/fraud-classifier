"""Async investigation orchestration with a fake investigator (no LLM calls)."""

from __future__ import annotations

import asyncio

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


async def test_late_strong_verdict_flips_verified_approve_to_decline():
    inv = SlowFakeInvestigator(fraud_prob=0.99, delay=0.05)
    svc = _service(inv)
    resp = await svc.score(Transaction(amount=500, card_id="c1"))
    # OTP verified before the agent finishes -> provisional APPROVE
    v = await svc.verify(resp.challenge_id, (await svc.store.get_challenge(resp.challenge_id)).dev_code)
    assert v.final_decision == DecisionOutcome.APPROVE
    assert "not yet complete" in v.final_reason

    await svc.wait_for_background()
    record = await svc.store.get_decision(resp.decision_id)
    assert record.final_decision == DecisionOutcome.DECLINE
    assert record.investigation.verdict.fraud_prob == 0.99


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
