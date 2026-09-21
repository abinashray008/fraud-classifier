"""Orchestrates: build state -> Jev classify -> policy -> (challenge + async investigation)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from app.agent.feature_store import FeatureStore
from app.agent.investigator import Investigator
from app.features.state_builder import build_state
from app.jev.classifier import FraudClassifier
from app.policy.decision import PolicyThresholds, decide
from app.policy.final import finalize
from app.schemas.transaction import (
    ChallengeStatus,
    DecisionOutcome,
    DecisionRecord,
    InvestigationRecord,
    ScoreResponse,
    Transaction,
    VerifyResponse,
)
from app.stepup.otp import OtpService
from app.store.memory import MemoryStore

logger = logging.getLogger(__name__)


class DecisionService:
    def __init__(
        self,
        *,
        classifier: FraudClassifier,
        store: MemoryStore,
        otp: OtpService,
        thresholds: PolicyThresholds,
        feature_store: FeatureStore,
        investigator: Investigator | None = None,
        agent_decline_prob: float = 0.90,
    ) -> None:
        self.classifier = classifier
        self.store = store
        self.otp = otp
        self.thresholds = thresholds
        self.feature_store = feature_store
        self.investigator = investigator
        self.agent_decline_prob = agent_decline_prob
        self._tasks: set[asyncio.Task] = set()

    # Scoring -----------------------------------------------------------------------
    async def score(self, tx: Transaction) -> ScoreResponse:
        state = build_state(tx)
        answers = await self.classifier.classify(state)
        outcome, explanation = decide(answers, self.thresholds)

        record = DecisionRecord(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC),
            transaction=tx,
            state=state,
            jev=answers,
            decision=outcome,
            explanation=explanation,
        )

        dev_code: str | None = None
        if outcome == DecisionOutcome.STEP_UP:
            challenge = await self.otp.create_challenge(record.decision_id)
            record.challenge_id = challenge.challenge_id
            dev_code = challenge.dev_code
            if self.investigator is not None:
                record.investigation = InvestigationRecord(status="running", started_at=datetime.now(UTC))
                self._spawn(self._run_investigation(record.decision_id, tx, state, answers))
            else:
                record.investigation = InvestigationRecord(status="skipped")
        else:
            record.final_decision = outcome
            record.final_reason = "decided by Jev policy"
            record.investigation = InvestigationRecord(status="skipped")
            self.feature_store.record(tx, "approved" if outcome == DecisionOutcome.APPROVE else "declined")

        await self.store.put_decision(record)
        return ScoreResponse(
            decision_id=record.decision_id,
            decision=outcome,
            jev=answers,
            explanation=explanation,
            challenge_id=record.challenge_id,
            dev_otp_code=dev_code,
        )

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_investigation(self, decision_id: str, tx: Transaction, state: dict, answers) -> None:
        assert self.investigator is not None
        try:
            result = await self.investigator.investigate(tx, state, answers)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Investigator raised")
            result = InvestigationRecord(status="failed", error=str(exc))
        result.started_at = result.started_at or datetime.now(UTC)
        result.finished_at = datetime.now(UTC)

        record = await self.store.get_decision(decision_id)
        if record is None:
            return
        record.investigation = result
        # If the OTP already resolved, re-run the final policy so a late strong verdict
        # is reflected in the record (the client can fetch it via GET /decisions/{id}).
        if record.challenge_id:
            challenge = await self.store.get_challenge(record.challenge_id)
            if challenge and challenge.status != ChallengeStatus.PENDING:
                self._apply_final(record, challenge.status)
        await self.store.put_decision(record)

    # Verification ------------------------------------------------------------------
    async def verify(self, challenge_id: str, code: str) -> VerifyResponse:
        challenge = await self.otp.verify(challenge_id, code)
        record = await self.store.get_decision(challenge.decision_id)
        if record is None:
            raise KeyError(challenge.decision_id)
        self._apply_final(record, challenge.status)
        await self.store.put_decision(record)
        return VerifyResponse(
            decision_id=record.decision_id,
            challenge_status=challenge.status,
            final_decision=record.final_decision,
            final_reason=record.final_reason,
            attempts_remaining=max(challenge.max_attempts - challenge.attempts, 0),
            investigation=record.investigation,
        )

    def _apply_final(self, record: DecisionRecord, status: ChallengeStatus) -> None:
        verdict = record.investigation.verdict if record.investigation else None
        final, reason = finalize(status, verdict, self.agent_decline_prob, record.investigation.status)
        if final is None:
            return
        changed = record.final_decision != final
        record.final_decision = final
        record.final_reason = reason
        if changed:
            self.feature_store.record(
                record.transaction, "approved" if final == DecisionOutcome.APPROVE else "declined"
            )

    async def wait_for_background(self) -> None:
        """Test helper: wait for outstanding investigations."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
