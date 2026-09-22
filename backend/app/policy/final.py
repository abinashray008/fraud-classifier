"""Merge the OTP outcome with the investigation verdict into a final decision.

OTP failure/expiry declines. OTP success stays pending until every required
investigation completes with a verdict. Disabled investigation permits OTP-only
authorization; failed/missing required investigation needs retry or manual review.
"""

from __future__ import annotations

from app.schemas.transaction import ChallengeStatus, DecisionOutcome, Verdict


def finalize(
    challenge_status: ChallengeStatus,
    verdict: Verdict | None,
    agent_decline_prob: float = 0.90,
    investigation_status: str = "running",
    investigation_required: bool = True,
) -> tuple[DecisionOutcome | None, str | None]:
    if challenge_status == ChallengeStatus.PENDING:
        return None, None
    if challenge_status in (ChallengeStatus.FAILED, ChallengeStatus.EXPIRED):
        return DecisionOutcome.DECLINE, f"OTP {challenge_status.value.lower()}"
    # VERIFIED: never return a provisional authorization as a final approval.
    if investigation_required and (investigation_status != "completed" or verdict is None):
        detail = (
            "investigation failed; retry or manual review required"
            if investigation_status == "failed"
            else ("required investigation awaiting a completed verdict")
        )
        return None, f"OTP verified; {detail}"
    if verdict is None:
        return DecisionOutcome.APPROVE, "OTP verified; investigation tier disabled"
    if verdict.fraud_prob >= agent_decline_prob:
        return (
            DecisionOutcome.DECLINE,
            f"OTP verified but investigation fraud_prob={verdict.fraud_prob:.2f} "
            f">= {agent_decline_prob} ({verdict.pattern})",
        )
    return DecisionOutcome.APPROVE, (
        f"OTP verified; investigation fraud_prob={verdict.fraud_prob:.2f} ({verdict.pattern})"
    )
