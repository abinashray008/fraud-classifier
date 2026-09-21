"""Merge the OTP outcome with the investigation verdict into a final decision.

Rules (v1):
- OTP failed or expired            -> DECLINE
- OTP verified, no verdict yet     -> APPROVE (verdict recorded when it lands)
- OTP verified, verdict fraud_prob >= agent_decline_prob -> DECLINE
  (OTP can be intercepted in an account-takeover; strong agent evidence wins)
- OTP verified otherwise           -> APPROVE
"""

from __future__ import annotations

from app.schemas.transaction import ChallengeStatus, DecisionOutcome, Verdict


def finalize(
    challenge_status: ChallengeStatus,
    verdict: Verdict | None,
    agent_decline_prob: float = 0.90,
    investigation_status: str = "running",
) -> tuple[DecisionOutcome | None, str | None]:
    if challenge_status == ChallengeStatus.PENDING:
        return None, None
    if challenge_status in (ChallengeStatus.FAILED, ChallengeStatus.EXPIRED):
        return DecisionOutcome.DECLINE, f"OTP {challenge_status.value.lower()}"
    # VERIFIED
    if verdict is None:
        detail = {
            "skipped": "investigation tier disabled",
            "failed": "investigation failed; decided on OTP alone",
        }.get(investigation_status, "investigation not yet complete")
        return DecisionOutcome.APPROVE, f"OTP verified; {detail}"
    if verdict.fraud_prob >= agent_decline_prob:
        return (
            DecisionOutcome.DECLINE,
            f"OTP verified but investigation fraud_prob={verdict.fraud_prob:.2f} "
            f">= {agent_decline_prob} ({verdict.pattern})",
        )
    return DecisionOutcome.APPROVE, (
        f"OTP verified; investigation fraud_prob={verdict.fraud_prob:.2f} ({verdict.pattern})"
    )
