"""Map Jev answers to APPROVE / STEP_UP / DECLINE.

Automatic actions use `is_fraud.noul` against thresholds selected by
`eval/calibrate.py`. That sweep chooses an operating point; it does not
calibrate the probability. Whether `is_fraud.noul` matches observed fraud
rates is measured separately by `eval/calibration.py` (Brier score and a
reliability diagram).

`risk.confidence` is how concentrated the risk-level distribution is. It is
not the probability that the fraud answer is correct, so it never authorizes
a decline. A spread distribution means the risk question does not have enough
to go on; that, and any contradiction between the fraud Noul, the risk score,
and the pattern, goes to review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.schemas.transaction import DecisionOutcome, JevAnswers, PolicyExplanation

FRAUD_PATTERNS = frozenset({"card_testing", "stolen_card", "account_takeover"})

ReviewReason = Literal["contradictory", "insufficient_evidence", "ambiguous"]


@dataclass(frozen=True)
class PolicyThresholds:
    t_low: float = 0.20
    t_high: float = 0.80
    # TypeSafe's floor for "do not act": a Score or Choice distribution flatter
    # than this is insufficient evidence, not a reason to decline.
    evidence_min: float = 0.50
    # Risk scale is 0–4 (see questions.RISK_LEVELS). Below 2 is routine/minor;
    # at or above 3 is strong or textbook fraud. The middle does not contradict.
    low_risk_max: float = 2.0
    high_risk_min: float = 3.0
    # A decline needs at least one signal Noul at or above this. Missing signal
    # questions are not treated as a lack of evidence.
    signal_support: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.t_low <= self.t_high <= 1:
            raise ValueError("Require 0 <= t_low <= t_high <= 1")
        if not 0 <= self.evidence_min <= 1:
            raise ValueError("Require 0 <= evidence_min <= 1")
        if not 0 <= self.signal_support <= 1:
            raise ValueError("Require 0 <= signal_support <= 1")
        if not 0 <= self.low_risk_max <= self.high_risk_min:
            raise ValueError("Require 0 <= low_risk_max <= high_risk_min")


def _contradiction(answers: JevAnswers, th: PolicyThresholds) -> str | None:
    p = answers.is_fraud.noul
    score = answers.risk.score
    pattern = answers.pattern.choice
    if p > th.t_high and score < th.low_risk_max:
        return (
            f"is_fraud.p={p:.3f} > t_high={th.t_high} but risk.score={score:.2f} < {th.low_risk_max} (routine or minor)"
        )
    if p < th.t_low and score >= th.high_risk_min:
        return (
            f"is_fraud.p={p:.3f} < t_low={th.t_low} but risk.score={score:.2f} "
            f">= {th.high_risk_min} (strong or textbook fraud)"
        )
    if p > th.t_high and pattern == "legitimate":
        return f"is_fraud.p={p:.3f} > t_high={th.t_high} but pattern is legitimate"
    if p < th.t_low and pattern in FRAUD_PATTERNS:
        return f"is_fraud.p={p:.3f} < t_low={th.t_low} but pattern is {pattern}"
    return None


def _insufficient(answers: JevAnswers, th: PolicyThresholds) -> str | None:
    """Withhold an automatic action when the other answers are too thin to support it."""
    p = answers.is_fraud.noul
    if not (p < th.t_low or p > th.t_high):
        return None
    conf = answers.risk.confidence
    if conf < th.evidence_min:
        return (
            f"risk.confidence={conf:.2f} < {th.evidence_min} "
            "(spread risk-level distribution; not enough to place the transaction)"
        )
    pattern_conf = answers.pattern.confidence
    if pattern_conf < th.evidence_min:
        return f"pattern.confidence={pattern_conf:.2f} < {th.evidence_min} (no pattern dominates)"
    if p > th.t_high and answers.signals:
        supported = max(signal.noul for signal in answers.signals.values())
        if supported < th.signal_support:
            return f"is_fraud.p={p:.3f} > t_high={th.t_high} but every signal noul is below {th.signal_support}"
    return None


def decide(answers: JevAnswers, th: PolicyThresholds) -> tuple[DecisionOutcome, PolicyExplanation]:
    p = answers.is_fraud.noul
    contradiction = _contradiction(answers, th)
    insufficient = None if contradiction else _insufficient(answers, th)

    if contradiction:
        outcome: DecisionOutcome = DecisionOutcome.STEP_UP
        reason: ReviewReason = "contradictory"
        rule = f"{contradiction}; review"
    elif insufficient:
        outcome = DecisionOutcome.STEP_UP
        reason = "insufficient_evidence"
        rule = f"{insufficient}; review"
    elif p < th.t_low:
        outcome = DecisionOutcome.APPROVE
        reason = None
        rule = f"is_fraud.p={p:.3f} < t_low={th.t_low}"
    elif p > th.t_high:
        outcome = DecisionOutcome.DECLINE
        reason = None
        rule = (
            f"is_fraud.p={p:.3f} > t_high={th.t_high}; "
            f"risk.score={answers.risk.score:.2f} and pattern={answers.pattern.choice} agree"
        )
    else:
        outcome = DecisionOutcome.STEP_UP
        reason = "ambiguous"
        rule = f"t_low={th.t_low} <= is_fraud.p={p:.3f} <= t_high={th.t_high}; ambiguous"

    return outcome, PolicyExplanation(
        rule=rule,
        t_low=th.t_low,
        t_high=th.t_high,
        fraud_probability=p,
        risk_score=answers.risk.score,
        risk_confidence=answers.risk.confidence,
        review_reason=reason,
    )
