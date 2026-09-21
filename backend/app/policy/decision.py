"""Threshold policy mapping Jev answers to APPROVE / STEP_UP / DECLINE.

Thresholds are calibrated per pinned Jev model version by `eval/calibrate.py`.
Bumping `JEV_MODEL` without recalibrating can shift the operating point.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.transaction import DecisionOutcome, JevAnswers, PolicyExplanation


@dataclass(frozen=True)
class PolicyThresholds:
    t_low: float = 0.20
    t_high: float = 0.80
    c_min: float = 0.60

    def __post_init__(self) -> None:
        if not 0 <= self.t_low <= self.t_high <= 1:
            raise ValueError("Require 0 <= t_low <= t_high <= 1")
        if not 0 <= self.c_min <= 1:
            raise ValueError("Require 0 <= c_min <= 1")


def decide(answers: JevAnswers, th: PolicyThresholds) -> tuple[DecisionOutcome, PolicyExplanation]:
    p = answers.is_fraud.noul
    conf = answers.risk.confidence

    if p < th.t_low:
        outcome = DecisionOutcome.APPROVE
        rule = f"is_fraud.p={p:.3f} < t_low={th.t_low}"
    elif p > th.t_high and conf >= th.c_min:
        outcome = DecisionOutcome.DECLINE
        rule = f"is_fraud.p={p:.3f} > t_high={th.t_high} and risk.confidence={conf:.2f} >= c_min={th.c_min}"
    elif p > th.t_high:
        outcome = DecisionOutcome.STEP_UP
        rule = (
            f"is_fraud.p={p:.3f} > t_high={th.t_high} but risk.confidence={conf:.2f} "
            f"< c_min={th.c_min}; step up instead of hard decline"
        )
    else:
        outcome = DecisionOutcome.STEP_UP
        rule = f"t_low={th.t_low} <= is_fraud.p={p:.3f} <= t_high={th.t_high}; ambiguous"

    return outcome, PolicyExplanation(
        rule=rule,
        t_low=th.t_low,
        t_high=th.t_high,
        c_min=th.c_min,
        fraud_probability=p,
        risk_confidence=conf,
    )
