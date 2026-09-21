import pytest

from app.policy.decision import PolicyThresholds, decide
from app.policy.final import finalize
from app.schemas.transaction import ChallengeStatus, DecisionOutcome, Verdict
from tests.conftest import make_answers

TH = PolicyThresholds(t_low=0.2, t_high=0.8, c_min=0.6)


@pytest.mark.parametrize(
    "p,conf,expected",
    [
        (0.05, 0.9, DecisionOutcome.APPROVE),
        (0.19, 0.1, DecisionOutcome.APPROVE),
        (0.20, 0.9, DecisionOutcome.STEP_UP),
        (0.5, 0.9, DecisionOutcome.STEP_UP),
        (0.80, 0.9, DecisionOutcome.STEP_UP),
        (0.81, 0.9, DecisionOutcome.DECLINE),
        (0.95, 0.5, DecisionOutcome.STEP_UP),  # high p but low confidence -> step up
        (0.99, 0.6, DecisionOutcome.DECLINE),
    ],
)
def test_decide(p, conf, expected):
    outcome, expl = decide(make_answers(p, risk_conf=conf), TH)
    assert outcome == expected
    assert expl.fraud_probability == p
    assert expl.t_low == 0.2 and expl.t_high == 0.8


def test_threshold_validation():
    with pytest.raises(ValueError):
        PolicyThresholds(t_low=0.9, t_high=0.5)
    with pytest.raises(ValueError):
        PolicyThresholds(c_min=1.5)


def _verdict(p: float) -> Verdict:
    return Verdict(fraud_prob=p, pattern="stolen_card", rationale="r", evidence=["e"])


def test_finalize_pending():
    assert finalize(ChallengeStatus.PENDING, None) == (None, None)


@pytest.mark.parametrize("status", [ChallengeStatus.FAILED, ChallengeStatus.EXPIRED])
def test_finalize_otp_failure_declines(status):
    final, reason = finalize(status, _verdict(0.01))
    assert final == DecisionOutcome.DECLINE
    assert status.value.lower() in reason


def test_finalize_verified_without_verdict_approves():
    final, reason = finalize(ChallengeStatus.VERIFIED, None)
    assert final == DecisionOutcome.APPROVE
    assert "not yet complete" in reason
    _, reason = finalize(ChallengeStatus.VERIFIED, None, investigation_status="skipped")
    assert "disabled" in reason
    _, reason = finalize(ChallengeStatus.VERIFIED, None, investigation_status="failed")
    assert "failed" in reason


def test_finalize_verified_with_strong_verdict_declines():
    final, reason = finalize(ChallengeStatus.VERIFIED, _verdict(0.95), agent_decline_prob=0.9)
    assert final == DecisionOutcome.DECLINE
    assert "0.95" in reason


def test_finalize_verified_with_weak_verdict_approves():
    final, _ = finalize(ChallengeStatus.VERIFIED, _verdict(0.4))
    assert final == DecisionOutcome.APPROVE
