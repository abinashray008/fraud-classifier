import pytest

from app.policy.decision import PolicyThresholds, decide
from app.policy.final import finalize
from app.schemas.transaction import ChallengeStatus, DecisionOutcome, Verdict
from tests.conftest import make_answers

TH = PolicyThresholds(t_low=0.2, t_high=0.8, evidence_min=0.5)


@pytest.mark.parametrize(
    "p,conf,pattern,score,expected,reason",
    [
        (0.05, 0.9, "legitimate", 0.2, DecisionOutcome.APPROVE, None),
        (0.19, 0.9, "legitimate", 0.4, DecisionOutcome.APPROVE, None),
        (0.20, 0.9, "other", 1.0, DecisionOutcome.STEP_UP, "ambiguous"),
        (0.5, 0.9, "other", 2.0, DecisionOutcome.STEP_UP, "ambiguous"),
        (0.80, 0.9, "stolen_card", 3.2, DecisionOutcome.STEP_UP, "ambiguous"),
        (0.81, 0.9, "stolen_card", 3.4, DecisionOutcome.DECLINE, None),
        (0.99, 0.6, "stolen_card", 3.9, DecisionOutcome.DECLINE, None),
        # Spread risk distribution withholds the action. It does not authorize it.
        (0.95, 0.49, "stolen_card", 3.8, DecisionOutcome.STEP_UP, "insufficient_evidence"),
        (0.05, 0.2, "legitimate", 0.2, DecisionOutcome.STEP_UP, "insufficient_evidence"),
        # High concentration does not authorize a decline the other answers contradict.
        (0.95, 0.95, "legitimate", 3.8, DecisionOutcome.STEP_UP, "contradictory"),
        (0.95, 0.95, "stolen_card", 0.4, DecisionOutcome.STEP_UP, "contradictory"),
        (0.05, 0.95, "stolen_card", 0.2, DecisionOutcome.STEP_UP, "contradictory"),
        (0.05, 0.95, "legitimate", 3.5, DecisionOutcome.STEP_UP, "contradictory"),
    ],
)
def test_decide(p, conf, pattern, score, expected, reason):
    outcome, expl = decide(make_answers(p, risk_conf=conf, pattern=pattern, risk_score=score), TH)
    assert outcome == expected
    assert expl.review_reason == reason
    assert expl.fraud_probability == p
    assert expl.risk_score == score
    assert expl.risk_confidence == conf
    assert expl.t_low == 0.2 and expl.t_high == 0.8
    assert "c_min" not in expl.rule


def test_flat_pattern_distribution_goes_to_review():
    answers = make_answers(0.95, pattern="stolen_card", risk_score=3.8, risk_conf=0.9, pattern_confidence=0.2)
    outcome, expl = decide(answers, TH)
    assert outcome == DecisionOutcome.STEP_UP
    assert expl.review_reason == "insufficient_evidence"
    assert "no pattern dominates" in expl.rule


def test_decline_without_supporting_signal_goes_to_review():
    answers = make_answers(
        0.95,
        pattern="stolen_card",
        risk_score=3.8,
        risk_conf=0.9,
        signals={"amount_anomalous": 0.1, "velocity_spike": 0.2},
    )
    outcome, expl = decide(answers, TH)
    assert outcome == DecisionOutcome.STEP_UP
    assert expl.review_reason == "insufficient_evidence"
    assert "signal" in expl.rule


def test_decline_with_no_signal_questions_still_declines():
    answers = make_answers(0.95, pattern="stolen_card", risk_score=3.8, risk_conf=0.9, signals={})
    outcome, expl = decide(answers, TH)
    assert outcome == DecisionOutcome.DECLINE
    assert expl.review_reason is None


def test_one_supporting_signal_allows_decline():
    answers = make_answers(
        0.95,
        pattern="stolen_card",
        risk_score=3.8,
        risk_conf=0.9,
        signals={"amount_anomalous": 0.1, "velocity_spike": 0.8},
    )
    assert decide(answers, TH)[0] == DecisionOutcome.DECLINE


def test_threshold_validation():
    with pytest.raises(ValueError):
        PolicyThresholds(t_low=0.9, t_high=0.5)
    with pytest.raises(ValueError):
        PolicyThresholds(evidence_min=1.5)
    with pytest.raises(ValueError):
        PolicyThresholds(low_risk_max=3.5, high_risk_min=1.0)


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
