import pytest

from app.features.controls import decline_before_model
from app.features.state_builder import build_state
from eval.authorization_counterexamples import build_sample
from eval.load_ieee import row_history, row_to_transaction
from eval.run_eval import score_sample
from tests.conftest import FakeClassifier


@pytest.mark.parametrize("row", [row for _, row in build_sample().iterrows()], ids=build_sample()["TransactionID"])
def test_genuine_counterexamples_preserve_separate_authorization_controls(row):
    tx = row_to_transaction(row)
    state = build_state(tx, history=row_history(row))
    assert row["isFraud"] == 0
    assert bool(decline_before_model(tx)) == bool(row["expected_hard_decline"])
    assert state["card_history"]["prior_transaction_count"] == 24
    assert state["card_present"]["amount_vs_mean_prior_ratio"] == 1.04
    assert state["card_present"]["rules"]["probing_amount"] is False
    assert "scenario_explanation" not in str(state)
    assert "expected_hard_decline" not in str(state)
    assert "isFraud" not in str(state)
    assert "legit_" not in str(state)  # Case IDs must not leak labels either.


async def test_counterexamples_run_through_existing_eval_pipeline_without_label_leakage():
    classifier = FakeClassifier()
    sample = build_sample()
    answers = await score_sample(sample, classifier, concurrency=2, cache=None)
    assert len(answers) == len(sample) == 8
    assert set(answers["label"]) == {0}
    assert set(answers["transaction_id"]) == set(sample["TransactionID"])
    assert len(classifier.calls) == 8
    # Deliberately includes blocked cases: this evaluates fraud semantics,
    # independently of the live path's mandatory authorization declines.
    assert any(state["card_present"]["rules"].get("pin_failure") for state in classifier.calls)


async def test_legitimate_control_failure_cases_still_decline_before_live_classifier(client, fake_classifier):
    sample = build_sample()
    for _, row in sample[sample["expected_hard_decline"]].iterrows():
        tx = row_to_transaction(row)
        response = await client.post("/transactions/score", json=tx.model_dump(mode="json"))
        assert response.status_code == 200
        assert response.json()["decision"] == "DECLINE"
        assert response.json()["jev"] is None
    assert not fake_classifier.calls
