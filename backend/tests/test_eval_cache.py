"""Cache provenance and held-out report boundaries; no API calls required."""

from types import SimpleNamespace

import pandas as pd
import pytest

from app.jev.questions import build_questions
from eval import run_eval
from eval.metrics import frame_for
from tests.conftest import FakeClassifier
from tests.test_eval import _ieee_frame


class PinnedClassifier(FakeClassifier):
    model = "jev-1.13.0"

    def __init__(self):
        super().__init__()
        self.runnable = SimpleNamespace(questions=build_questions())
        self.resolved_model = self.model

    async def classify(self, state):
        answers = await super().classify(state)
        return answers.model_copy(update={"model": self.resolved_model})


def sample():
    return _ieee_frame(3).assign(split=["development", "calibration", "test"], sample_weight=1.0)


@pytest.mark.asyncio
async def test_cache_roundtrip_reuses_identical_inputs_and_scopes_output(tmp_path):
    classifier = PinnedClassifier()
    source = sample()
    first = await run_eval.score_sample(source, classifier, 2, None)
    path = tmp_path / "answers.parquet"
    first.to_parquet(path, index=False)
    second = await run_eval.score_sample(source.iloc[[2, 0]], classifier, 2, pd.read_parquet(path))
    assert len(classifier.calls) == 3
    assert second.transaction_id.tolist() == ["3", "1"]
    assert second.cache_key.tolist() == first.iloc[[2, 0]].cache_key.tolist()
    assert second.split.tolist() == ["test", "development"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("TransactionAmt", 999.0), ("transactions_last_1h", 20)])
async def test_changed_transaction_or_history_rescores(field, value):
    classifier = PinnedClassifier()
    source = sample()
    first = await run_eval.score_sample(source, classifier, 2, None)
    if field not in source:
        source[field] = None
        first = await run_eval.score_sample(source, classifier, 2, first)
    count = len(classifier.calls)
    source.loc[0, field] = value
    second = await run_eval.score_sample(source, classifier, 2, first)
    assert len(classifier.calls) == count + 1
    assert second.loc[0, "cache_key"] != first.loc[0, "cache_key"]
    if field == "TransactionAmt":
        assert second.loc[0, "p_fraud"] == pytest.approx(0.999)


@pytest.mark.asyncio
async def test_current_reporting_metadata_replaces_cached_metadata():
    classifier = PinnedClassifier()
    source = sample()
    first = await run_eval.score_sample(source, classifier, 2, None)
    source.loc[0, ["split", "sample_weight", "isFraud"]] = ["test", 3.5, 1]
    second = await run_eval.score_sample(source, classifier, 2, first)
    assert len(classifier.calls) == 3
    assert second.loc[0, "split"] == "test"
    assert second.loc[0, "sample_weight"] == 3.5
    assert second.loc[0, "label"] == 1
    third = await run_eval.score_sample(source.drop(columns=["split", "sample_weight"]), classifier, 2, second)
    assert len(classifier.calls) == 3
    assert "split" not in third and "sample_weight" not in third
    with pytest.raises(ValueError, match="split metadata"):
        frame_for(third, "test")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["questions", "features", "version", "model"])
async def test_scoring_definition_changes_invalidate_cache(monkeypatch, change):
    classifier = PinnedClassifier()
    source = sample()
    first = await run_eval.score_sample(source, classifier, 2, None)
    if change == "questions":
        classifier.runnable.questions["is_fraud"] = classifier.runnable.questions["is_fraud"].model_copy(
            update={"instructions": "Updated fraud definition"}
        )
    elif change == "features":
        monkeypatch.setattr(run_eval, "feature_fingerprint", lambda: "updated-features")
    elif change == "version":
        monkeypatch.setattr(run_eval, "CACHE_VERSION", run_eval.CACHE_VERSION + 1)
    else:
        classifier.model = classifier.resolved_model = "jev-1.14.0"
    second = await run_eval.score_sample(source, classifier, 2, first)
    assert len(classifier.calls) == 6
    assert set(first.cache_key).isdisjoint(second.cache_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["cache_key", "input_fingerprint", "questions_fingerprint", "features_fingerprint"])
async def test_legacy_or_incomplete_cache_is_rebuilt(missing):
    classifier = PinnedClassifier()
    source = sample()
    first = await run_eval.score_sample(source, classifier, 2, None)
    legacy = first.drop(columns=[missing, "split", "sample_weight"])
    legacy["p_fraud"] = -1
    second = await run_eval.score_sample(source, classifier, 2, legacy)
    assert len(classifier.calls) == 6
    assert (second.p_fraud >= 0).all()
    assert second.split.tolist() == source.split.tolist()
    assert second.sample_weight.tolist() == source.sample_weight.tolist()


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,resolved", [("jev-latest", "jev-1.13.0"), ("jev-1.13.0", "jev-1.14.0")])
async def test_alias_or_unexpected_resolved_model_is_not_reused(requested, resolved):
    classifier = PinnedClassifier()
    classifier.model, classifier.resolved_model = requested, resolved
    first = await run_eval.score_sample(sample(), classifier, 2, None)
    second = await run_eval.score_sample(sample(), classifier, 2, first)
    assert len(classifier.calls) == 6
    assert set(second.model) == {resolved}
    assert set(second.requested_model) == {requested}


@pytest.mark.asyncio
async def test_failed_rescore_does_not_restore_stale_answer(monkeypatch):
    classifier = PinnedClassifier()
    source = sample()
    first = await run_eval.score_sample(source, classifier, 2, None)
    source.loc[0, "TransactionAmt"] = 999.0

    async def fail(state):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(classifier, "classify", fail)
    second = await run_eval.score_sample(source, classifier, 2, first)
    assert second.transaction_id.tolist() == ["2", "3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["cache_version", "input_fingerprint", "model", "cache_key"])
async def test_mixed_cache_rebuilds_only_rows_without_provenance(missing):
    classifier = PinnedClassifier()
    source = sample()
    first = await run_eval.score_sample(source, classifier, 2, None)
    first[missing] = first[missing].astype(object)
    first.loc[0, missing] = pd.NA
    first.loc[0, "p_fraud"] = -1
    second = await run_eval.score_sample(source, classifier, 2, first)
    assert len(classifier.calls) == 4
    assert (second.p_fraud >= 0).all()


@pytest.mark.parametrize("split", ["calibration", "test"])
@pytest.mark.parametrize("values", [None, [None, "test"], ["calibration", ""], ["legacy", "test"]])
def test_held_out_reports_reject_legacy_and_invalid_splits(split, values):
    frame = pd.DataFrame({"transaction_id": ["1", "2"]})
    if values is not None:
        frame["split"] = values
    with pytest.raises(ValueError, match="split metadata.*Rebuild"):
        frame_for(frame, split)


def test_held_out_reports_reject_overlapping_transactions_and_absent_period():
    frame = pd.DataFrame({"transaction_id": ["1", "1"], "split": ["calibration", "test"]})
    with pytest.raises(ValueError, match="multiple splits"):
        frame_for(frame, "test")
    with pytest.raises(ValueError, match="no 'test' rows"):
        frame_for(frame.iloc[:1], "test")
