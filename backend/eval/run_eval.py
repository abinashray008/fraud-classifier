"""Score a labeled sample with Jev and cache per-row answers.

Answers are cached by input, questions, feature definitions, and resolved model.
Legacy caches are rebuilt; mutable model aliases are always rescored.

Usage:
    TYPESAFE_API_KEY=... python -m eval.run_eval --sample eval/reports/sample.parquet \
        --model jev-1.13.0 --concurrency 8 --out eval/reports/answers_jev-1.13.0.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.features.state_builder import build_state
from app.jev.classifier import JevFraudClassifier
from app.jev.questions import SIGNAL_QUESTIONS, build_questions
from app.schemas.transaction import JevAnswers
from eval.load_ieee import row_history, row_to_transaction

CACHE_VERSION = 1


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def feature_fingerprint() -> str:
    """Invalidate results when feature computation, mapping, or schemas change."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / "eval/load_ieee.py", root / "app/jev/classifier.py"]
    for directory in ("app/features", "app/schemas"):
        paths.extend(sorted((root / directory).glob("*.py")))
    return _fingerprint({str(p.relative_to(root)): p.read_text() for p in paths})


def _cache_key(provenance: dict, model: str) -> str:
    return _fingerprint({**provenance, "resolved_model": model})


def _cell(row: pd.Series, key: str):
    if key not in row.index:
        return None
    value = row[key]
    if value is None or pd.isna(value):
        return None
    return value


def answers_to_row(
    tx_id: str,
    label: int,
    a: JevAnswers,
    *,
    split: str | None = None,
    sample_weight: float | None = None,
) -> dict:
    row = {
        "transaction_id": tx_id,
        "label": label,
        "model": a.model,
        "p_fraud": a.is_fraud.noul,
        "risk_score": a.risk.score,
        "risk_confidence": a.risk.confidence,
        "pattern": a.pattern.choice,
        "pattern_confidence": a.pattern.confidence,
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
        "latency_ms": a.latency_ms,
        "risk_probabilities": json.dumps(a.risk.probabilities),
        "pattern_probabilities": json.dumps(a.pattern.probabilities),
    }
    for q in SIGNAL_QUESTIONS:
        row[f"signal_{q}"] = a.signals[q].noul if q in a.signals else None
    if split is not None:
        row["split"] = split
    if sample_weight is not None:
        row["sample_weight"] = sample_weight
    return row


async def score_sample(
    sample: pd.DataFrame,
    classifier: JevFraudClassifier,
    concurrency: int,
    cache: pd.DataFrame | None,
) -> pd.DataFrame:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if sample["TransactionID"].isna().any() or sample["TransactionID"].astype(str).duplicated().any():
        raise ValueError("sample must have unique, non-null TransactionID values")
    runnable = getattr(classifier, "runnable", None)
    questions = runnable.questions if runnable is not None else build_questions()
    context = {
        "cache_version": CACHE_VERSION,
        "questions_fingerprint": _fingerprint(
            {name: q.model_dump(mode="json", exclude_none=True) for name, q in questions.items()}
        ),
        "features_fingerprint": feature_fingerprint(),
    }
    cached = {}
    required = {*context, "input_fingerprint", "cache_key", "model", "transaction_id"}
    # Only an explicitly pinned version can be resolved without a fresh request.
    # A previous response cannot tell us what jev-latest resolves to today.
    pinned = re.fullmatch(r"jev-\d+\.\d+\.\d+", classifier.model) is not None
    if pinned and cache is not None and required <= set(cache.columns):
        valid = cache[list(required)].notna().all(axis=1) & cache["model"].eq(classifier.model)
        for name, value in context.items():
            valid &= cache[name].eq(value)
        for record in cache.loc[valid.fillna(False)].to_dict(orient="records"):
            provenance = {**context, "input_fingerprint": record["input_fingerprint"]}
            if record["cache_key"] == _cache_key(provenance, classifier.model):
                cached[(str(record["transaction_id"]), record["cache_key"])] = record

    sem = asyncio.Semaphore(concurrency)
    rows: list[dict | None] = [None] * len(sample)
    errors = 0
    hits = 0
    scored = 0
    print(f"checking {len(sample)} rows for compatible cached answers with {classifier.model}")

    async def one(index: int, row: pd.Series) -> None:
        nonlocal errors, hits, scored
        tx = row_to_transaction(row)
        state = build_state(tx, history=row_history(row))
        # Include raw inputs as well as the exact state (which can round values).
        # Reporting metadata is refreshed from the current sample on every hit.
        inputs = row.drop(labels=["isFraud", "split", "sample_weight"], errors="ignore")
        provenance = {
            **context,
            "input_fingerprint": _fingerprint(
                {
                    "input": json.loads(inputs.sort_index().to_json(date_format="iso", double_precision=15)),
                    "state": state,
                }
            ),
        }
        key = _cache_key(provenance, classifier.model)
        hit = cached.get((tx.transaction_id or "", key))
        if hit is not None:
            result = dict(hit)
            hits += 1
        else:
            async with sem:
                try:
                    answers = await classifier.classify(state)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"  ! {tx.transaction_id}: {type(exc).__name__}: {exc}")
                    return
            result = answers_to_row(tx.transaction_id or "", int(row["isFraud"]), answers)
            result.update(provenance)
            result["cache_key"] = _cache_key(provenance, answers.model)
            scored += 1
            if scored % 100 == 0:
                print(f"  scored {scored} rows")
        result["requested_model"] = classifier.model
        result["label"] = int(row["isFraud"])
        result.pop("split", None)
        result.pop("sample_weight", None)
        split = _cell(row, "split")
        weight = _cell(row, "sample_weight")
        if split is not None:
            result["split"] = str(split)
        if weight is not None:
            result["sample_weight"] = float(weight)
        rows[index] = result

    started = time.perf_counter()
    await asyncio.gather(*(one(i, r) for i, (_, r) in enumerate(sample.iterrows())))
    elapsed = time.perf_counter() - started
    print(f"{hits} cached, scored {scored} rows in {elapsed:.1f}s ({errors} errors)")
    # Never carry stale, failed, or out-of-sample rows into a report.
    return pd.DataFrame([row for row in rows if row is not None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=Path("eval/reports/sample.parquet"))
    ap.add_argument("--model", default=None, help="Jev model version; defaults to JEV_MODEL")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    settings = get_settings()
    model = args.model or settings.jev_model
    out = args.out or Path(f"eval/reports/answers_{model}.parquet")

    sample = pd.read_parquet(args.sample)
    if args.limit:
        sample = sample.head(args.limit)
    cache = pd.read_parquet(out) if out.exists() else None

    classifier = JevFraudClassifier(
        api_key=settings.typesafe_api_key,
        model=model,
        timeout=settings.jev_timeout_seconds * 4,
        max_retries=3,
    )
    result = asyncio.run(score_sample(sample, classifier, args.concurrency, cache))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"wrote {len(result)} answers to {out}")


if __name__ == "__main__":
    main()
