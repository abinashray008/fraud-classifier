"""Score a labeled sample with Jev and cache per-row answers.

Answers are cached to disk keyed by (model, transaction_id) so threshold sweeps and
metric reports never re-spend tokens.

Usage:
    TYPESAFE_API_KEY=... python -m eval.run_eval --sample eval/reports/sample.parquet \
        --model jev-1.13.0 --concurrency 8 --out eval/reports/answers_jev-1.13.0.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.features.state_builder import build_state
from app.jev.classifier import JevFraudClassifier
from app.jev.questions import SIGNAL_QUESTIONS
from app.schemas.transaction import JevAnswers
from eval.load_ieee import row_history, row_to_transaction


def _cell(row: pd.Series, key: str):
    if key not in row.index:
        return None
    value = row[key]
    if value is None or (isinstance(value, float) and pd.isna(value)):
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
    done = set()
    if cache is not None and len(cache):
        done = set(cache.loc[cache["model"] == classifier.model, "transaction_id"].astype(str))
    todo = sample[~sample["TransactionID"].astype(str).isin(done)]
    print(f"{len(done)} cached, {len(todo)} to score with {classifier.model}")

    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    errors = 0

    async def one(row: pd.Series) -> None:
        nonlocal errors
        tx = row_to_transaction(row)
        state = build_state(tx, history=row_history(row))
        async with sem:
            try:
                answers = await classifier.classify(state)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"  ! {tx.transaction_id}: {type(exc).__name__}: {exc}")
                return
        split = _cell(row, "split")
        weight = _cell(row, "sample_weight")
        rows.append(
            answers_to_row(
                tx.transaction_id or "",
                int(row["isFraud"]),
                answers,
                split=None if split is None else str(split),
                sample_weight=None if weight is None else float(weight),
            )
        )
        if len(rows) % 100 == 0:
            print(f"  scored {len(rows)}/{len(todo)}")

    started = time.perf_counter()
    await asyncio.gather(*(one(r) for _, r in todo.iterrows()))
    elapsed = time.perf_counter() - started
    print(f"scored {len(rows)} rows in {elapsed:.1f}s ({errors} errors)")

    new = pd.DataFrame(rows)
    if cache is not None and len(cache):
        return pd.concat([cache, new], ignore_index=True).drop_duplicates(
            subset=["model", "transaction_id"], keep="last"
        )
    return new


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
