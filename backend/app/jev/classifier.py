"""Thin adapter around `TypeSafeClassifier`.

Isolates the alpha `langchain-typesafe` API from the rest of the app, adds a
timeout/retry envelope, and maps the response into `JevAnswers`. Tests inject a
fake via the `FraudClassifier` protocol.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol

from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeRateLimitError,
)
from langchain_typesafe.types import ClassificationResponse

from app.jev.questions import (
    PATTERN_QUESTION,
    PRIMARY_QUESTION,
    RISK_QUESTION,
    SIGNAL_QUESTIONS,
    build_questions,
)
from app.schemas.transaction import ChoiceResult, JevAnswers, NoulResult, ScoreResult

logger = logging.getLogger(__name__)

RETRYABLE = (TypeSafeAPIConnectionError, TypeSafeAPITimeoutError, TypeSafeRateLimitError)


class FraudClassifier(Protocol):
    model: str

    async def classify(self, state: dict[str, Any]) -> JevAnswers: ...


def map_response(response: ClassificationResponse, latency_ms: float | None = None) -> JevAnswers:
    """Convert a raw `ClassificationResponse` into the app's `JevAnswers`."""
    nouls = response.nouls
    scores = response.scores
    choices = response.choices
    risk = scores[RISK_QUESTION]
    pattern = choices[PATTERN_QUESTION]
    return JevAnswers(
        model=response.model,
        is_fraud=NoulResult(noul=nouls[PRIMARY_QUESTION].noul),
        risk=ScoreResult(
            score=risk.score,
            legend={int(k): str(v) for k, v in risk.legend.items()},
            probabilities={int(k): float(v) for k, v in risk.probabilities.items()},
            confidence=risk.confidence,
        ),
        pattern=ChoiceResult(
            choice=pattern.choice,
            probabilities=pattern.probabilities,
            confidence=pattern.confidence,
        ),
        signals={q: NoulResult(noul=nouls[q].noul) for q in SIGNAL_QUESTIONS if q in nouls},
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
        request_id=response.request_id,
    )


class JevFraudClassifier:
    """Production classifier backed by TypeSafe's Jev model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jev-latest",
        timeout: float = 5.0,
        max_retries: int = 1,
        base_url: str | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if async_client is not None:
            kwargs["async_client"] = async_client  # e.g. httpx2.AsyncClient(transport=MockTransport)
        self._classifier = TypeSafeClassifier(
            questions=build_questions(),
            model=model,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

    @property
    def runnable(self) -> TypeSafeClassifier:
        """Expose the underlying Runnable for batch/eval usage."""
        return self._classifier

    async def classify(self, state: dict[str, Any]) -> JevAnswers:
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                response = await self._classifier.ainvoke(state)
                latency_ms = (time.perf_counter() - started) * 1000
                return map_response(response, latency_ms)
            except RETRYABLE as exc:
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                backoff = 0.2 * (2 ** (attempt - 1))
                logger.warning("Jev call failed (%s); retrying in %.1fs", type(exc).__name__, backoff)
                await asyncio.sleep(backoff)
