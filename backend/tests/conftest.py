from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.jev.questions import RISK_LEVELS
from app.main import create_app
from app.schemas.transaction import ChoiceResult, JevAnswers, NoulResult, ScoreResult


def make_answers(
    p_fraud: float,
    *,
    risk_conf: float = 0.9,
    pattern: str = "legitimate",
    risk_score: float | None = None,
) -> JevAnswers:
    score = risk_score if risk_score is not None else p_fraud * 4
    return JevAnswers(
        model="jev-test",
        is_fraud=NoulResult(noul=p_fraud),
        risk=ScoreResult(
            score=score,
            legend={i: lvl for i, lvl in enumerate(RISK_LEVELS)},
            probabilities={i: (1.0 if i == round(score) else 0.0) for i in range(len(RISK_LEVELS))},
            confidence=risk_conf,
        ),
        pattern=ChoiceResult(choice=pattern, probabilities={pattern: 1.0}, confidence=0.95),
        signals={"amount_anomalous": NoulResult(noul=p_fraud)},
        input_tokens=120,
        output_tokens=30,
        latency_ms=42.0,
    )


class FakeClassifier:
    """Deterministic classifier: fraud probability = state['transaction']['amount_usd'] / 1000."""

    model = "jev-test"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def classify(self, state: dict[str, Any]) -> JevAnswers:
        self.calls.append(state)
        amount = state["transaction"]["amount_usd"]
        p = max(0.0, min(1.0, amount / 1000))
        return make_answers(p, risk_conf=0.9)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        TYPESAFE_API_KEY="test-key",
        POLICY_T_LOW=0.2,
        POLICY_T_HIGH=0.8,
        POLICY_C_MIN=0.6,
        OTP_DEV_MODE=True,
        OTP_MAX_ATTEMPTS=3,
        AGENT_ENABLED=False,
        SERVE_FRONTEND=False,
        _env_file=None,
    )


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
async def client(settings: Settings, fake_classifier: FakeClassifier):
    app = create_app(settings=settings, classifier=fake_classifier)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            ac.app = app  # type: ignore[attr-defined]
            yield ac
