"""Exercise the real TypeSafeClassifier wire path against a mock Jev endpoint."""

from __future__ import annotations

import json

import httpx2
import pytest
from langchain_typesafe.client import TypeSafeAPITimeoutError, TypeSafeAuthenticationError

from app.features.state_builder import build_state
from app.jev.classifier import JevFraudClassifier
from app.jev.questions import PATTERN_LABELS, RISK_LEVELS, SIGNAL_QUESTIONS
from app.schemas.transaction import Transaction


def fake_jev_answers(p_fraud: float = 0.72) -> dict:
    risk_probs = {"0": 0.05, "1": 0.10, "2": 0.25, "3": 0.40, "4": 0.20}
    return {
        "model": "jev-1.13.0",
        "answers": {
            "is_fraud": {"type": "noul", "noul": p_fraud},
            "risk": {
                "type": "score",
                "score": sum(int(k) * v for k, v in risk_probs.items()),
                "legend": {str(i): lvl for i, lvl in enumerate(RISK_LEVELS)},
                "probabilities": risk_probs,
                "confidence": 0.61,
            },
            "pattern": {
                "type": "choice",
                "choice": "stolen_card",
                "probabilities": {k: (0.7 if k == "stolen_card" else 0.06) for k in PATTERN_LABELS},
                "confidence": 0.8,
            },
            **{q: {"type": "noul", "noul": 0.5} for q in SIGNAL_QUESTIONS},
        },
        "usage": {"input_tokens": 311, "output_tokens": 62},
    }


class MockJev:
    def __init__(self, *, fail_first: int = 0, status: int = 200) -> None:
        self.requests: list[dict] = []
        self.fail_first = fail_first
        self.status = status

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        if self.fail_first > 0:
            self.fail_first -= 1
            raise httpx2.ReadTimeout("simulated timeout", request=request)
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "nope"}})
        return httpx2.Response(200, json=fake_jev_answers(), headers={"x-typesafe-request-id": "req_123"})

    def classifier(self, **kw) -> JevFraudClassifier:
        return JevFraudClassifier(
            api_key="test-key",
            model="jev-1.13.0",
            async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)),
            **kw,
        )


async def test_payload_and_response_mapping():
    mock = MockJev()
    clf = mock.classifier()
    tx = Transaction(amount=1250, card4="visa", P_emaildomain="gmail.com", R_emaildomain="mailinator.com", D2=0.0)
    state = build_state(tx)

    answers = await clf.classify(state)

    # Request shape matches the Jev API: POST /v1/systemone with model/state/questions.
    req = mock.requests[0]
    assert req["url"].endswith("/v1/systemone")
    assert req["headers"]["authorization"] == "Bearer test-key"
    assert req["body"]["model"] == "jev-1.13.0"
    assert req["body"]["state"] == state
    qs = req["body"]["questions"]
    assert qs["is_fraud"]["type"] == "noul" and "criteria" in qs["is_fraud"]
    assert qs["risk"]["type"] == "score" and len(qs["risk"]["criteria"]) == 5
    assert qs["pattern"]["type"] == "choice" and set(qs["pattern"]["criteria"]) == set(PATTERN_LABELS)
    assert set(SIGNAL_QUESTIONS) <= set(qs)

    # Response mapping.
    assert answers.model == "jev-1.13.0"
    assert answers.is_fraud.noul == 0.72
    assert answers.risk.confidence == 0.61
    assert answers.risk.legend[4] == RISK_LEVELS[4]
    assert answers.risk.probabilities[3] == 0.40
    assert answers.pattern.choice == "stolen_card"
    assert set(answers.signals) == set(SIGNAL_QUESTIONS)
    assert answers.input_tokens == 311
    assert answers.request_id == "req_123"
    assert answers.latency_ms is not None and answers.latency_ms >= 0


async def test_retries_on_timeout_then_succeeds():
    mock = MockJev(fail_first=1)
    clf = mock.classifier(max_retries=1)
    answers = await clf.classify({"transaction": {"amount_usd": 5}})
    assert answers.is_fraud.noul == 0.72
    assert len(mock.requests) == 2


async def test_gives_up_after_max_retries():
    mock = MockJev(fail_first=3)
    clf = mock.classifier(max_retries=1)
    with pytest.raises(TypeSafeAPITimeoutError):
        await clf.classify({"transaction": {"amount_usd": 5}})
    assert len(mock.requests) == 2


async def test_auth_error_is_not_retried():
    mock = MockJev(status=401)
    clf = mock.classifier(max_retries=2)
    with pytest.raises(TypeSafeAuthenticationError):
        await clf.classify({"transaction": {"amount_usd": 5}})
    assert len(mock.requests) == 1
