"""Domain schemas: transaction input, Jev answers, decisions, challenges, verdicts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    """Readable subset of an IEEE-CIS transaction plus optional identity fields.

    Field names mirror the dataset where sensible so the eval loader and the API
    share one schema. Raw C/D velocity columns are carried under descriptive names.
    """

    transaction_id: str | None = None
    card_id: str | None = Field(default=None, description="Stable per-card key for history lookups")

    # Core transaction fields
    amount: float = Field(alias="TransactionAmt", ge=0)
    product_code: str | None = Field(default=None, alias="ProductCD")
    card_network: str | None = Field(default=None, alias="card4", description="visa/mastercard/...")
    card_type: str | None = Field(default=None, alias="card6", description="debit/credit")
    purchaser_email_domain: str | None = Field(default=None, alias="P_emaildomain")
    recipient_email_domain: str | None = Field(default=None, alias="R_emaildomain")
    billing_region: str | None = Field(default=None, alias="addr1")
    billing_country: str | None = Field(default=None, alias="addr2")
    distance_billing_to_purchase: float | None = Field(default=None, alias="dist1")
    timestamp_delta_seconds: int | None = Field(
        default=None,
        alias="TransactionDT",
        description="Seconds since dataset reference time; used to derive hour of day",
    )
    transaction_time: datetime | None = None

    # Identity table fields
    device_type: str | None = Field(default=None, alias="DeviceType")
    device_info: str | None = Field(default=None, alias="DeviceInfo")

    # Velocity / recency (IEEE-CIS C and D columns, described)
    card_txn_count: float | None = Field(default=None, alias="C1")
    addr_match_count: float | None = Field(default=None, alias="C2")
    email_txn_count: float | None = Field(default=None, alias="C13")
    device_txn_count: float | None = Field(default=None, alias="C14")
    days_since_prev_txn: float | None = Field(default=None, alias="D1")
    days_since_card_first_seen: float | None = Field(default=None, alias="D2")
    days_since_prev_txn_same_addr: float | None = Field(default=None, alias="D4")
    days_since_prev_txn_same_amount: float | None = Field(default=None, alias="D10")
    days_since_device_first_seen: float | None = Field(default=None, alias="D15")

    # Optional context the caller may already know (real-time feature store)
    card_avg_amount: float | None = None
    card_known_devices: list[str] | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator(
        "billing_region",
        "billing_country",
        "product_code",
        "card_network",
        "card_type",
        "device_type",
        "device_info",
        "purchaser_email_domain",
        "recipient_email_domain",
        mode="before",
    )
    @classmethod
    def _coerce_code_to_str(cls, v):
        """IEEE-CIS encodes region codes as floats (e.g. 204.0); normalize to '204'."""
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)


class DecisionOutcome(StrEnum):
    APPROVE = "APPROVE"
    STEP_UP = "STEP_UP"
    DECLINE = "DECLINE"


class NoulResult(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceResult(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreResult(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[int, str]
    probabilities: dict[int, float]
    confidence: float


class JevAnswers(BaseModel):
    """Typed view over the Jev response for the fraud question set."""

    model: str
    is_fraud: NoulResult
    risk: ScoreResult
    pattern: ChoiceResult
    signals: dict[str, NoulResult] = Field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    request_id: str | None = None


class PolicyExplanation(BaseModel):
    rule: str
    t_low: float
    t_high: float
    c_min: float
    fraud_probability: float
    risk_confidence: float


class Verdict(BaseModel):
    """Structured output from the investigation agent."""

    fraud_prob: float = Field(ge=0, le=1)
    pattern: Literal[
        "legitimate",
        "card_testing",
        "stolen_card",
        "account_takeover",
        "friendly_fraud_risk",
        "other",
    ]
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class InvestigationRecord(BaseModel):
    status: Literal["not_started", "running", "completed", "failed", "skipped"] = "not_started"
    verdict: Verdict | None = None
    model_route: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)
    blocked_tool_calls: list[dict] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ChallengeStatus(StrEnum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class Challenge(BaseModel):
    challenge_id: str
    decision_id: str
    status: ChallengeStatus = ChallengeStatus.PENDING
    attempts: int = 0
    max_attempts: int
    expires_at: datetime
    created_at: datetime
    # Only populated in dev mode.
    dev_code: str | None = None


class DecisionRecord(BaseModel):
    decision_id: str
    created_at: datetime
    transaction: Transaction
    state: dict
    jev: JevAnswers
    decision: DecisionOutcome
    explanation: PolicyExplanation
    challenge_id: str | None = None
    final_decision: DecisionOutcome | None = None
    final_reason: str | None = None
    investigation: InvestigationRecord = Field(default_factory=InvestigationRecord)


class ScoreResponse(BaseModel):
    decision_id: str
    decision: DecisionOutcome
    jev: JevAnswers
    explanation: PolicyExplanation
    challenge_id: str | None = None
    # Dev mode only
    dev_otp_code: str | None = None


class VerifyRequest(BaseModel):
    code: str


class VerifyResponse(BaseModel):
    decision_id: str
    challenge_status: ChallengeStatus
    final_decision: DecisionOutcome | None
    final_reason: str | None
    attempts_remaining: int
    investigation: InvestigationRecord
