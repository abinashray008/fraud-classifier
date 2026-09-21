"""Domain schemas: transaction input, Jev answers, decisions, challenges, verdicts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator


class Transaction(BaseModel):
    """IEEE-CIS transaction input plus optional caller-supplied history context.

    Names follow the dataset where Vesta published a meaning (amount, product
    code, card network/type, email domains, device type/info, address codes,
    TransactionDT). C1–C14 and D1–D15 are kept under their opaque names:
    Vesta documented them only as counts and timedeltas and masked the rest.
    They are accepted for eval dumps and are not interpreted as device age,
    device counts, or similar. ProductCD values (W/C/H/S/R) are category codes
    with no published mapping — do not gloss H as hotel, R as recurring, etc.

    Jev state is built from the verified fields above plus features computed
    from this card's transaction history (see `app.features.history`).
    """

    transaction_id: str | None = None
    card_id: str | None = Field(default=None, description="Stable per-card key for history lookups")

    # Core transaction fields (meanings published by Vesta / visible in values)
    amount: float = Field(alias="TransactionAmt", ge=0)
    product_code: str | None = Field(
        default=None,
        alias="ProductCD",
        description="Opaque product category code (W/C/H/S/R). No published mapping.",
    )
    card_network: str | None = Field(
        default=None, alias="card4", description="visa/mastercard/... as it appears in the data"
    )
    card_type: str | None = Field(
        default=None, alias="card6", description="debit/credit as it appears in the data"
    )
    purchaser_email_domain: str | None = Field(default=None, alias="P_emaildomain")
    recipient_email_domain: str | None = Field(default=None, alias="R_emaildomain")
    billing_region: str | None = Field(
        default=None,
        alias="addr1",
        description="IEEE-CIS addr1 (address code). Treated as a region-like code; mapping unpublished.",
    )
    billing_country: str | None = Field(
        default=None,
        alias="addr2",
        description="IEEE-CIS addr2 (address code). Treated as a country-like code; mapping unpublished.",
    )
    dist1: float | None = Field(
        default=None,
        alias="dist1",
        validation_alias=AliasChoices("dist1", "distance_billing_to_purchase"),
        description="IEEE-CIS dist1. Vesta labeled it a distance; unit and endpoints unpublished.",
    )
    timestamp_delta_seconds: int | None = Field(
        default=None,
        alias="TransactionDT",
        description="Seconds since an unpublished reference time; used only as hour-of-day = (dt/3600) % 24",
    )
    transaction_time: datetime | None = None

    # Identity table fields
    device_type: str | None = Field(default=None, alias="DeviceType")
    device_info: str | None = Field(default=None, alias="DeviceInfo")

    # Opaque IEEE-CIS C (counts) and D (timedeltas). Meanings masked; not sent to Jev.
    c1: float | None = Field(default=None, alias="C1")
    c2: float | None = Field(default=None, alias="C2")
    c13: float | None = Field(default=None, alias="C13")
    c14: float | None = Field(default=None, alias="C14")
    d1: float | None = Field(default=None, alias="D1")
    d2: float | None = Field(default=None, alias="D2")
    d4: float | None = Field(default=None, alias="D4")
    d10: float | None = Field(default=None, alias="D10")
    d15: float | None = Field(default=None, alias="D15")

    # Card-present authorization. Set when the cardholder is at a merchant terminal
    # and the acquirer has sent the request through the card network for an
    # approve or decline. Omitted for card-not-present and IEEE-CIS rows.
    channel: Literal["card_present", "card_not_present"] | None = None
    entry_mode: Literal["swipe", "chip", "contactless", "fallback_swipe", "keyed"] | None = None
    card_status: Literal["open", "lost", "stolen", "expired", "blocked"] | None = None
    cvm_result: Literal["pin_verified", "pin_failed", "signature", "no_cvm"] | None = None
    pin_tries_exceeded: bool | None = None
    track_cvv: Literal["match", "mismatch", "not_present"] | None = Field(
        default=None,
        description="CVV1/CVC1 check on the magstripe track. Mismatch is a counterfeit signal.",
    )
    merchant_id: str | None = None
    merchant_name: str | None = None
    mcc: str | None = Field(default=None, description="ISO 18245 merchant category code")
    merchant_country: str | None = None
    merchant_city: str | None = None
    terminal_id: str | None = None
    terminal_attended: bool | None = Field(
        default=None,
        description="True at an attended POS. False at an unattended terminal (ATM, pump, kiosk).",
    )
    cardholder_country: str | None = None
    available_credit_usd: float | None = Field(default=None, ge=0)
    single_purchase_limit_usd: float | None = Field(default=None, ge=0)

    # Optional context for offline `build_state(tx)` calls. Live scoring ignores these
    # and reads the feature store instead.
    card_avg_amount: float | None = Field(
        default=None,
        description="Mean prior amount in USD. Live scoring does not use this; it reads the feature store.",
    )
    card_known_devices: list[str] | None = Field(
        default=None,
        description="DeviceInfo values previously seen on this card. Live scoring does not use this.",
    )

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
        "merchant_id",
        "merchant_name",
        "mcc",
        "merchant_country",
        "merchant_city",
        "terminal_id",
        "cardholder_country",
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
    fraud_probability: float
    risk_score: float
    # Concentration of the risk-level distribution. Not P(the fraud answer is correct).
    risk_confidence: float
    review_reason: Literal["contradictory", "insufficient_evidence", "ambiguous"] | None = None


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
