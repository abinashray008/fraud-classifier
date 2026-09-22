"""In-memory feature store stub backing the investigation tools.

In production this would be a real feature store / warehouse. Here it is seeded
with a few synthetic card histories so the agent has something to look up, and it
learns from every scored transaction (`record`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Literal
from uuid import uuid4

from app.features.card_present import is_card_present
from app.features.history import (
    AUTHORIZATION_COUNTER_KEYS,
    PriorTxn,
    compute_card_history_features,
    compute_merchant_features,
    distinct_device_ids,
)
from app.schemas.transaction import Transaction

# Synthetic IDs model keys issued by a trusted device-enrollment service.
# DeviceInfo strings remain descriptions, including model/build strings.
DEMO_GOOD_CARD = "card_good_001"
DEMO_TRUSTED_DEVICE = "demo:device:good-001"
DEMO_SHARED_DEVICE = "demo:device:shared-001"
_RECENT_ATTEMPTS = 5
_AUTHORIZATION_DECISIONS = ("pending", "approved", "declined")
_FRAUD_OUTCOMES = ("legitimate", "fraud", "chargeback")


@dataclass
class HistoricalTxn:
    amount: float
    timestamp: datetime
    device_info: str | None
    email_domain: str | None
    billing_region: str | None
    authorization_decision: str  # pending | approved | declined
    merchant_id: str | None = None
    merchant_country: str | None = None
    device_id: str | None = None
    record_id: str | None = None
    fraud_outcome: str | None = None
    outcome_source: str | None = None
    outcome_evidence_ref: str | None = None
    outcome_verified_at: datetime | None = None


@dataclass(frozen=True)
class DeviceOwnershipEvent:
    device_id: str
    verified: bool
    at: datetime
    source: str
    evidence_ref: str


@dataclass
class CardProfile:
    card_id: str
    history: list[HistoricalTxn] = field(default_factory=list)
    device_ownership: list[DeviceOwnershipEvent] = field(default_factory=list)
    known_email_domains: set[str] = field(default_factory=set)
    home_region: str | None = None

    def trusted_device_ids(self, at: datetime) -> list[str]:
        latest: dict[str, bool] = {}
        for event in sorted(self.device_ownership, key=lambda e: e.at):
            if event.at <= _aware(at):
                latest[event.device_id] = event.verified
        return sorted(device_id for device_id, verified in latest.items() if verified)

    @property
    def known_devices(self) -> set[str]:
        """Compatibility view of independently verified, currently active ownership."""
        return set(self.trusted_device_ids(datetime.now(UTC)))


KNOWN_BAD_DOMAINS = {"mailinator.com", "guerrillamail.com", "yopmail.com", "tempmail.com"}
HIGH_RISK_DOMAINS = {"anonymous.com", "protonmail.com", "mail.ru"}


class FeatureStore:
    def __init__(self) -> None:
        self.cards: dict[str, CardProfile] = {}
        self.device_index: dict[str, set[str]] = defaultdict(set)  # trusted device_id -> card_ids
        self.device_chargebacks: dict[str, int] = defaultdict(int)
        self._transactions: dict[tuple[str, str], HistoricalTxn] = {}

    # Seeding -----------------------------------------------------------------
    def seed_demo(self) -> None:
        now = datetime.now(UTC)
        good = CardProfile(card_id=DEMO_GOOD_CARD, home_region="315")
        for i in range(12):
            good.history.append(
                HistoricalTxn(
                    amount=40 + (i % 4) * 15,
                    timestamp=now - timedelta(days=90 - i * 7),
                    device_info="SM-G950F Build/R16NW",
                    device_id=DEMO_TRUSTED_DEVICE,
                    email_domain="gmail.com",
                    billing_region="315",
                    authorization_decision="approved",
                )
            )
        good.known_email_domains.add("gmail.com")
        self.cards[good.card_id] = good
        self.verify_device_ownership(
            good.card_id,
            DEMO_TRUSTED_DEVICE,
            source="demo_enrollment_registry",
            evidence_ref="demo:enrollment:good-001",
            verified_at=now - timedelta(days=91),
        )
        self.device_index[DEMO_TRUSTED_DEVICE].add(good.card_id)

        # Established spending, but every prior device string is the generic
        # label "Windows" — a description, not a device ID.
        ato = CardProfile(card_id="card_ato_002", home_region="204")
        for i in range(8):
            ato.history.append(
                HistoricalTxn(
                    amount=25 + i * 3,
                    timestamp=now - timedelta(days=120 - i * 12),
                    device_info="Windows",
                    email_domain="outlook.com",
                    billing_region="204",
                    authorization_decision="approved",
                )
            )
        ato.known_email_domains.add("outlook.com")
        self.cards[ato.card_id] = ato

        # Card-present history: grocery swipes in the US, plus one 40 minutes ago
        # so a follow-up in another country trips the 2-hour travel rule.
        swipe = CardProfile(card_id="card_swipe_001")
        for i in range(6):
            swipe.history.append(
                HistoricalTxn(
                    amount=40 + (i % 3) * 8,
                    timestamp=now - timedelta(days=60 - i * 9),
                    device_info=None,
                    email_domain=None,
                    billing_region=None,
                    authorization_decision="approved",
                    merchant_id="merch_grocery_88",
                    merchant_country="US",
                )
            )
        swipe.history.append(
            HistoricalTxn(
                amount=46.0,
                timestamp=now - timedelta(minutes=40),
                device_info=None,
                email_domain=None,
                billing_region=None,
                authorization_decision="approved",
                merchant_id="merch_grocery_88",
                merchant_country="US",
            )
        )
        self.cards[swipe.card_id] = swipe

        # Independent synthetic dispute evidence, separate from authorizations.
        for i in range(1, 5):
            card_id, record_id = f"card_x{i}", f"demo:transaction:shared-{i}"
            self.record(
                Transaction(
                    amount=20,
                    card_id=card_id,
                    device_id=DEMO_SHARED_DEVICE,
                    DeviceInfo="SM-G9650 Build/R16NW",
                    transaction_time=now - timedelta(days=7),
                ),
                "approved",
                record_id=record_id,
            )
            if i <= 3:
                self.record_verified_outcome(
                    card_id,
                    record_id,
                    "chargeback",
                    source="demo_dispute_feed",
                    evidence_ref=f"demo:dispute:{i}",
                    verified_at=now - timedelta(days=1),
                )

    # Writes --------------------------------------------------------------------
    def record(
        self, tx: Transaction, authorization_decision: str = "pending", *, record_id: str | None = None
    ) -> HistoricalTxn | None:
        """Upsert one authorization attempt. A decision never verifies ownership or fraud."""
        if authorization_decision not in _AUTHORIZATION_DECISIONS:
            raise ValueError("Use record_verified_outcome for independent fraud outcomes")
        if not tx.card_id:
            return None
        key = (tx.card_id, record_id or tx.transaction_id or uuid4().hex)
        existing = self._transactions.get(key)
        if existing is not None:
            if existing.device_id != tx.device_id or existing.amount != tx.amount:
                raise ValueError("Transaction identity and amount cannot change during a decision update")
            existing.authorization_decision = authorization_decision
            return existing
        profile = self.cards.setdefault(tx.card_id, CardProfile(card_id=tx.card_id))
        row = HistoricalTxn(
            record_id=key[1],
            amount=tx.amount,
            timestamp=tx.transaction_time or datetime.now(UTC),
            device_info=tx.device_info,
            device_id=tx.device_id,
            email_domain=tx.purchaser_email_domain,
            billing_region=tx.billing_region,
            authorization_decision=authorization_decision,
            merchant_id=tx.merchant_id,
            merchant_country=tx.merchant_country.strip().upper() if tx.merchant_country else None,
        )
        profile.history.append(row)
        self._transactions[key] = row
        if tx.device_id:
            self.device_index[tx.device_id].add(tx.card_id)
        return row

    def verify_device_ownership(
        self,
        card_id: str,
        device_id: str,
        *,
        source: str,
        evidence_ref: str,
        verified_at: datetime | None = None,
    ) -> None:
        """Trusted integration only: enrollment/authentication bound to this card AND device.

        The adapter must verify the evidence before calling; neither model output nor
        this demo's unbound OTP is ownership evidence. Not exposed as a public tool/API.
        """
        self._ownership_event(card_id, device_id, True, source, evidence_ref, verified_at)

    def revoke_device_ownership(
        self,
        card_id: str,
        device_id: str,
        *,
        source: str,
        evidence_ref: str,
        revoked_at: datetime | None = None,
    ) -> None:
        """Revoke ownership until a later independently verified enrollment/authentication."""
        self._ownership_event(card_id, device_id, False, source, evidence_ref, revoked_at)

    def _ownership_event(self, card_id, device_id, verified, source, evidence_ref, at) -> None:
        if not all(v and v.strip() for v in (card_id, device_id, source, evidence_ref)):
            raise ValueError("Ownership events require card/device IDs, source and evidence reference")
        profile = self.cards.setdefault(card_id, CardProfile(card_id=card_id))
        for event in profile.device_ownership:
            if event.source == source and event.evidence_ref == evidence_ref:
                if event.device_id != device_id or event.verified != verified:
                    raise ValueError("Ownership evidence reference already used for a different event")
                return  # Replaying an old verification must never undo a later revocation.
        profile.device_ownership.append(
            DeviceOwnershipEvent(
                device_id=device_id,
                verified=verified,
                at=_aware(at or datetime.now(UTC)),
                source=source,
                evidence_ref=evidence_ref,
            )
        )

    def record_verified_outcome(
        self,
        card_id: str,
        record_id: str,
        outcome: Literal["legitimate", "fraud", "chargeback"],
        *,
        source: str,
        evidence_ref: str,
        verified_at: datetime | None = None,
    ) -> None:
        """Trusted adjudication/chargeback feed only; never called by the classifier or agent."""
        if outcome not in _FRAUD_OUTCOMES or not source.strip() or not evidence_ref.strip():
            raise ValueError("Verified outcomes require a supported outcome, source and evidence reference")
        row = self._transactions[(card_id, record_id)]
        if row.device_id:
            self.device_chargebacks[row.device_id] += int(outcome == "chargeback") - int(
                row.fraud_outcome == "chargeback"
            )
        row.fraud_outcome = outcome
        row.outcome_source = source
        row.outcome_evidence_ref = evidence_ref
        row.outcome_verified_at = _aware(verified_at or datetime.now(UTC))

    # Reads (used by tools) ----------------------------------------------------
    def card_history(self, card_id: str, limit: int = 10) -> dict:
        profile = self.cards.get(card_id)
        if profile is None:
            return {"card_id": card_id, "found": False, "note": "No history for this card"}
        hist = sorted(profile.history, key=lambda h: h.timestamp, reverse=True)[:limit]
        amounts = [h.amount for h in profile.history] or [0.0]
        return {
            "card_id": card_id,
            "found": True,
            "transaction_count": len(profile.history),
            "average_amount_usd": round(mean(amounts), 2),
            "max_amount_usd": round(max(amounts), 2),
            "known_devices": sorted(profile.known_devices),
            "device_ownership_events": [
                {
                    "device_id": e.device_id,
                    "verified": e.verified,
                    "at": e.at.isoformat(),
                    "source": e.source,
                    "evidence_ref": e.evidence_ref,
                }
                for e in profile.device_ownership
            ],
            "known_email_domains": sorted(profile.known_email_domains),
            "home_region": profile.home_region,
            "chargebacks": sum(1 for h in profile.history if h.fraud_outcome == "chargeback"),
            "recent": [
                {
                    "record_id": h.record_id,
                    "amount_usd": h.amount,
                    "when": h.timestamp.isoformat(),
                    "device_info": h.device_info,
                    "device_id": h.device_id,
                    "email_domain": h.email_domain,
                    "region": h.billing_region,
                    "authorization_decision": h.authorization_decision,
                    "fraud_outcome": h.fraud_outcome or "unknown",
                    "outcome_source": h.outcome_source,
                    "outcome_evidence_ref": h.outcome_evidence_ref,
                    **({"merchant_id": h.merchant_id} if h.merchant_id else {}),
                    **({"merchant_country": h.merchant_country} if h.merchant_country else {}),
                }
                for h in hist
            ],
        }

    def device_history(self, device_id: str | None = None) -> dict:
        if not device_id or device_id not in self.device_index:
            return {
                "device_id": device_id,
                "identifier_kind": "unknown",
                "distinct_cards_seen": None,
                "chargebacks_on_device": None,
                "shared_across_many_cards": None,
                "note": "No recorded trusted device ID. DeviceInfo cannot establish identity.",
            }
        cards = self.device_index[device_id]
        return {
            "device_id": device_id,
            "identifier_kind": "device_id",
            "distinct_cards_seen": len(cards),
            "chargebacks_on_device": self.device_chargebacks.get(device_id, 0),
            "shared_across_many_cards": len(cards) >= 3,
        }

    def email_domain_reputation(self, domain: str) -> dict:
        d = (domain or "").lower()
        if d in KNOWN_BAD_DOMAINS:
            rep = "disposable"
        elif d in HIGH_RISK_DOMAINS:
            rep = "high_risk"
        elif d in {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com"}:
            rep = "free_webmail"
        elif d:
            rep = "corporate_or_isp"
        else:
            rep = "unknown"
        return {"domain": d, "reputation": rep}

    def velocity_stats(self, card_id: str, window_hours: int = 24) -> dict:
        profile = self.cards.get(card_id)
        if profile is None:
            return {"card_id": card_id, "found": False}
        now = datetime.now(UTC)
        if window_hours <= 0:
            raise ValueError("window_hours must be positive")
        window = [
            h for h in profile.history if timedelta(0) <= now - _aware(h.timestamp) <= timedelta(hours=window_hours)
        ]
        distinct = distinct_device_ids([h.device_id for h in window])
        return {
            "card_id": card_id,
            "found": True,
            "window_hours": window_hours,
            "transactions_in_window": len(window),  # Compatibility alias for attempts.
            **{f"{name}_in_window": count for name, count in _authorization_counts(window, now).items()},
            "total_amount_in_window_usd": round(sum(h.amount for h in window), 2),
            "distinct_devices_in_window": distinct,
        }

    def history_features(self, tx: Transaction) -> dict:
        """Server-side history, verified ownership, and independent outcomes for Jev.

        Caller-supplied averages and device lists are not consulted. Missing
        card identity leaves the evidence unset (unknown) rather than zero.
        Device descriptions are never looked up as identities.

        `as_of` is `transaction_time` when present, otherwise now. TransactionDT is a
        dataset-relative timedelta and is not comparable to store timestamps.
        """
        as_of = tx.transaction_time or datetime.now(UTC)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)

        if not tx.card_id:
            features: dict = _unknown_card_history()
        else:
            profile = self.cards.get(tx.card_id)
            past = _past(profile, as_of) if profile is not None else []
            prior = [
                PriorTxn(
                    amount=h.amount,
                    timestamp=h.timestamp,
                    device_info=h.device_info,
                    device_id=h.device_id,
                    email_domain=h.email_domain,
                    fraud_outcome=(
                        h.fraud_outcome if h.outcome_verified_at and h.outcome_verified_at <= as_of else None
                    ),
                    merchant_id=h.merchant_id,
                    merchant_country=h.merchant_country,
                )
                for h in past
            ]
            features = compute_card_history_features(
                prior=prior,
                amount=tx.amount,
                at=as_of,
                device_info=tx.device_info,
                device_id=tx.device_id,
                email_domain=tx.purchaser_email_domain,
            )
            for label, duration in (("1h", timedelta(hours=1)), ("24h", timedelta(days=1)), ("7d", timedelta(days=7))):
                window = [h for h in past if as_of - _aware(h.timestamp) <= duration]
                features.update(
                    {f"{name}_last_{label}": count for name, count in _authorization_counts(window, as_of).items()}
                )
            if is_card_present(tx):
                features.update(
                    compute_merchant_features(
                        prior=prior,
                        at=as_of,
                        merchant_id=tx.merchant_id,
                        merchant_country=tx.merchant_country,
                        amount=tx.amount,
                    )
                )
            trusted = profile.trusted_device_ids(as_of) if profile else []
            features["trusted_device_ids"] = trusted
            features["recent_attempts"] = _recent_attempts(past, as_of)
            features["authorization_decisions"] = {
                name: sum(h.authorization_decision == name for h in past) for name in _AUTHORIZATION_DECISIONS
            }
            features["confirmed_outcomes"] = _confirmed_outcomes(past, as_of)
            features["current_device_is_trusted"] = tx.device_id in trusted if tx.device_id else None

        if tx.device_id:
            cards = self.device_index.get(tx.device_id, set())
            features["device_distinct_cards_seen"] = len(cards)
            features["device_chargebacks"] = self.device_chargebacks.get(tx.device_id, 0)
        else:
            features["device_distinct_cards_seen"] = None
            features["device_chargebacks"] = None
        return features


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _unknown_card_history() -> dict:
    """No card id was supplied, so card evidence was not looked up."""
    return {
        **dict.fromkeys(AUTHORIZATION_COUNTER_KEYS),
        "history_found": None,
        "prior_transaction_count": None,
        "days_since_first_transaction": None,
        "days_since_previous_transaction": None,
        "mean_amount_usd_prior": None,
        "transactions_last_1h": None,
        "transactions_last_24h": None,
        "transactions_last_7d": None,
        "amount_usd_last_1h": None,
        "amount_usd_last_24h": None,
        "amount_usd_last_7d": None,
        "distinct_devices_last_24h": None,
        "distinct_devices_last_7d": None,
        "device_seen_before_on_this_card": None,
        "email_domain_seen_before_on_this_card": None,
        "matching_amount_count_last_24h": None,
        "chargebacks_on_card": None,
        "trusted_device_ids": None,
        "recent_attempts": None,
        "confirmed_outcomes": None,
        "authorization_decisions": None,
        "current_device_is_trusted": None,
    }


def _past(profile: CardProfile, as_of: datetime) -> list[HistoricalTxn]:
    return [h for h in profile.history if _aware(h.timestamp) < as_of]


def _authorization_counts(rows: list[HistoricalTxn], as_of: datetime) -> dict[str, int]:
    """Separate observed attempts, current approvals, and independently confirmed fraud.

    Confirmed fraud counts only adjudicated fraud labels known at lookup time.
    An unlabeled attempt or a chargeback alone is not confirmed fraud.
    """
    return {
        "attempts": len(rows),
        "approvals": sum(h.authorization_decision == "approved" for h in rows),
        "confirmed_fraud": sum(
            h.fraud_outcome == "fraud" and h.outcome_verified_at is not None and h.outcome_verified_at <= as_of
            for h in rows
        ),
    }


def _confirmed_outcomes(past: list[HistoricalTxn], as_of: datetime) -> dict[str, int] | None:
    verified = [h for h in past if h.outcome_verified_at and h.outcome_verified_at <= as_of]
    if past and not verified:
        return None
    return {name: sum(h.fraud_outcome == name for h in verified) for name in _FRAUD_OUTCOMES}


def _recent_attempts(past: list[HistoricalTxn], as_of: datetime) -> list[dict]:
    ordered = sorted(past, key=lambda h: _aware(h.timestamp), reverse=True)[:_RECENT_ATTEMPTS]
    return [
        {
            "amount_usd": round(h.amount, 2),
            "days_ago": round((as_of - _aware(h.timestamp)).total_seconds() / 86400.0, 1),
            "device_info": h.device_info,
            "device_id": h.device_id,
            "authorization_decision": h.authorization_decision,
            "fraud_outcome": (
                h.fraud_outcome if h.outcome_verified_at and h.outcome_verified_at <= as_of else "unknown"
            ),
            **({"merchant_id": h.merchant_id} if h.merchant_id else {}),
            **({"merchant_country": h.merchant_country} if h.merchant_country else {}),
        }
        for h in ordered
    ]
