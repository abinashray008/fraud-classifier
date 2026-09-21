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

from app.features.card_present import is_card_present
from app.features.history import (
    PriorTxn,
    compute_card_history_features,
    compute_merchant_features,
    is_device_fingerprint,
)
from app.schemas.transaction import Transaction

# Seeded card the legit preset uses. The device string is a build fingerprint;
# "Windows" / "iOS Device" are descriptions and are not stored as identities.
DEMO_GOOD_CARD = "card_good_001"
DEMO_TRUSTED_DEVICE = "SM-G950F Build/R16NW"
_RECENT_ATTEMPTS = 5
_OUTCOMES = ("approved", "declined", "chargeback")


@dataclass
class HistoricalTxn:
    amount: float
    timestamp: datetime
    device_info: str | None
    email_domain: str | None
    billing_region: str | None
    outcome: str  # approved | declined | chargeback
    merchant_id: str | None = None
    merchant_country: str | None = None


@dataclass
class CardProfile:
    card_id: str
    history: list[HistoricalTxn] = field(default_factory=list)
    known_devices: set[str] = field(default_factory=set)
    known_email_domains: set[str] = field(default_factory=set)
    home_region: str | None = None


KNOWN_BAD_DOMAINS = {"mailinator.com", "guerrillamail.com", "yopmail.com", "tempmail.com"}
HIGH_RISK_DOMAINS = {"anonymous.com", "protonmail.com", "mail.ru"}


class FeatureStore:
    def __init__(self) -> None:
        self.cards: dict[str, CardProfile] = {}
        self.device_index: dict[str, set[str]] = defaultdict(set)  # device_info -> card_ids
        self.device_chargebacks: dict[str, int] = defaultdict(int)

    # Seeding -----------------------------------------------------------------
    def seed_demo(self) -> None:
        now = datetime.now(UTC)
        good = CardProfile(card_id=DEMO_GOOD_CARD, home_region="315")
        for i in range(12):
            good.history.append(
                HistoricalTxn(
                    amount=40 + (i % 4) * 15,
                    timestamp=now - timedelta(days=90 - i * 7),
                    device_info=DEMO_TRUSTED_DEVICE,
                    email_domain="gmail.com",
                    billing_region="315",
                    outcome="approved",
                )
            )
        good.known_devices.add(DEMO_TRUSTED_DEVICE)
        good.known_email_domains.add("gmail.com")
        self.cards[good.card_id] = good
        self.device_index[DEMO_TRUSTED_DEVICE].add(good.card_id)

        # Established spending, but every prior device string is the generic
        # label "Windows" — a description, not a trusted fingerprint.
        ato = CardProfile(card_id="card_ato_002", home_region="204")
        for i in range(8):
            ato.history.append(
                HistoricalTxn(
                    amount=25 + i * 3,
                    timestamp=now - timedelta(days=120 - i * 12),
                    device_info="Windows",
                    email_domain="outlook.com",
                    billing_region="204",
                    outcome="approved",
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
                    outcome="approved",
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
                outcome="approved",
                merchant_id="merch_grocery_88",
                merchant_country="US",
            )
        )
        self.cards[swipe.card_id] = swipe

        # A device seen across many cards with chargebacks: classic fraud farm.
        self.device_index["SM-G9650 Build/R16NW"].update({"card_x1", "card_x2", "card_x3", "card_x4"})
        self.device_chargebacks["SM-G9650 Build/R16NW"] = 3

    # Writes --------------------------------------------------------------------
    def record(self, tx: Transaction, outcome: str = "approved") -> None:
        if not tx.card_id:
            return
        profile = self.cards.setdefault(tx.card_id, CardProfile(card_id=tx.card_id))
        profile.history.append(
            HistoricalTxn(
                amount=tx.amount,
                timestamp=tx.transaction_time or datetime.now(UTC),
                device_info=tx.device_info,
                email_domain=tx.purchaser_email_domain,
                billing_region=tx.billing_region,
                outcome=outcome,
                merchant_id=tx.merchant_id,
                merchant_country=tx.merchant_country.strip().upper() if tx.merchant_country else None,
            )
        )
        if is_device_fingerprint(tx.device_info):
            self.device_index[tx.device_info].add(tx.card_id)
            if outcome == "approved":
                profile.known_devices.add(tx.device_info)

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
            "known_email_domains": sorted(profile.known_email_domains),
            "home_region": profile.home_region,
            "chargebacks": sum(1 for h in profile.history if h.outcome == "chargeback"),
            "recent": [
                {
                    "amount_usd": h.amount,
                    "when": h.timestamp.isoformat(),
                    "device": h.device_info,
                    "email_domain": h.email_domain,
                    "region": h.billing_region,
                    "outcome": h.outcome,
                    **({"merchant_id": h.merchant_id} if h.merchant_id else {}),
                    **({"merchant_country": h.merchant_country} if h.merchant_country else {}),
                }
                for h in hist
            ],
        }

    def device_history(self, device_info: str) -> dict:
        if not is_device_fingerprint(device_info):
            return {
                "device_info": device_info,
                "identifier_kind": "description",
                "distinct_cards_seen": None,
                "chargebacks_on_device": None,
                "shared_across_many_cards": None,
                "note": "Generic device description, not a unique fingerprint.",
            }
        cards = self.device_index.get(device_info, set())
        return {
            "device_info": device_info,
            "identifier_kind": "fingerprint",
            "distinct_cards_seen": len(cards),
            "chargebacks_on_device": self.device_chargebacks.get(device_info, 0),
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
        window = [h for h in profile.history if now - h.timestamp <= timedelta(hours=window_hours)]
        labels = [h.device_info for h in window if h.device_info]
        fingerprints = {d for d in labels if is_device_fingerprint(d)}
        distinct: int | None = None if labels and not fingerprints else len(fingerprints)
        return {
            "card_id": card_id,
            "found": True,
            "window_hours": window_hours,
            "transactions_in_window": len(window),
            "total_amount_in_window_usd": round(sum(h.amount for h in window), 2),
            "distinct_devices_in_window": distinct,
        }

    def history_features(self, tx: Transaction) -> dict:
        """Trusted card and device evidence for Jev's first decision.

        Caller-supplied averages and device lists are not consulted. Missing
        card identity leaves the evidence unset (unknown) rather than zero.
        Generic device strings are not looked up as fingerprints.

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
                    email_domain=h.email_domain,
                    outcome=h.outcome,
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
                email_domain=tx.purchaser_email_domain,
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
            trusted = _trusted_device_ids(past)
            features["trusted_device_ids"] = trusted
            features["recent_attempts"] = _recent_attempts(past, as_of)
            features["confirmed_outcomes"] = _confirmed_outcomes(past)
            if is_device_fingerprint(tx.device_info):
                features["current_device_is_trusted"] = tx.device_info in trusted
            else:
                features["current_device_is_trusted"] = None

        if is_device_fingerprint(tx.device_info):
            cards = self.device_index.get(tx.device_info, set())
            features["device_distinct_cards_seen"] = len(cards)
            features["device_chargebacks"] = self.device_chargebacks.get(tx.device_info, 0)
        elif tx.device_info:
            features["device_distinct_cards_seen"] = None
            features["device_chargebacks"] = None
        return features


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _unknown_card_history() -> dict:
    """No card id was supplied, so card evidence was not looked up."""
    return {
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
        "current_device_is_trusted": None,
    }


def _past(profile: CardProfile, as_of: datetime) -> list[HistoricalTxn]:
    return [h for h in profile.history if _aware(h.timestamp) < as_of]


def _trusted_device_ids(past: list[HistoricalTxn]) -> list[str]:
    return sorted(
        {
            h.device_info
            for h in past
            if h.outcome == "approved" and is_device_fingerprint(h.device_info) and h.device_info
        }
    )


def _confirmed_outcomes(past: list[HistoricalTxn]) -> dict[str, int] | None:
    if past and not any(h.outcome in _OUTCOMES for h in past):
        return None
    return {name: sum(1 for h in past if h.outcome == name) for name in _OUTCOMES}


def _recent_attempts(past: list[HistoricalTxn], as_of: datetime) -> list[dict]:
    ordered = sorted(past, key=lambda h: _aware(h.timestamp), reverse=True)[:_RECENT_ATTEMPTS]
    return [
        {
            "amount_usd": round(h.amount, 2),
            "days_ago": round((as_of - _aware(h.timestamp)).total_seconds() / 86400.0, 1),
            "device_info": h.device_info,
            "outcome": h.outcome if h.outcome in _OUTCOMES else "unknown",
            **({"merchant_id": h.merchant_id} if h.merchant_id else {}),
            **({"merchant_country": h.merchant_country} if h.merchant_country else {}),
        }
        for h in ordered
    ]
