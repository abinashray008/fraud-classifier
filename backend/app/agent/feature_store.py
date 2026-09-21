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

from app.schemas.transaction import Transaction


@dataclass
class HistoricalTxn:
    amount: float
    timestamp: datetime
    device_info: str | None
    email_domain: str | None
    billing_region: str | None
    outcome: str  # approved | declined | chargeback


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
        good = CardProfile(card_id="card_good_001", home_region="315")
        for i in range(12):
            good.history.append(
                HistoricalTxn(
                    amount=40 + (i % 4) * 15,
                    timestamp=now - timedelta(days=90 - i * 7),
                    device_info="iOS Device",
                    email_domain="gmail.com",
                    billing_region="315",
                    outcome="approved",
                )
            )
        good.known_devices.add("iOS Device")
        good.known_email_domains.add("gmail.com")
        self.cards[good.card_id] = good
        self.device_index["iOS Device"].add(good.card_id)

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
        ato.known_devices.add("Windows")
        ato.known_email_domains.add("outlook.com")
        self.cards[ato.card_id] = ato
        self.device_index["Windows"].add(ato.card_id)

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
            )
        )
        if tx.device_info:
            self.device_index[tx.device_info].add(tx.card_id)

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
                }
                for h in hist
            ],
        }

    def device_history(self, device_info: str) -> dict:
        cards = self.device_index.get(device_info, set())
        return {
            "device_info": device_info,
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
        return {
            "card_id": card_id,
            "found": True,
            "window_hours": window_hours,
            "transactions_in_window": len(window),
            "total_amount_in_window_usd": round(sum(h.amount for h in window), 2),
            "distinct_devices_in_window": len({h.device_info for h in window if h.device_info}),
        }
