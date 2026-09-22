"""Card-history features with explicit units and time windows.

IEEE-CIS C and D columns are counts and timedeltas whose individual meanings
were masked (Vesta's data dictionary; confirmed by the competition winner's
write-up). This module does not interpret those columns. It computes features
from the card's own prior transactions so a semantic classifier can read them
as facts: counts, USD amounts, days, and named windows (1h / 24h / 7d).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Any

# Keys produced from a card's prior transactions. Device-index features
# (`device_distinct_cards_seen`, `device_chargebacks`) are store-only extras.
CARD_HISTORY_KEYS = (
    "history_found",
    "prior_transaction_count",
    "days_since_first_transaction",
    "days_since_previous_transaction",
    "mean_amount_usd_prior",
    "transactions_last_1h",
    "transactions_last_24h",
    "transactions_last_7d",
    "amount_usd_last_1h",
    "amount_usd_last_24h",
    "amount_usd_last_7d",
    "distinct_devices_last_24h",
    "distinct_devices_last_7d",
    "device_seen_before_on_this_card",
    "email_domain_seen_before_on_this_card",
    "matching_amount_count_last_24h",
    "chargebacks_on_card",
)

# Live-store counters; offline IEEE-CIS rows have no authorization decision feed.
AUTHORIZATION_COUNTER_KEYS = tuple(
    f"{counter}_last_{window}"
    for window in ("1h", "24h", "7d")
    for counter in ("attempts", "approvals", "confirmed_fraud")
)

_WINDOW_1H = timedelta(hours=1)
_WINDOW_24H = timedelta(hours=24)
_WINDOW_7D = timedelta(days=7)
_AMOUNT_EPS = 0.01  # USD; treat amounts within a cent as the same


def device_identifier_kind(device_info: str | None, device_id: str | None = None) -> str | None:
    """Identity comes only from the trusted integration's explicit device_id."""
    if device_id:
        return "device_id"
    return "description" if device_info and device_info.strip() else None


def distinct_device_ids(device_ids: list[str | None]) -> int | None:
    """Exact count, unknown if any event lacks identity; an empty window is zero."""
    if any(not device_id for device_id in device_ids):
        return None
    return len(set(device_ids))


@dataclass(frozen=True)
class PriorTxn:
    amount: float
    timestamp: datetime
    device_info: str | None = None
    email_domain: str | None = None
    fraud_outcome: str | None = None
    merchant_id: str | None = None
    merchant_country: str | None = None
    device_id: str | None = None


# A merchant-country change inside this window cannot be ordinary cardholder travel.
_TRAVEL_WINDOW = timedelta(hours=2)


def compute_merchant_features(
    *,
    prior: list[PriorTxn],
    at: datetime,
    merchant_id: str | None = None,
    merchant_country: str | None = None,
    amount: float | None = None,
) -> dict[str, Any]:
    """Merchant velocity for a card-present authorization.

    Counts include the current merchant. `merchant_country_changed_within_2h`
    is true only when the previous merchant country is known, differs, and the
    gap is under two hours. It is left unset when either country is missing.
    """
    at = _aware(at)
    past = [p for p in prior if _aware(p.timestamp) < at]
    past.sort(key=lambda p: _aware(p.timestamp))
    last_1h = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_1H]
    last_24h = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_24H]

    merchant_ids = {p.merchant_id for p in last_1h if p.merchant_id}
    if merchant_id:
        merchant_ids.add(merchant_id)
    countries = {(p.merchant_country or "").upper() for p in last_24h if p.merchant_country}
    if merchant_country:
        countries.add(merchant_country.upper())

    located = [p for p in past if p.merchant_id or p.merchant_country]
    previous_country = None
    minutes: float | None = None
    changed: bool | None = None
    if located:
        last = located[-1]
        previous_country = last.merchant_country.upper() if last.merchant_country else None
        minutes = round((at - _aware(last.timestamp)).total_seconds() / 60.0, 1)
        if merchant_country and previous_country:
            gap = at - _aware(last.timestamp)
            changed = previous_country != merchant_country.upper() and gap < _TRAVEL_WINDOW

    seen: bool | None = None
    if merchant_id:
        seen = any(p.merchant_id == merchant_id for p in past)

    at_merchant = [p.amount for p in past if merchant_id and p.merchant_id == merchant_id]
    mean_at_merchant = round(mean(at_merchant), 2) if at_merchant else None
    vs_merchant = None
    if mean_at_merchant and amount is not None and mean_at_merchant > 0:
        vs_merchant = round(amount / mean_at_merchant, 2)

    return {
        "merchant_seen_before_on_this_card": seen,
        "distinct_merchants_last_1h": len(merchant_ids),
        "distinct_merchant_countries_last_24h": len(countries),
        "minutes_since_previous_merchant": minutes,
        "previous_merchant_country": previous_country,
        "merchant_country_changed_within_2h": changed,
        "mean_amount_usd_at_this_merchant": mean_at_merchant,
        "amount_vs_this_merchant_ratio": vs_merchant,
    }


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def compute_card_history_features(
    *,
    prior: list[PriorTxn],
    amount: float,
    at: datetime,
    device_info: str | None = None,
    device_id: str | None = None,
    email_domain: str | None = None,
) -> dict[str, Any]:
    """Return features from `prior` transactions strictly before `at`.

    Transaction counts and amounts include every prior authorization attempt,
    including pending and declined attempts. The current transaction and anything
    at or after `at` are excluded.
    Timedelta fields are omitted (None) when there is no earlier transaction.
    Empty history still returns counts of 0. DeviceInfo is descriptive only.
    Device identity requires an explicit device_id from a trusted integration;
    a missing historical ID cannot establish that the current device is new.
    """
    at = _aware(at)
    past = [p for p in prior if _aware(p.timestamp) < at]
    past.sort(key=lambda p: _aware(p.timestamp))

    last_1h = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_1H]
    last_24h = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_24H]
    last_7d = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_7D]

    amounts = [p.amount for p in past]
    device_seen: bool | None = None
    if device_id:
        if any(p.device_id == device_id for p in past):
            device_seen = True
        elif all(p.device_id for p in past):
            device_seen = False
    email_seen: bool | None = None
    if email_domain:
        d = email_domain.lower()
        email_seen = any((p.email_domain or "").lower() == d for p in past)

    first_ts = _aware(past[0].timestamp) if past else None
    last_ts = _aware(past[-1].timestamp) if past else None

    return {
        "history_found": bool(past),
        "prior_transaction_count": len(past),
        "days_since_first_transaction": (round((at - first_ts).total_seconds() / 86400.0, 1) if first_ts else None),
        "days_since_previous_transaction": (round((at - last_ts).total_seconds() / 86400.0, 1) if last_ts else None),
        "mean_amount_usd_prior": round(mean(amounts), 2) if amounts else None,
        "transactions_last_1h": len(last_1h),
        "transactions_last_24h": len(last_24h),
        "transactions_last_7d": len(last_7d),
        "amount_usd_last_1h": round(sum(p.amount for p in last_1h), 2),
        "amount_usd_last_24h": round(sum(p.amount for p in last_24h), 2),
        "amount_usd_last_7d": round(sum(p.amount for p in last_7d), 2),
        "distinct_devices_last_24h": distinct_device_ids([p.device_id for p in last_24h]),
        "distinct_devices_last_7d": distinct_device_ids([p.device_id for p in last_7d]),
        "device_seen_before_on_this_card": device_seen,
        "email_domain_seen_before_on_this_card": email_seen,
        "matching_amount_count_last_24h": sum(1 for p in last_24h if abs(p.amount - amount) <= _AMOUNT_EPS),
        "chargebacks_on_card": sum(1 for p in past if p.fraud_outcome == "chargeback"),
    }
