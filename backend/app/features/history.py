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

_WINDOW_1H = timedelta(hours=1)
_WINDOW_24H = timedelta(hours=24)
_WINDOW_7D = timedelta(days=7)
_AMOUNT_EPS = 0.01  # USD; treat amounts within a cent as the same

# IEEE-CIS DeviceInfo mixes unique build strings ("SM-G9650 Build/R16NW") with
# OS labels shared by many people. Those labels are not fingerprints.
GENERIC_DEVICE_DESCRIPTIONS = frozenset(
    {
        "windows",
        "ios",
        "ios device",
        "macos",
        "mac os",
        "mac os x",
        "linux",
        "android",
        "other",
        "mobile",
        "desktop",
    }
)


def is_device_fingerprint(device_info: str | None) -> bool:
    """True when `device_info` can identify one device.

    Bare OS names such as "Windows" or "iOS Device" are descriptions.
    """
    if device_info is None:
        return False
    text = device_info.strip()
    if not text:
        return False
    return text.lower() not in GENERIC_DEVICE_DESCRIPTIONS


def device_identifier_kind(device_info: str | None) -> str | None:
    if device_info is None or not device_info.strip():
        return None
    return "fingerprint" if is_device_fingerprint(device_info) else "description"


def _distinct_fingerprints(rows: list[PriorTxn]) -> int | None:
    """Count unique fingerprints. Description-only rows are unknown, not zero devices."""
    labels = [p.device_info for p in rows if p.device_info]
    if not labels:
        return 0
    fingerprints = {d for d in labels if is_device_fingerprint(d)}
    if not fingerprints:
        return None
    return len(fingerprints)


@dataclass(frozen=True)
class PriorTxn:
    amount: float
    timestamp: datetime
    device_info: str | None = None
    email_domain: str | None = None
    outcome: str | None = None
    merchant_id: str | None = None
    merchant_country: str | None = None


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
    email_domain: str | None = None,
) -> dict[str, Any]:
    """Return features from `prior` transactions strictly before `at`.

    The current transaction and anything at or after `at` are excluded.
    Timedelta fields are omitted (None) when there is no earlier transaction.
    Empty history still returns counts of 0. A fingerprint not seen before is
    false. A generic description such as "Windows" leaves device identity unset
    (unknown): it is not evidence the same device was, or was not, seen.
    """
    at = _aware(at)
    past = [p for p in prior if _aware(p.timestamp) < at]
    past.sort(key=lambda p: _aware(p.timestamp))

    last_1h = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_1H]
    last_24h = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_24H]
    last_7d = [p for p in past if at - _aware(p.timestamp) <= _WINDOW_7D]

    amounts = [p.amount for p in past]
    device_seen: bool | None = None
    if is_device_fingerprint(device_info):
        device_seen = any(p.device_info == device_info for p in past)
    email_seen: bool | None = None
    if email_domain:
        d = email_domain.lower()
        email_seen = any((p.email_domain or "").lower() == d for p in past)

    first_ts = _aware(past[0].timestamp) if past else None
    last_ts = _aware(past[-1].timestamp) if past else None

    return {
        "history_found": bool(past),
        "prior_transaction_count": len(past),
        "days_since_first_transaction": (
            round((at - first_ts).total_seconds() / 86400.0, 1) if first_ts else None
        ),
        "days_since_previous_transaction": (
            round((at - last_ts).total_seconds() / 86400.0, 1) if last_ts else None
        ),
        "mean_amount_usd_prior": round(mean(amounts), 2) if amounts else None,
        "transactions_last_1h": len(last_1h),
        "transactions_last_24h": len(last_24h),
        "transactions_last_7d": len(last_7d),
        "amount_usd_last_1h": round(sum(p.amount for p in last_1h), 2),
        "amount_usd_last_24h": round(sum(p.amount for p in last_24h), 2),
        "amount_usd_last_7d": round(sum(p.amount for p in last_7d), 2),
        "distinct_devices_last_24h": _distinct_fingerprints(last_24h),
        "distinct_devices_last_7d": _distinct_fingerprints(last_7d),
        "device_seen_before_on_this_card": device_seen,
        "email_domain_seen_before_on_this_card": email_seen,
        "matching_amount_count_last_24h": sum(
            1 for p in last_24h if abs(p.amount - amount) <= _AMOUNT_EPS
        ),
        "chargebacks_on_card": sum(1 for p in past if p.outcome == "chargeback"),
    }
