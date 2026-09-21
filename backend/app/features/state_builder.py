"""Build the Jev `state` from a Transaction.

Jev is a semantic classifier: it reasons over readable fields, not opaque
columns. This module trims the transaction to decision-relevant fields, renames
them descriptively, and adds derived features (hour of day, amount anomaly,
email domain class, distance bucket, velocity flags). Irrelevant context degrades
accuracy and costs tokens, so `None` values are dropped.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.schemas.transaction import Transaction

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "icloud.com",
    "live.com",
    "msn.com",
    "ymail.com",
    "mail.com",
    "protonmail.com",
    "comcast.net",
    "att.net",
    "sbcglobal.net",
    "verizon.net",
    "yahoo.com.mx",
    "hotmail.fr",
    "yahoo.fr",
    "gmail",
    "yahoo",
    "hotmail",
    "outlook",
}

DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com",
    "guerrillamail.com",
    "10minutemail.com",
    "tempmail.com",
    "yopmail.com",
    "trashmail.com",
    "anonymous.com",
}

PRODUCT_CODE_DESCRIPTIONS = {
    "W": "web purchase of physical goods",
    "C": "card-not-present digital purchase",
    "H": "hotel / travel",
    "S": "service subscription",
    "R": "recurring billing",
}

# IEEE-CIS TransactionDT is seconds since an unspecified reference. The community
# consensus places the reference at 2017-12-01 00:00 UTC; hour-of-day only needs the
# offset modulo 24h so the exact date does not matter for this feature.
_REFERENCE_EPOCH = datetime(2017, 12, 1)


def classify_email_domain(domain: str | None) -> str | None:
    if not domain:
        return None
    d = domain.lower().strip()
    if d in DISPOSABLE_EMAIL_DOMAINS:
        return "disposable"
    if d in FREE_EMAIL_DOMAINS or d.split(".")[0] in FREE_EMAIL_DOMAINS:
        return "free_webmail"
    if d.endswith((".edu", ".gov", ".mil")):
        return "institutional"
    return "corporate_or_isp"


def bucket_distance(dist: float | None) -> str | None:
    if dist is None:
        return None
    if dist <= 5:
        return "local (<=5)"
    if dist <= 50:
        return "regional (5-50)"
    if dist <= 500:
        return "distant (50-500)"
    return "far (>500)"


def bucket_amount(amount: float) -> str:
    if amount < 10:
        return "micro (<10)"
    if amount < 100:
        return "small (10-100)"
    if amount < 500:
        return "medium (100-500)"
    if amount < 2000:
        return "large (500-2000)"
    return "very large (>2000)"


def hour_of_day(tx: Transaction) -> int | None:
    if tx.transaction_time is not None:
        return tx.transaction_time.hour
    if tx.timestamp_delta_seconds is not None:
        return int((tx.timestamp_delta_seconds // 3600) % 24)
    return None


def _is_new_card(tx: Transaction) -> bool | None:
    if tx.days_since_card_first_seen is None:
        return None
    return tx.days_since_card_first_seen <= 1


def _is_new_device(tx: Transaction) -> bool | None:
    if tx.days_since_device_first_seen is not None:
        return tx.days_since_device_first_seen <= 1
    if tx.device_info and tx.card_known_devices is not None:
        return tx.device_info not in tx.card_known_devices
    return None


def _round(v: float | None, nd: int = 2) -> float | None:
    return None if v is None else round(v, nd)


def build_state(tx: Transaction) -> dict[str, Any]:
    """Return a compact, human-readable JSON state for Jev."""
    hour = hour_of_day(tx)
    p_domain_type = classify_email_domain(tx.purchaser_email_domain)
    r_domain_type = classify_email_domain(tx.recipient_email_domain)
    email_mismatch = (
        tx.recipient_email_domain is not None
        and tx.purchaser_email_domain is not None
        and tx.recipient_email_domain.lower() != tx.purchaser_email_domain.lower()
    )

    amount_ratio = None
    if tx.card_avg_amount and tx.card_avg_amount > 0:
        amount_ratio = tx.amount / tx.card_avg_amount

    transaction: dict[str, Any] = {
        "amount_usd": _round(tx.amount),
        "amount_bucket": bucket_amount(tx.amount),
        "product": PRODUCT_CODE_DESCRIPTIONS.get(tx.product_code or "", tx.product_code),
        "hour_of_day_utc": hour,
        "is_night_time": (hour is not None and (hour < 6 or hour >= 23)) if hour is not None else None,
        "card_network": tx.card_network,
        "card_type": tx.card_type,
        "purchaser_email_domain": tx.purchaser_email_domain,
        "purchaser_email_domain_type": p_domain_type,
        "recipient_email_domain": tx.recipient_email_domain,
        "recipient_email_domain_type": r_domain_type,
        "purchaser_recipient_email_mismatch": email_mismatch if tx.recipient_email_domain else None,
        "billing_region_code": tx.billing_region,
        "billing_country_code": tx.billing_country,
        "distance_billing_to_purchase": _round(tx.distance_billing_to_purchase),
        "distance_bucket": bucket_distance(tx.distance_billing_to_purchase),
    }

    device: dict[str, Any] = {
        "device_type": tx.device_type,
        "device_info": tx.device_info,
        "device_is_new_for_card": _is_new_device(tx),
        "days_since_device_first_seen": _round(tx.days_since_device_first_seen, 1),
        "transactions_from_this_device": _round(tx.device_txn_count, 0),
    }

    velocity: dict[str, Any] = {
        "card_transaction_count": _round(tx.card_txn_count, 0),
        "email_transaction_count": _round(tx.email_txn_count, 0),
        "address_match_count": _round(tx.addr_match_count, 0),
        "days_since_previous_transaction": _round(tx.days_since_prev_txn, 1),
        "days_since_card_first_seen": _round(tx.days_since_card_first_seen, 1),
        "card_is_new": _is_new_card(tx),
        "days_since_prev_txn_same_address": _round(tx.days_since_prev_txn_same_addr, 1),
        "days_since_prev_txn_same_amount": _round(tx.days_since_prev_txn_same_amount, 1),
        "repeat_of_recent_amount": (
            tx.days_since_prev_txn_same_amount is not None and tx.days_since_prev_txn_same_amount <= 1
        )
        if tx.days_since_prev_txn_same_amount is not None
        else None,
        "card_average_amount_usd": _round(tx.card_avg_amount),
        "amount_vs_card_average_ratio": _round(amount_ratio),
    }

    state = {
        "transaction": _prune(transaction),
        "device": _prune(device),
        "velocity": _prune(velocity),
    }
    return {k: v for k, v in state.items() if v}


def _prune(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
