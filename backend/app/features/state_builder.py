"""Build the Jev `state` from a Transaction plus optional card history.

Jev is a semantic classifier: it reasons over readable fields. Invented labels
for masked IEEE-CIS columns (C14 as "device transaction count", D15 as
"device age", ProductCD H as "hotel / travel") would be presented as facts and
can steer it toward confident but unsupported conclusions.

This module therefore:
- keeps published names (amount, card network/type, email domains,
  DeviceType/Info, addr1/addr2, dist1)
- omits ProductCD. The letters W/C/H/S/R have no published mapping, so they
  are not sent to Jev
- does not gloss C/D columns
- attaches features computed from the card's prior transactions, with units
  and time windows in the field names (USD, days, 1h / 24h / 7d)
- when the request is a card-present authorization, attaches that section
  (`card_present`) including the real-time rule facts
- puts issuer card status on the transaction itself, so a card-not-present
  request does not lose it by omitting the card-present section
"""

from __future__ import annotations

from typing import Any

from app.features.card_present import build_card_present
from app.features.history import AUTHORIZATION_COUNTER_KEYS, device_identifier_kind
from app.schemas.transaction import Transaction

UNKNOWN = "unknown"

# Evidence claims. When history was retrieved and the value is missing, the
# state says "unknown" so a missing fact is not read as a clean negative.
_DEVICE_EVIDENCE = frozenset(
    {
        "device_identifier_kind",
        "device_seen_before_on_this_card",
        "device_is_new_for_card",
        "current_device_is_trusted",
        "device_distinct_cards_seen",
        "device_chargebacks",
    }
)
_HISTORY_EVIDENCE = frozenset(
    {
        *AUTHORIZATION_COUNTER_KEYS,
        "history_found",
        "prior_transaction_count",
        "card_is_new",
        "days_since_first_transaction",
        "days_since_previous_transaction",
        "mean_amount_usd_prior",
        "amount_vs_mean_prior_ratio",
        "transactions_last_1h",
        "transactions_last_24h",
        "transactions_last_7d",
        "amount_usd_last_1h",
        "amount_usd_last_24h",
        "amount_usd_last_7d",
        "distinct_devices_last_24h",
        "distinct_devices_last_7d",
        "matching_amount_count_last_24h",
        "email_domain_seen_before_on_this_card",
        "chargebacks_on_card",
        "trusted_device_ids",
        "recent_attempts",
        "confirmed_outcomes",
        "authorization_decisions",
    }
)

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

# IEEE-CIS TransactionDT is seconds since an unspecified reference. Hour-of-day
# only needs the offset modulo 24h, so the exact epoch does not matter.


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


def bucket_dist1(dist: float | None) -> str | None:
    """Numeric bins for dist1. Not geographic labels — unit/endpoints unpublished."""
    if dist is None:
        return None
    if dist <= 5:
        return "<=5"
    if dist <= 50:
        return "5-50"
    if dist <= 500:
        return "50-500"
    return ">500"


def bucket_amount(amount: float) -> str:
    if amount < 10:
        return "micro (<10 USD)"
    if amount < 100:
        return "small (10-100 USD)"
    if amount < 500:
        return "medium (100-500 USD)"
    if amount < 2000:
        return "large (500-2000 USD)"
    return "very large (>2000 USD)"


def hour_of_day(tx: Transaction) -> int | None:
    if tx.transaction_time is not None:
        return tx.transaction_time.hour
    if tx.timestamp_delta_seconds is not None:
        return int((tx.timestamp_delta_seconds // 3600) % 24)
    return None


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _round(v: float | None, nd: int = 2) -> float | None:
    return None if v is None else round(v, nd)


def _merge_history(tx: Transaction, history: dict[str, Any] | None) -> dict[str, Any]:
    """Use retrieved history as-is.

    Caller-supplied averages apply only when no history dict
    was retrieved (unit tests and offline callers). A provided dict wins even
    when some values are missing — those stay unknown rather than being filled
    from the request.
    """
    if history is not None:
        return dict(history)
    merged: dict[str, Any] = {}
    if tx.card_avg_amount and tx.card_avg_amount > 0:
        merged["mean_amount_usd_prior"] = tx.card_avg_amount
    return merged


def build_state(tx: Transaction, history: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a compact JSON state for Jev. `history` is trusted card evidence.

    When `history` is provided, missing evidence is the string "unknown".
    When it is omitted, absent fields are dropped.
    """
    history_provided = history is not None
    hour = hour_of_day(tx)
    p_domain_type = classify_email_domain(tx.purchaser_email_domain)
    r_domain_type = classify_email_domain(tx.recipient_email_domain)
    email_mismatch = (
        tx.recipient_email_domain is not None
        and tx.purchaser_email_domain is not None
        and tx.recipient_email_domain.lower() != tx.purchaser_email_domain.lower()
    )

    hist = _merge_history(tx, history)
    mean_prior = _number(hist.get("mean_amount_usd_prior"))
    amount_ratio = tx.amount / mean_prior if mean_prior and mean_prior > 0 else None

    prior_count = hist.get("prior_transaction_count")
    card_is_new = None if not isinstance(prior_count, int) or isinstance(prior_count, bool) else prior_count == 0
    device_seen = hist.get("device_seen_before_on_this_card") if tx.device_id else None

    transaction: dict[str, Any] = {
        "amount_usd": _round(tx.amount),
        "amount_bucket": bucket_amount(tx.amount),
        "hour_of_day_utc": hour,
        "is_night_time": (hour is not None and (hour < 6 or hour >= 23)) if hour is not None else None,
        "card_network": tx.card_network,
        "card_type": tx.card_type,
        "channel": tx.channel,
        "card_status": tx.card_status,
        "purchaser_email_domain": tx.purchaser_email_domain,
        "purchaser_email_domain_type": p_domain_type,
        "recipient_email_domain": tx.recipient_email_domain,
        "recipient_email_domain_type": r_domain_type,
        "purchaser_recipient_email_mismatch": email_mismatch if tx.recipient_email_domain else None,
        "addr1": tx.billing_region,
        "addr2": tx.billing_country,
        "dist1": _round(tx.dist1),
        "dist1_bucket": bucket_dist1(tx.dist1),
    }

    device: dict[str, Any] = {
        "device_type": tx.device_type,
        "device_info": tx.device_info,
        "device_id": tx.device_id,
        "device_identifier_kind": device_identifier_kind(tx.device_info, tx.device_id),
        "device_seen_before_on_this_card": device_seen if isinstance(device_seen, bool) else None,
        "device_is_new_for_card": None if not isinstance(device_seen, bool) else not device_seen,
        "current_device_is_trusted": hist.get("current_device_is_trusted") if tx.device_id else None,
        "device_distinct_cards_seen": hist.get("device_distinct_cards_seen") if tx.device_id else None,
        "device_chargebacks": hist.get("device_chargebacks") if tx.device_id else None,
    }

    card_history: dict[str, Any] = {
        **{key: hist.get(key) for key in AUTHORIZATION_COUNTER_KEYS},
        "history_found": hist.get("history_found"),
        "prior_transaction_count": prior_count,
        "card_is_new": card_is_new,
        "days_since_first_transaction": _round(hist.get("days_since_first_transaction"), 1),
        "days_since_previous_transaction": _round(hist.get("days_since_previous_transaction"), 1),
        "mean_amount_usd_prior": _round(mean_prior),
        "amount_vs_mean_prior_ratio": _round(amount_ratio),
        "transactions_last_1h": hist.get("transactions_last_1h"),
        "transactions_last_24h": hist.get("transactions_last_24h"),
        "transactions_last_7d": hist.get("transactions_last_7d"),
        "amount_usd_last_1h": _round(hist.get("amount_usd_last_1h")),
        "amount_usd_last_24h": _round(hist.get("amount_usd_last_24h")),
        "amount_usd_last_7d": _round(hist.get("amount_usd_last_7d")),
        "distinct_devices_last_24h": hist.get("distinct_devices_last_24h"),
        "distinct_devices_last_7d": hist.get("distinct_devices_last_7d"),
        "matching_amount_count_last_24h": hist.get("matching_amount_count_last_24h"),
        "email_domain_seen_before_on_this_card": hist.get("email_domain_seen_before_on_this_card"),
        "chargebacks_on_card": hist.get("chargebacks_on_card"),
        "trusted_device_ids": hist.get("trusted_device_ids"),
        "recent_attempts": hist.get("recent_attempts"),
        "confirmed_outcomes": hist.get("confirmed_outcomes"),
        "authorization_decisions": hist.get("authorization_decisions"),
    }

    state = {
        "transaction": _prune(transaction),
        "device": _finalize(device, _DEVICE_EVIDENCE, history_provided),
        "card_history": _finalize(card_history, _HISTORY_EVIDENCE, history_provided),
        "card_present": build_card_present(tx, hist, history_provided=history_provided),
    }
    return {k: v for k, v in state.items() if v}


def _prune(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _finalize(d: dict[str, Any], evidence: frozenset[str], history_provided: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        if value is None:
            if history_provided and key in evidence:
                out[key] = UNKNOWN
            continue
        out[key] = value
    return out
