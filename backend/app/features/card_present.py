"""Card-present authorization facts for Jev.

A swipe, chip, or tap at a merchant is an authorization request: the terminal
sends it through the acquirer and the card network, and the issuer approves or
declines. This module turns that request into named facts. It does not invent
meanings for masked IEEE-CIS columns; those rows never carry these fields, and
the section is omitted.

`rules` are the real-time checks. Hard controls (unusable card, track-CVV
mismatch, PIN failure, over limit) also decline in `authorization_decline_reason`
so a low fraud score cannot approve them. Amount, fallback, travel, cash-like
MCC, and merchant bursts stay in `rules` for Jev to weigh as fraud evidence.
`over_limit` is a credit control. `amount_far_above_history` and
`probing_amount` are fraud checks on the authorization amount.
"""

from __future__ import annotations

from typing import Any

from app.schemas.transaction import Transaction

# ISO 18245 codes issuers treat as cash-equivalent on a card-present auth.
CASH_LIKE_MCCS = frozenset({"4829", "6010", "6011", "6051", "7995"})

_UNUSABLE = frozenset({"lost", "stolen", "expired", "blocked"})
_PRESENTMENT_ATTRS = (
    "entry_mode",
    "card_status",
    "cvm_result",
    "pin_tries_exceeded",
    "track_cvv",
    "merchant_id",
    "merchant_name",
    "mcc",
    "merchant_country",
    "merchant_city",
    "terminal_id",
    "terminal_attended",
    "cardholder_country",
    "available_credit_usd",
    "single_purchase_limit_usd",
)

ENTRY_MODE_MEANING = {
    "swipe": "magstripe swipe at the merchant terminal",
    "chip": "EMV chip insert at the merchant terminal",
    "contactless": "contactless tap at the merchant terminal",
    "fallback_swipe": "chip failed and the terminal fell back to a magstripe swipe",
    "keyed": "card number keyed at the merchant terminal",
}

# Evidence looked up from prior card-present activity. Missing after a lookup
# is "unknown", so Jev does not read a gap as a clean negative.
MERCHANT_EVIDENCE = frozenset(
    {
        "merchant_seen_before_on_this_card",
        "distinct_merchants_last_1h",
        "distinct_merchant_countries_last_24h",
        "minutes_since_previous_merchant",
        "previous_merchant_country",
        "merchant_country_changed_within_2h",
        "mean_amount_usd_at_this_merchant",
        "amount_vs_this_merchant_ratio",
    }
)

# Fraud checks on the authorization amount, separate from the credit-limit control.
# 4× the card's prior mean is a spent-card spike. At most $1, or the same
# sub-$10 amount already seen twice in 24h, is a presentment probe.
FAR_ABOVE_MEAN = 4.0
PROBE_AMOUNT_USD = 1.0
REPEATED_SMALL_USD = 10.0

HARD_RULES = {
    "card_unusable": "card is lost, stolen, expired, or blocked",
    "track_cvv_mismatch": "magstripe track CVV does not match",
    "pin_failure": "PIN failed or the PIN try limit is exceeded",
    "over_limit": "amount is above available credit or the single-purchase limit",
}

UNKNOWN = "unknown"


def is_card_present(tx: Transaction) -> bool:
    if tx.channel == "card_not_present":
        return False
    if tx.channel == "card_present":
        return True
    return any(getattr(tx, name) is not None for name in _PRESENTMENT_ATTRS)


def _country(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().upper()
    return text or None


def _mcc(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _amount_facts(tx: Transaction, hist: dict[str, Any], *, history_provided: bool) -> dict[str, Any]:
    """Authorization amount and how it compares with this card's prior tickets."""
    facts: dict[str, Any] = {"amount_usd": round(tx.amount, 2)}
    mean_prior = _number(hist.get("mean_amount_usd_prior"))
    if mean_prior and mean_prior > 0:
        facts["mean_prior_amount_usd"] = round(mean_prior, 2)
        facts["amount_vs_mean_prior_ratio"] = round(tx.amount / mean_prior, 2)
    elif history_provided:
        facts["mean_prior_amount_usd"] = UNKNOWN
        facts["amount_vs_mean_prior_ratio"] = UNKNOWN

    spent_1h = _number(hist.get("amount_usd_last_1h"))
    if spent_1h is not None:
        facts["amount_usd_last_1h_including_this"] = round(spent_1h + tx.amount, 2)
    elif history_provided:
        facts["amount_usd_last_1h_including_this"] = UNKNOWN
    return facts


def _rules(tx: Transaction, hist: dict[str, Any]) -> dict[str, bool]:
    rules: dict[str, bool] = {}
    if tx.card_status is not None:
        rules["card_unusable"] = tx.card_status in _UNUSABLE
    if tx.track_cvv is not None:
        rules["track_cvv_mismatch"] = tx.track_cvv == "mismatch"
    if tx.cvm_result is not None or tx.pin_tries_exceeded is not None:
        rules["pin_failure"] = tx.cvm_result == "pin_failed" or tx.pin_tries_exceeded is True
    if tx.available_credit_usd is not None or tx.single_purchase_limit_usd is not None:
        over = False
        if tx.available_credit_usd is not None and tx.amount > tx.available_credit_usd:
            over = True
        if tx.single_purchase_limit_usd is not None and tx.amount > tx.single_purchase_limit_usd:
            over = True
        rules["over_limit"] = over
    if tx.entry_mode is not None:
        rules["magstripe_fallback"] = tx.entry_mode == "fallback_swipe"
        if tx.cvm_result in {"signature", "no_cvm"} and tx.entry_mode in {"swipe", "fallback_swipe"}:
            rules["swipe_without_pin"] = True
        elif tx.cvm_result is not None and tx.entry_mode in {"swipe", "fallback_swipe"}:
            rules["swipe_without_pin"] = False

    merchant_country = _country(tx.merchant_country)
    cardholder_country = _country(tx.cardholder_country)
    if merchant_country and cardholder_country:
        rules["cross_border"] = merchant_country != cardholder_country

    mcc = _mcc(tx.mcc)
    if mcc is not None:
        rules["cash_like_mcc"] = mcc in CASH_LIKE_MCCS

    changed = hist.get("merchant_country_changed_within_2h")
    if isinstance(changed, bool):
        rules["merchant_country_changed_within_2h"] = changed
    merchants = hist.get("distinct_merchants_last_1h")
    if isinstance(merchants, int) and not isinstance(merchants, bool):
        rules["merchant_burst_1h"] = merchants >= 3

    mean_prior = _number(hist.get("mean_amount_usd_prior"))
    if mean_prior and mean_prior > 0:
        rules["amount_far_above_history"] = tx.amount / mean_prior >= FAR_ABOVE_MEAN
    merchant_ratio = _number(hist.get("amount_vs_this_merchant_ratio"))
    if merchant_ratio is not None:
        rules["amount_far_above_this_merchant"] = merchant_ratio >= FAR_ABOVE_MEAN

    repeated = hist.get("matching_amount_count_last_24h")
    repeated_small = (
        isinstance(repeated, int)
        and not isinstance(repeated, bool)
        and tx.amount < REPEATED_SMALL_USD
        and repeated >= 2
    )
    rules["probing_amount"] = tx.amount <= PROBE_AMOUNT_USD or repeated_small
    return rules


def build_card_present(
    tx: Transaction,
    hist: dict[str, Any],
    *,
    history_provided: bool,
) -> dict[str, Any] | None:
    """Return the `card_present` state section, or None when this is not one."""
    if not is_card_present(tx):
        return None

    merchant_country = _country(tx.merchant_country)
    cardholder_country = _country(tx.cardholder_country)
    mcc = _mcc(tx.mcc)
    facts: dict[str, Any] = {
        "channel": "card_present",
        "authorization_path": "merchant terminal → acquirer → card network → issuer approve or decline",
        "network": tx.card_network,
        "entry_mode": tx.entry_mode,
        "entry_mode_meaning": ENTRY_MODE_MEANING.get(tx.entry_mode) if tx.entry_mode else None,
        "card_status": tx.card_status,
        "cvm_result": tx.cvm_result,
        "pin_tries_exceeded": tx.pin_tries_exceeded,
        "track_cvv": tx.track_cvv,
        "merchant_id": tx.merchant_id,
        "merchant_name": tx.merchant_name,
        "mcc": mcc,
        "merchant_country": merchant_country,
        "merchant_city": tx.merchant_city,
        "terminal_id": tx.terminal_id,
        "terminal_attended": tx.terminal_attended,
        "cardholder_country": cardholder_country,
        "available_credit_usd": tx.available_credit_usd,
        "single_purchase_limit_usd": tx.single_purchase_limit_usd,
        **_amount_facts(tx, hist, history_provided=history_provided),
    }
    for key in MERCHANT_EVIDENCE:
        value = hist.get(key)
        if value is None and history_provided:
            facts[key] = UNKNOWN
        elif value is not None:
            facts[key] = value

    rules = _rules(tx, hist)
    section = {key: value for key, value in facts.items() if value is not None}
    if rules:
        section["rules"] = rules
    return section


def authorization_decline_reason(state: dict[str, Any]) -> str | None:
    """Hard controls that must decline the network response.

    Jev still scores the same state. These checks are binary issuer controls:
    a low fraud probability does not make a lost card or a bad track CVV approvable.
    """
    rules = (state.get("card_present") or {}).get("rules") or {}
    fired = [f"{name} ({text})" for name, text in HARD_RULES.items() if rules.get(name) is True]
    if not fired:
        return None
    return "; ".join(fired)
