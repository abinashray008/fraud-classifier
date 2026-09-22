"""Hard authorization controls that do not depend on the fraud model.

Issuer controls (unusable card, credit limit) apply on either channel and are
evaluated before the classifier is called. A model timeout cannot turn a stolen
or blocked card into an approval, and an explicit card-not-present channel
cannot drop them by omitting the card-present section.

Presentment controls (track CVV, PIN) apply only when the transaction is
card-present. Terminal-only fields combined with `card_not_present` are
rejected as contradictory input.
"""

from __future__ import annotations

from typing import Any

UNUSABLE_STATUSES = frozenset({"lost", "stolen", "expired", "blocked"})

# Facts that exist only at a merchant terminal. They contradict channel=card_not_present.
PRESENTMENT_ONLY_FIELDS = (
    "entry_mode",
    "cvm_result",
    "pin_tries_exceeded",
    "track_cvv",
    "terminal_id",
    "terminal_attended",
)

ISSUER_CONTROLS = {
    "card_unusable": "card is lost, stolen, expired, or blocked",
    "over_limit": "amount is above available credit or the single-purchase limit",
}

PRESENTMENT_CONTROLS = {
    "track_cvv_mismatch": "magstripe track CVV does not match",
    "pin_failure": "PIN failed or the PIN try limit is exceeded",
}


def contradictory_channel(tx: Any) -> str | None:
    """Reject a card-not-present request that also carries terminal facts."""
    if getattr(tx, "channel", None) != "card_not_present":
        return None
    present = [name for name in PRESENTMENT_ONLY_FIELDS if getattr(tx, name, None) is not None]
    if not present:
        return None
    return "channel card_not_present contradicts card-present fields: " + ", ".join(present)


def issuer_control_flags(tx: Any) -> dict[str, bool]:
    """Channel-independent controls. Absent inputs are omitted, not treated as clear."""
    flags: dict[str, bool] = {}
    status = getattr(tx, "card_status", None)
    if status is not None:
        flags["card_unusable"] = status in UNUSABLE_STATUSES
    available = getattr(tx, "available_credit_usd", None)
    purchase_limit = getattr(tx, "single_purchase_limit_usd", None)
    if available is not None or purchase_limit is not None:
        amount = tx.amount
        over = False
        if available is not None and amount > available:
            over = True
        if purchase_limit is not None and amount > purchase_limit:
            over = True
        flags["over_limit"] = over
    return flags


def presentment_control_flags(tx: Any) -> dict[str, bool]:
    """Channel-specific controls. Callers apply them only on a card-present auth."""
    flags: dict[str, bool] = {}
    track_cvv = getattr(tx, "track_cvv", None)
    if track_cvv is not None:
        flags["track_cvv_mismatch"] = track_cvv == "mismatch"
    cvm = getattr(tx, "cvm_result", None)
    pin_tries = getattr(tx, "pin_tries_exceeded", None)
    if cvm is not None or pin_tries is not None:
        flags["pin_failure"] = cvm == "pin_failed" or pin_tries is True
    return flags


def decline_before_model(tx: Any) -> str | None:
    """Hard declines that must be decided before the classifier is called.

    Issuer controls always run. Presentment controls run only when this request
    is a card-present authorization.
    """
    from app.features.card_present import is_card_present

    fired = _fired(issuer_control_flags(tx), ISSUER_CONTROLS)
    if is_card_present(tx):
        fired.extend(_fired(presentment_control_flags(tx), PRESENTMENT_CONTROLS))
    if not fired:
        return None
    return "; ".join(fired)


def _fired(flags: dict[str, bool], texts: dict[str, str]) -> list[str]:
    return [f"{name} ({texts[name]})" for name, on in flags.items() if on]
