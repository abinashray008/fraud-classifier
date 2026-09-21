"""The fraud question set evaluated by Jev in a single call.

All questions share the same state and run in parallel, so adding a question
costs only its own tokens. Question IDs are stable keys used by the policy and UI;
the full judgment lives in `instructions`.
"""

from __future__ import annotations

from langchain_typesafe import Choice, Noul, NoulCriteria, Score

PRIMARY_QUESTION = "is_fraud"
RISK_QUESTION = "risk"
PATTERN_QUESTION = "pattern"
SIGNAL_QUESTIONS = (
    "amount_anomalous",
    "new_device_for_card",
    "velocity_spike",
    "email_mismatch_suspicious",
)

PATTERN_LABELS = {
    "legitimate": "A normal purchase by the genuine cardholder on a known device and address.",
    "card_testing": (
        "Small or micro amounts, often repeated, on a newly seen card or device; "
        "fraudsters probing whether stolen card details work."
    ),
    "stolen_card": (
        "Card-not-present purchase using stolen card details: new device, mismatched or "
        "throwaway email, large distance from billing address, unusual amount."
    ),
    "account_takeover": (
        "An established card or account suddenly used from a new device or location "
        "with changed email or shipping details."
    ),
    "friendly_fraud_risk": (
        "Genuine cardholder but a pattern prone to chargebacks: subscriptions, "
        "recurring billing, or digital goods with inconsistent history."
    ),
    "other": "None of the listed patterns fits.",
}

RISK_LEVELS = [
    "Routine purchase by a long-established card, typical amount, known device, "
    "consistent email and address; nothing unusual.",
    "Mostly normal with one minor oddity (e.g. slightly high amount or unusual hour) "
    "that a genuine cardholder plausibly explains.",
    "Several soft signals together (new device, free webmail, distant purchase) but no decisive indicator; ambiguous.",
    "Strong indicators of fraud such as a brand-new card and device, email mismatch, "
    "far distance and an anomalous amount; likely fraudulent.",
    "Textbook stolen-card or account-takeover pattern: multiple hard signals aligned with no legitimate explanation.",
]


def build_questions() -> dict:
    return {
        PRIMARY_QUESTION: Noul(
            instructions=(
                "This card transaction is fraudulent, i.e. it was not authorized by the "
                "genuine cardholder. Judge from the transaction, device and velocity fields."
            ),
            criteria=NoulCriteria(
                true="The transaction is unauthorized (stolen card, account takeover, card testing).",
                false="The transaction was made by the genuine cardholder.",
            ),
        ),
        RISK_QUESTION: Score(
            instructions="How risky is this card transaction, from routine to textbook fraud?",
            criteria=RISK_LEVELS,
        ),
        PATTERN_QUESTION: Choice(
            instructions="Which fraud pattern best describes this transaction?",
            criteria=PATTERN_LABELS,
        ),
        "amount_anomalous": Noul(
            instructions=(
                "The transaction amount is anomalous for this card, either far above the "
                "card's average or an unusually tiny probing amount."
            )
        ),
        "new_device_for_card": Noul(
            instructions="The transaction comes from a device not previously associated with this card."
        ),
        "velocity_spike": Noul(
            instructions=(
                "The card, email or device shows a burst of recent activity or repeated "
                "amounts inconsistent with normal spending cadence."
            )
        ),
        "email_mismatch_suspicious": Noul(
            instructions=(
                "The purchaser and recipient email domains differ or use free/disposable "
                "webmail in a way that suggests the buyer is not the cardholder."
            )
        ),
    }
