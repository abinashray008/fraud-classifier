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
    "presentment_invalid",
    "merchant_anomaly",
)

PATTERN_LABELS = {
    "legitimate": "A normal purchase by the genuine cardholder on a known device and address.",
    "card_testing": (
        "Small or micro amounts, often repeated, on a newly seen card or device, or a "
        "burst of swipes at several merchants in an hour; fraudsters probing whether "
        "stolen card details work."
    ),
    "stolen_card": (
        "Stolen card details or a counterfeit presentment. Card-not-present: new device, "
        "mismatched or throwaway email, large distance from billing address, unusual amount. "
        "Card-present: magstripe fallback, track CVV mismatch, a swipe far from the "
        "cardholder's recent merchant country, or an amount several times this card's "
        "usual ticket."
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
                "genuine cardholder. Judge from the transaction, device, and "
                "card_history fields (counts, USD amounts, days, 1h/24h/7d windows). "
                "When card_present is included, this is an in-person authorization that "
                "arrived through the card network (merchant terminal → acquirer → network "
                "→ issuer). Read card_present.amount_usd and card_present.rules. "
                "card_unusable, track_cvv_mismatch, pin_failure, or magstripe_fallback "
                "means the presentment is not a genuine cardholder purchase. "
                "amount_far_above_history (amount at least 4 times the card's prior mean), "
                "amount_far_above_this_merchant, and probing_amount (at most 1 USD, or the "
                "same sub-10 USD amount already seen twice in 24 hours) are fraud evidence "
                "from the authorization amount. A ratio near 1 supports a genuine cardholder. "
                "merchant_country_changed_within_2h, merchant_burst_1h, cash_like_mcc, "
                "cross_border, and swipe_without_pin support that judgment and are not "
                "enough on their own. over_limit is a credit control, separate from whether "
                "the amount is unusual for this card. If card_present is absent, ignore it."
            ),
            criteria=NoulCriteria(
                true="The transaction is unauthorized (stolen card, account takeover, card testing).",
                false="The transaction was made by the genuine cardholder.",
            ),
        ),
        RISK_QUESTION: Score(
            instructions=(
                "How risky is this card transaction, from routine to textbook fraud? "
                "When card_present is present, weigh entry mode, card status, track CVV, "
                "PIN, merchant, card_present.amount_usd, amount_vs_mean_prior_ratio, and "
                "card_present.rules. A lost or stolen card, a track CVV mismatch, a PIN "
                "failure, or a chip-to-magstripe fallback is at least strong fraud. "
                "amount_far_above_history or probing_amount raises fraud risk. Amount over "
                "available credit is an authorization problem, not by itself textbook fraud. "
                "If card_present is absent, ignore it."
            ),
            criteria=RISK_LEVELS,
        ),
        PATTERN_QUESTION: Choice(
            instructions=(
                "Which fraud pattern best describes this transaction? "
                "When card_present is present, use entry mode, track CVV, card status, "
                "amount_usd, and the merchant and amount rules in that section."
            ),
            criteria=PATTERN_LABELS,
        ),
        "amount_anomalous": Noul(
            instructions=(
                "The transaction amount is anomalous for this card, either far above the "
                "card's average or an unusually tiny probing amount. On a card-present "
                "authorization, read card_present.amount_usd, amount_vs_mean_prior_ratio, "
                "and amount_vs_this_merchant_ratio. This is true when amount_far_above_history, "
                "amount_far_above_this_merchant, or probing_amount is true. over_limit is a "
                "credit control and does not by itself make the amount anomalous."
            )
        ),
        "new_device_for_card": Noul(
            instructions=(
                "The transaction comes from a device fingerprint not previously associated "
                "with this card. A generic description such as Windows is not a fingerprint. "
                "If device identity is unknown, this is not established."
            )
        ),
        "velocity_spike": Noul(
            instructions=(
                "The card, email or device shows a burst of recent activity or repeated "
                "amounts inconsistent with normal spending cadence. On a card-present "
                "authorization, three or more distinct merchants in the last hour "
                "(merchant_burst_1h) is the same kind of burst."
            )
        ),
        "email_mismatch_suspicious": Noul(
            instructions=(
                "The purchaser and recipient email domains differ or use free/disposable "
                "webmail in a way that suggests the buyer is not the cardholder."
            )
        ),
        "presentment_invalid": Noul(
            instructions=(
                "The card-present authorization failed a control that means this is not "
                "a genuine presentment: card_unusable, track_cvv_mismatch, pin_failure, "
                "or magstripe_fallback is true. If card_present is absent, this is false."
            )
        ),
        "merchant_anomaly": Noul(
            instructions=(
                "The card-present merchant context is anomalous: cash_like_mcc, "
                "cross_border, merchant_country_changed_within_2h, or merchant_burst_1h "
                "is true. One of these alone is a soft signal. If card_present is absent, "
                "this is false."
            )
        ),
    }
