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
    "legitimate": (
        "A purchase authorized by the genuine cardholder, including a purchase that an issuer must decline "
        "for an expired card, failed verification, or insufficient credit. A known device is not required."
    ),
    "card_testing": (
        "Small or micro amounts, often repeated, on a newly seen card or device, or a "
        "burst of swipes at several merchants in an hour; fraudsters probing whether "
        "stolen card details work."
    ),
    "stolen_card": (
        "Stolen card details or a counterfeit presentment. Card-not-present: new device, "
        "mismatched or throwaway email, large distance from billing address, unusual amount. "
        "Card-present: corroborating evidence of unauthorized use, such as incompatible merchant travel "
        "together with anomalous activity. Fallback or failed verification alone does not establish this pattern."
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
    "Mostly normal with a plausible legitimate explanation for an oddity, such as a technical "
    "chip-read failure causing fallback, a mistyped PIN, slightly high amount or unusual hour.",
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
                "authorization_decisions are prior policy actions, not confirmed fraud outcomes or ownership. "
                "Only independent confirmed_outcomes and verified device ownership establish those facts. "
                "When card_present is included, this is an in-person authorization that "
                "arrived through the card network (merchant terminal → acquirer → network "
                "→ issuer). Read card_present.amount_usd and card_present.rules. "
                "Authorization eligibility and cardholder authorization are different questions. "
                "An expired or blocked card, insufficient credit, failed PIN, or track CVV mismatch "
                "can require an issuer decline without proving that the cardholder did not authorize the purchase. "
                "Hard controls are enforced separately before live scoring; do not infer a fraud label from them. "
                "magstripe_fallback can follow a damaged chip or terminal read error during a genuine purchase. "
                "A genuine cardholder can mistype a PIN or present an expired card. Fallback and failed "
                "verification are contextual evidence; weigh corroborating activity and plausible technical "
                "explanations, never treat them alone as proof of unauthorized use. "
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
                false=(
                    "The genuine cardholder authorized the purchase, even if authorization controls require a decline."
                ),
            ),
        ),
        RISK_QUESTION: Score(
            instructions=(
                "How risky is this card transaction, from routine to textbook fraud? "
                "When card_present is present, weigh entry mode, merchant, "
                "card_present.amount_usd, amount_vs_mean_prior_ratio, and "
                "card_present.rules. Estimate unauthorized-use risk, not the chance an issuer declines. "
                "Fallback can be a technical chip/terminal problem; PIN failure can be a typing error. "
                "Neither sets a minimum fraud-risk level. Expiry, blocking, credit limits and verification "
                "failures can prevent authorization of a genuine purchase. Weigh independent corroborating "
                "evidence and legitimate explanations before assigning high fraud risk. "
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
                "amount_usd, and the merchant and amount rules in that section. A technical fallback, "
                "failed verification or issuer decline alone does not establish a stolen-card pattern. "
                "The legitimate pattern remains possible even when an authorization must decline."
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
                "The transaction comes from an explicit trusted device_id not previously associated "
                "with this card. DeviceInfo, including OS, model and build strings, is descriptive only. "
                "If device identity is unknown, this is not established."
            )
        ),
        "velocity_spike": Noul(
            instructions=(
                "The card, email or device shows a burst of recent activity or repeated "
                "amounts inconsistent with normal spending cadence. attempts_last_* and "
                "transactions_last_* count all attempts, including pending OTPs and declines. "
                "approvals_last_* counts approvals separately; confirmed_fraud_last_* counts only "
                "independently verified fraud labels, not model declines. Zero confirmed fraud "
                "does not establish that unlabeled attempts are legitimate. On a card-present "
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
                "The card-present data establishes an authorization control or verification failure, "
                "such as card_unusable, track_cvv_mismatch or pin_failure. This signal describes "
                "eligibility/verification, not whether the genuine cardholder authorized the purchase. "
                "magstripe_fallback alone does not establish invalid presentment: a chip or terminal "
                "can fail during a legitimate purchase. If card_present is absent, this is false."
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
