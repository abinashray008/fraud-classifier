"""Read-only investigation tools bound to a FeatureStore."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from app.agent.feature_store import FeatureStore


def build_tools(store: FeatureStore) -> list[BaseTool]:
    @tool
    def get_card_history(card_id: str) -> str:
        """Return card history, authorization decisions, independently verified fraud outcomes,
        active verified device ownership and its provenance. An approval is only an
        authorization action; it does not verify ownership or legitimacy."""
        return json.dumps(store.card_history(card_id))

    @tool
    def get_device_history(device_id: str) -> str:
        """Return distinct cards and chargebacks for the explicit trusted device_id.
        Use only device_id from the transaction or retrieved history. DeviceInfo,
        OS labels, model/build strings and user agents are descriptions, never IDs.
        Missing or unrecorded IDs return unknown."""
        return json.dumps(store.device_history(device_id))

    @tool
    def email_domain_reputation(domain: str) -> str:
        """Classify an email domain as disposable, high_risk, free_webmail, corporate_or_isp
        or unknown."""
        return json.dumps(store.email_domain_reputation(domain))

    @tool
    def velocity_stats(card_id: str, window_hours: int = 24) -> str:
        """Return separate attempt, approval and independently confirmed-fraud counts,
        attempted amount and distinct devices within the last `window_hours` hours.
        Attempts include pending OTPs and declines; transactions_in_window is an
        alias for attempts_in_window. Zero confirmed fraud does not clear unlabeled attempts."""
        return json.dumps(store.velocity_stats(card_id, window_hours))

    return [get_card_history, get_device_history, email_domain_reputation, velocity_stats]
