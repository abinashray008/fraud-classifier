"""Read-only investigation tools bound to a FeatureStore."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from app.agent.feature_store import FeatureStore


def build_tools(store: FeatureStore) -> list[BaseTool]:
    @tool
    def get_card_history(card_id: str) -> str:
        """Return the card's transaction history: counts, average amount, known devices,
        known email domains, home region, chargebacks and the most recent transactions."""
        return json.dumps(store.card_history(card_id))

    @tool
    def get_device_history(device_info: str) -> str:
        """Return how many distinct cards a device fingerprint has been seen with and
        how many chargebacks are associated with it."""
        return json.dumps(store.device_history(device_info))

    @tool
    def email_domain_reputation(domain: str) -> str:
        """Classify an email domain as disposable, high_risk, free_webmail, corporate_or_isp
        or unknown."""
        return json.dumps(store.email_domain_reputation(domain))

    @tool
    def velocity_stats(card_id: str, window_hours: int = 24) -> str:
        """Return transaction count, total amount and distinct devices for the card within
        the last `window_hours` hours."""
        return json.dumps(store.velocity_stats(card_id, window_hours))

    return [get_card_history, get_device_history, email_domain_reputation, velocity_stats]
