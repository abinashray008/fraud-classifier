"""Dependency container wired at app startup."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.agent.feature_store import FeatureStore
from app.services.decision_service import DecisionService
from app.store.memory import MemoryStore


@dataclass
class Container:
    store: MemoryStore
    feature_store: FeatureStore
    decision_service: DecisionService


def get_container(request: Request) -> Container:
    return request.app.state.container
