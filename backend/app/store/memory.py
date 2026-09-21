"""In-memory store for decisions and challenges. Persistence is deferred to a later phase."""

from __future__ import annotations

import asyncio

from app.schemas.transaction import Challenge, DecisionRecord


class MemoryStore:
    def __init__(self) -> None:
        self._decisions: dict[str, DecisionRecord] = {}
        self._challenges: dict[str, Challenge] = {}
        self._lock = asyncio.Lock()

    # Decisions -----------------------------------------------------------------
    async def put_decision(self, record: DecisionRecord) -> None:
        async with self._lock:
            self._decisions[record.decision_id] = record

    async def get_decision(self, decision_id: str) -> DecisionRecord | None:
        return self._decisions.get(decision_id)

    async def list_decisions(self, limit: int = 50) -> list[DecisionRecord]:
        items = sorted(self._decisions.values(), key=lambda d: d.created_at, reverse=True)
        return items[:limit]

    # Challenges ----------------------------------------------------------------
    async def put_challenge(self, challenge: Challenge) -> None:
        async with self._lock:
            self._challenges[challenge.challenge_id] = challenge

    async def get_challenge(self, challenge_id: str) -> Challenge | None:
        return self._challenges.get(challenge_id)

    def clear(self) -> None:
        self._decisions.clear()
        self._challenges.clear()
