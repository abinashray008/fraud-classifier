"""OTP challenge lifecycle with a pluggable delivery provider.

The mock provider logs the code (and, in dev mode, the API returns it) so the
flow is testable end to end without an SMS vendor. Swap in a Twilio-style
provider by implementing `OtpProvider`.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.schemas.transaction import Challenge, ChallengeStatus
from app.store.memory import MemoryStore

logger = logging.getLogger(__name__)


class OtpProvider(Protocol):
    async def send(self, decision_id: str, code: str) -> None: ...


class MockSmsProvider:
    """Logs the OTP instead of sending it."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send(self, decision_id: str, code: str) -> None:
        self.sent.append((decision_id, code))
        logger.info("[MOCK SMS] decision=%s otp=%s", decision_id, code)


def _hash(code: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{code}".encode()).hexdigest()


class OtpService:
    def __init__(
        self,
        store: MemoryStore,
        provider: OtpProvider,
        *,
        ttl_seconds: int = 300,
        max_attempts: int = 3,
        code_length: int = 6,
        dev_mode: bool = True,
        clock=None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_attempts = max_attempts
        self.code_length = code_length
        self.dev_mode = dev_mode
        self._clock = clock or (lambda: datetime.now(UTC))
        self._secrets: dict[str, tuple[str, str]] = {}  # challenge_id -> (salt, hash)

    def _generate_code(self) -> str:
        upper = 10**self.code_length
        return f"{secrets.randbelow(upper):0{self.code_length}d}"

    async def create_challenge(self, decision_id: str) -> Challenge:
        now = self._clock()
        code = self._generate_code()
        salt = secrets.token_hex(8)
        challenge = Challenge(
            challenge_id=f"chal_{uuid.uuid4().hex[:12]}",
            decision_id=decision_id,
            status=ChallengeStatus.PENDING,
            attempts=0,
            max_attempts=self.max_attempts,
            expires_at=now + self.ttl,
            created_at=now,
            dev_code=code if self.dev_mode else None,
        )
        self._secrets[challenge.challenge_id] = (salt, _hash(code, salt))
        await self.store.put_challenge(challenge)
        await self.provider.send(decision_id, code)
        return challenge

    async def verify(self, challenge_id: str, code: str) -> Challenge:
        challenge = await self.store.get_challenge(challenge_id)
        if challenge is None:
            raise KeyError(challenge_id)
        if challenge.status != ChallengeStatus.PENDING:
            return challenge

        now = self._clock()
        if now >= challenge.expires_at:
            challenge.status = ChallengeStatus.EXPIRED
            await self.store.put_challenge(challenge)
            return challenge

        salt, expected = self._secrets[challenge_id]
        challenge.attempts += 1
        if hmac.compare_digest(_hash(code.strip(), salt), expected):
            challenge.status = ChallengeStatus.VERIFIED
            self._secrets.pop(challenge_id, None)
        elif challenge.attempts >= challenge.max_attempts:
            challenge.status = ChallengeStatus.FAILED
            self._secrets.pop(challenge_id, None)
        await self.store.put_challenge(challenge)
        return challenge
