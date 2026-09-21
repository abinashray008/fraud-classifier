from datetime import UTC, datetime, timedelta

import pytest

from app.schemas.transaction import ChallengeStatus
from app.stepup.otp import MockSmsProvider, OtpService
from app.store.memory import MemoryStore


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def svc():
    clock = Clock()
    provider = MockSmsProvider()
    service = OtpService(MemoryStore(), provider, ttl_seconds=60, max_attempts=2, dev_mode=True, clock=clock)
    return service, provider, clock


async def test_create_and_verify_success(svc):
    service, provider, _ = svc
    ch = await service.create_challenge("dec_1")
    assert ch.status == ChallengeStatus.PENDING
    assert len(ch.dev_code) == 6
    assert provider.sent == [("dec_1", ch.dev_code)]

    verified = await service.verify(ch.challenge_id, ch.dev_code)
    assert verified.status == ChallengeStatus.VERIFIED
    assert verified.attempts == 1
    # idempotent after resolution
    again = await service.verify(ch.challenge_id, "000000")
    assert again.status == ChallengeStatus.VERIFIED


async def test_wrong_code_then_fail_after_max_attempts(svc):
    service, _, _ = svc
    ch = await service.create_challenge("dec_2")
    r1 = await service.verify(ch.challenge_id, "bad")
    assert r1.status == ChallengeStatus.PENDING and r1.attempts == 1
    r2 = await service.verify(ch.challenge_id, "bad")
    assert r2.status == ChallengeStatus.FAILED and r2.attempts == 2
    # correct code no longer helps
    r3 = await service.verify(ch.challenge_id, ch.dev_code)
    assert r3.status == ChallengeStatus.FAILED


async def test_expiry(svc):
    service, _, clock = svc
    ch = await service.create_challenge("dec_3")
    clock.now += timedelta(seconds=61)
    r = await service.verify(ch.challenge_id, ch.dev_code)
    assert r.status == ChallengeStatus.EXPIRED


async def test_unknown_challenge(svc):
    service, _, _ = svc
    with pytest.raises(KeyError):
        await service.verify("nope", "123456")


async def test_dev_mode_off_hides_code():
    service = OtpService(MemoryStore(), MockSmsProvider(), dev_mode=False)
    ch = await service.create_challenge("dec_4")
    assert ch.dev_code is None
