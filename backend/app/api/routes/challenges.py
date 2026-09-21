from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import Container, get_container
from app.schemas.transaction import VerifyRequest, VerifyResponse

router = APIRouter(prefix="/challenges", tags=["challenges"])


@router.post("/{challenge_id}/verify", response_model=VerifyResponse)
async def verify_challenge(
    challenge_id: str, body: VerifyRequest, c: Container = Depends(get_container)
) -> VerifyResponse:
    try:
        return await c.decision_service.verify(challenge_id, body.code)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="challenge not found") from exc
