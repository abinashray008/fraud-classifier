from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import Container, get_container
from app.schemas.transaction import ScoreResponse, Transaction

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("/score", response_model=ScoreResponse)
async def score_transaction(tx: Transaction, c: Container = Depends(get_container)) -> ScoreResponse:
    try:
        return await c.decision_service.score(tx)
    except Exception as exc:  # noqa: BLE001
        # Classifier / upstream failures surface as 502 so callers can fall back.
        raise HTTPException(status_code=502, detail=f"classification failed: {exc}") from exc
