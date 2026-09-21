from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import Container, get_container
from app.schemas.transaction import DecisionRecord

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.get("", response_model=list[DecisionRecord])
async def list_decisions(limit: int = 50, c: Container = Depends(get_container)) -> list[DecisionRecord]:
    return await c.store.list_decisions(limit=limit)


@router.get("/{decision_id}", response_model=DecisionRecord)
async def get_decision(decision_id: str, c: Container = Depends(get_container)) -> DecisionRecord:
    record = await c.store.get_decision(decision_id)
    if record is None:
        raise HTTPException(status_code=404, detail="decision not found")
    return record
