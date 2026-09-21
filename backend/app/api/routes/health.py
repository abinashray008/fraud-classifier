from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "jev_model": settings.jev_model,
        "jev_configured": settings.jev_configured,
        "agent_configured": settings.agent_configured,
        "otp_dev_mode": settings.otp_dev_mode,
        "policy": {
            "t_low": settings.policy_t_low,
            "t_high": settings.policy_t_high,
            "evidence_min": settings.policy_evidence_min,
        },
    }
