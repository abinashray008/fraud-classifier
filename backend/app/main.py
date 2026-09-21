"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.feature_store import FeatureStore
from app.api.deps import Container
from app.api.routes import challenges, decisions, health, transactions
from app.config import Settings, get_settings
from app.jev.classifier import FraudClassifier, JevFraudClassifier
from app.policy.decision import PolicyThresholds
from app.services.decision_service import DecisionService
from app.stepup.otp import MockSmsProvider, OtpService
from app.store.memory import MemoryStore

logger = logging.getLogger(__name__)

API_ROUTERS = (health.router, transactions.router, challenges.router, decisions.router)


def build_container(settings: Settings, classifier: FraudClassifier | None = None) -> Container:
    store = MemoryStore()
    feature_store = FeatureStore()
    feature_store.seed_demo()

    if classifier is None:
        if not settings.jev_configured:
            raise RuntimeError("TYPESAFE_API_KEY is not set; cannot create the Jev classifier.")
        classifier = JevFraudClassifier(
            api_key=settings.typesafe_api_key,
            model=settings.jev_model,
            timeout=settings.jev_timeout_seconds,
            max_retries=settings.jev_max_retries,
        )

    otp = OtpService(
        store,
        MockSmsProvider(),
        ttl_seconds=settings.otp_ttl_seconds,
        max_attempts=settings.otp_max_attempts,
        code_length=settings.otp_code_length,
        dev_mode=settings.otp_dev_mode,
    )

    investigator = None
    if settings.agent_configured:
        from app.agent.investigator import AgentInvestigator

        investigator = AgentInvestigator(
            feature_store=feature_store,
            llm_fast=settings.llm_fast,
            llm_powerful=settings.llm_powerful or None,
            typesafe_api_key=settings.typesafe_api_key,
            jev_model=settings.jev_model,
            tool_risk_threshold=settings.agent_tool_risk_threshold,
            max_steps=settings.agent_max_steps,
        )
    else:
        logger.info("Investigation agent disabled (set LLM_FAST and TYPESAFE_API_KEY to enable).")

    service = DecisionService(
        classifier=classifier,
        store=store,
        otp=otp,
        thresholds=PolicyThresholds(
            t_low=settings.policy_t_low,
            t_high=settings.policy_t_high,
            c_min=settings.policy_c_min,
        ),
        feature_store=feature_store,
        investigator=investigator,
        agent_decline_prob=settings.policy_agent_decline_prob,
    )
    return Container(store=store, feature_store=feature_store, decision_service=service)


def create_app(settings: Settings | None = None, classifier: FraudClassifier | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container(settings, classifier)
        yield

    app = FastAPI(title="Jev Fraud Classifier", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in API_ROUTERS:
        app.include_router(router)
        # Same routes under /api so the Vite client (BASE=/api) works without a proxy.
        app.include_router(router, prefix="/api")

    _mount_frontend(app, settings)
    return app


def resolve_frontend_dist(settings: Settings) -> Path | None:
    """Return the Vite build directory when it exists and serving is enabled."""
    if not settings.serve_frontend:
        return None
    candidates: list[Path] = []
    if settings.frontend_dist:
        candidates.append(Path(settings.frontend_dist).expanduser())
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "frontend" / "dist")
    candidates.append(Path("/app/frontend/dist"))
    seen: set[str] = set()
    for raw in candidates:
        key = str(raw)
        if key in seen:
            continue
        seen.add(key)
        if raw.is_dir() and (raw / "index.html").is_file():
            return raw.resolve()
    return None


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    dist = resolve_frontend_dist(settings)
    if dist is None:
        return

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = dist / "index.html"

    @app.get("/")
    async def spa_index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (dist / full_path).resolve()
        try:
            candidate.relative_to(dist)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Not Found") from exc
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)

    logger.info("Serving frontend from %s", dist)


app = create_app()
