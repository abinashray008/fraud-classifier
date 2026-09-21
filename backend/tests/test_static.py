from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app, resolve_frontend_dist


def _settings(**kwargs) -> Settings:
    values = dict(
        TYPESAFE_API_KEY="test-key",
        AGENT_ENABLED=False,
        SERVE_FRONTEND=True,
        _env_file=None,
    )
    values.update(kwargs)
    return Settings(**values)


def test_resolve_skips_when_disabled():
    assert resolve_frontend_dist(_settings(SERVE_FRONTEND=False, FRONTEND_DIST="/nope")) is None


def test_resolve_uses_explicit_dist(tmp_path: Path):
    (tmp_path / "index.html").write_text("<html>ui</html>")
    found = resolve_frontend_dist(_settings(FRONTEND_DIST=str(tmp_path)))
    assert found == tmp_path.resolve()


@pytest.fixture
async def spa_client(tmp_path: Path, fake_classifier):
    (tmp_path / "index.html").write_text("<html>spa</html>")
    (tmp_path / "favicon.svg").write_text("<svg />")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('ok')")
    app = create_app(
        settings=_settings(FRONTEND_DIST=str(tmp_path)),
        classifier=fake_classifier,
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


async def test_spa_index_and_assets(spa_client):
    root = await spa_client.get("/")
    assert root.status_code == 200
    assert "spa" in root.text

    asset = await spa_client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text

    icon = await spa_client.get("/favicon.svg")
    assert icon.status_code == 200

    unknown = await spa_client.get("/not-a-real-route")
    assert unknown.status_code == 200
    assert "spa" in unknown.text


async def test_spa_does_not_shadow_api(spa_client):
    health = await spa_client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    api_health = await spa_client.get("/api/health")
    assert api_health.status_code == 200

    docs = await spa_client.get("/docs")
    assert docs.status_code == 200

    missing = await spa_client.get("/api/does-not-exist")
    assert missing.status_code == 404
