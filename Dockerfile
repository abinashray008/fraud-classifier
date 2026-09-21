# syntax=docker/dockerfile:1

# --- frontend ---------------------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Client calls /api/*; FastAPI exposes the same routes under /api in production.
RUN npm run build

# --- backend + static UI ----------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    FRONTEND_DIST=/app/frontend/dist \
    SERVE_FRONTEND=true

COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/app ./app
COPY backend/eval ./eval
RUN uv sync --frozen --no-dev --extra google --extra openai --extra anthropic

COPY --from=frontend /src/dist /app/frontend/dist

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render / Railway inject PORT. Local compose defaults to 8000.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
