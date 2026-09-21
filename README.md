# Jev Fraud Classifier

Real-time card-transaction fraud classification built on the harness pattern from
[Building a Harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev):
a fast **System One** model (Jev, via `langchain-typesafe`) makes every live decision in a
single call, and a **System Two** LLM agent, supervised by Jev, only runs for the ambiguous slice.

```
POST /transactions/score
        |
        v
  StateBuilder  -> trims to readable fields, derives hour/amount-ratio/email-class/velocity flags
        |
        v
  Jev (one call, 7 questions in parallel)
     is_fraud: Noul     risk: Score(5 levels)     pattern: Choice(6)     4 signal Nouls
        |
        v
  Policy: p < t_low -> APPROVE | p > t_high & conf >= c_min -> DECLINE | else STEP_UP
                                                                          |
                                     +------------------------------------+
                                     |                                    |
                            OTP challenge (sync)              Investigation agent (async)
                            mock SMS, TTL, attempts           create_agent + read-only tools
                                     |                        ModelRouterMiddleware (Jev Choice)
                                     |                        JevToolGuard (Jev Noul per tool call)
                                     v                                    |
                       POST /challenges/{id}/verify  <--------------------+
                       final = f(OTP outcome, agent verdict)
```

## Layout

```
backend/
  app/
    config.py               pydantic-settings; all knobs via env (.env)
    schemas/transaction.py  Transaction (readable IEEE-CIS subset), JevAnswers, DecisionRecord, ...
    features/state_builder.py   Transaction -> compact Jev state
    jev/questions.py        the question set (Noul / Score / Choice)
    jev/classifier.py       TypeSafeClassifier adapter: retries, timeout, typed mapping
    policy/decision.py      threshold policy       policy/final.py   OTP + verdict merge
    stepup/otp.py           OtpService + MockSmsProvider (OtpProvider protocol)
    agent/                  feature_store stub, tools, guardrail (JevToolGuard), investigator
    services/decision_service.py   orchestration + background investigation tasks
    api/routes/             /health  /transactions/score  /challenges/{id}/verify  /decisions
  eval/                     IEEE-CIS loader, cached scoring, metrics, calibrate, regression
  tests/                    54 tests; no network required
frontend/                   Vite + React + TS console (simulator, inspector, OTP, investigation)
Dockerfile                  multi-stage: Vite build + FastAPI serving the SPA
docker-compose.yml          local one-command run (UI + API on :8000)
render.yaml / railway.toml  hosted Docker deploy
```

## Quick start

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 20+.

```bash
cp .env.example .env            # add TYPESAFE_API_KEY (and LLM_FAST + provider key for the agent tier)

# backend
cd backend
uv venv && uv pip install -e ".[eval,dev]"          # add ,openai or ,anthropic for the agent tier
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --reload --port 8000

# frontend (separate shell)
cd frontend
npm install
npm run dev                                          # http://localhost:5173, proxies /api -> :8000
```

Without `LLM_FAST` the app runs Jev-only: `STEP_UP` still issues an OTP, and the investigation
panel shows `skipped`.

## Deploy (share a URL)

One process serves the API and the built React app. The UI calls `/api/*`; FastAPI exposes the
same routes at `/` (for curl) and under `/api`.

Set secrets in `.env` first (`TYPESAFE_API_KEY`, and `GOOGLE_API_KEY` / `LLM_FAST` for the agent).

### Local: Docker Compose

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/). From the repo root:

```bash
docker compose up --build
```

Open http://localhost:8000. Stop with Ctrl+C (add `-d` to run in the background).

### Render

1. Push this repo to GitHub.
2. [Render](https://render.com) → New → Blueprint → select the repo (`render.yaml`).
3. Fill `TYPESAFE_API_KEY` and `GOOGLE_API_KEY` (leave other provider keys empty if unused).
4. Deploy. Share the `https://….onrender.com` URL.

Or New → Web Service → Docker, with health check `/health` and the same env vars.

### Railway

```bash
# one-time: npm i -g @railway/cli && railway login
railway init
railway variables set TYPESAFE_API_KEY=... GOOGLE_API_KEY=... LLM_FAST=google_genai:gemini-2.5-flash OTP_DEV_MODE=true CORS_ORIGINS=*
railway up
```

Or New Project → Deploy from GitHub; Railway picks up `Dockerfile` + `railway.toml`. Set the
same variables in the dashboard. Health check is `/health`.

The store is in-memory: a restart wipes decisions and OTP challenges, and you should run a
single instance.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | model, policy thresholds, which tiers are configured |
| `POST` | `/transactions/score` | score a transaction; returns decision, full Jev answers, policy rule, and a `challenge_id` (+ `dev_otp_code` in dev mode) for `STEP_UP` |
| `POST` | `/challenges/{id}/verify` | submit the OTP; returns challenge status, final decision, and the investigation record |
| `GET`  | `/decisions/{id}` | full record including the async investigation verdict once complete |
| `GET`  | `/decisions?limit=` | recent decisions |

Transaction fields accept either descriptive names (`amount`, `card_network`, `purchaser_email_domain`)
or IEEE-CIS column names (`TransactionAmt`, `card4`, `P_emaildomain`, `C1`, `D15`, ...).

Example:

```bash
curl -s localhost:8000/transactions/score -H 'content-type: application/json' -d '{
  "card_id": "card_new_777", "amount": 1250, "card4": "visa", "card6": "credit",
  "P_emaildomain": "gmail.com", "R_emaildomain": "mailinator.com",
  "dist1": 812, "TransactionDT": 270000, "DeviceInfo": "SM-G9650 Build/R16NW",
  "D2": 0, "D15": 0, "C1": 1
}'
```

## Decision policy

`policy/decision.py`, thresholds from env:

- `APPROVE` if `is_fraud.noul < POLICY_T_LOW`
- `DECLINE` if `is_fraud.noul > POLICY_T_HIGH` and `risk.confidence >= POLICY_C_MIN`
- otherwise `STEP_UP` (a high probability with low confidence steps up rather than hard-declining)

`policy/final.py` after OTP:

- OTP failed / expired -> `DECLINE`
- OTP verified -> `APPROVE`, unless the agent verdict has `fraud_prob >= POLICY_AGENT_DECLINE_PROB` -> `DECLINE`
  (OTP can be intercepted in an account takeover; strong evidence wins)
- If the agent finishes after verification, the record is re-finalized; poll `GET /decisions/{id}`

## Investigation agent

`agent/investigator.py` builds a LangChain `create_agent` with:

- **Tools** (`agent/tools.py`): `get_card_history`, `get_device_history`, `email_domain_reputation`,
  `velocity_stats`, backed by an in-memory `FeatureStore` stub seeded with demo cards
  (`card_good_001`, `card_ato_002`) and a shared fraud-farm device.
- **`ModelRouterMiddleware`** (from `langchain_typesafe.experimental`): a Jev `Choice` picks
  `LLM_FAST` vs `LLM_POWERFUL` per case. Only enabled when both are set and differ.
- **`JevToolGuard`** (`agent/guardrail.py`): a Jev `Noul` per tool call ("out of scope / side
  effects / PII?"). Blocks at `AGENT_TOOL_RISK_THRESHOLD` (default 0.30) and records blocked calls.
  This mirrors the library's `AutoModeMiddleware` but with a configurable threshold and visibility.
- **Structured output**: `response_format=Verdict` -> `{fraud_prob, pattern, rationale, evidence[]}`.

The agent never runs on the request path and never runs for clear `APPROVE`/`DECLINE` cases.

## Evaluation and calibration

Jev is a semantic model, so the harness uses only the human-readable IEEE-CIS columns
(`TransactionAmt, ProductCD, card4, card6, P/R_emaildomain, addr1/2, dist1, DeviceType/Info,
C1/C2/C13/C14, D1/D2/D4/D10/D15`) and derives descriptive features in `state_builder.py`.
Expect weaker signal than a GBM on the full 400-column feature set; the eval reports set expectations.

1. Download `train_transaction.csv` and `train_identity.csv` from
   [Kaggle IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data) into `data/ieee/`.
2. Sample, score (answers are cached per `(model, transaction_id)` so sweeps are free), report, calibrate:

```bash
cd backend
.venv/bin/python -m eval.load_ieee --data-dir ../data/ieee --n 5000 --fraud-frac 0.3
.venv/bin/python -m eval.run_eval --model jev-1.13.0 --concurrency 8
.venv/bin/python -m eval.metrics   --answers eval/reports/answers_jev-1.13.0.parquet --t-low 0.2 --t-high 0.8
.venv/bin/python -m eval.calibrate --answers eval/reports/answers_jev-1.13.0.parquet \
    --max-step-up 0.15 --max-fnr 0.05 --max-fpr 0.01
```

`calibrate` prints `POLICY_T_LOW` / `POLICY_T_HIGH` for your constraints; paste them into `.env`
together with the pinned `JEV_MODEL`.

3. Before bumping `JEV_MODEL`, score the same sample with the candidate and gate on drift:

```bash
.venv/bin/python -m eval.run_eval --model jev-1.14.0
.venv/bin/python -m eval.regression --baseline eval/reports/answers_jev-1.13.0.parquet \
    --candidate eval/reports/answers_jev-1.14.0.parquet --t-low 0.2 --t-high 0.8   # exit 1 on drift
```

### Calibration status

The shipped defaults (`t_low=0.20`, `t_high=0.80`, `c_min=0.60`) are **uncalibrated placeholders**.
The harness was validated end to end on synthetic IEEE-shaped data (`tests/test_eval.py`), but the
real sweep requires a `TYPESAFE_API_KEY` and the Kaggle files, which are not part of this repo.
Run step 2 above and record the output here.

## Notes on `langchain-typesafe` (0.0.1a2)

Pinned because the package is alpha. Differences from the blog post's snippets worth knowing:

- `questions` are passed to the `TypeSafeClassifier` constructor; `invoke(state)` / `ainvoke(state)`
  take only the state.
- `AutoModeMiddleware` uses a fixed 0.5 threshold; this repo's `JevToolGuard` is the configurable equivalent.
- The experimental middleware needs the `[experimental]` extra (pulls in `langchain` + `langgraph`).

All integration points are isolated in `jev/classifier.py` and `agent/guardrail.py`.

## Not in v1 (deferred by design)

Persistence (in-memory store only), analyst review queue, LangSmith dashboards (tracing works out of the
box if `LANGSMITH_TRACING=true` is set), real SMS provider.
