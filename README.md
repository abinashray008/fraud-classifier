# Jev Fraud Classifier

Real-time card-transaction fraud classification built on the harness pattern from
[Building a Harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev):
a fast **System One** model (Jev, via `langchain-typesafe`) makes every live decision in a
single call, and a **System Two** LLM agent, supervised by Jev, only runs for the ambiguous slice.

```
POST /transactions/score
        |
        v
  StateBuilder  -> verified IEEE fields + card-history features; card_present rules on an in-person auth
        |
        v
  Hard controls, before the model
     issuer (either channel): unusable card, over credit limit -> DECLINE
     card-present only: track CVV mismatch, PIN failure -> DECLINE
        |
        v
  Jev (one call, 7 questions in parallel)
     is_fraud: Noul     risk: Score(5 levels)     pattern: Choice(6)     4 signal Nouls
        |
        v
  Policy: p < t_low -> APPROVE | p > t_high -> DECLINE | contradictory or thin evidence -> STEP_UP
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
    schemas/transaction.py  Transaction (IEEE-CIS subset; C/D kept opaque), JevAnswers, ...
    features/history.py     card-history features with units and time windows
    features/state_builder.py   Transaction + history -> compact Jev state
    jev/questions.py        the question set (Noul / Score / Choice)
    jev/classifier.py       TypeSafeClassifier adapter: retries, timeout, typed mapping
    policy/decision.py      threshold policy       policy/final.py   OTP + verdict merge
    stepup/otp.py           OtpService + MockSmsProvider (OtpProvider protocol)
    agent/                  feature_store stub, tools, guardrail (JevToolGuard), investigator
    services/decision_service.py   orchestration + background investigation tasks
    api/routes/             /health  /transactions/score  /challenges/{id}/verify  /decisions
  eval/                     IEEE-CIS loader, cached scoring, metrics, threshold sweep, probability calibration, regression
  tests/                    no network required
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
C1–C14 and D1–D15 are accepted under those opaque names and are **not** interpreted
(Vesta masked their meanings). ProductCD is passed through as `W`/`C`/`H`/`S`/`R` with
no product-type gloss. Card history (counts, USD, days, 1h/24h/7d windows) is computed
from the in-memory feature store at score time.

Example:

```bash
curl -s localhost:8000/transactions/score -H 'content-type: application/json' -d '{
  "card_id": "card_new_777", "amount": 1250, "card4": "visa", "card6": "credit",
  "P_emaildomain": "gmail.com", "R_emaildomain": "mailinator.com",
  "dist1": 812, "TransactionDT": 270000, "DeviceInfo": "SM-G9650 Build/R16NW"
}'
```

## Decision policy

`policy/decision.py`, thresholds from env:

- `APPROVE` if `is_fraud.noul < POLICY_T_LOW`
- `DECLINE` if `is_fraud.noul > POLICY_T_HIGH`
- `STEP_UP` when the probability sits between the thresholds, or when the answers contradict, or when the evidence is too thin to act

A contradiction is the fraud Noul pointing one way and the risk score or the pattern pointing the other: a high fraud probability with a routine risk score or a `legitimate` pattern, or a low fraud probability with a strong risk score or a fraud pattern (`card_testing`, `stolen_card`, `account_takeover`).

Thin evidence is a spread distribution. `risk.confidence` and `pattern.confidence` measure how concentrated that answer's own distribution is. TypeSafe documents confidence as that concentration statistic, not as the probability the answer is correct, and a Noul does not carry one. A value below `POLICY_EVIDENCE_MIN` (default 0.50, the "do not act" floor) sends the case to review. It never authorizes a decline. A decline also goes to review when every signal Noul is below 0.50; missing signal questions are not treated as missing evidence.

Whether the old extra gate (`DECLINE` only when `risk.confidence` was also high) improves outcomes is measured, not assumed. `eval.metrics` and `eval.calibrate` report `confidence_gate`. The gate can only turn a decline into a step-up. It improves the decline set only when the rows it removes are less often fraud than the declines it keeps (`removes_disproportionately_legitimate`).

`policy/final.py` after OTP:

- OTP failed / expired -> `DECLINE`
- OTP verified with a required investigation still running, failed, or missing a verdict -> pending (`final_decision: null`). Failure requires retry or manual review; it never silently approves.
- OTP verified with a completed required investigation -> `DECLINE` when `fraud_prob >= POLICY_AGENT_DECLINE_PROB`, otherwise `APPROVE`.
- When investigation is explicitly disabled, OTP verification can authorize `APPROVE`. It does not establish device ownership.
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

## Card-present authorization

A swipe, chip insert, or tap at a merchant is an authorization request the acquirer
sent through the card network for an approve or decline. Send `channel: "card_present"`
with the terminal facts: `entry_mode` (`swipe`, `chip`, `contactless`, `fallback_swipe`,
`keyed`), `cvm_result`, `pin_tries_exceeded`, `track_cvv`, merchant id / name / MCC /
country / city, and `cardholder_country`. `card_status` and the credit limits are issuer
facts: they apply on `card_not_present` as well. Combining terminal facts with
`channel: "card_not_present"` is rejected.

Those terminal facts reach Jev as a `card_present` section, including `rules` computed
from the request and from this card's earlier merchants. `is_fraud`, `risk`, `pattern`,
and the `presentment_invalid` and `merchant_anomaly` signals are instructed to judge
from that section. Rows without those fields, including IEEE-CIS, omit it. Issuer
`card_status` is on the transaction itself, so an explicit card-not-present channel
does not drop it.

Hard controls are evaluated before the model call. A timeout cannot approve them, and
the score response has no Jev answers when one fires:

- `card_unusable` — lost, stolen, expired, or blocked (either channel)
- `over_limit` — amount is above available credit or the single-purchase limit (either channel)
- `track_cvv_mismatch` — magstripe CVV1/CVC1 does not match (card-present only)
- `pin_failure` — PIN failed, or the PIN try limit is exceeded (card-present only)

These controls determine authorization eligibility, not whether the genuine cardholder
consented to the purchase. An expired card, a mistyped PIN, or insufficient credit can
require declining a legitimate purchase. Fraud prompts and risk/pattern questions keep
that distinction explicit. `presentment_invalid` describes a control or verification
failure; it is not a fraud label, and fallback alone does not make it true.

Chip-to-magstripe fallback can result from a damaged chip or terminal read problem,
as described in the [U.S. Payments Forum fallback guidance](https://www.uspaymentsforum.org/emv-implementation-guidance-fallback-transactions/).
The model treats fallback and failed verification as context, considers legitimate
explanations, and requires corroboration before inferring unauthorized use. Fallback
sets no minimum fraud-risk level.

Jev also reads `card_present.amount_usd` against this card's prior mean
(`amount_vs_mean_prior_ratio`) and against prior tickets at the same merchant.
Those comparisons are fraud evidence, separate from the credit-limit control:

- `amount_far_above_history` — amount is at least 4× the card's prior mean
- `amount_far_above_this_merchant` — amount is at least 4× the prior mean at this merchant
- `probing_amount` — amount is at most 1 USD, or the same sub-10 USD amount already appeared twice in 24 hours

These stay with Jev as well: chip-to-magstripe fallback, a swipe without PIN, a cash-like MCC
(`4829`, `6010`, `6011`, `6051`, `7995`), a cross-border merchant, a merchant-country
change inside 2 hours, and 3 or more distinct merchants in the last hour.

## IEEE-CIS fields (what Jev is allowed to believe)

Vesta published meanings for a small subset of columns and **deliberately masked**
the rest. The [competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
says C1–C14 are counts and D1–D15 are timedeltas, “such as how many addresses are
found to be associated with the payment card” / “days between previous transaction”,
with the actual mapping unpublished. The [1st-place write-up](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284)
treats those columns as unlabeled numerics rather than named semantics.

This harness is a **semantic** classifier, so guessed labels would be presented to
Jev as facts. Incorrect names (C14 = device transaction count, D15 = device age,
ProductCD `H` = hotel/travel) can produce confident, unsupported conclusions.

| Input | What Jev sees |
| --- | --- |
| `TransactionAmt`, `card4`, `card6`, `P/R_emaildomain`, `DeviceType`, `DeviceInfo` | Published / visible-in-values names (`amount_usd`, `card_network`, …) |
| `ProductCD` | Opaque code `W`/`C`/`H`/`S`/`R` — no product-type gloss |
| `addr1`, `addr2`, `dist1` | Opaque codes / a distance whose unit and endpoints were not published |
| `TransactionDT` | Hour of day only: `(dt / 3600) % 24` |
| `C1`–`C14`, `D1`–`D15` | **Not sent to Jev.** Kept on the schema for eval dumps |
| Card history | Computed from prior transactions of the same `card_id`: `prior_transaction_count`, `days_since_*`, `mean_amount_usd_prior`, `transactions_last_{1h,24h,7d}`, `amount_usd_last_*`, `distinct_devices_last_*`, `matching_amount_count_last_24h`, `device_seen_before_on_this_card`. Live scoring uses `FeatureStore`; eval computes the same keys time-respecting `TransactionDT` |

Jev is a semantic model, so expect weaker signal than a GBM on the full 400-column
feature set; the eval reports set expectations.

## Evaluation, action thresholds, and probability calibration

Card history is computed on the full timeline first, using only transactions with an earlier
`TransactionDT`. A later amount cannot change an earlier card average.

The sample is then cut on time into three periods:

- **development** — earliest 50% of the `TransactionDT` span
- **calibration** — the next 25% (`split=calibration`); `eval.calibrate` reads only this period and chooses `t_low` / `t_high`. The split name is the time cut. The command selects action thresholds; it does not calibrate probabilities.
- **test** — the last 25%, untouched by threshold selection; `eval.metrics`, `eval.calibration`, and `eval.regression` read only this period

The IEEE-CIS train file is about **3.5% fraud** (20,663 / 590,540). The loader draws a
representative sample inside each period so precision, PR-AUC, and step-up rate match that
prevalence. `--fraud-frac` oversamples and stores `sample_weight`; metrics and calibration
apply those weights. Do not report those three numbers on an unweighted 30% fraud sample.

1. Download `train_transaction.csv` and `train_identity.csv` from
   [Kaggle IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data) into `data/ieee/`.
2. Sample, score (answers are cached by input, questions, feature definitions, and resolved model so sweeps are free), select thresholds on the calibration period, then report on the test period:

```bash
cd backend
.venv/bin/python -m eval.load_ieee --data-dir ../data/ieee --n 5000
.venv/bin/python -m eval.run_eval --model jev-1.13.0 --concurrency 8
.venv/bin/python -m eval.calibrate --answers eval/reports/answers_jev-1.13.0.parquet \
    --max-step-up 0.15 --max-fnr 0.05 --max-fpr 0.01
.venv/bin/python -m eval.calibration --answers eval/reports/answers_jev-1.13.0.parquet \
    --out eval/reports/reliability.svg
.venv/bin/python -m eval.metrics   --answers eval/reports/answers_jev-1.13.0.parquet --t-low 0.2 --t-high 0.8
```

Legacy answer caches without fingerprints are rescored automatically by `eval.run_eval`.
Each output contains only the current sample and refreshes labels, splits, and weights from it.
Pin a model version to reuse answers; aliases such as `jev-latest` are always rescored.
Report commands reject missing, null, or unknown splits and transactions assigned to multiple
splits. For legacy samples without splits, rerun `eval.load_ieee` and then `eval.run_eval`;
do not assign old answers to calibration and test by hand.

`calibrate` prints `POLICY_T_LOW` / `POLICY_T_HIGH` / `POLICY_EVIDENCE_MIN` for your constraints; paste them into `.env` together with the pinned `JEV_MODEL`. It will not use test rows. It also prints `confidence_gate`, the outcome delta of the historical `risk.confidence` decline gate.

`calibration` scores `is_fraud.noul` against labels on the held-out test period: Brier score (mean squared error of the probability) and a reliability diagram (`--out` writes the SVG; the same bins are in the metrics JSON). A stated probability of 0.8 should land in a bin whose observed fraud rate is about 0.8. That is a property of a set of predictions, not a guarantee about one transaction.

3. Before bumping `JEV_MODEL`, score the same sample with the candidate and gate on drift:

```bash
.venv/bin/python -m eval.run_eval --model jev-1.14.0
.venv/bin/python -m eval.regression --baseline eval/reports/answers_jev-1.13.0.parquet \
    --candidate eval/reports/answers_jev-1.14.0.parquet --t-low 0.2 --t-high 0.8   # exit 1 on drift
```

### Calibration status

The shipped defaults (`t_low=0.20`, `t_high=0.80`, `evidence_min=0.50`) are **placeholders**.
`evidence_min` is the spread-distribution review floor, not a fitted probability threshold.
The harness was validated end to end on synthetic IEEE-shaped data (`tests/test_eval.py`), but the
real threshold sweep and the reliability diagram require a `TYPESAFE_API_KEY` and the Kaggle files,
which are not part of this repo. Run step 2 above and record the output here.

## Notes on `langchain-typesafe` (0.0.1a2)

Pinned because the package is alpha. Differences from the blog post's snippets worth knowing:

- `questions` are passed to the `TypeSafeClassifier` constructor; `invoke(state)` / `ainvoke(state)`
  take only the state.
- Choice and Score `confidence` is computed from that answer's probability distribution: a peaked
  distribution is high, a flat one is low. It is not a separate estimate that the answer is correct.
  Noul answers do not include it; `is_fraud.noul` is already P(fraud). See the decision policy above.
- `AutoModeMiddleware` uses a fixed 0.5 threshold; this repo's `JevToolGuard` is the configurable equivalent.
- The experimental middleware needs the `[experimental]` extra (pulls in `langchain` + `langgraph`).

All integration points are isolated in `jev/classifier.py` and `agent/guardrail.py`.

## Not in v1 (deferred by design)

Persistence (in-memory store only), analyst review queue, LangSmith dashboards (tracing works out of the
box if `LANGSMITH_TRACING=true` is set), real SMS provider.

### Device identity provenance

`DeviceInfo` / `device_info` always describes a device, OS, model or software build.
It is never used to establish identity, even if the string looks specific or random.
Android's [build fingerprint definition](https://source.android.com/docs/compatibility/12/android-12-cdd#3_2_2_build_parameters)
identifies a software build, not an individual physical device.

`device_id` is a separate, optional opaque key. A trusted server integration must
supply it from a verified device-enrollment service (for example, an issuer registry
that maps a verified device credential to a stable ID). Namespace IDs by issuer or
provider to avoid collisions and keep them stable across cards and software updates.
Never copy or hash DeviceInfo, a user agent, or a model/build string into this field.
Omit it when provenance cannot be established. The demo accepts trusted caller input;
it does not authenticate enrollment or make arbitrary client-supplied IDs trustworthy.
A production gateway must discard end-user device_id claims and populate the field
from its verified enrollment lookup.

Only explicit IDs feed device reuse, trusted-device lists, cross-card sharing and
chargeback lookups. Missing IDs leave identity evidence unknown; an exact distinct
count is unknown when any transaction in its window lacks an ID. Empty windows have
zero devices. Legacy `card_known_devices` descriptions no longer supply identity
evidence. IEEE-CIS has no such device ID, so its build strings remain descriptive.
The investigation tool `get_device_history` takes `device_id`; unrecorded IDs return
unknown. Demo seeds and presets use synthetic `demo:device:*` IDs, independently of
their display descriptions. Existing descriptions must never be backfilled as IDs.

### Authorization history, ownership, and fraud outcomes

Authorization decisions (`pending`, `approved`, `declined`) are policy actions.
They appear under `authorization_decisions` and never populate `confirmed_outcomes`
or establish device ownership. Each decision has one history row keyed by its
`decision_id`; OTP retries and later decision updates modify that row without
inflating transaction counts or velocity. Every card-identified attempt is recorded
before waiting on the model or OTP provider, including pending step-ups and attempts
whose processing later fails. The current attempt is excluded from its own scoring
snapshot. Attempts without a card ID cannot be aggregated into per-card history.

Live history exposes `attempts_last_{1h,24h,7d}`, `approvals_last_{1h,24h,7d}` and
`confirmed_fraud_last_{1h,24h,7d}` separately. Existing `transactions_last_*` fields
remain aliases for attempt counts; amount totals and repeated-amount checks also
include pending and declined attempts. Investigation `velocity_stats` exposes the
same separation with `*_in_window` keys. Windows use the original attempt time,
so resolving an old OTP cannot move its attempt into a newer window. Approval
counts reflect the current authorization decisions. Confirmed-fraud counts include
only independently adjudicated `fraud` labels known at lookup time; a decline or
chargeback alone does not establish fraud. Zero confirmed labels does not mean
unlabeled attempts are legitimate. Offline IEEE-CIS evaluation has no authorization
feed, so these additional live counters remain unknown there.

Device trust comes only from the internal `FeatureStore.verify_device_ownership`
method, called by a trusted enrollment/authentication adapter after verifying evidence
bound to both the card and device. It requires a source, evidence reference, and
verification time. `revoke_device_ownership` appends a revocation; later approvals
cannot undo it. A new independent verification is required to restore trust. Ownership
events are retained, and historical lookups honor their effective times. The seeded
legitimate device has an explicit synthetic enrollment event. The demo OTP is not
bound to device enrollment and never establishes trust.

`record_verified_outcome` accepts independent adjudication or chargeback evidence
with source, reference, and verification time. `confirmed_outcomes` contains only
`legitimate`, `fraud`, and `chargeback` labels from this feed; it is unknown when no
past transaction has a verified label. Corrections update the label and chargeback
counts without adding transactions. Classifier and investigation predictions never
call this method. These internal integration methods are not public HTTP endpoints
or agent tools; production adapters must authenticate their evidence sources.

Required investigations keep authorization pending until a completed verdict and
successful OTP are both available. The console follows the stored decision as the
investigation finishes. Failed investigations remain pending for retry/manual review;
this demo does not provide a retry or manual-review endpoint.

### Legitimate authorization-failure counterexamples

`eval/fixtures/authorization_counterexamples.json` contains eight synthetic genuine
purchases: a routine chip baseline, two technical fallback cases, an expired card,
a mistyped PIN, exhausted PIN retries, a still-locked card, and insufficient credit.
Each has a fraud label of zero and a separate expected hard-control decline flag.
Explanations and labels are evaluation metadata and are excluded from model state.

Run from `backend/` with the evaluation dependencies installed:

```bash
python -m eval.authorization_counterexamples --out eval/reports/authorization_counterexamples.parquet
python -m eval.run_eval --sample eval/reports/authorization_counterexamples.parquet --out eval/reports/counterexamples_prompt_v2.parquet
```

The second command calls the configured Jev model. It intentionally scores even
control-blocked inputs directly to check fraud semantics; live authorization still
declines those cases before invoking the model. Inspect fraud probability, risk,
pattern, and `presentment_invalid` independently of the expected issuer action.
Compare routine chip and fallback cases for an unjustified risk floor. This small,
all-legitimate stress set is not a calibration dataset or a prevalence estimate.
Prompt and feature revisions automatically invalidate cached answers. Keep separate
output files to compare revisions. Unit tests verify fixture plumbing
and live control behavior with a fake classifier; they do not measure Jev's accuracy.
