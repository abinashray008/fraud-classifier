import { useCallback, useEffect, useState } from 'react'
import { api, ApiError } from './api/client'
import type { DecisionRecord, Health, ScoreResponse, TransactionInput, VerifyResponse } from './api/types'
import { InvestigationPanel } from './components/InvestigationPanel'
import { OtpChallenge } from './components/OtpChallenge'
import { DecisionInspector } from './pages/DecisionInspector'
import { Simulator } from './pages/Simulator'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [score, setScore] = useState<ScoreResponse | null>(null)
  const [record, setRecord] = useState<DecisionRecord | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  const refresh = useCallback(async (id: string) => {
    try {
      setRecord(await api.decision(id))
    } catch {
      /* transient */
    }
  }, [])

  // Poll while the background investigation is running.
  useEffect(() => {
    if (!score || record?.investigation.status !== 'running') return
    const t = setInterval(() => refresh(score.decision_id), 1500)
    return () => clearInterval(t)
  }, [score, record?.investigation.status, refresh])

  const submit = async (tx: TransactionInput) => {
    setBusy(true)
    setError(null)
    setScore(null)
    setRecord(null)
    try {
      const s = await api.score(tx)
      setScore(s)
      await refresh(s.decision_id)
    } catch (err) {
      setError(err instanceof ApiError ? `${err.status}: ${err.message}` : String(err))
    } finally {
      setBusy(false)
    }
  }

  const onVerify = async (_r: VerifyResponse) => {
    if (score) await refresh(score.decision_id)
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>Jev Fraud Classifier</h1>
          <p className="muted">
            System One (Jev) decides in one call; ambiguous cases step up to OTP while a guarded LLM agent investigates.
          </p>
        </div>
        <div className="health mono small">
          {health ? (
            <>
              <span>model {health.jev_model}</span>
              <span className={health.jev_configured ? 'ok' : 'warn'}>jev {health.jev_configured ? 'on' : 'no key'}</span>
              <span className={health.agent_configured ? 'ok' : 'muted'}>agent {health.agent_configured ? 'on' : 'off'}</span>
              <span>
                t_low {health.policy.t_low} | t_high {health.policy.t_high}
              </span>
            </>
          ) : (
            <span className="warn">backend unreachable</span>
          )}
        </div>
      </header>

      <main>
        <div className="col">
          <Simulator onSubmit={submit} busy={busy} />
        </div>
        <div className="col">
          {error && <div className="card error">{error}</div>}
          {!score && !error && (
            <div className="card placeholder muted">
              Pick a preset or edit the form, then score. Results, probabilities and the policy rule appear here.
            </div>
          )}
          {score && (
            <>
              <DecisionInspector score={score} record={record} />
              {score.decision === 'STEP_UP' && score.challenge_id && (
                <OtpChallenge challengeId={score.challenge_id} devCode={score.dev_otp_code} onResult={onVerify} />
              )}
              {record && <InvestigationPanel inv={record.investigation} />}
            </>
          )}
        </div>
      </main>
    </div>
  )
}
