import { useState } from 'react'
import type { DecisionRecord, VerifyResponse } from '../api/types'
import { api, ApiError } from '../api/client'
import { DecisionBadge } from './DecisionBadge'

interface Props {
  challengeId: string
  devCode: string | null
  onResult: (r: VerifyResponse) => void
  record: DecisionRecord | null
}

export function OtpChallenge({ challengeId, devCode, onResult, record }: Props) {
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<VerifyResponse | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const r = await api.verify(challengeId, code)
      setResult(r)
      onResult(r)
      setCode('')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const finalDecision = record ? record.final_decision : result?.final_decision ?? null
  const finalReason = record ? record.final_reason : result?.final_reason
  const resolved = result && result.challenge_status !== 'PENDING'

  return (
    <div className="card otp">
      <h3>Step-up: one-time passcode</h3>
      <p className="muted">
        Jev found this transaction ambiguous. The cardholder receives an OTP while the investigation
        agent works in the background.
      </p>
      {devCode && (
        <div className="devcode">
          <span className="muted">dev mode - code sent via mock SMS:</span> <code>{devCode}</code>
        </div>
      )}
      {!resolved ? (
        <form onSubmit={submit} className="otp-form">
          <input
            inputMode="numeric"
            pattern="[0-9]*"
            maxLength={8}
            placeholder="Enter code"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            disabled={busy}
            autoFocus
          />
          <button type="submit" disabled={busy || code.length === 0}>
            {busy ? 'Verifying...' : 'Verify'}
          </button>
          {result && result.challenge_status === 'PENDING' && (
            <span className="warn">Incorrect. {result.attempts_remaining} attempt(s) remaining.</span>
          )}
          {error && <span className="warn">{error}</span>}
        </form>
      ) : (
        <div className="otp-result">
          <div>
            OTP <strong>{result.challenge_status}</strong>
          </div>
          <div className="final">
            {finalDecision ? 'Final decision: ' : 'Authorization: '}<DecisionBadge decision={finalDecision} />
          </div>
          <div className="muted">{finalReason}</div>
        </div>
      )}
    </div>
  )
}
