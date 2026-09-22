import type { DecisionRecord, JevAnswers, ScoreResponse } from '../api/types'
import { DecisionBadge } from '../components/DecisionBadge'
import { ProbabilityBar } from '../components/ProbabilityBar'

interface Props {
  score: ScoreResponse
  record: DecisionRecord | null
}

export function DecisionInspector({ score, record }: Props) {
  const { jev, explanation } = score
  const final = record?.final_decision ?? (score.decision !== 'STEP_UP' ? score.decision : null)

  return (
    <div className="card inspector">
      <div className="card-head">
        <h2>{jev ? 'Jev decision' : 'Decision'}</h2>
        <div className="badges">
          <DecisionBadge decision={score.decision} size="lg" />
          {score.decision === 'STEP_UP' && (
            <>
              <span className="muted">final</span>
              <DecisionBadge decision={final} size="lg" />
            </>
          )}
        </div>
      </div>

      <div className="meta mono">
        {jev ? <span>{jev.model}</span> : <span>model not called</span>}
        {jev?.latency_ms != null && <span>{jev.latency_ms.toFixed(0)} ms</span>}
        {jev?.input_tokens != null && <span>{jev.input_tokens} in / {jev.output_tokens ?? 0} out tokens</span>}
        <span>{score.decision_id}</span>
      </div>

      {record && <CardPresentAmount state={record.state} />}

      <section>
        {jev && <ProbabilityBar label="is_fraud (Noul)" value={jev.is_fraud.noul} />}
        <div className="thresholds mono small">
          <span>t_low {explanation.t_low}</span>
          <span>t_high {explanation.t_high}</span>
          {explanation.review_reason && <span>review {explanation.review_reason}</span>}
        </div>
        <p className="rule">
          <span className="muted">policy:</span> {explanation.rule}
        </p>
        {record?.final_reason && (
          <p className="rule">
            <span className="muted">final:</span> {record.final_reason}
          </p>
        )}
      </section>

      {jev && <JevAnswersView jev={jev} />}

      {record && (
        <details>
          <summary>{jev ? 'State sent to Jev' : 'State (model not called)'}</summary>
          <pre className="state">{JSON.stringify(record.state, null, 2)}</pre>
        </details>
      )}
    </div>
  )
}

function JevAnswersView({ jev }: { jev: JevAnswers }) {
  const levels = Object.keys(jev.risk.legend)
    .map(Number)
    .sort((a, b) => a - b)
  const maxLevel = levels[levels.length - 1] ?? 4
  const patterns = Object.entries(jev.pattern.probabilities).sort((a, b) => b[1] - a[1])

  return (
    <>
      <section>
        <h3>
          Risk score (Score) <span className="mono">{jev.risk.score.toFixed(2)} / {maxLevel}</span>{' '}
          <span className="muted small" title="How concentrated the risk-level distribution is. Not a fraud-probability confidence.">
            level concentration {jev.risk.confidence.toFixed(2)}
          </span>
        </h3>
        <div className="dist">
          {levels.map((lvl) => {
            const p = jev.risk.probabilities[String(lvl)] ?? 0
            return (
              <div key={lvl} className="dist-col" title={jev.risk.legend[String(lvl)]}>
                <div className="dist-bar" style={{ height: `${Math.max(2, p * 100)}%` }} />
                <span className="mono small">{lvl}</span>
                <span className="mono tiny">{(p * 100).toFixed(0)}%</span>
              </div>
            )
          })}
        </div>
        <p className="muted small legend">
          {maxLevel}: {jev.risk.legend[String(maxLevel)]}
        </p>
      </section>

      <section>
        <h3>
          Pattern (Choice) <code>{jev.pattern.choice}</code>{' '}
          <span className="muted small" title="How concentrated the pattern distribution is.">
            distribution concentration {jev.pattern.confidence.toFixed(2)}
          </span>
        </h3>
        {patterns.map(([name, p]) => (
          <ProbabilityBar key={name} label={name} value={p} tone="neutral" />
        ))}
      </section>

      {Object.keys(jev.signals).length > 0 && (
        <section>
          <h3>Signals (Nouls)</h3>
          {Object.entries(jev.signals).map(([name, s]) => (
            <ProbabilityBar key={name} label={name} value={s.noul} />
          ))}
        </section>
      )}
    </>
  )
}

const AMOUNT_RULES: Array<{ key: string; label: string }> = [
  { key: 'amount_far_above_history', label: '4× card mean' },
  { key: 'amount_far_above_this_merchant', label: '4× this merchant' },
  { key: 'probing_amount', label: 'probing amount' },
  { key: 'over_limit', label: 'over limit' },
]

function money(value: unknown): string {
  if (value === 'unknown') return 'unknown'
  if (typeof value !== 'number') return '—'
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function ratio(value: unknown): string {
  if (value === 'unknown') return 'unknown'
  if (typeof value !== 'number') return '—'
  return `${value.toFixed(2)}×`
}

function CardPresentAmount({ state }: { state: Record<string, unknown> }) {
  const present = state.card_present
  if (!present || typeof present !== 'object') return null
  const cp = present as Record<string, unknown>
  if (typeof cp.amount_usd !== 'number') return null
  const rules =
    cp.rules && typeof cp.rules === 'object' ? (cp.rules as Record<string, unknown>) : {}

  return (
    <section className="auth-amount">
      <h3>Authorization amount</h3>
      <p className="amount-figure mono">{money(cp.amount_usd)}</p>
      <ul className="facts">
        <li>
          vs card mean {money(cp.mean_prior_amount_usd)} ({ratio(cp.amount_vs_mean_prior_ratio)})
        </li>
        <li>
          vs this merchant {money(cp.mean_amount_usd_at_this_merchant)} ({ratio(cp.amount_vs_this_merchant_ratio)})
        </li>
        <li>last hour including this {money(cp.amount_usd_last_1h_including_this)}</li>
      </ul>
      <div className="rule-flags">
        {AMOUNT_RULES.map((rule) =>
          typeof rules[rule.key] === 'boolean' ? (
            <span key={rule.key} className={`badge ${rules[rule.key] ? 'badge-decline' : 'badge-pending'}`}>
              {rule.label}
              {rules[rule.key] ? '' : ' clear'}
            </span>
          ) : null,
        )}
      </div>
    </section>
  )
}
