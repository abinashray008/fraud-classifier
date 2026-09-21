import type { DecisionRecord, ScoreResponse } from '../api/types'
import { DecisionBadge } from '../components/DecisionBadge'
import { ProbabilityBar } from '../components/ProbabilityBar'

interface Props {
  score: ScoreResponse
  record: DecisionRecord | null
}

export function DecisionInspector({ score, record }: Props) {
  const { jev, explanation } = score
  const levels = Object.keys(jev.risk.legend)
    .map(Number)
    .sort((a, b) => a - b)
  const maxLevel = levels[levels.length - 1] ?? 4
  const patterns = Object.entries(jev.pattern.probabilities).sort((a, b) => b[1] - a[1])
  const final = record?.final_decision ?? (score.decision !== 'STEP_UP' ? score.decision : null)

  return (
    <div className="card inspector">
      <div className="card-head">
        <h2>Jev decision</h2>
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
        <span>{jev.model}</span>
        {jev.latency_ms != null && <span>{jev.latency_ms.toFixed(0)} ms</span>}
        {jev.input_tokens != null && <span>{jev.input_tokens} in / {jev.output_tokens ?? 0} out tokens</span>}
        <span>{score.decision_id}</span>
      </div>

      <section>
        <ProbabilityBar label="is_fraud (Noul)" value={jev.is_fraud.noul} />
        <div className="thresholds mono small">
          <span>t_low {explanation.t_low}</span>
          <span>t_high {explanation.t_high}</span>
          <span>c_min {explanation.c_min}</span>
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

      <section>
        <h3>
          Risk score (Score) <span className="mono">{jev.risk.score.toFixed(2)} / {maxLevel}</span>{' '}
          <span className="muted small">confidence {jev.risk.confidence.toFixed(2)}</span>
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
          <span className="muted small">confidence {jev.pattern.confidence.toFixed(2)}</span>
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

      {record && (
        <details>
          <summary>State sent to Jev</summary>
          <pre className="state">{JSON.stringify(record.state, null, 2)}</pre>
        </details>
      )}
    </div>
  )
}
