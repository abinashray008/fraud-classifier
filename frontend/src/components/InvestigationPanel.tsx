import type { InvestigationRecord } from '../api/types'
import { ProbabilityBar } from './ProbabilityBar'

export function InvestigationPanel({ inv }: { inv: InvestigationRecord }) {
  return (
    <div className="card">
      <div className="card-head">
        <h3>Investigation agent (LLM tier)</h3>
        <span className={`status status-${inv.status}`}>{inv.status}</span>
      </div>

      {inv.status === 'skipped' && (
        <p className="muted">
          Not run. Either the decision was clear (Jev decided outright) or the agent tier is disabled
          (set <code>LLM_FAST</code>).
        </p>
      )}
      {inv.status === 'running' && <p className="muted">Gathering evidence with read-only tools...</p>}
      {inv.status === 'failed' && <p className="warn">{inv.error}</p>}

      {inv.model_route && (
        <div className="kv">
          <span className="muted">Model route (Jev Choice)</span>
          <code>{inv.model_route}</code>
        </div>
      )}

      {inv.verdict && (
        <div className="verdict">
          <ProbabilityBar label={`Agent fraud probability (${inv.verdict.pattern})`} value={inv.verdict.fraud_prob} />
          <p>{inv.verdict.rationale}</p>
          {inv.verdict.evidence.length > 0 && (
            <ul>
              {inv.verdict.evidence.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {inv.tool_calls.length > 0 && (
        <details open>
          <summary>Tool calls ({inv.tool_calls.length})</summary>
          <ul className="tools">
            {inv.tool_calls.map((tc, i) => {
              const blocked = tc.result?.includes('blocked by Jev guardrail')
              return (
                <li key={i} className={blocked ? 'blocked' : ''}>
                  <code>
                    {tc.name}({JSON.stringify(tc.args)})
                  </code>
                  {blocked && <span className="badge badge-decline badge-sm">blocked</span>}
                  {tc.result && !blocked && <pre>{tc.result}</pre>}
                </li>
              )
            })}
          </ul>
        </details>
      )}

      {inv.blocked_tool_calls.length > 0 && (
        <div className="guard">
          <h4>Jev guardrail blocked {inv.blocked_tool_calls.length} call(s)</h4>
          <ul>
            {inv.blocked_tool_calls.map((b, i) => (
              <li key={i}>
                <code>{b.name}</code> {JSON.stringify(b.args)}{' '}
                <span className="mono">p(risky)={b.probability?.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
