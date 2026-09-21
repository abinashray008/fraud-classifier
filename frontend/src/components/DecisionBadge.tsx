import type { DecisionOutcome } from '../api/types'

export function DecisionBadge({ decision, size = 'md' }: { decision: DecisionOutcome | null; size?: 'md' | 'lg' }) {
  if (!decision) return <span className={`badge badge-pending badge-${size}`}>PENDING</span>
  const cls = decision === 'APPROVE' ? 'badge-approve' : decision === 'DECLINE' ? 'badge-decline' : 'badge-stepup'
  return <span className={`badge ${cls} badge-${size}`}>{decision.replace('_', ' ')}</span>
}
