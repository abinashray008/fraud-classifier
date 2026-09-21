interface Props {
  label: string
  value: number // 0..1
  tone?: 'risk' | 'neutral'
  hint?: string
}

export function ProbabilityBar({ label, value, tone = 'risk', hint }: Props) {
  const pct = Math.round(value * 100)
  const cls = tone === 'risk' ? (value >= 0.7 ? 'bar-high' : value >= 0.35 ? 'bar-mid' : 'bar-low') : 'bar-neutral'
  return (
    <div className="pbar" title={hint}>
      <div className="pbar-head">
        <span>{label}</span>
        <span className="mono">{value.toFixed(3)}</span>
      </div>
      <div className="pbar-track">
        <div className={`pbar-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}
