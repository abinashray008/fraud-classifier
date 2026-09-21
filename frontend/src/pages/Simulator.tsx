import { useState } from 'react'
import type { TransactionInput } from '../api/types'
import { PRESETS } from '../presets'

interface Props {
  onSubmit: (tx: TransactionInput) => Promise<void>
  busy: boolean
}

type Field = {
  key: keyof TransactionInput
  label: string
  type: 'text' | 'number'
  group: 'transaction' | 'device' | 'velocity'
}

const FIELDS: Field[] = [
  { key: 'card_id', label: 'Card id (feature-store key)', type: 'text', group: 'transaction' },
  { key: 'amount', label: 'Amount (USD)', type: 'number', group: 'transaction' },
  { key: 'product_code', label: 'Product code (W/C/H/S/R)', type: 'text', group: 'transaction' },
  { key: 'card_network', label: 'Card network', type: 'text', group: 'transaction' },
  { key: 'card_type', label: 'Card type (debit/credit)', type: 'text', group: 'transaction' },
  { key: 'purchaser_email_domain', label: 'Purchaser email domain', type: 'text', group: 'transaction' },
  { key: 'recipient_email_domain', label: 'Recipient email domain', type: 'text', group: 'transaction' },
  { key: 'billing_region', label: 'Billing region code', type: 'text', group: 'transaction' },
  { key: 'distance_billing_to_purchase', label: 'Distance billing to purchase', type: 'number', group: 'transaction' },
  { key: 'timestamp_delta_seconds', label: 'Timestamp offset (s)', type: 'number', group: 'transaction' },
  { key: 'device_type', label: 'Device type', type: 'text', group: 'device' },
  { key: 'device_info', label: 'Device info', type: 'text', group: 'device' },
  { key: 'device_txn_count', label: 'Txns from device', type: 'number', group: 'device' },
  { key: 'days_since_device_first_seen', label: 'Days since device first seen', type: 'number', group: 'device' },
  { key: 'card_txn_count', label: 'Card txn count', type: 'number', group: 'velocity' },
  { key: 'email_txn_count', label: 'Email txn count', type: 'number', group: 'velocity' },
  { key: 'days_since_prev_txn', label: 'Days since previous txn', type: 'number', group: 'velocity' },
  { key: 'days_since_card_first_seen', label: 'Days since card first seen', type: 'number', group: 'velocity' },
  { key: 'days_since_prev_txn_same_amount', label: 'Days since same amount', type: 'number', group: 'velocity' },
  { key: 'card_avg_amount', label: 'Card average amount', type: 'number', group: 'velocity' },
]

const GROUPS: Array<Field['group']> = ['transaction', 'device', 'velocity']

export function Simulator({ onSubmit, busy }: Props) {
  const [tx, setTx] = useState<TransactionInput>(PRESETS[0].tx)
  const [presetId, setPresetId] = useState(PRESETS[0].id)
  const [showJson, setShowJson] = useState(false)

  const applyPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)
    if (p) {
      setPresetId(id)
      setTx(p.tx)
    }
  }

  const update = (key: keyof TransactionInput, raw: string, type: Field['type']) => {
    setTx((prev) => {
      const next = { ...prev }
      if (raw === '') {
        delete next[key]
      } else if (type === 'number') {
        const n = Number(raw)
        if (!Number.isNaN(n)) (next as Record<string, unknown>)[key] = n
      } else {
        ;(next as Record<string, unknown>)[key] = raw
      }
      return next
    })
  }

  return (
    <div className="card simulator">
      <div className="card-head">
        <h2>Transaction simulator</h2>
        <button className="link" onClick={() => setShowJson((s) => !s)}>
          {showJson ? 'form' : 'json'}
        </button>
      </div>

      <div className="presets">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            className={`preset ${presetId === p.id ? 'active' : ''}`}
            onClick={() => applyPreset(p.id)}
            title={p.description}
          >
            {p.label}
          </button>
        ))}
      </div>
      <p className="muted small">{PRESETS.find((p) => p.id === presetId)?.description}</p>

      {showJson ? (
        <textarea
          className="json"
          value={JSON.stringify(tx, null, 2)}
          onChange={(e) => {
            try {
              setTx(JSON.parse(e.target.value))
            } catch {
              /* keep typing */
            }
          }}
          rows={22}
        />
      ) : (
        GROUPS.map((g) => (
          <fieldset key={g}>
            <legend>{g}</legend>
            <div className="grid">
              {FIELDS.filter((f) => f.group === g).map((f) => (
                <label key={f.key}>
                  <span>{f.label}</span>
                  <input
                    type={f.type}
                    step="any"
                    value={(tx[f.key] as string | number | undefined) ?? ''}
                    onChange={(e) => update(f.key, e.target.value, f.type)}
                  />
                </label>
              ))}
            </div>
          </fieldset>
        ))
      )}

      <button className="primary" disabled={busy} onClick={() => onSubmit(tx)}>
        {busy ? 'Scoring with Jev...' : 'Score transaction'}
      </button>
    </div>
  )
}
