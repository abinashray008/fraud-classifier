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
  type: 'text' | 'number' | 'boolean'
  group: 'transaction' | 'device' | 'card present'
}

const FIELDS: Field[] = [
  { key: 'card_id', label: 'Card id (feature-store key)', type: 'text', group: 'transaction' },
  { key: 'amount', label: 'Amount (USD)', type: 'number', group: 'transaction' },
  { key: 'card_network', label: 'Card network', type: 'text', group: 'transaction' },
  { key: 'card_type', label: 'Card type (debit/credit)', type: 'text', group: 'transaction' },
  { key: 'card_status', label: 'Card status, either channel (open/lost/stolen/expired/blocked)', type: 'text', group: 'transaction' },
  { key: 'available_credit_usd', label: 'Available credit (USD), either channel', type: 'number', group: 'transaction' },
  { key: 'single_purchase_limit_usd', label: 'Single-purchase limit (USD), either channel', type: 'number', group: 'transaction' },
  { key: 'purchaser_email_domain', label: 'Purchaser email domain', type: 'text', group: 'transaction' },
  { key: 'recipient_email_domain', label: 'Recipient email domain', type: 'text', group: 'transaction' },
  { key: 'billing_region', label: 'addr1 (address code)', type: 'text', group: 'transaction' },
  { key: 'dist1', label: 'dist1 (distance; unit unpublished)', type: 'number', group: 'transaction' },
  { key: 'timestamp_delta_seconds', label: 'TransactionDT (s since reference)', type: 'number', group: 'transaction' },
  { key: 'device_type', label: 'Device type', type: 'text', group: 'device' },
  { key: 'device_id', label: 'Trusted device ID', type: 'text', group: 'device' },
  { key: 'device_info', label: 'Device description', type: 'text', group: 'device' },
  { key: 'amount', label: 'Authorization amount (USD)', type: 'number', group: 'card present' },
  { key: 'channel', label: 'Channel (card_present / card_not_present)', type: 'text', group: 'card present' },
  { key: 'entry_mode', label: 'Entry mode (swipe/chip/contactless/fallback_swipe)', type: 'text', group: 'card present' },
  { key: 'cvm_result', label: 'CVM (pin_verified/pin_failed/signature/no_cvm)', type: 'text', group: 'card present' },
  { key: 'track_cvv', label: 'Track CVV (match/mismatch/not_present)', type: 'text', group: 'card present' },
  { key: 'pin_tries_exceeded', label: 'PIN tries exceeded', type: 'boolean', group: 'card present' },
  { key: 'merchant_id', label: 'Merchant id', type: 'text', group: 'card present' },
  { key: 'merchant_name', label: 'Merchant name', type: 'text', group: 'card present' },
  { key: 'mcc', label: 'MCC', type: 'text', group: 'card present' },
  { key: 'merchant_country', label: 'Merchant country', type: 'text', group: 'card present' },
  { key: 'merchant_city', label: 'Merchant city', type: 'text', group: 'card present' },
  { key: 'cardholder_country', label: 'Cardholder country', type: 'text', group: 'card present' },
  { key: 'terminal_attended', label: 'Terminal attended', type: 'boolean', group: 'card present' },
]

const GROUPS: Array<Field['group']> = ['transaction', 'device', 'card present']

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
      } else if (type === 'boolean') {
        ;(next as Record<string, unknown>)[key] = raw === 'true'
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
                  {f.type === 'boolean' ? (
                    <select
                      value={tx[f.key] === true ? 'true' : tx[f.key] === false ? 'false' : ''}
                      onChange={(e) => update(f.key, e.target.value, f.type)}
                    >
                      <option value="">—</option>
                      <option value="true">true</option>
                      <option value="false">false</option>
                    </select>
                  ) : (
                    <input
                      type={f.type}
                      step="any"
                      value={(tx[f.key] as string | number | undefined) ?? ''}
                      onChange={(e) => update(f.key, e.target.value, f.type)}
                    />
                  )}
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
