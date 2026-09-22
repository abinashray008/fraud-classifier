export type DecisionOutcome = 'APPROVE' | 'STEP_UP' | 'DECLINE'
export type ChallengeStatus = 'PENDING' | 'VERIFIED' | 'FAILED' | 'EXPIRED'

export interface TransactionInput {
  transaction_id?: string
  card_id?: string
  amount: number
  product_code?: string
  card_network?: string
  card_type?: string
  purchaser_email_domain?: string
  recipient_email_domain?: string
  billing_region?: string
  billing_country?: string
  dist1?: number
  timestamp_delta_seconds?: number
  device_type?: string
  device_info?: string
  device_id?: string
  /** Opaque IEEE-CIS count/timedelta columns; accepted but not interpreted. */
  c1?: number
  c2?: number
  c13?: number
  c14?: number
  d1?: number
  d2?: number
  d4?: number
  d10?: number
  d15?: number
  card_avg_amount?: number
  card_known_devices?: string[]
  /** In-person authorization routed through the card network. */
  channel?: 'card_present' | 'card_not_present'
  entry_mode?: 'swipe' | 'chip' | 'contactless' | 'fallback_swipe' | 'keyed'
  card_status?: 'open' | 'lost' | 'stolen' | 'expired' | 'blocked'
  cvm_result?: 'pin_verified' | 'pin_failed' | 'signature' | 'no_cvm'
  pin_tries_exceeded?: boolean
  track_cvv?: 'match' | 'mismatch' | 'not_present'
  merchant_id?: string
  merchant_name?: string
  mcc?: string
  merchant_country?: string
  merchant_city?: string
  terminal_id?: string
  terminal_attended?: boolean
  cardholder_country?: string
  available_credit_usd?: number
  single_purchase_limit_usd?: number
}

export interface NoulResult {
  type: 'noul'
  noul: number
}
export interface ChoiceResult {
  type: 'choice'
  choice: string
  probabilities: Record<string, number>
  confidence: number
}
export interface ScoreResult {
  type: 'score'
  score: number
  legend: Record<string, string>
  probabilities: Record<string, number>
  confidence: number
}

export interface JevAnswers {
  model: string
  is_fraud: NoulResult
  risk: ScoreResult
  pattern: ChoiceResult
  signals: Record<string, NoulResult>
  input_tokens?: number | null
  output_tokens?: number | null
  latency_ms?: number | null
  request_id?: string | null
}

export interface PolicyExplanation {
  rule: string
  t_low: number
  t_high: number
  /** Absent when a hard control declined before the model was called. */
  fraud_probability: number | null
  risk_score: number | null
  /** Concentration of the risk-level distribution. Not P(the fraud answer is correct). */
  risk_confidence: number | null
  review_reason: 'contradictory' | 'insufficient_evidence' | 'ambiguous' | null
}

export interface Verdict {
  fraud_prob: number
  pattern: string
  rationale: string
  evidence: string[]
}

export interface ToolCallRecord {
  name: string
  args: Record<string, unknown>
  result?: string | null
  probability?: number
}

export interface InvestigationRecord {
  status: 'not_started' | 'running' | 'completed' | 'failed' | 'skipped'
  verdict: Verdict | null
  model_route: string | null
  tool_calls: ToolCallRecord[]
  blocked_tool_calls: ToolCallRecord[]
  error: string | null
  started_at: string | null
  finished_at: string | null
}

export interface ScoreResponse {
  decision_id: string
  decision: DecisionOutcome
  jev: JevAnswers | null
  explanation: PolicyExplanation
  challenge_id: string | null
  dev_otp_code: string | null
}

export interface VerifyResponse {
  decision_id: string
  challenge_status: ChallengeStatus
  final_decision: DecisionOutcome | null
  final_reason: string | null
  attempts_remaining: number
  investigation: InvestigationRecord
}

export interface DecisionRecord {
  decision_id: string
  created_at: string
  transaction: Record<string, unknown>
  state: Record<string, unknown>
  jev: JevAnswers | null
  decision: DecisionOutcome
  explanation: PolicyExplanation
  challenge_id: string | null
  final_decision: DecisionOutcome | null
  final_reason: string | null
  investigation_required: boolean
  investigation: InvestigationRecord
}

export interface Health {
  status: string
  jev_model: string
  jev_configured: boolean
  agent_configured: boolean
  otp_dev_mode: boolean
  policy: { t_low: number; t_high: number; evidence_min: number }
}