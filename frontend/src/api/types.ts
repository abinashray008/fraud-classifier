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
  distance_billing_to_purchase?: number
  timestamp_delta_seconds?: number
  device_type?: string
  device_info?: string
  card_txn_count?: number
  addr_match_count?: number
  email_txn_count?: number
  device_txn_count?: number
  days_since_prev_txn?: number
  days_since_card_first_seen?: number
  days_since_prev_txn_same_addr?: number
  days_since_prev_txn_same_amount?: number
  days_since_device_first_seen?: number
  card_avg_amount?: number
  card_known_devices?: string[]
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
  c_min: number
  fraud_probability: number
  risk_confidence: number
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
  jev: JevAnswers
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
  jev: JevAnswers
  decision: DecisionOutcome
  explanation: PolicyExplanation
  challenge_id: string | null
  final_decision: DecisionOutcome | null
  final_reason: string | null
  investigation: InvestigationRecord
}

export interface Health {
  status: string
  jev_model: string
  jev_configured: boolean
  agent_configured: boolean
  otp_dev_mode: boolean
  policy: { t_low: number; t_high: number; c_min: number }
}
