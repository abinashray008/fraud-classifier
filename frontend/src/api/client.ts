import type {
  DecisionRecord,
  Health,
  ScoreResponse,
  TransactionInput,
  VerifyResponse,
} from './types'

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail))
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail: unknown = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

export const api = {
  health: () => request<Health>('/health'),
  score: (tx: TransactionInput) =>
    request<ScoreResponse>('/transactions/score', { method: 'POST', body: JSON.stringify(tx) }),
  verify: (challengeId: string, code: string) =>
    request<VerifyResponse>(`/challenges/${challengeId}/verify`, {
      method: 'POST',
      body: JSON.stringify({ code }),
    }),
  decision: (id: string) => request<DecisionRecord>(`/decisions/${id}`),
  decisions: (limit = 25) => request<DecisionRecord[]>(`/decisions?limit=${limit}`),
}
