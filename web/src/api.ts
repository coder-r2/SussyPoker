import type { DetectorState, Direction, GameEvent, GameReport, GameState, HandReplay, Highlight, Strategy } from "./types";

// Game API (FastAPI). Locally: `uvicorn api.main:app --port 8000`. In production the Hugging Face Space URL is set
// at build time with VITE_API_BASE.
export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(API_BASE + path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, detail);
  }
  return r.json() as Promise<T>;
}

export interface NewSession { strategy: Strategy; direction?: Direction; intensity: number; partner_seat?: number | null }

export const api = {
  health: () => call<{ status: string; model_version: string; git_sha: string }>("/api/health"),
  create: (req: NewSession) => call<{ session_id: string; state: GameState }>("/api/sessions", req),
  state: (id: string) => call<{ state: GameState }>(`/api/sessions/${id}`),
  act: (id: string, action: string, amount?: number) =>
    call<{ state: GameState; hand_complete: boolean; detector: DetectorState | null; events: GameEvent[] }>(`/api/sessions/${id}/action`,
      { action, amount }),
  autoplay: (id: string, hands: number, user_mode: "colluding" | "honest") =>
    call<{ state: GameState; detector: DetectorState }>(`/api/sessions/${id}/autoplay`, { hands, user_mode }),
  detector: (id: string) => call<DetectorState>(`/api/sessions/${id}/detector`),
  report: (id: string) => call<GameReport>(`/api/sessions/${id}/report`),
  hand: (id: string, handId: string) => call<HandReplay>(`/api/sessions/${id}/hands/${handId}`),
};

/** Same rule as suspoker/game/replay.py: actions where one partner acts on the other. */
export function pairHighlights(replay: HandReplay, a: string, b: string): Highlight[] {
  const out: Highlight[] = [];
  const aggressor: Record<string, string> = {};
  const folded = new Set<string>();
  const verb: Record<string, string> = { fold: "folds to", call: "calls", raise: "re-raises", all_in: "goes all-in against" };
  for (const x of replay.actions) {
    const mate = x.player_id === a ? b : x.player_id === b ? a : null;
    const aggressive = x.action === "bet" || x.action === "raise" || (x.action === "all_in" && x.amount > x.to_call);
    if (mate && !folded.has(mate)) {
      if (x.to_call > 0 && aggressor[x.street] === mate) {
        out.push({ action_no: x.action_no, text: `${verb[x.action] ?? x.action} the partner's bet` });
      } else if (aggressive) {
        out.push({ action_no: x.action_no, text: "bets or raises with the partner in the pot" });
      }
    }
    if (aggressive) aggressor[x.street] = x.player_id;
    if (x.action === "fold") folded.add(x.player_id);
  }
  return out;
}
