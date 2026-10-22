import { useCallback, useEffect, useMemo, useState } from "react";
import { avatar, prettyName } from "../avatars";
import type { HandReplay, Highlight, ReplayAction, TableView } from "../types";
import { Table } from "./Table";

export { Card, SEAT_POS } from "./Table";

const STREET_CARDS: Record<string, number> = { preflop: 0, flop: 3, turn: 4, river: 5 };

export function describe(a: ReplayAction): string {
  if (a.action === "raise") return `raises to ${a.amount_to}`;
  if (a.action === "bet") return `bets ${a.amount}`;
  if (a.action === "call") return `calls ${a.amount}`;
  if (a.action === "all_in") return `all-in ${a.amount}`;
  return a.action === "fold" ? "folds" : "checks";
}

/** Table snapshot after the first `step` actions of a replay (all cards face-up: investigator view). */
function viewAt(h: HandReplay, step: number): TableView {
  const sb = (h.button_seat + 1) % 6;
  const bb = (h.button_seat + 2) % 6;
  const seats = new Map(h.seats.map((s) => {
    const blind = s.seat_no === sb ? h.small_blind : s.seat_no === bb ? h.big_blind : 0;
    return [s.player_id, { seat: s.seat_no, name: s.name || s.player_id.replace(/^SIM-/, ""), stack: s.starting_stack - blind,
      committed: blind, folded: false, cards: s.cards.includes("??") ? null : s.cards, last_action: null as string | null,
      net: null as number | null }];
  }));
  let street = "preflop";
  for (const a of h.actions.slice(0, step)) {
    if (a.street !== street) {
      street = a.street;
      seats.forEach((s) => { s.committed = 0; });
    }
    const s = seats.get(a.player_id)!;
    s.stack -= a.amount;
    s.committed = a.amount_to;
    s.last_action = describe(a);
    if (a.action === "fold") s.folded = true;
  }
  const done = step >= h.actions.length;
  if (done) {
    for (const r of h.seats) {
      const s = seats.get(r.player_id)!;
      s.net = r.net_chips;
      s.stack = r.starting_stack + r.net_chips;
      s.committed = 0;
    }
  }
  return {
    seats: [...seats.values()].sort((a, b) => a.seat - b.seat),
    board: h.board.slice(0, done ? h.board.length : STREET_CARDS[step > 0 ? h.actions[step - 1].street : "preflop"]),
    pot: done ? h.final_pot : h.actions[step].pot_before,
    button_seat: h.button_seat,
    to_act: done ? null : h.seats.find((s) => s.player_id === h.actions[step].player_id)?.seat_no ?? null,
  };
}

export function Replayer({ hand, pair, highlights = [], autoStart = false }: { hand: HandReplay; pair?: [string, string];
  highlights?: Highlight[]; autoStart?: boolean }) {
  const n = hand.actions.length;
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(autoStart);
  const [speed, setSpeed] = useState(1);
  const hl = useMemo(() => new Map(highlights.map((x) => [x.action_no, x.text])), [highlights]);
  const view = useMemo(() => viewAt(hand, step), [hand, step]);

  useEffect(() => { setStep(0); setPlaying(autoStart); }, [hand.hand_id, autoStart]);
  useEffect(() => {
    if (!playing) return;
    if (step >= n) { setPlaying(false); return; }
    const t = setTimeout(() => setStep((s) => Math.min(s + 1, n)), 1000 / speed);
    return () => clearTimeout(t);
  }, [playing, step, n, speed]);

  const go = useCallback((s: number) => { setPlaying(false); setStep(Math.max(0, Math.min(n, s))); }, [n]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowRight") { e.preventDefault(); go(step + 1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); go(step - 1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, step]);

  const seatOf = (pid: string) => hand.seats.find((s) => s.player_id === pid);
  const labels: Record<number, { text: string; kind: "you" | "partner" }> = {};
  if (pair) {
    const a = seatOf(pair[0]);
    const b = seatOf(pair[1]);
    if (a) labels[a.seat_no] = { text: "A", kind: "you" };
    if (b) labels[b.seat_no] = { text: "B", kind: "partner" };
  }
  const last = step > 0 ? hand.actions[step - 1] : null;
  const name = (pid: string) => prettyName(seatOf(pid)?.name || pid.replace(/^SIM-/, ""));
  let lastStreet = "";

  return (
    <div className="replayer">
      <div>
        <Table view={view} labels={labels} stepKey={step} showNet={step >= n}
          bubble={last ? { seat: seatOf(last.player_id)!.seat_no, text: describe(last) } : null} />
        <div className="controls-bar" role="toolbar" aria-label="Replay controls">
          <button className="btn small" onClick={() => go(0)} aria-label="First action" disabled={step === 0}>⏮</button>
          <button className="btn small" onClick={() => go(step - 1)} aria-label="Previous action" disabled={step === 0}>◀</button>
          <button className="btn primary" onClick={() => (step >= n ? (setStep(0), setPlaying(true)) : setPlaying(!playing))}>
            {playing ? "⏸ Pause" : step >= n ? "↺ Replay" : "▶ Play"}
          </button>
          <button className="btn small" onClick={() => go(step + 1)} aria-label="Next action" disabled={step >= n}>▶</button>
          <button className="btn small" onClick={() => go(n)} aria-label="Last action" disabled={step >= n}>⏭</button>
          <button className="btn small" onClick={() => setSpeed(speed === 1 ? 2 : 1)} title="Playback speed">{speed === 1 ? "🐢 1×" : "🐇 2×"}</button>
          <span className="muted small num">move {step}/{n} · ← →</span>
        </div>
        <p className="small muted">{hand.hand_id} · blinds {hand.small_blind}/{hand.big_blind} · every card face-up, like a casino's
          security camera{pair && <> · <strong style={{ color: "var(--seat-a)" }}>A</strong> {name(pair[0])} and{" "}
            <strong style={{ color: "var(--seat-b)" }}>B</strong> {name(pair[1])}</>}</p>
      </div>
      <div className="log" aria-label="Action log">
        <ol>
          {hand.actions.map((a, i) => {
            const header = a.street !== lastStreet ? (lastStreet = a.street) : null;
            const note = hl.get(a.action_no);
            const cls = [i === step - 1 ? "current" : "", i >= step ? "future" : "", note ? "hl" : ""].join(" ");
            return [
              header && <li key={`h-${a.street}`} className="street">{a.street}</li>,
              <li key={a.action_no} className={cls} onClick={() => go(i + 1)} title={note}>
                <span className="avatar sm">{avatar(seatOf(a.player_id)?.name ?? "")}</span>
                <span><strong>{name(a.player_id)}</strong> {describe(a)}{note && <span className="note">🚩 {note}</span>}</span>
              </li>,
            ];
          })}
        </ol>
      </div>
    </div>
  );
}
