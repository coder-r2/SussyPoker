import { useCallback, useEffect, useMemo, useState } from "react";
import { shortId } from "../format";
import type { Highlight, HandReplay, ReplayAction } from "../types";

// Seat positions on the oval (percent of the table box), seats 0-5 clockwise from the bottom.
const SEAT_POS = [
  [50, 91], [11, 70], [11, 30], [50, 9], [89, 30], [89, 70],
];
const SUIT = { s: "♠", h: "♥", d: "♦", c: "♣" } as Record<string, string>;
const STREET_CARDS = { preflop: 0, flop: 3, turn: 4, river: 5 } as Record<string, number>;
const VERB: Record<string, string> = { fold: "folds", check: "checks", call: "calls", bet: "bets", raise: "raises to",
  all_in: "all-in" };

export function Card({ card, big }: { card?: string; big?: boolean }) {
  if (!card) return <span className={`pcard empty${big ? " big" : ""}`} aria-hidden />;
  const suit = card[1];
  const red = suit === "h" || suit === "d";
  return (
    <span className={`pcard${red ? " red" : ""}${big ? " big" : ""}`} aria-label={card}>
      {card[0] === "T" ? "10" : card[0]}{SUIT[suit]}
    </span>
  );
}

function describe(a: ReplayAction): string {
  if (a.action === "raise") return `raises to ${a.amount_to}`;
  if (a.action === "bet") return `bets ${a.amount}`;
  if (a.action === "call") return `calls ${a.amount}`;
  if (a.action === "all_in") return `all-in ${a.amount}`;
  return VERB[a.action];
}

interface SeatState { stack: number; commit: number; folded: boolean; last: string }

/** Table state after the first `step` actions. */
function stateAt(h: HandReplay, step: number) {
  const sb = (h.button_seat + 1) % 6;
  const bb = (h.button_seat + 2) % 6;
  const seats = new Map<string, SeatState>();
  for (const s of h.seats) {
    const blind = s.seat_no === sb ? h.small_blind : s.seat_no === bb ? h.big_blind : 0;
    seats.set(s.player_id, { stack: s.starting_stack - blind, commit: blind, folded: false,
      last: blind ? (s.seat_no === sb ? `SB ${blind}` : `BB ${blind}`) : "" });
  }
  let street = "preflop";
  for (const a of h.actions.slice(0, step)) {
    if (a.street !== street) {
      street = a.street;
      seats.forEach((s) => { s.commit = 0; s.last = s.folded ? s.last : ""; });
    }
    const s = seats.get(a.player_id)!;
    s.stack -= a.amount;
    s.commit = a.amount_to;
    s.last = describe(a);
    if (a.action === "fold") s.folded = true;
  }
  const done = step >= h.actions.length;
  const pot = done ? h.final_pot : h.actions[step].pot_before;
  const boardN = done ? h.board.length : STREET_CARDS[step > 0 ? h.actions[step - 1].street : "preflop"];
  const actor = done ? null : h.actions[step].player_id;
  return { seats, pot, boardN, actor, done, street };
}

export function Replayer({ hand, pair, highlights = [] }: { hand: HandReplay; pair?: [string, string];
  highlights?: Highlight[] }) {
  const n = hand.actions.length;
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const hl = useMemo(() => new Map(highlights.map((x) => [x.action_no, x.text])), [highlights]);
  const st = useMemo(() => stateAt(hand, step), [hand, step]);

  useEffect(() => { setStep(0); setPlaying(false); }, [hand.hand_id]);
  useEffect(() => {
    if (!playing) return;
    if (step >= n) { setPlaying(false); return; }
    const t = setTimeout(() => setStep((s) => Math.min(s + 1, n)), 900 / speed);
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

  const name = (pid: string) => hand.seats.find((s) => s.player_id === pid)?.name || shortId(pid);
  let lastStreet = "";

  return (
    <div>
      <div className="replayer">
        <div>
          <div className="felt-wrap" role="img" aria-label={`Table for hand ${hand.hand_id}`}>
            <div className="felt" />
            <div className="felt-center">
              <div className="row" style={{ gap: 3 }}>
                {[0, 1, 2, 3, 4].map((i) => <Card key={i} big card={i < st.boardN ? hand.board[i] : undefined} />)}
              </div>
              <span className="pot num">Pot {st.pot}</span>
            </div>
            {hand.seats.map((s) => {
              const ss = st.seats.get(s.player_id)!;
              const [x, y] = SEAT_POS[s.seat_no];
              const cls = ["seat", ss.folded ? "folded" : "", st.actor === s.player_id ? "acting" : "",
                pair?.[0] === s.player_id ? "pair-a" : "", pair?.[1] === s.player_id ? "pair-b" : ""].join(" ");
              return (
                <div key={s.player_id} className={cls} style={{ left: `${x}%`, top: `${y}%` }}>
                  {s.seat_no === hand.button_seat && <span className="marker" title="Dealer button">D</span>}
                  <div className="who">{pair?.[0] === s.player_id ? "A · " : pair?.[1] === s.player_id ? "B · " : ""}
                    {s.name || shortId(s.player_id)}</div>
                  <div><Card card={s.cards[0]} /><Card card={s.cards[1]} /></div>
                  <div className="num">{st.done ? (
                    <span className={s.net_chips > 0 ? "badge-ok" : s.net_chips < 0 ? "badge-bad" : ""}>
                      {s.net_chips > 0 ? "+" : ""}{s.net_chips}
                    </span>) : <>stack {ss.stack}</>}</div>
                  <div className="last">{ss.last}</div>
                </div>
              );
            })}
          </div>
          <div className="controls-bar" role="toolbar" aria-label="Replay controls">
            <button className="btn" onClick={() => go(0)} aria-label="First action" disabled={step === 0}>⏮</button>
            <button className="btn" onClick={() => go(step - 1)} aria-label="Previous action" disabled={step === 0}>◀</button>
            <button className="btn primary" onClick={() => (step >= n ? (setStep(0), setPlaying(true)) : setPlaying(!playing))}>
              {playing ? "Pause" : "Play"}
            </button>
            <button className="btn" onClick={() => go(step + 1)} aria-label="Next action" disabled={step >= n}>▶</button>
            <button className="btn" onClick={() => go(n)} aria-label="Last action" disabled={step >= n}>⏭</button>
            <button className="btn" onClick={() => setSpeed(speed === 1 ? 2 : 1)} title="Playback speed">{speed}×</button>
            <span className="muted small num">action {step}/{n} · keys ← →</span>
          </div>
          <p className="small muted">
            {hand.hand_id} · {hand.table_id} · blinds {hand.small_blind}/{hand.big_blind} · all cards face-up (investigator
            view){pair && <> · <span style={{ color: "var(--seat-a)" }}>■</span> A = {name(pair[0])},{" "}
              <span style={{ color: "var(--seat-b)" }}>■</span> B = {name(pair[1])}</>}
          </p>
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
                  <strong>{name(a.player_id)}</strong> {describe(a)}
                  <span className="muted num"> · pot {a.pot_before}</span>
                  {note && <span className="note">⚑ {note}</span>}
                </li>,
              ];
            })}
            {n === 0 && <li className="future">No actions recorded for this hand.</li>}
          </ol>
        </div>
      </div>
    </div>
  );
}
