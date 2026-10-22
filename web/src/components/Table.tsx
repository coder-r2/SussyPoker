import { avatar, prettyName } from "../avatars";
import type { TableView } from "../types";

// Seat positions (percent of the table box), seats 0-5 clockwise from the bottom; seat 0 is the viewer / player A.
export const SEAT_POS = [[50, 89], [10, 69], [10, 30], [50, 10], [90, 30], [90, 69]];
const SUIT: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };

export function Card({ card, big }: { card?: string | null; big?: boolean }) {
  if (!card) return <span className={`pcard empty${big ? " big" : ""}`} aria-hidden />;
  if (card === "??") return <span className={`pcard back${big ? " big" : ""}`} aria-label="hidden card" />;
  const suit = card[1];
  return (
    <span className={`pcard${suit === "h" || suit === "d" ? " red" : ""}${big ? " big" : ""}`} aria-label={card}>
      {card[0] === "T" ? "10" : card[0]}{SUIT[suit]}
    </span>
  );
}

const THINKING = "…";

function bubbleKind(text: string): string {
  if (text === THINKING) return "thinking";
  if (text.startsWith("fold")) return "fold";
  if (text.startsWith("all-in")) return "allin";
  if (text.startsWith("raise")) return "raise";
  if (text.startsWith("bet")) return "bet";
  return "";
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** A poker table drawn from a snapshot. `bubble` shows the latest action over a seat ("…" = still thinking);
 * `stepKey` restarts animations. */
export function Table({ view, labels = {}, bubble, stepKey = 0, showNet = false }: {
  view: TableView; labels?: Record<number, { text: string; kind: "you" | "partner" }>; bubble?: { seat: number; text: string } | null;
  stepKey?: number | string; showNet?: boolean }) {
  const best = showNet ? Math.max(...view.seats.map((s) => s.net ?? 0)) : 0;
  return (
    <div className="table-area" role="img" aria-label="Poker table">
      <div className="felt" />
      <div className="felt-logo">SUSPOKER</div>
      <div className="felt-center">
        <div className="row" style={{ gap: 4 }}>
          {[0, 1, 2, 3, 4].map((i) => <Card key={`${i}-${view.board[i] ?? ""}`} big card={view.board[i]} />)}
        </div>
        <span className="pot num" key={`pot-${view.pot}`}><span className="chipstack gold" /> {view.pot}</span>
      </div>
      {view.seats.map((s) => {
        const [x, y] = SEAT_POS[s.seat];
        const label = labels[s.seat];
        const winner = showNet && (s.net ?? 0) > 0 && (s.net ?? 0) === best;
        const cls = ["seat", s.folded ? "folded" : "", view.to_act === s.seat ? "acting" : "", label?.kind ?? "",
          winner ? "winner" : ""].join(" ");
        return (
          <div key={s.seat} className={cls} style={{ left: `${x}%`, top: `${y}%` }}>
            {label && <span className="tagline">{label.text}</span>}
            {s.seat === view.button_seat && <span className="dealer" title="Dealer button">D</span>}
            <div className="head"><span className="avatar sm">{avatar(s.name)}</span>{prettyName(s.name)}</div>
            <div key={`${stepKey}-${(s.cards ?? []).join("")}`}>
              {s.cards ? <><Card card={s.cards[0]} /><Card card={s.cards[1]} /></> : <><Card card="??" /><Card card="??" /></>}
            </div>
            <div className="stack">🪙 {s.stack}</div>
          </div>
        );
      })}
      {view.seats.filter((s) => s.committed > 0).map((s) => {
        const [x, y] = SEAT_POS[s.seat];
        return (
          <span key={`chips-${s.seat}-${s.committed}`} className="bet-chips" style={{ left: `${lerp(x, 50, 0.42)}%`, top: `${lerp(y, 50, 0.42)}%` }}>
            <span className="chipstack" /> {s.committed}
          </span>
        );
      })}
      {bubble && (() => {
        const [x, y] = SEAT_POS[bubble.seat];
        return <span key={`b-${stepKey}`} className={`bubble ${bubbleKind(bubble.text)}`} style={{ left: `${x}%`, top: `${y - 13.5}%` }}>
          {bubble.text === THINKING ? <span className="dots" aria-label="thinking"><i /><i /><i /></span> : bubble.text}</span>;
      })()}
      {showNet && view.seats.filter((s) => (s.net ?? 0) !== 0 && !s.folded || (s.net ?? 0) > 0).map((s) => {
        const [x, y] = SEAT_POS[s.seat];
        const n = s.net ?? 0;
        return <span key={`net-${stepKey}-${s.seat}`} className={`float-net ${n > 0 ? "win" : "loss"}`} style={{ left: `${x}%`, top: `${y}%` }}>
          {n > 0 ? `+${n}` : n}</span>;
      })}
    </div>
  );
}
