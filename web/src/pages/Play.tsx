import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { API_BASE, ApiError, api, pairHighlights } from "../api";
import { Card, Replayer, SEAT_POS } from "../components/Replayer";
import { FamilyChip, RiskBar, Status, Term } from "../components/ui";
import { FAMILY_LABEL, fmt, styleLabel } from "../format";
import type { DetectorState, Direction, GameReport, GameState, HandReplay, Strategy } from "../types";

const STRATEGIES: { id: Strategy; title: string; partner: string; you: string }[] = [
  { id: "soft_play", title: "Soft play", partner: "never bets or raises into you, checks hands down with you, and folds medium hands to your bets",
    you: "go easy on your partner: don't bet into them, check it down when you're heads-up" },
  { id: "directed_transfer", title: "Directed transfer", partner: "folds hands that are ahead to the receiver and pays off with hands that are behind",
    you: "move chips one way: the receiver bets, the giver folds or pays" },
  { id: "coordinated_isolation", title: "Coordinated isolation", partner: "squeezes third players out when you open, folds behind you otherwise, never raises you",
    you: "open light and let your partner clear the pot of everyone else" },
  { id: "honest", title: "Honest (control)", partner: "plays its normal style; there is no collusion",
    you: "just play: see how often an innocent pair looks suspicious" },
];
const SESSION_KEY = "suspoker-session";

type Phase = "checking" | "offline" | "setup" | "playing" | "report";

export function Play() {
  const [phase, setPhase] = useState<Phase>("checking");
  const [state, setState] = useState<GameState | null>(null);
  const [detector, setDetector] = useState<DetectorState | null>(null);
  const [report, setReport] = useState<GameReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [replay, setReplay] = useState<HandReplay | null>(null);

  // cold start: the free server sleeps when idle; poll /health until it answers (SPEC §7.9)
  useEffect(() => {
    let live = true;
    let tries = 0;
    const poll = async () => {
      try {
        await api.health();
        if (!live) return;
        const saved = (() => { try { return sessionStorage.getItem(SESSION_KEY); } catch { return null; } })();
        if (saved) {
          try {
            const r = await api.state(saved);
            setState(r.state);
            setDetector(await api.detector(saved));
            setPhase("playing");
            return;
          } catch { /* expired */ }
        }
        setPhase("setup");
      } catch {
        tries += 1;
        if (!live) return;
        setPhase(tries > 1 ? "offline" : "checking");
        setTimeout(poll, 4000);
      }
    };
    poll();
    return () => { live = false; };
  }, []);

  const guard = useCallback(async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        try { sessionStorage.removeItem(SESSION_KEY); } catch { /* ignore */ }
        setPhase("setup");
        setError("Your session expired (30 minutes without play). Start a new one.");
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(null);
    }
  }, []);

  const start = (strategy: Strategy, intensity: number, direction: Direction, partnerSeat: number | null) =>
    guard("Dealing…", async () => {
      const r = await api.create({ strategy, intensity, direction, partner_seat: partnerSeat });
      try { sessionStorage.setItem(SESSION_KEY, r.session_id); } catch { /* ignore */ }
      setState(r.state);
      setDetector(await api.detector(r.session_id));
      setReport(null);
      setPhase("playing");
    });

  const act = (action: string, amount?: number) => state && guard("", async () => {
    const r = await api.act(state.session_id, action, amount);
    setState(r.state);
    if (r.detector) setDetector(r.detector);
  });

  const autoplay = (hands: number, mode: "colluding" | "honest") => state && guard("Auto-playing…", async () => {
    let left = Math.min(hands, state.max_hands - state.hands_played);
    let s = state;
    while (left > 0) {
      const step = Math.min(10, left);
      setBusy(`Auto-playing… ${hands - left} / ${hands}`);
      const r = await api.autoplay(s.session_id, step, mode);
      s = r.state;
      setState(r.state);
      setDetector(r.detector);
      left -= step;
      if (s.finished) break;
    }
  });

  const finish = () => state && guard("Writing the report…", async () => {
    setReport(await api.report(state.session_id));
    setPhase("report");
  });

  const openHand = (handId: string) => state && guard("", async () => setReplay(await api.hand(state.session_id, handId)));

  return (
    <div>
      <h1>Beat the Detector</h1>
      {error && <p className="badge-bad" role="alert">{error}</p>}
      {phase === "checking" && <p className="muted">Connecting to the game server…</p>}
      {phase === "offline" && (
        <div className="card" style={{ maxWidth: 680 }}>
          <h3 style={{ marginTop: 0 }}><span className="spinner" aria-hidden /> Waking up the detector (~60 s)…</h3>
          <p className="small muted" style={{ marginBottom: 0 }}>The game server sleeps when nobody is playing. This page
            retries every few seconds. Server: <code>{API_BASE}</code></p>
        </div>
      )}
      {phase === "setup" && <Setup onStart={start} busy={!!busy} />}
      {phase === "playing" && state && (
        <PlayingView state={state} detector={detector} busy={busy} onAct={act} onAutoplay={autoplay}
          onFinish={finish} onOpenHand={openHand} />
      )}
      {phase === "report" && report && state && (
        <ReportView report={report} state={state} onOpenHand={openHand} onRestart={() => {
          try { sessionStorage.removeItem(SESSION_KEY); } catch { /* ignore */ }
          setState(null); setDetector(null); setReport(null); setPhase("setup");
        }} />
      )}
      {replay && state && (
        <div className="modal-back" onClick={() => setReplay(null)} role="dialog" aria-modal="true">
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="row" style={{ marginBottom: 8 }}>
              <strong>Hand {replay.hand_id}</strong><span className="spacer" />
              <button className="btn" onClick={() => setReplay(null)}>Close ✕</button>
            </div>
            <Replayer hand={replay} pair={pairOf(replay, state)} highlights={pairHighlights(replay, ...pairOf(replay, state))} />
          </div>
        </div>
      )}
    </div>
  );
}

function pairOf(replay: HandReplay, state: GameState): [string, string] {
  const id = (seat: number) => replay.seats.find((s) => s.seat_no === seat)?.player_id ?? "";
  return [id(state.you_seat), id(state.partner_seat)];
}

// ------------------------------------------------------------------ setup
function Setup({ onStart, busy }: { onStart: (s: Strategy, i: number, d: Direction, p: number | null) => void; busy: boolean }) {
  const [strategy, setStrategy] = useState<Strategy>("soft_play");
  const [intensity, setIntensity] = useState(50);
  const [direction, setDirection] = useState<Direction>("partner_to_you");
  const [seat, setSeat] = useState<string>("random");
  return (
    <div style={{ maxWidth: 820 }}>
      <p>
        You sit at a six-player No-Limit Hold'em table (blinds 1/2, 200-chip stacks) with five bots. One of them is your
        secret partner and follows the scheme you pick. After every hand the detector rescores all 15 pairs at the
        table. Your goal: collude without becoming the pair an investigator looks at first.
      </p>
      <div className="grid grid-2">
        {STRATEGIES.map((s) => (
          <label key={s.id} className="card" style={{ cursor: "pointer", borderColor: strategy === s.id ? "var(--accent)" : undefined }}>
            <div className="row"><input type="radio" name="strategy" checked={strategy === s.id} onChange={() => setStrategy(s.id)} />
              <strong>{s.title}</strong></div>
            <p className="small" style={{ margin: "6px 0 0" }}><span className="muted">Partner:</span> {s.partner}.</p>
            <p className="small" style={{ margin: "2px 0 0" }}><span className="muted">You:</span> {s.you}.</p>
          </label>
        ))}
      </div>
      <div className="controls" style={{ marginTop: 14 }}>
        {strategy === "directed_transfer" && (
          <label>Chips flow
            <select value={direction} onChange={(e) => setDirection(e.target.value as Direction)}>
              <option value="partner_to_you">from your partner to you</option>
              <option value="you_to_partner">from you to your partner</option>
            </select>
          </label>
        )}
        {strategy !== "honest" && (
          <label style={{ flex: "1 1 280px" }}><Term k="intensity">Collusion intensity</Term>
            <input type="range" min={0} max={100} step={5} value={intensity} style={{ flex: 1 }}
              onChange={(e) => setIntensity(Number(e.target.value))} />
            <span className="num">{intensity}%</span>
          </label>
        )}
        <label>Partner's seat
          <select value={seat} onChange={(e) => setSeat(e.target.value)}>
            <option value="random">random</option>
            {[1, 2, 3, 4, 5].map((s) => <option key={s} value={s}>seat {s}{s === 1 || s === 5 ? " (next to you)" : ""}</option>)}
          </select>
        </label>
      </div>
      <button className="btn primary" disabled={busy} onClick={() => onStart(strategy, intensity / 100, direction,
        seat === "random" ? null : Number(seat))}>{busy ? "Dealing…" : "Sit down"}</button>
      <p className="small muted" style={{ marginTop: 12 }}>Everything is simulated; nothing is stored. Sessions end after 500
        hands or 30 minutes without play.</p>
    </div>
  );
}

// ------------------------------------------------------------------ playing
function PlayingView({ state, detector, busy, onAct, onAutoplay, onFinish, onOpenHand }: {
  state: GameState; detector: DetectorState | null; busy: string | null; onAct: (a: string, amt?: number) => void;
  onAutoplay: (n: number, m: "colluding" | "honest") => void; onFinish: () => void; onOpenHand: (h: string) => void }) {
  const [mode, setMode] = useState<"colluding" | "honest">(state.strategy === "honest" ? "honest" : "colluding");
  const strat = STRATEGIES.find((s) => s.id === state.strategy)!;
  return (
    <div>
      <div className="banner small">
        <strong>{strat.title}</strong>{state.strategy !== "honest" && <> at {fmt.pct(state.intensity, 0)} intensity</>}
        {state.direction && <> · chips flow {state.direction === "partner_to_you" ? "partner → you" : "you → partner"}</>}
        {" "}· your partner is <strong>{state.partner_name}</strong> (seat {state.partner_seat}) · hand {state.hands_played + 1} of{" "}
        {state.max_hands}
      </div>
      <div className="game">
        <div>
          <GameTable state={state} />
          {state.legal ? <Controls state={state} disabled={!!busy} onAct={onAct} />
            : <p className="muted">{state.finished ? "The session is over (500 hands)." : busy ?? "Waiting…"}</p>}
          <div className="controls-bar">
            <span className="small muted">Auto-play your seat:</span>
            <select value={mode} onChange={(e) => setMode(e.target.value as "colluding" | "honest")} disabled={!!busy}>
              <option value="colluding" disabled={state.strategy === "honest"}>colluding with your partner</option>
              <option value="honest">honestly</option>
            </select>
            <button className="btn" disabled={!!busy || state.finished} onClick={() => onAutoplay(10, mode)}>+10 hands</button>
            <button className="btn" disabled={!!busy || state.finished} onClick={() => onAutoplay(50, mode)}>+50 hands</button>
            <span className="spacer" />
            <button className="btn" disabled={!!busy} onClick={onFinish}>End session & see report</button>
          </div>
          {busy && busy.startsWith("Auto") && <Progress label={busy} />}
          {state.last_hand && <LastHandView state={state} onOpenHand={onOpenHand} />}
        </div>
        <DetectorPanel detector={detector} />
      </div>
    </div>
  );
}

function Progress({ label }: { label: string }) {
  const m = label.match(/(\d+) \/ (\d+)/);
  const pct = m ? (Number(m[1]) / Number(m[2])) * 100 : 50;
  return (
    <div className="small muted" aria-live="polite">
      {label}
      <div style={{ height: 6, background: "var(--surface-2)", borderRadius: 3, marginTop: 4 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: "var(--accent)", borderRadius: 3, transition: "width .3s" }} />
      </div>
    </div>
  );
}

function GameTable({ state }: { state: GameState }) {
  const h = state.hand;
  return (
    <div className="felt-wrap" role="img" aria-label="Poker table">
      <div className="felt" />
      <div className="felt-center">
        <div className="row" style={{ gap: 3 }}>{[0, 1, 2, 3, 4].map((i) => <Card key={i} big card={h?.board[i]} />)}</div>
        <span className="pot num">Pot {h?.pot ?? 0}</span>
      </div>
      {state.seats.map((s) => {
        const [x, y] = SEAT_POS[s.seat];
        const cls = ["seat", s.folded ? "folded" : "", state.to_act === s.seat ? "acting" : "", s.is_you ? "pair-a" : "",
          s.is_partner ? "pair-b" : ""].join(" ");
        return (
          <div key={s.seat} className={cls} style={{ left: `${x}%`, top: `${y}%` }}>
            {h && s.seat === h.button_seat && <span className="marker" title="Dealer button">D</span>}
            <div className="who">{s.is_you ? "You" : s.name}{s.is_partner && <span className="muted"> · partner</span>}</div>
            <div>{s.cards ? <><Card card={s.cards[0]} /><Card card={s.cards[1]} /></> : <><Card card="??" /><Card card="??" /></>}</div>
            <div className="num">stack {s.stack}{s.committed > 0 && <span className="muted"> · in {s.committed}</span>}</div>
            <div className="last">{s.last_action ?? ""}</div>
          </div>
        );
      })}
    </div>
  );
}

function Controls({ state, disabled, onAct }: { state: GameState; disabled: boolean; onAct: (a: string, amt?: number) => void }) {
  const l = state.legal!;
  const r = l.raise;
  const [amount, setAmount] = useState<number>(r?.min_to ?? 0);
  useEffect(() => { setAmount(r?.min_to ?? 0); }, [r?.min_to, state.hand?.hand_id]);
  const facing = l.committed + l.call_amount; // highest commitment on this street
  const preset = (f: number) => r ? Math.min(r.max_to, Math.max(r.min_to, Math.round(facing + f * (l.pot + l.call_amount)))) : 0;
  const verb = facing === 0 || (state.hand?.street !== "preflop" && l.call_amount === 0) ? "Bet" : "Raise to";
  return (
    <div className="card" style={{ marginTop: 10 }}>
      <div className="row">
        {l.fold && <button className="btn" disabled={disabled} onClick={() => onAct("fold")}>Fold</button>}
        {l.check && <button className="btn" disabled={disabled} onClick={() => onAct("check")}>Check</button>}
        {l.call && <button className="btn" disabled={disabled} onClick={() => onAct("call")}>Call {l.call_amount}</button>}
        {r && <button className="btn primary" disabled={disabled} onClick={() => onAct(amount >= r.max_to ? "all_in" : "raise", amount)}>
          {amount >= r.max_to ? `All-in ${r.max_to}` : `${verb} ${amount}`}</button>}
      </div>
      {r && (
        <div className="row" style={{ marginTop: 8 }}>
          <input type="range" min={r.min_to} max={r.max_to} value={amount} style={{ flex: "1 1 160px" }} aria-label="Bet size"
            onChange={(e) => setAmount(Number(e.target.value))} />
          <button className="btn small" onClick={() => setAmount(preset(0.5))}>½ pot</button>
          <button className="btn small" onClick={() => setAmount(preset(1))}>Pot</button>
          <button className="btn small" onClick={() => setAmount(r.max_to)}>All-in</button>
        </div>
      )}
      {state.hand && state.hand.log.length > 0 && (
        <p className="small muted" style={{ marginBottom: 0, marginTop: 8 }}>
          This hand: {state.hand.log.slice(-6).map((a) => `${a.name === "you" ? "you" : a.name} ${a.action}${a.amount ? " " + a.amount : ""}`).join(" · ")}
        </p>
      )}
    </div>
  );
}

function LastHandView({ state, onOpenHand }: { state: GameState; onOpenHand: (h: string) => void }) {
  const h = state.last_hand!;
  const winners = h.results.filter((r) => r.net > 0);
  const you = h.results.find((r) => r.seat === state.you_seat);
  return (
    <div className="card small" style={{ marginTop: 10 }}>
      <div className="row">
        <strong>Last hand</strong> <button className="linkish" onClick={() => onOpenHand(h.hand_id)}>{h.hand_id}</button>
        <span className="muted">pot {h.pot}</span>
        {h.board.length > 0 && <span>{h.board.map((c) => <Card key={c} card={c} />)}</span>}
      </div>
      <div style={{ marginTop: 4 }}>
        {winners.map((w) => <span key={w.seat}>{w.name === "you" ? "You" : w.name} won {w.net}{w.cards && <> with {w.cards.join(" ")}</>}. </span>)}
        {you && you.net < 0 && <span>You lost {-you.net}.</span>}
      </div>
    </div>
  );
}

function RiskMeter({ risk, threshold }: { risk: number; threshold: number }) {
  // log scale: almost every pair scores far below 0.01
  const pos = (v: number) => (Math.log10(Math.max(v, 1e-4)) + 4) / 4 * 100;
  return (
    <div style={{ position: "relative", height: 14, background: "var(--surface-2)", borderRadius: 7, margin: "6px 0 2px" }}
      title={`risk ${risk.toFixed(4)} (log scale); review threshold ${threshold}`}>
      <div style={{ height: "100%", width: `${pos(risk)}%`, background: risk >= threshold ? "var(--up)" : "var(--accent)", borderRadius: 7 }} />
      <div style={{ position: "absolute", top: -3, bottom: -3, left: `${pos(threshold)}%`, width: 2, background: "var(--ink)" }} />
    </div>
  );
}

function DetectorPanel({ detector }: { detector: DetectorState | null }) {
  if (!detector) return <div className="card"><Status /></div>;
  if (detector.status === "insufficient") {
    return (
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Detector</h3>
        <p>Insufficient data ({detector.hands}/{detector.needed})</p>
        <div style={{ height: 8, background: "var(--surface-2)", borderRadius: 4 }}>
          <div style={{ height: "100%", width: `${(detector.hands / detector.needed) * 100}%`, background: "var(--accent)", borderRadius: 4 }} />
        </div>
        <p className="small muted">Like the competition, the detector gives no verdict on a pair with fewer than {detector.needed} shared hands.
          Auto-play gets you there quickly.</p>
      </div>
    );
  }
  const you = detector.pairs.find((p) => p.you_and_partner)!;
  return (
    <div className="card detector">
      <h3 style={{ marginTop: 0 }}>Detector · after {detector.hands} hands</h3>
      <div className="small muted">You + partner</div>
      <div className="row" style={{ alignItems: "baseline" }}>
        <span className="hero num" style={{ fontSize: "2.2rem" }}>#{you.rank}</span>
        <span className="muted">of {detector.pairs.length} pairs</span>
        <span className="spacer" />
        <span className="num"><Term k="risk">risk</Term> {fmt.risk(you.risk)}</span>
      </div>
      <RiskMeter risk={you.risk} threshold={detector.threshold} />
      <div className="small muted">{you.flagged ? <span className="badge-bad">⚑ above the review threshold</span> : "below the review threshold"}{" "}
        (log scale, threshold {detector.threshold})</div>
      <div style={{ margin: "8px 0" }}><FamilyChip family={you.behavior} prob={you.probs[you.behavior as keyof typeof you.probs]} /></div>
      {you.drivers.length > 0 && (
        <>
          <div className="small muted">Top reasons it suspects you</div>
          <ol className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>{you.drivers.map((d) => <li key={d.label}>{d.label}</li>)}</ol>
        </>
      )}
      <h3>Pair leaderboard</h3>
      <div className="table-wrap">
        <table className="data small">
          <tbody>
            {detector.pairs.map((p) => (
              <tr key={p.pair_id} style={p.you_and_partner ? { outline: "2px solid var(--accent)", outlineOffset: -2 } : undefined}>
                <td className="num">#{p.rank}</td>
                <td>{p.players.map((x) => x.is_you ? "You" : x.name + (x.is_partner ? " (partner)" : "")).join(" & ")}</td>
                <td><RiskBar risk={p.risk} /></td>
                <td>{p.flagged ? <span className="badge-bad" title="above the review threshold">⚑</span> : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ report
function ReportView({ report, state, onOpenHand, onRestart }: { report: GameReport; state: GameState;
  onOpenHand: (h: string) => void; onRestart: () => void }) {
  const icon = { caught: "⚑", suspected: "⚠", clear: "✓", insufficient: "…" }[report.verdict];
  const cls = report.verdict === "clear" ? "badge-ok" : report.verdict === "insufficient" ? "" : "badge-bad";
  return (
    <div>
      <div className="card" style={{ marginBottom: 14 }}>
        <h2 style={{ marginTop: 0 }} className={cls}>{icon} {report.headline}</h2>
        <p className="small muted" style={{ marginBottom: 0 }}>{report.hands} hands played · strategy{" "}
          {STRATEGIES.find((s) => s.id === state.strategy)?.title}{state.strategy !== "honest" && <> at {fmt.pct(state.intensity, 0)} intensity</>}
          {report.you && <> · risk {fmt.risk(report.you.risk)} · predicted {FAMILY_LABEL[report.you.behavior]}</>}</p>
      </div>
      {report.evidence.length > 0 && (
        <>
          <h3>The evidence an investigator would see</h3>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>#</th><th>Hand</th><th><Term k="evidence_score">Score</Term></th><th>Why this hand</th><th>Your partner colluded here?</th></tr></thead>
              <tbody>
                {report.evidence.map((e) => (
                  <tr key={e.hand_id}>
                    <td className="num">{e.rank}</td>
                    <td><button className="linkish" onClick={() => onOpenHand(e.hand_id)}>{e.hand_id}</button></td>
                    <td className="num">{e.score.toFixed(3)}</td>
                    <td>{e.tags.map((t) => <span key={t.id} className="tag">{t.text}</span>)}</td>
                    <td>{state.strategy === "honest" ? <span className="muted">n/a</span> : e.coordinated ? "✓ yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {report.benign_alternative && (
        <>
          <h3>Your best defence (an innocent explanation)</h3>
          <div className="card"><strong>{report.benign_alternative.title}</strong>
            <p className="small" style={{ marginBottom: 0 }}>{report.benign_alternative.text}</p></div>
        </>
      )}
      <h3>The table, revealed</h3>
      <ul className="small">
        {report.bots.map((b) => <li key={b.seat}>Seat {b.seat}: {b.name}, {styleLabel(b.style)}{b.is_partner && <strong> (your partner)</strong>}</li>)}
      </ul>
      {state.strategy !== "honest" && <p className="small">Your partner's collusion changed its decisions in {report.coordinated_hands.length} hands.
        All cards are now visible in the replays.</p>}
      <div className="row"><button className="btn primary" onClick={onRestart}>Play again</button>
        <Link to="/model" className="small">How the detector works →</Link></div>
    </div>
  );
}
