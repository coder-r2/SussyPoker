import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { API_BASE, ApiError, api, pairHighlights } from "../api";
import { avatar, prettyName } from "../avatars";
import { Replayer } from "../components/Replayer";
import { Table } from "../components/Table";
import { FamilyChip, Status, Term } from "../components/ui";
import { FAMILY_COLOR, FAMILY_EMOJI, fmt, styleLabel } from "../format";
import type { DetectorState, Direction, GameEvent, GameReport, GameState, HandReplay, Strategy, TableView } from "../types";

const STRATEGIES: { id: Strategy; emoji: string; title: string; partner: string; you: string }[] = [
  { id: "directed_transfer", emoji: "💸", title: "Chip dump", partner: "folds good hands to the receiver and pays off with bad ones",
    you: "move the chips one way without anyone noticing" },
  { id: "soft_play", emoji: "🤝", title: "Soft play", partner: "never bets into you and checks it down with you",
    you: "go easy on your partner, hard on everyone else" },
  { id: "coordinated_isolation", emoji: "🎯", title: "Squeeze play", partner: "re-raises to push the others out when you open",
    you: "open light and let your partner clear the table" },
  { id: "honest", emoji: "😇", title: "Play it straight", partner: "plays normally, no cheating at all",
    you: "see whether an innocent pair ever looks guilty" },
];
const SESSION_KEY = "suspoker-session";
const DELAY = { action: 850, you: 380, new_hand: 900, hand_end: 2300 };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Phase = "checking" | "offline" | "setup" | "playing" | "report";
type Shown = { view: TableView; bubble: { seat: number; text: string } | null; showNet: boolean; key: number };

function viewFromState(s: GameState): TableView | null {
  if (!s.hand) return null;
  return { seats: s.seats.map((x) => ({ ...x })), board: s.hand.board, pot: s.hand.pot, button_seat: s.hand.button_seat,
    to_act: s.to_act, street: s.hand.street };
}

function resultLine(view: TableView): string {
  const best = Math.max(...view.seats.map((s) => s.net ?? 0));
  const w = view.seats.find((s) => (s.net ?? 0) === best && best > 0);
  if (!w) return "Split pot";
  return w.is_you ? `🏆 You win ${best}!` : `🏆 ${prettyName(w.name)} wins ${best}${w.is_partner ? " (your partner)" : ""}`;
}

export function Play() {
  const [phase, setPhase] = useState<Phase>("checking");
  const [state, setState] = useState<GameState | null>(null);
  const [detector, setDetector] = useState<DetectorState | null>(null);
  const [report, setReport] = useState<GameReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [replay, setReplay] = useState<HandReplay | null>(null);
  const [shown, setShown] = useState<Shown | null>(null);
  const [toast, setToast] = useState<{ text: string; key: number } | null>(null);
  const [fast, setFast] = useState(false);
  const skip = useRef(false);
  const fastRef = useRef(false);
  fastRef.current = fast;

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
        setError("Your session expired (30 minutes without play). Start a new one!");
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(null);
    }
  }, []);

  /** Play the server's step-by-step snapshots with a delay per move, then show the final state. */
  const animate = useCallback(async (events: GameEvent[], final: GameState, det: DetectorState | null) => {
    skip.current = false;
    for (let i = 0; i < events.length && !skip.current; i++) {
      const e = events[i];
      setShown({ view: e.table, bubble: e.type === "action" && e.seat !== undefined ? { seat: e.seat, text: e.text ?? "" } : null,
        showNet: e.type === "hand_end", key: i });
      if (e.type === "hand_end") setToast({ text: resultLine(e.table), key: Date.now() });
      const ms = e.type === "action" ? (e.seat === 0 ? DELAY.you : DELAY.action) : DELAY[e.type];
      await sleep(fastRef.current ? ms * 0.35 : ms);
    }
    setShown(null);
    setState(final);
    if (det) setDetector(det);
  }, []);

  const start = (strategy: Strategy, intensity: number, direction: Direction, partnerSeat: number | null) =>
    guard("Shuffling…", async () => {
      const r = await api.create({ strategy, intensity, direction, partner_seat: partnerSeat });
      try { sessionStorage.setItem(SESSION_KEY, r.session_id); } catch { /* ignore */ }
      setState(r.state);
      setDetector(await api.detector(r.session_id));
      setReport(null);
      setPhase("playing");
    });

  const act = (action: string, amount?: number) => state && guard("playing", async () => {
    const r = await api.act(state.session_id, action, amount);
    await animate(r.events ?? [], r.state, r.detector);
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
      {error && <p className="badge-bad" role="alert">⚠️ {error}</p>}
      {phase === "checking" && <p className="muted"><span className="spinner" /> Connecting to the game server…</p>}
      {phase === "offline" && (
        <div className="card" style={{ maxWidth: 680, margin: "40px auto", textAlign: "center" }}>
          <div style={{ fontSize: "3rem" }}>😴</div>
          <h2 style={{ marginTop: 0 }}><span className="spinner" /> Waking up the detector (~60 s)…</h2>
          <p className="small muted" style={{ marginBottom: 0 }}>The game server naps when nobody is playing. This page keeps
            trying every few seconds. Server: <code>{API_BASE}</code></p>
        </div>
      )}
      {phase === "setup" && <Setup onStart={start} busy={!!busy} />}
      {phase === "playing" && state && (
        <PlayingView state={state} shown={shown} detector={detector} busy={busy} fast={fast} setFast={setFast}
          onSkip={() => { skip.current = true; }} onAct={act} onAutoplay={autoplay} onFinish={finish} onOpenHand={openHand} />
      )}
      {phase === "report" && report && state && (
        <ReportView report={report} state={state} onOpenHand={openHand} onRestart={() => {
          try { sessionStorage.removeItem(SESSION_KEY); } catch { /* ignore */ }
          setState(null); setDetector(null); setReport(null); setPhase("setup");
        }} />
      )}
      {toast && <div key={toast.key} className="toast" onAnimationEnd={() => setToast(null)}>{toast.text}</div>}
      {replay && state && (
        <div className="modal-back" onClick={() => setReplay(null)} role="dialog" aria-modal="true">
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="row" style={{ marginBottom: 8 }}>
              <strong className="display" style={{ fontSize: "1.2rem" }}>🎬 Hand {replay.hand_id}</strong><span className="spacer" />
              <button className="btn small" onClick={() => setReplay(null)}>Close ✕</button>
            </div>
            <Replayer hand={replay} pair={pairOf(replay, state)} autoStart highlights={pairHighlights(replay, ...pairOf(replay, state))} />
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
  const [strategy, setStrategy] = useState<Strategy>("directed_transfer");
  const [intensity, setIntensity] = useState(50);
  const [direction, setDirection] = useState<Direction>("partner_to_you");
  const [seat, setSeat] = useState<number | null>(null);
  const often = intensity <= 25 ? "now and then" : intensity <= 60 ? "fairly often" : intensity < 100 ? "almost always" : "every single chance";
  return (
    <div>
      <div className="hero" style={{ gridTemplateColumns: "1fr", padding: "30px 30px" }}>
        <div>
          <span className="pill" style={{ background: "rgba(255,255,255,.2)", color: "#fff" }}>🎮 Beat the Detector</span>
          <h1 style={{ marginTop: 10, marginBottom: 6 }}>Pick your scheme</h1>
          <p className="lead" style={{ marginBottom: 0 }}>Six seats, five bots, one secret partner. After every hand the detector re-ranks
            all 15 pairs at the table. Don't be #1.</p>
        </div>
      </div>
      <div className="grid grid-4" style={{ marginTop: 18 }}>
        {STRATEGIES.map((s) => (
          <button key={s.id} className={`card scheme-pick ${strategy === s.id ? "selected" : ""}`} onClick={() => setStrategy(s.id)}
            style={{ textAlign: "left", font: "inherit", color: "inherit", position: "relative", overflow: "hidden" }} aria-pressed={strategy === s.id}>
            <span className="ribbon" style={{ position: "absolute", inset: "0 0 auto 0", height: 6,
              background: s.id === "honest" ? "var(--brand-3)" : FAMILY_COLOR[s.id] }} />
            <div style={{ fontSize: "2.2rem" }}>{s.emoji}</div>
            <h3 style={{ margin: "4px 0" }}>{s.title}</h3>
            <p className="small" style={{ margin: 0 }}><span className="muted">Partner:</span> {s.partner}.</p>
            <p className="small" style={{ margin: "4px 0 0" }}><span className="muted">You:</span> {s.you}.</p>
          </button>
        ))}
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <div className="grid grid-3" style={{ alignItems: "end" }}>
          {strategy === "directed_transfer" && (
            <div><div className="small muted" style={{ marginBottom: 6 }}>Which way do the chips flow?</div>
              <div className="seg">
                <button className={direction === "partner_to_you" ? "on" : ""} onClick={() => setDirection("partner_to_you")}>💰 Partner → you</button>
                <button className={direction === "you_to_partner" ? "on" : ""} onClick={() => setDirection("you_to_partner")}>🎁 You → partner</button>
              </div></div>
          )}
          {strategy !== "honest" && (
            <div><div className="small muted" style={{ marginBottom: 6 }}><Term k="intensity">How often your partner cheats</Term>:{" "}
              <strong style={{ color: "var(--ink)" }}>{often}</strong> ({intensity}%)</div>
              <input type="range" min={5} max={100} step={5} value={intensity} style={{ width: "100%" }}
                onChange={(e) => setIntensity(Number(e.target.value))} aria-label="Collusion intensity" /></div>
          )}
          <div><div className="small muted" style={{ marginBottom: 6 }}>Where does your partner sit?</div>
            <div className="seg">
              <button className={seat === null ? "on" : ""} onClick={() => setSeat(null)}>🎲 Random</button>
              {[1, 2, 3, 4, 5].map((s) => <button key={s} className={seat === s ? "on" : ""} onClick={() => setSeat(s)}
                title={s === 1 || s === 5 ? "right next to you" : undefined}>{s}</button>)}
            </div></div>
        </div>
        <div className="row" style={{ marginTop: 18 }}>
          <button className="btn primary big" disabled={busy} onClick={() => onStart(strategy, intensity / 100, direction, seat)}>
            {busy ? "Shuffling…" : "🃏 Sit down"}</button>
          <span className="small muted">Blinds 1/2, 200 chips each. Nothing is saved; a session ends after 500 hands or 30 idle minutes.</span>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ playing
function PlayingView({ state, shown, detector, busy, fast, setFast, onSkip, onAct, onAutoplay, onFinish, onOpenHand }: {
  state: GameState; shown: Shown | null; detector: DetectorState | null; busy: string | null; fast: boolean;
  setFast: (f: boolean) => void; onSkip: () => void; onAct: (a: string, amt?: number) => void;
  onAutoplay: (n: number, m: "colluding" | "honest") => void; onFinish: () => void; onOpenHand: (h: string) => void }) {
  const [mode, setMode] = useState<"colluding" | "honest">(state.strategy === "honest" ? "honest" : "colluding");
  const strat = STRATEGIES.find((s) => s.id === state.strategy)!;
  const view = shown?.view ?? viewFromState(state);
  const animating = shown !== null;
  const labels = { [state.you_seat]: { text: "You", kind: "you" as const }, [state.partner_seat]: { text: "Partner", kind: "partner" as const } };
  return (
    <div>
      <div className="row" style={{ marginBottom: 10 }}>
        <span className="pill hot">{strat.emoji} {strat.title}</span>
        {state.strategy !== "honest" && <span className="chip">cheats {fmt.pct(state.intensity, 0)} of the time</span>}
        {state.direction && <span className="chip">{state.direction === "partner_to_you" ? "💰 partner → you" : "🎁 you → partner"}</span>}
        <span className="chip"><span className="avatar sm">{avatar(state.partner_name)}</span> partner: {prettyName(state.partner_name)}</span>
        <span className="spacer" />
        <span className="small muted num">hand {state.hands_played + 1} / {state.max_hands}</span>
      </div>
      <div className="game">
        <div>
          {view ? <Table view={view} labels={labels} bubble={shown?.bubble ?? null} stepKey={shown?.key ?? `s${state.hands_played}`}
            showNet={shown?.showNet ?? false} /> : <p className="muted">The session is over.</p>}
          <div className="card" style={{ marginTop: 10, padding: 14 }}>
            {animating ? (
              <div className="row">
                <span className="thinking">Bots are playing <i /><i /><i /></span>
                <span className="spacer" />
                <button className="btn small" onClick={() => setFast(!fast)} title="Animation speed">{fast ? "🐇 Fast" : "🐢 Normal"}</button>
                <button className="btn small" onClick={onSkip}>Skip ⏭</button>
              </div>
            ) : state.legal ? <Controls state={state} disabled={!!busy} onAct={onAct} />
              : <p className="muted" style={{ margin: 0 }}>{state.finished ? "That's 500 hands: the session is over." : busy ?? "Waiting…"}</p>}
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <span className="small muted">⚡ Auto-play your seat</span>
            <div className="seg">
              <button className={mode === "colluding" ? "on" : ""} disabled={state.strategy === "honest"} onClick={() => setMode("colluding")}>😈 cheating</button>
              <button className={mode === "honest" ? "on" : ""} onClick={() => setMode("honest")}>😇 honest</button>
            </div>
            <button className="btn small cool" disabled={!!busy || animating || state.finished} onClick={() => onAutoplay(10, mode)}>+10 hands</button>
            <button className="btn small cool" disabled={!!busy || animating || state.finished} onClick={() => onAutoplay(50, mode)}>+50 hands</button>
            <span className="spacer" />
            <button className="btn small" disabled={!!busy || animating} onClick={onFinish}>🏁 End & see verdict</button>
          </div>
          {busy && busy.startsWith("Auto") && <Progress label={busy} />}
          {state.last_hand && !animating && (
            <p className="small muted" style={{ marginTop: 10 }}>Last hand: <button className="linkish" onClick={() => onOpenHand(state.last_hand!.hand_id)}>
              🎬 replay {state.last_hand.hand_id}</button></p>
          )}
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
    <div className="small muted" aria-live="polite" style={{ marginTop: 10 }}>
      {label}
      <div style={{ height: 8, background: "var(--surface-2)", borderRadius: 4, marginTop: 4 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: "var(--grad-cool)", borderRadius: 4, transition: "width .3s" }} />
      </div>
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
  const verb = l.call_amount === 0 && state.hand?.street !== "preflop" ? "Bet" : "Raise to";
  return (
    <div>
      <div className="action-bar">
        <span className="small muted" style={{ marginRight: 4 }}>Your move:</span>
        {l.fold && <button className="btn danger" disabled={disabled} onClick={() => onAct("fold")}>✋ Fold</button>}
        {l.check && <button className="btn call" disabled={disabled} onClick={() => onAct("check")}>👌 Check</button>}
        {l.call && <button className="btn call" disabled={disabled} onClick={() => onAct("call")}>📞 Call {l.call_amount}</button>}
        {r && <button className="btn warm" disabled={disabled} onClick={() => onAct(amount >= r.max_to ? "all_in" : "raise", amount)}>
          {amount >= r.max_to ? `🔥 All-in ${r.max_to}` : `⬆️ ${verb} ${amount}`}</button>}
      </div>
      {r && (
        <div className="row" style={{ marginTop: 10 }}>
          <input type="range" min={r.min_to} max={r.max_to} value={amount} style={{ flex: "1 1 160px" }} aria-label="Bet size"
            onChange={(e) => setAmount(Number(e.target.value))} />
          <button className="btn small" onClick={() => setAmount(preset(0.5))}>½ pot</button>
          <button className="btn small" onClick={() => setAmount(preset(1))}>Pot</button>
          <button className="btn small" onClick={() => setAmount(r.max_to)}>All-in</button>
        </div>
      )}
    </div>
  );
}

function SuspicionMeter({ risk, threshold }: { risk: number; threshold: number }) {
  const pos = (v: number) => Math.min(100, Math.max(0, ((Math.log10(Math.max(v, 1e-4)) + 4) / 4) * 100)); // log scale
  return (
    <div style={{ margin: "10px 0 4px" }}>
      <div className="meter" title={`suspicion ${risk.toFixed(4)}; investigation line ${threshold}`}>
        <span className="mark" style={{ left: `${pos(threshold)}%` }} />
        <span className="needle" style={{ left: `${pos(risk)}%` }} />
      </div>
      <div className="row small muted" style={{ justifyContent: "space-between", marginTop: 4 }}><span>😇 innocent</span><span>🚨 guilty</span></div>
    </div>
  );
}

function DetectorPanel({ detector }: { detector: DetectorState | null }) {
  if (!detector) return <div className="card"><Status /></div>;
  if (detector.status === "insufficient") {
    return (
      <div className="card" style={{ textAlign: "center" }}>
        <div style={{ fontSize: "2.6rem" }} className="big-emoji">👁️</div>
        <h3 style={{ marginTop: 0 }}>The detector is watching…</h3>
        <p className="small muted">It needs {detector.needed} hands before it judges anyone ({detector.hands}/{detector.needed}).</p>
        <div style={{ height: 10, background: "var(--surface-2)", borderRadius: 5 }}>
          <div style={{ height: "100%", width: `${(detector.hands / detector.needed) * 100}%`, background: "var(--grad)", borderRadius: 5, transition: "width .5s" }} />
        </div>
        <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>Tip: auto-play gets you there in seconds.</p>
      </div>
    );
  }
  const you = detector.pairs.find((p) => p.you_and_partner)!;
  const status = you.rank === 1 && you.flagged ? { t: "🚨 Under investigation!", c: "var(--critical)" }
    : you.flagged || you.rank <= 3 ? { t: "👀 Getting noticed…", c: "#d97706" } : { t: "😎 Flying under the radar", c: "var(--good-text)" };
  const medal = you.rank === 1 ? "🥇" : you.rank === 2 ? "🥈" : you.rank === 3 ? "🥉" : "";
  return (
    <div className="card">
      <div className="row"><h3 style={{ margin: 0 }}>🕵️ The detector</h3><span className="spacer" />
        <span className="small muted">after {detector.hands} hands</span></div>
      <div className="display" style={{ fontSize: "1.3rem", fontWeight: 700, color: status.c, margin: "10px 0 2px" }} key={status.t}>{status.t}</div>
      <div className="row" style={{ alignItems: "baseline" }}>
        <span className="stat-hero" style={{ fontSize: "2.6rem" }} key={you.rank}>{medal}#{you.rank}</span>
        <span className="muted">most suspicious of {detector.pairs.length} pairs</span>
      </div>
      <SuspicionMeter risk={you.risk} threshold={detector.threshold} />
      <div className="row small" style={{ marginTop: 8 }}><span className="muted">It thinks you're doing:</span>
        <FamilyChip family={you.behavior} prob={you.probs[you.behavior as keyof typeof you.probs]} /></div>
      {you.drivers.length > 0 && (
        <>
          <div className="small muted" style={{ marginTop: 10 }}>What gives you away</div>
          <ol className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>{you.drivers.map((d) => <li key={d.label}>{d.label}</li>)}</ol>
        </>
      )}
      <h3>🏆 Most suspicious pairs</h3>
      <div className="leader">
        <ol>
          {detector.pairs.slice(0, 8).map((p) => (
            <li key={p.pair_id} className={p.you_and_partner ? "me" : ""}>
              <span className="num" style={{ width: 26, fontWeight: 700 }}>{p.rank === 1 ? "🥇" : p.rank === 2 ? "🥈" : p.rank === 3 ? "🥉" : `#${p.rank}`}</span>
              {p.players.map((x) => <span key={x.seat} className="avatar sm" title={x.is_you ? "You" : prettyName(x.name)}>{x.is_you ? "😎" : avatar(x.name)}</span>)}
              <span className="small" style={{ flex: 1 }}>{p.players.map((x) => (x.is_you ? "You" : prettyName(x.name))).join(" & ")}</span>
              <span className="small num">{fmt.risk(p.risk)}</span>
              {p.flagged && <span title="above the investigation line">🚩</span>}
            </li>
          ))}
        </ol>
        {you.rank > 8 && <p className="small muted" style={{ marginBottom: 0 }}>…you're #{you.rank}. Nicely hidden.</p>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ report
function ReportView({ report, state, onOpenHand, onRestart }: { report: GameReport; state: GameState;
  onOpenHand: (h: string) => void; onRestart: () => void }) {
  const stamp = { caught: "CAUGHT!", suspected: "SUSPECTED", clear: "GOT AWAY", insufficient: "TOO SOON" }[report.verdict];
  const cls = report.verdict === "clear" ? "clear" : report.verdict === "caught" ? "caught" : "suspected";
  const strat = STRATEGIES.find((s) => s.id === state.strategy)!;
  return (
    <div>
      <div className="card" style={{ textAlign: "center", padding: 30 }}>
        <div className={`stamp ${cls}`}>{stamp}</div>
        <h2 style={{ marginTop: 18 }}>{report.headline}</h2>
        <p className="muted" style={{ marginBottom: 0 }}>{report.hands} hands · {strat.emoji} {strat.title}
          {state.strategy !== "honest" && <> · cheating {fmt.pct(state.intensity, 0)} of the time</>}
          {report.you && <> · suspicion {fmt.risk(report.you.risk)} · guessed {FAMILY_EMOJI[report.you.behavior]}</>}</p>
      </div>
      {report.evidence.length > 0 && (
        <>
          <h2>🎬 The evidence against you</h2>
          <div className="grid grid-2">
            {report.evidence.map((e, i) => (
              <div key={e.hand_id} className="card" style={{ animation: `fadeUp .4s ${i * 0.08}s both` }}>
                <div className="row"><span className="pill">#{e.rank}</span>
                  <button className="btn small" onClick={() => onOpenHand(e.hand_id)}>▶ {e.hand_id}</button><span className="spacer" />
                  {state.strategy !== "honest" && (e.coordinated ? <span className="badge-ok small">✓ your partner cheated here</span>
                    : <span className="small muted">no cheating here</span>)}</div>
                <div style={{ marginTop: 8 }}>{e.tags.map((t) => <span key={t.id} className="tag">{t.text}</span>)}</div>
              </div>
            ))}
          </div>
        </>
      )}
      {report.benign_alternative && (
        <>
          <h2>😇 Your best excuse</h2>
          <div className="card"><strong>{report.benign_alternative.title}</strong>
            <p className="small" style={{ marginBottom: 0 }}>{report.benign_alternative.text}</p></div>
        </>
      )}
      <h2>🎭 The table, unmasked</h2>
      <div className="grid grid-3">
        {report.bots.map((b) => (
          <div key={b.seat} className="card flat row" style={{ flexWrap: "nowrap", ...(b.is_partner ? { borderColor: "var(--seat-b)", borderWidth: 2 } : {}) }}>
            <span className="avatar lg">{avatar(b.name)}</span>
            <div><strong>{prettyName(b.name)}</strong>{b.is_partner && <span className="pill hot" style={{ marginLeft: 6 }}>partner</span>}
              <div className="small muted">seat {b.seat} · plays {styleLabel(b.style)}</div></div>
          </div>
        ))}
      </div>
      {state.strategy !== "honest" && <p className="small muted" style={{ marginTop: 12 }}>Your partner bent the rules in {report.coordinated_hands.length} hands.
        Every card is now visible in the replays.</p>}
      <div className="row" style={{ marginTop: 18 }}>
        <button className="btn primary big" onClick={onRestart}>🔁 Play again</button>
        <Link to="/how" className="btn">💡 How does the detector work?</Link>
      </div>
    </div>
  );
}
