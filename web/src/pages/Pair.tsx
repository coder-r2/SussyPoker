import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { avatar, prettyName } from "../avatars";
import { Clues, TimelineCharts } from "../components/charts";
import { Replayer } from "../components/Replayer";
import { FamilyChip, ProbBars, Status, Term, TruthLabel } from "../components/ui";
import { loadHand, loadPair, useData } from "../data";
import { FAMILY_EMOJI, FAMILY_LABEL, fmt, styleLabel } from "../format";
import type { PairDetail } from "../types";

export function Pair() {
  const { pairId = "" } = useParams();
  const { data: p, error } = useData(() => loadPair(pairId), pairId);
  const [open, setOpen] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  useEffect(() => {
    setRevealed(false);
  }, [pairId]);
  if (!p) return <Status error={error} />;
  const [a, b] = p.players;
  const coord = p.truth.coordinated_hands ?? [];
  const evIds = p.evidence.map((e) => e.hand_id);
  const replayable = new Set(p.replays);
  const handBtn = (h: string) => replayable.has(h)
    ? <button className="btn small" onClick={() => setOpen(h)}>▶ {h}</button> : <span className="muted small">{h}</span>;

  return (
    <div>
      <p className="small"><Link to="/alerts">← All case files</Link></p>
      <div className="card" style={{ background: "var(--grad)", color: "#fff", border: 0 }}>
        <div className="row" style={{ gap: 18 }}>
          <span className="pill" style={{ background: "rgba(255,255,255,.2)", color: "#fff" }}>Case {p.pair_id}</span>
          <span className="small" style={{ opacity: 0.9 }}>Suspect #{p.rank} · {p.shared_hands} hands together · {p.table_id}</span>
        </div>
        <div className="row" style={{ gap: 14, marginTop: 14 }}>
          <Suspect name={a.name} id={a.player_id} tag="A" />
          <span className="display" style={{ fontSize: "1.8rem" }}>🤝?</span>
          <Suspect name={b.name} id={b.player_id} tag="B" />
          <span className="spacer" />
          <div style={{ textAlign: "right" }}>
            <div className="small" style={{ opacity: 0.9 }}><Term k="risk"><span style={{ color: "#fff" }}>Suspicion</span></Term></div>
            <div className="stat-hero">{fmt.risk(p.risk)}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>🧠 The detector's guess</h3>
          <div style={{ marginBottom: 10 }}><FamilyChip family={p.behavior} /></div>
          <ProbBars probs={p.probs} />
        </div>
        <div className="card" style={{ textAlign: "center", display: "grid", placeItems: "center", minHeight: 180 }}>
          {!revealed ? (
            <div>
              <h3 style={{ marginTop: 0 }}>🔒 The truth</h3>
              <p className="muted">Look at the clues first. Then find out if they were really cheating.</p>
              <button className="btn warm big" onClick={() => setRevealed(true)}>Reveal the truth</button>
            </div>
          ) : <TruthReveal p={p} />}
        </div>
      </div>

      <h2>🔎 Clues</h2>
      <div className="card"><Clues shap={p.shap} /></div>

      <h2>📈 Their history together</h2>
      <div className="card">
        <p className="small muted">A = {prettyName(a.name)}, B = {prettyName(b.name)}. Hover to see each hand.</p>
        <TimelineCharts tl={p.timeline} evidence={evIds} coordinated={revealed ? coord : []} />
      </div>

      <h2>🎬 Evidence hands</h2>
      <p className="small muted">The five hands the detector would show an investigator first. Press play to watch one.</p>
      <div className="grid grid-2">
        {p.evidence.map((e, i) => (
          <div key={e.hand_id} className="card" style={{ animation: `fadeUp .4s ${i * 0.08}s both` }}>
            <div className="row">
              <span className="pill">#{e.rank}</span>{handBtn(e.hand_id)}
              <span className="small muted">pot {fmt.bb(e.pot_bb)}</span>
              <span className="spacer" />
              {revealed && p.truth.role === "planted" && (e.coordinated
                ? <span className="badge-ok reveal-card">✓ real cheating</span> : <span className="badge-bad reveal-card">✗ false lead</span>)}
            </div>
            <div style={{ marginTop: 8 }}>{e.tags.map((t) => <span key={t.id} className="tag">{t.text}</span>)}</div>
          </div>
        ))}
      </div>

      <h2>😇 Could it be innocent?</h2>
      {p.benign.length ? (
        <div className="grid grid-2">
          {p.benign.map((x) => (
            <div key={x.id} className="card">
              <h3 style={{ marginTop: 0 }}>{x.title}</h3>
              <p className="small" style={{ marginBottom: 0 }}>{x.text}</p>
            </div>
          ))}
        </div>
      ) : <p className="muted">None of the usual honest explanations fits this pair.</p>}

      {revealed && p.truth.role === "planted" && coord.length > 0 && (
        <>
          <h3>Every hand where they really cheated ({coord.length})</h3>
          <div className="small" style={{ display: "flex", flexWrap: "wrap", gap: "6px 10px" }}>
            {coord.map((h) => <span key={h}>{evIds.includes(h) && "⭐"}{handBtn(h)}</span>)}
          </div>
        </>
      )}

      {open && <HandModal handId={open} pair={p} onClose={() => setOpen(null)} />}
    </div>
  );
}

function Suspect({ name, id, tag }: { name: string; id: string; tag: string }) {
  return (
    <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
      <span className="avatar lg" style={{ background: "rgba(255,255,255,.95)" }}>{avatar(name)}</span>
      <div><div className="display" style={{ fontSize: "1.25rem", fontWeight: 600 }}>{tag} · {prettyName(name)}</div>
        <div className="small" style={{ opacity: 0.85 }}>{id}</div></div>
    </div>
  );
}

function TruthReveal({ p }: { p: PairDetail }) {
  const t = p.truth;
  const [a, b] = p.players;
  const guilty = t.role === "planted";
  const name = (pid?: string | null) => (pid === a.player_id ? prettyName(a.name) : pid === b.player_id ? prettyName(b.name) : "");
  return (
    <div className="reveal-card">
      <div className={`stamp ${guilty ? "caught" : "clear"}`}>{guilty ? "GUILTY" : "INNOCENT"}</div>
      <div style={{ margin: "14px 0 6px" }}><TruthLabel truth={t} /></div>
      {guilty && (
        <p className="small" style={{ marginBottom: 0 }}>
          {t.giver && <>{name(t.giver)} fed chips to {name(t.receiver)}. </>}They cheated in {(t.coordinated_hands ?? []).length} hands,
          about {fmt.pct(t.intensity ?? 0, 0)} of the chances they got. The detector guessed{" "}
          {t.behavior_correct ? <strong className="badge-ok">the scheme right ✓</strong>
            : <strong className="badge-bad">{FAMILY_EMOJI[p.behavior]} {FAMILY_LABEL[p.behavior]} ✗</strong>}.
        </p>
      )}
      {t.role === "decoy" && <p className="small" style={{ marginBottom: 0 }}>A decoy: innocent, but built on purpose to look shady.</p>}
      {t.role === "innocent" && <p className="small" style={{ marginBottom: 0 }}>Nobody told these two to cheat.</p>}
      <p className="small muted" style={{ marginTop: 8, marginBottom: 0 }}>Their styles: {prettyName(a.name)} plays {styleLabel(a.style)}; {prettyName(b.name)} plays {styleLabel(b.style)}.</p>
    </div>
  );
}

function HandModal({ handId, pair, onClose }: { handId: string; pair: PairDetail; onClose: () => void }) {
  const { data, error } = useData(() => loadHand(handId), handId);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-back" onClick={onClose} role="dialog" aria-modal="true" aria-label={`Replay of ${handId}`}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ marginBottom: 8 }}>
          <strong className="display" style={{ fontSize: "1.2rem" }}>🎬 Hand {handId}</strong>
          <Link className="small" to={`/hand/${handId}?pair=${pair.pair_id}`}>open as a page</Link>
          <span className="spacer" />
          <button className="btn small" onClick={onClose} aria-label="Close">Close ✕</button>
        </div>
        {data ? <Replayer hand={data} pair={[pair.players[0].player_id, pair.players[1].player_id]} autoStart
          highlights={pair.highlights[handId] ?? []} /> : <Status error={error} />}
      </div>
    </div>
  );
}
