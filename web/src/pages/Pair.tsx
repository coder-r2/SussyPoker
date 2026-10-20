import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { TimelineCharts, Waterfall } from "../components/charts";
import { SimBanner } from "../components/Layout";
import { Replayer } from "../components/Replayer";
import { FamilyChip, ProbBars, Status, Term, TruthLabel } from "../components/ui";
import { loadHand, loadPair, useData } from "../data";
import { FAMILY_LABEL, fmt, styleLabel } from "../format";
import type { PairDetail } from "../types";

export function Pair() {
  const { pairId = "" } = useParams();
  const { data: p, error } = useData(() => loadPair(pairId), pairId);
  const [open, setOpen] = useState<string | null>(null);
  if (!p) return <Status error={error} />;
  const [a, b] = p.players;
  const coord = p.truth.coordinated_hands ?? [];
  const evIds = p.evidence.map((e) => e.hand_id);
  const replayable = new Set(p.replays);
  const others = p.replays.filter((h) => !evIds.includes(h) && !coord.includes(h));
  const handLink = (h: string) => replayable.has(h)
    ? <button className="linkish" onClick={() => setOpen(h)}>{h}</button> : <span className="muted">{h}</span>;

  return (
    <div>
      <p className="small"><Link to="/alerts">← All alerts</Link></p>
      <h1>{p.pair_id}</h1>
      <SimBanner text="Simulated pair from the SimRoom, scored by the Kaggle-trained models. The answer key is shown so you can judge the detector." />
      <p className="muted">
        <strong style={{ color: "var(--ink)" }}>A</strong> {a.name} ({a.player_id}) and{" "}
        <strong style={{ color: "var(--ink)" }}>B</strong> {b.name} ({b.player_id}) · {p.table_id} ·{" "}
        <Term k="shared">{p.shared_hands} shared hands</Term> · <Term k="rank">rank {p.rank}</Term>
      </p>

      <div className="grid grid-2">
        <div className="card">
          <div className="row" style={{ alignItems: "flex-start", gap: 24 }}>
            <div>
              <div className="small muted"><Term k="risk">Risk</Term></div>
              <div className="hero num">{fmt.risk(p.risk)}</div>
              <div style={{ marginTop: 6 }}><FamilyChip family={p.behavior} /></div>
              <div className="small muted" style={{ marginTop: 6 }}><Term k="novelty">novelty</Term> {p.novelty_pct.toFixed(0)}th pct</div>
            </div>
            <div style={{ flex: 1 }}>
              <div className="small muted" style={{ marginBottom: 4 }}><Term k="behavior">Family probabilities</Term></div>
              <ProbBars probs={p.probs} />
            </div>
          </div>
        </div>
        <TruthCard p={p} />
      </div>

      <h2>Why flagged</h2>
      <div className="card"><Waterfall shap={p.shap} risk={p.risk} /></div>

      <h2>Timeline</h2>
      <div className="card">
        <p className="small muted">A = {a.name}, B = {b.name}. Hover for each shared hand.</p>
        <TimelineCharts tl={p.timeline} evidence={evIds} coordinated={coord} />
      </div>

      <h2>Evidence hands</h2>
      <p className="small muted">The 5 hands the evidence model would show an investigator first. Click one to replay it.</p>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>#</th><th>Hand</th><th><Term k="evidence_score">Score</Term></th><th>Why this hand</th>
            <th>Pot</th><th><Term k="coordinated">Really coordinated?</Term></th></tr></thead>
          <tbody>
            {p.evidence.map((e) => (
              <tr key={e.hand_id}>
                <td className="num">{e.rank}</td>
                <td>{handLink(e.hand_id)}</td>
                <td className="num">{e.score.toFixed(3)}</td>
                <td>{e.tags.map((t) => <span key={t.id} className="tag">{t.text}</span>)}</td>
                <td className="num">{fmt.bb(e.pot_bb)}</td>
                <td>{p.truth.role !== "planted" ? <span className="muted">n/a (innocent pair)</span>
                  : e.coordinated ? <span className="badge-ok">✓ yes</span> : <span className="badge-bad">✗ no</span>}</td>
              </tr>
            ))}
            {p.evidence.length === 0 && <tr><td colSpan={6} className="muted">No candidate hands.</td></tr>}
          </tbody>
        </table>
      </div>

      {p.truth.role === "planted" && (
        <>
          <h3>Hands where they actually coordinated ({coord.length})</h3>
          <p className="small muted">
            From the simulator's log: hands where the collusion changed at least one of their decisions.
            {" "}{evIds.filter((h) => coord.includes(h)).length} of the 5 evidence hands are among them.
            Replays are exported for the first five.
          </p>
          <div className="small" style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px" }}>
            {coord.map((h) => (
              <span key={h} style={{ whiteSpace: "nowrap" }}>
                {evIds.includes(h) && <span title="also chosen as evidence">★ </span>}{handLink(h)}
              </span>
            ))}
            {coord.length === 0 && <span className="muted">None: the overlay never changed a decision.</span>}
          </div>
        </>
      )}

      <h2>Innocent explanation?</h2>
      {p.benign.length ? (
        <div className="grid grid-2">
          {p.benign.map((x) => (
            <div key={x.id} className="card">
              <h3 style={{ marginTop: 0 }}>{x.title}</h3>
              <p className="small">{x.text}</p>
              <p className="small muted"><Term k="benign">strength</Term> {x.strength.toFixed(0)} / 100</p>
            </div>
          ))}
        </div>
      ) : <p className="muted">No common innocent pattern is extreme for this pair, so the usual honest explanations do not apply.</p>}

      {others.length > 0 && (
        <>
          <h3>Other replays: their biggest chip swings</h3>
          <div className="small" style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px" }}>
            {others.map((h) => <span key={h}>{handLink(h)}</span>)}
          </div>
        </>
      )}

      {open && <HandModal handId={open} pair={p} onClose={() => setOpen(null)} />}
    </div>
  );
}

function TruthCard({ p }: { p: PairDetail }) {
  const t = p.truth;
  const [a, b] = p.players;
  const name = (pid?: string | null) => (pid === a.player_id ? `A (${a.name})` : pid === b.player_id ? `B (${b.name})` : "");
  return (
    <div className="card">
      <div className="small muted"><Term k="truth">Ground truth</Term></div>
      <div style={{ margin: "6px 0 8px" }}><TruthLabel truth={t} /></div>
      {t.role === "planted" && (
        <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
          {t.giver && <li>Chips flow from {name(t.giver)} to {name(t.receiver)}</li>}
          <li><Term k="intensity">Intensity</Term> {fmt.pct(t.intensity ?? 0, 0)}, active in table hands{" "}
            {(t.windows ?? []).map(([s, e]) => `${s}–${e}`).join(", ")}</li>
          <li>{(t.coordinated_hands ?? []).length} hands where they actually coordinated</li>
          <li>Predicted family {t.behavior_correct ? <span className="badge-ok">✓ correct</span>
            : <span className="badge-bad">✗ wrong ({FAMILY_LABEL[p.behavior]})</span>}</li>
        </ul>
      )}
      {t.role === "decoy" && <p className="small">Innocent, but built to look suspicious. A good detector should not flag it.</p>}
      {t.role === "innocent" && <p className="small">Innocent: no collusion was planted for this pair.</p>}
      <p className="small muted" style={{ marginBottom: 0 }}>Styles: A {styleLabel(a.style)}; B {styleLabel(b.style)}.</p>
    </div>
  );
}

function HandModal({ handId, pair, onClose }: { handId: string; pair: PairDetail; onClose: () => void }) {
  const { data, error } = useData(() => loadHand(handId), handId);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-back" onClick={onClose} role="dialog" aria-modal="true" aria-label={`Replay of ${handId}`}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ marginBottom: 8 }}>
          <strong>Hand {handId}</strong>
          <Link className="small" to={`/hand/${handId}?pair=${pair.pair_id}`}>open as a page</Link>
          <span className="spacer" />
          <button className="btn" onClick={onClose} aria-label="Close">Close ✕</button>
        </div>
        {data ? <Replayer hand={data} pair={[pair.players[0].player_id, pair.players[1].player_id]}
          highlights={pair.highlights[handId] ?? []} /> : <Status error={error} />}
      </div>
    </div>
  );
}
