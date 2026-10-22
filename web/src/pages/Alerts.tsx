import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { FamilyChip, RiskBar, Status, Term, TruthLabel, Verdict, Who } from "../components/ui";
import { loadAlerts, useData } from "../data";
import { FAMILY_EMOJI, FAMILY_LABEL, fmt } from "../format";
import type { Alert, Behavior } from "../types";

const PAGE = 50;
const SLIDER_MIN = -4.5; // the line is set on a log scale: most pairs score far below 0.01
type SortKey = "rank" | "shared_hands" | "table_id" | "pair_id";

export function Alerts() {
  const { data, error } = useData(loadAlerts, "alerts");
  const [query, setQuery] = useState("");
  const [behavior, setBehavior] = useState<Behavior | "all">("all");
  const [decoysOnly, setDecoysOnly] = useState(false);
  const [answers, setAnswers] = useState(false);
  const [logT, setLogT] = useState<number | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "rank", asc: true });
  const [page, setPage] = useState(0);

  const threshold = logT === null ? data?.threshold ?? 0.5 : 10 ** logT;
  useEffect(() => {
    setPage(0);
  }, [query, behavior, decoysOnly, sort]);

  const score = useMemo(() => {
    if (!data) return null;
    const flagged = data.alerts.filter((a) => a.risk >= threshold);
    const planted = data.alerts.filter((a) => a.truth.role === "planted").length;
    const caught = flagged.filter((a) => a.truth.role === "planted").length;
    return { flagged: flagged.length, caught, innocent: flagged.length - caught, missed: planted - caught, planted };
  }, [data, threshold]);

  const rows = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    const r = data.alerts.filter((a) =>
      (behavior === "all" || a.behavior === behavior) && (!decoysOnly || a.truth.role === "decoy")
      && (!q || a.pair_id.toLowerCase().includes(q) || a.players.some((p) =>
        p.player_id.toLowerCase().includes(q) || p.name.toLowerCase().includes(q))));
    const dir = sort.asc ? 1 : -1;
    return [...r].sort((a, b) => {
      const va = a[sort.key], vb = b[sort.key];
      return (va < vb ? -1 : va > vb ? 1 : a.rank - b.rank) * dir;
    });
  }, [data, query, behavior, decoysOnly, sort]);

  if (!data || !score) return <Status error={error} />;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const shown = rows.slice(page * PAGE, (page + 1) * PAGE);
  const head = (key: SortKey, label: string, term?: string) => (
    <th aria-sort={sort.key === key ? (sort.asc ? "ascending" : "descending") : "none"}>
      <button onClick={() => setSort({ key, asc: sort.key === key ? !sort.asc : key !== "shared_hands" })}>
        {term ? <Term k={term}>{label}</Term> : label}{sort.key === key ? (sort.asc ? " ▲" : " ▼") : ""}
      </button>
    </th>
  );

  return (
    <div>
      <span className="pill hot">🕵️ Case files</span>
      <h1 style={{ marginTop: 10 }}>Who's working together?</h1>
      <p className="muted" style={{ maxWidth: 760 }}>
        Our bots played thousands of hands in a simulated card room, and a few of them were secretly told to cheat. The
        detector ranked all {fmt.int(data.alerts.length)} pairs by suspicion. Open a case to see the clues, replay the hands,
        and reveal the truth.
      </p>

      <div className="card" style={{ marginBottom: 16, background: "var(--surface-2)", boxShadow: "none" }}>
        <h3 style={{ marginTop: 0 }}>🎯 Where would you draw the line?</h3>
        <div className="controls" style={{ marginBottom: 6 }}>
          <label style={{ flex: "1 1 320px" }}><Term k="threshold">Investigate pairs above</Term>
            <input type="range" min={SLIDER_MIN} max={0} step={0.01} style={{ flex: 1 }}
              value={Math.log10(Math.max(threshold, 10 ** SLIDER_MIN))} onChange={(e) => setLogT(Number(e.target.value))}
              aria-valuetext={threshold.toFixed(4)} />
            <span className="num" style={{ minWidth: 64, fontWeight: 700 }}>{fmt.risk(threshold)}</span>
          </label>
          <button className="btn small" onClick={() => setLogT(null)}>Reset</button>
        </div>
        <div className="row" aria-live="polite" style={{ gap: 18 }}>
          <span>You'd investigate <strong className="num">{score.flagged}</strong> pairs:</span>
          <span className="badge-ok">✅ {score.caught} cheaters caught</span>
          <span className="badge-bad">🙅 {score.innocent} innocent pairs bothered</span>
          <span className="muted">🏃 {score.missed} cheaters got away</span>
        </div>
      </div>

      <div className="controls" role="search">
        <input type="search" placeholder="🔎 Find a pair or bot…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search" />
        <select value={behavior} onChange={(e) => setBehavior(e.target.value as Behavior | "all")} aria-label="Scheme">
          <option value="all">All schemes</option>
          {(["directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"] as Behavior[]).map((b) =>
            <option key={b} value={b}>{FAMILY_EMOJI[b]} {FAMILY_LABEL[b]}</option>)}
        </select>
        <label><input type="checkbox" checked={decoysOnly} onChange={(e) => setDecoysOnly(e.target.checked)} /> 🎭 Decoys only</label>
        <span className="spacer" />
        <button className={`btn ${answers ? "warm" : ""}`} onClick={() => setAnswers(!answers)} aria-pressed={answers}>
          {answers ? "🙈 Hide the answers" : "👀 Show the answers"}</button>
      </div>

      <div className="table-wrap">
        <table className="data alerts">
          <thead>
            <tr>
              {head("rank", "#", "rank")}
              <th>Suspects</th>
              {head("shared_hands", "Hands", "shared")}
              <th><Term k="risk">Suspicion</Term></th>
              <th><Term k="behavior">Detector's guess</Term></th>
              <th><Term k="truth">The truth</Term></th>
              <th>Verdict</th>
              <th />
            </tr>
          </thead>
          <tbody key={`${page}-${sort.key}-${sort.asc}`}>
            {shown.map((a, i) => <AlertRow key={a.pair_id} a={a} i={i} flagged={a.risk >= threshold} answers={answers} />)}
            {shown.length === 0 && <tr><td colSpan={8} className="muted">No pairs match.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <span className="muted small num">{fmt.int(rows.length)} pairs · page {page + 1} of {pages}</span>
        <button className="btn small" onClick={() => setPage(page - 1)} disabled={page === 0}>← Previous</button>
        <button className="btn small" onClick={() => setPage(page + 1)} disabled={page >= pages - 1}>Next →</button>
      </div>
    </div>
  );
}

function AlertRow({ a, i, flagged, answers }: { a: Alert; i: number; flagged: boolean; answers: boolean }) {
  const medal = a.rank === 1 ? "🥇" : a.rank === 2 ? "🥈" : a.rank === 3 ? "🥉" : "";
  return (
    <tr style={{ animationDelay: `${Math.min(i, 20) * 0.02}s` }}>
      <td data-label="Rank" className="num"><strong>{medal || a.rank}</strong></td>
      <td data-label="Suspects"><div className="row" style={{ gap: 4 }}><Who name={a.players[0].name} /><span className="muted">&</span>
        <Who name={a.players[1].name} /></div><div className="small muted">{a.pair_id} · {a.table_id}</div></td>
      <td data-label="Hands" className="num">{a.shared_hands}</td>
      <td data-label="Suspicion"><RiskBar risk={a.risk} /></td>
      <td data-label="Guess"><FamilyChip family={a.behavior} /></td>
      <td data-label="Truth">{answers ? <TruthLabel truth={a.truth} /> : <span className="muted">🔒 hidden</span>}</td>
      <td data-label="Verdict">{answers ? <Verdict truth={a.truth} flagged={flagged} /> : <span className="muted">?</span>}</td>
      <td>{a.detail ? <Link className="btn small" to={`/pair/${a.pair_id}`}>Open case →</Link>
        : <span className="small muted" title="Only the top 100, the planted pairs and the decoys have a case file">no file</span>}</td>
    </tr>
  );
}
