import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { SimBanner } from "../components/Layout";
import { FamilyChip, RiskBar, Status, Term, TruthLabel, Verdict } from "../components/ui";
import { loadAlerts, useData } from "../data";
import { FAMILY_LABEL, fmt } from "../format";
import type { Alert, Behavior } from "../types";

const PAGE = 50;
const SLIDER_MIN = -4.5; // threshold slider is logarithmic: most pairs score far below 0.01
type SortKey = "rank" | "shared_hands" | "table_id" | "pair_id";

export function Alerts() {
  const { data, error } = useData(loadAlerts, "alerts");
  const [query, setQuery] = useState("");
  const [behavior, setBehavior] = useState<Behavior | "all">("all");
  const [decoysOnly, setDecoysOnly] = useState(false);
  const [logT, setLogT] = useState<number | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "rank", asc: true });
  const [page, setPage] = useState(0);

  const threshold = logT === null ? data?.threshold ?? 0.5 : 10 ** logT;
  useEffect(() => {
    setPage(0);
  }, [query, behavior, decoysOnly, sort]);

  const readout = useMemo(() => {
    if (!data) return null;
    const flagged = data.alerts.filter((a) => a.risk >= threshold);
    const planted = data.alerts.filter((a) => a.truth.role === "planted").length;
    const tp = flagged.filter((a) => a.truth.role === "planted").length;
    const decoys = flagged.filter((a) => a.truth.role === "decoy").length;
    return { flagged: flagged.length, tp, planted, decoys,
      precision: flagged.length ? tp / flagged.length : 0, recall: planted ? tp / planted : 0 };
  }, [data, threshold]);

  const rows = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    let r = data.alerts.filter((a) =>
      (behavior === "all" || a.behavior === behavior) && (!decoysOnly || a.truth.role === "decoy")
      && (!q || a.pair_id.toLowerCase().includes(q) || a.players.some((p) =>
        p.player_id.toLowerCase().includes(q) || p.name.toLowerCase().includes(q))));
    const dir = sort.asc ? 1 : -1;
    r = [...r].sort((a, b) => {
      const va = a[sort.key], vb = b[sort.key];
      return (va < vb ? -1 : va > vb ? 1 : a.rank - b.rank) * dir;
    });
    return r;
  }, [data, query, behavior, decoysOnly, sort]);

  if (!data || !readout) return <Status error={error} />;
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
      <h1>Alerts</h1>
      <SimBanner />
      <p className="small muted" style={{ maxWidth: 780 }}>
        Every SimRoom pair with at least 38 shared hands, ranked by the Kaggle-trained model's risk. The
        "ground truth" column is the simulator's answer key: planted colluders, innocent decoys built to look
        suspicious, and everyone else. ✓/✗ judges each row at the current threshold.
      </p>

      <div className="controls" role="search">
        <label>Search <input type="search" placeholder="pair, player or bot name" value={query}
          onChange={(e) => setQuery(e.target.value)} /></label>
        <label>Predicted behavior
          <select value={behavior} onChange={(e) => setBehavior(e.target.value as Behavior | "all")}>
            <option value="all">all</option>
            {(["directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"] as Behavior[]).map((b) =>
              <option key={b} value={b}>{FAMILY_LABEL[b]}</option>)}
          </select>
        </label>
        <label><input type="checkbox" checked={decoysOnly} onChange={(e) => setDecoysOnly(e.target.checked)} /> Show decoys only</label>
      </div>
      <div className="controls">
        <label style={{ flex: "1 1 320px" }}>
          <Term k="threshold">Risk threshold</Term>
          <input type="range" min={SLIDER_MIN} max={0} step={0.01} style={{ flex: 1 }}
            value={Math.log10(Math.max(threshold, 10 ** SLIDER_MIN))} onChange={(e) => setLogT(Number(e.target.value))}
            aria-valuetext={threshold.toFixed(4)} />
          <span className="num" style={{ minWidth: 64 }}>{fmt.risk(threshold)}</span>
        </label>
        <button className="btn small" onClick={() => setLogT(null)}>Reset to break-even</button>
      </div>
      <div className="readout" aria-live="polite">
        <span><strong className="num">{readout.flagged}</strong> flagged</span>
        <span><Term k="precision">Precision</Term> <strong className="num">{fmt.pct(readout.precision, 0)}</strong></span>
        <span><Term k="recall">Recall</Term> <strong className="num">{fmt.pct(readout.recall, 0)}</strong>{" "}
          <span className="muted">({readout.tp} of {readout.planted} planted)</span></span>
        <span><strong className="num">{readout.decoys}</strong> decoys flagged</span>
      </div>

      <div className="table-wrap">
        <table className="data alerts">
          <thead>
            <tr>
              {head("rank", "Rank", "rank")}
              {head("pair_id", "Pair")}
              <th>Players</th>
              {head("table_id", "Table")}
              {head("shared_hands", "Shared", "shared")}
              <th><Term k="risk">Risk</Term></th>
              <th><Term k="behavior">Predicted</Term></th>
              <th><Term k="truth">Ground truth</Term></th>
              <th>✓/✗</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((a) => <AlertRow key={a.pair_id} a={a} flagged={a.risk >= threshold} />)}
            {shown.length === 0 && <tr><td colSpan={9} className="muted">No pairs match.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <span className="muted small num">{fmt.int(rows.length)} pairs · page {page + 1} of {pages}</span>
        <button onClick={() => setPage(page - 1)} disabled={page === 0}>Previous</button>
        <button onClick={() => setPage(page + 1)} disabled={page >= pages - 1}>Next</button>
      </div>
    </div>
  );
}

function AlertRow({ a, flagged }: { a: Alert; flagged: boolean }) {
  return (
    <tr>
      <td data-label="Rank" className="num">{a.rank}</td>
      <td data-label="Pair">
        {a.detail ? <Link to={`/pair/${a.pair_id}`}>{a.pair_id}</Link>
          : <span title="No detail page exported for this pair (only the top 100, planted pairs and decoys)">{a.pair_id}</span>}
      </td>
      <td data-label="Players" className="small">{a.players.map((p) => p.name).join(" & ")}</td>
      <td data-label="Table">{a.table_id}</td>
      <td data-label="Shared" className="num">{a.shared_hands}</td>
      <td data-label="Risk"><RiskBar risk={a.risk} /></td>
      <td data-label="Predicted"><FamilyChip family={a.behavior} /></td>
      <td data-label="Truth"><TruthLabel truth={a.truth} />
        {a.truth.role === "planted" && a.truth.family !== a.behavior &&
          <span className="small badge-bad" title="predicted family differs from the planted one"> ≠</span>}</td>
      <td data-label="Verdict"><Verdict truth={a.truth} flagged={flagged} /></td>
    </tr>
  );
}
