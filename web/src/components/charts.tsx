import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt } from "../format";
import type { ShapWaterfall, Timeline } from "../types";
import { Term } from "./ui";

const AXIS = { stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 12 } };

// ------------------------------------------------------------------ clues: the SHAP contributions, in plain words
export function Clues({ shap }: { shap: ShapWaterfall }) {
  const max = Math.max(...shap.features.map((f) => Math.abs(f.shap)), 1e-9);
  const val = (v: number | null) => (v === null ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
  return (
    <div>
      <div className="row small" style={{ marginBottom: 10 }}>
        <span className="chip"><span className="dot" style={{ background: "var(--up)" }} />points to cheating</span>
        <span className="chip"><span className="dot" style={{ background: "var(--down)" }} />points to innocence</span>
        <span className="muted">longer bar = stronger clue</span>
      </div>
      <div className="wf">
        {shap.features.map((f, i) => (
          <div key={f.feature} style={{ display: "contents" }}>
            <div className="wf-label" style={{ animation: `fadeUp .4s ${i * 0.05}s both` }}>{f.label}
              <div className="muted" style={{ fontSize: "0.74rem" }}>this pair {val(f.value)} · a typical pair {val(f.median)}</div></div>
            <div style={{ height: 12, background: "var(--surface-2)", borderRadius: 6, overflow: "hidden" }}
              title={`${f.label}: ${fmt.signed(f.shap)} (log-odds)`}>
              <div style={{ height: "100%", width: `${(Math.abs(f.shap) / max) * 100}%`, borderRadius: 6, transformOrigin: "left",
                background: f.shap >= 0 ? "var(--up)" : "var(--down)", animation: `grow .7s ${0.1 + i * 0.05}s both` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ timeline: two stacked charts, one shared x-axis
interface TlPoint { i: number; hand_id: string; hand_no: number; flow: number; cum: number; susp: number; ev: boolean; coord: boolean }

function TlTooltip({ active, payload }: { active?: boolean; payload?: { payload: TlPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="card flat small" style={{ padding: "8px 10px" }}>
      <div><strong className="num">{fmt.bb(p.cum)}</strong> <span className="muted">moved A → B so far</span></div>
      <div><strong className="num">{p.susp.toFixed(3)}</strong> <span className="muted">suspicion</span></div>
      <div className="muted">hand #{p.i} together · {p.hand_id}{p.ev ? " · 🔍 evidence" : ""}{p.coord ? " · 🚩 they cheated" : ""}</div>
    </div>
  );
}

export function TimelineCharts({ tl, evidence, coordinated }: { tl: Timeline; evidence: string[]; coordinated: string[] }) {
  const ev = new Set(evidence);
  const co = new Set(coordinated);
  const data: TlPoint[] = tl.hand_ids.map((h, k) => ({ i: k + 1, hand_id: h, hand_no: tl.table_hand_no[k],
    flow: tl.flow_bb[k], cum: tl.cum_flow_bb[k], susp: tl.suspicion[k], ev: ev.has(h), coord: co.has(h) }));
  const evPts = data.filter((d) => d.ev);
  const dot = { r: 6, fill: "var(--brand-2)", stroke: "var(--surface)", strokeWidth: 2 };
  return (
    <div>
      <div className="row small" style={{ marginBottom: 6 }}>
        <span className="chip"><span className="dot" style={{ background: "var(--brand-2)" }} />evidence hand picked by the detector</span>
        {coordinated.length > 0 && <span className="chip"><span className="dot" style={{ background: "var(--up)" }} />hand where they really cheated</span>}
      </div>
      <div className="small"><Term k="cum_flow">Chips moving between them (big blinds)</Term></div>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={data} syncId="tl" margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="i" {...AXIS} hide />
          <YAxis {...AXIS} width={52} tickFormatter={(v: number) => v.toFixed(0)} />
          <Tooltip content={<TlTooltip />} cursor={{ stroke: "var(--axis)" }} />
          <Line type="linear" dataKey="cum" stroke="var(--brand)" strokeWidth={2.5} dot={false} animationDuration={1200} />
          {evPts.map((p) => <ReferenceDot key={p.hand_id} x={p.i} y={p.cum} {...dot} />)}
        </LineChart>
      </ResponsiveContainer>
      <div className="small"><Term k="suspicion">How suspicious their recent hands look (0–1)</Term></div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} syncId="tl" margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="i" {...AXIS} label={{ value: "hands played together", position: "insideBottomRight", offset: -2, fill: "var(--muted)", fontSize: 12 }} />
          <YAxis {...AXIS} width={52} domain={[0, "auto"]} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip content={<TlTooltip />} cursor={{ stroke: "var(--axis)" }} />
          <Line type="linear" dataKey="susp" stroke="var(--brand-3)" strokeWidth={2.5} dot={false} animationDuration={1200} />
          {evPts.map((p) => <ReferenceDot key={p.hand_id} x={p.i} y={p.susp} {...dot} />)}
        </LineChart>
      </ResponsiveContainer>
      {coordinated.length > 0 && (
        <div aria-label="Hands where they cheated" style={{ position: "relative", height: 14, margin: "2px 12px 0 52px", background: "var(--surface-2)", borderRadius: 4 }}>
          {data.filter((d) => d.coord).map((d) => (
            <span key={d.hand_id} title={`they cheated: ${d.hand_id}`} style={{ position: "absolute", top: 1, bottom: 1, width: 3, borderRadius: 2,
              left: `${((d.i - 1) / Math.max(data.length - 1, 1)) * 100}%`, background: "var(--up)" }} />
          ))}
        </div>
      )}
    </div>
  );
}
