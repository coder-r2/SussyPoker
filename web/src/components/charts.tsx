import { useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fmt, sigmoid } from "../format";
import type { PrCurve, ShapWaterfall, Timeline } from "../types";
import { Term } from "./ui";

const AXIS = { stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 12 } };

// ------------------------------------------------------------------ SHAP waterfall (HTML rows, one shared scale)
export function Waterfall({ shap, risk }: { shap: ShapWaterfall; risk: number }) {
  const rows = useMemo(() => {
    let cum = shap.base;
    const out = shap.features.map((f) => {
      const start = cum;
      cum += f.shap;
      return { ...f, start, end: cum };
    });
    const restStart = cum;
    cum += shap.rest;
    return { feats: out, rest: { start: restStart, end: cum } };
  }, [shap]);
  const pts = [shap.base, shap.raw, ...rows.feats.flatMap((r) => [r.start, r.end]), rows.rest.end];
  const lo = Math.min(...pts, 0);
  const hi = Math.max(...pts, 0);
  const pad = (hi - lo) * 0.04;
  const x = (v: number) => ((v - lo + pad) / (hi - lo + 2 * pad)) * 100;
  const bar = (a: number, b: number, color: string, title: string) => (
    <div style={{ position: "relative", height: 16 }} title={title}>
      <div style={{ position: "absolute", left: `${x(0)}%`, top: -4, bottom: -4, width: 1, background: "var(--axis)" }} />
      <div style={{ position: "absolute", left: `${x(Math.min(a, b))}%`, width: `${Math.max(Math.abs(x(b) - x(a)), 0.4)}%`,
        top: 2, height: 12, background: color, borderRadius: 3 }} />
    </div>
  );
  const val = (v: number | null) => (v === null ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
  return (
    <div>
      <div className="legend">
        <span><span className="sw" style={{ background: "var(--up)" }} />pushes risk up</span>
        <span><span className="sw" style={{ background: "var(--down)" }} />pushes risk down</span>
        <span>units: <Term k="shap">log-odds</Term></span>
      </div>
      <div className="wf">
        <div className="muted wf-label">Average pair (base value)</div>
        {bar(lo, shap.base, "var(--axis)", `base ${shap.base.toFixed(2)}`)}
        <div className="num muted">{shap.base.toFixed(2)}</div>
        {rows.feats.map((r) => (
          <FragmentRow key={r.feature} label={r.label}
            sub={`this pair ${val(r.value)} · typical ${val(r.median)}`}
            bar={bar(r.start, r.end, r.shap >= 0 ? "var(--up)" : "var(--down)",
              `${r.label}: ${fmt.signed(r.shap)} (value ${val(r.value)}, typical ${val(r.median)})`)}
            value={fmt.signed(r.shap)} />
        ))}
        <div className="muted wf-label">All other features</div>
        {bar(rows.rest.start, rows.rest.end, rows.rest.end >= rows.rest.start ? "var(--up)" : "var(--down)",
          `other features ${fmt.signed(shap.rest)}`)}
        <div className="num muted">{fmt.signed(shap.rest)}</div>
        <div className="wf-label"><strong>This pair</strong> <span className="muted small">(raw score)</span></div>
        {bar(lo, shap.raw, "var(--ink-2)", `raw ${shap.raw.toFixed(2)}`)}
        <div className="num"><strong>{shap.raw.toFixed(2)}</strong></div>
      </div>
      <p className="small muted" style={{ marginTop: 8 }}>
        A raw score of {shap.raw.toFixed(2)} log-odds is a probability of about {fmt.pct(sigmoid(shap.raw))} before
        calibration; the calibrated <Term k="risk">risk</Term> is {fmt.risk(risk)}. "Typical" is the median over all
        scored SimRoom pairs.
      </p>
    </div>
  );
}

function FragmentRow({ label, sub, bar, value }: { label: string; sub: string; bar: JSX.Element; value: string }) {
  return (
    <>
      <div className="wf-label">{label}<div className="muted" style={{ fontSize: "0.76rem" }}>{sub}</div></div>
      {bar}
      <div className="num">{value}</div>
    </>
  );
}

// ------------------------------------------------------------------ timeline: two small multiples, one x-axis
interface TlPoint { i: number; hand_id: string; hand_no: number; flow: number; cum: number; susp: number; ev: boolean;
  coord: boolean }

function TlTooltip({ active, payload }: { active?: boolean; payload?: { payload: TlPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="viz-tooltip">
      <div><span className="v num">{fmt.bb(p.cum)}</span> <span className="muted">running flow A→B</span></div>
      <div><span className="v num">{p.susp.toFixed(3)}</span> <span className="muted">rolling suspicion</span></div>
      <div className="muted">shared hand #{p.i} · table hand {p.hand_no} · {p.hand_id}</div>
      <div className="muted">this hand: {fmt.bb(p.flow)}{p.ev ? " · evidence hand" : ""}{p.coord ? " · coordinated (truth)" : ""}</div>
    </div>
  );
}

export function TimelineCharts({ tl, evidence, coordinated }: { tl: Timeline; evidence: string[]; coordinated: string[] }) {
  const ev = new Set(evidence);
  const co = new Set(coordinated);
  const data: TlPoint[] = tl.hand_ids.map((h, k) => ({ i: k + 1, hand_id: h, hand_no: tl.table_hand_no[k],
    flow: tl.flow_bb[k], cum: tl.cum_flow_bb[k], susp: tl.suspicion[k], ev: ev.has(h), coord: co.has(h) }));
  const evPts = data.filter((d) => d.ev);
  const dot = { r: 5, fill: "var(--ink)", stroke: "var(--surface)", strokeWidth: 2 };
  return (
    <div>
      <div className="legend">
        <span><span className="dot" style={{ background: "var(--ink)", marginRight: 5 }} />evidence hand chosen by the model</span>
        {coordinated.length > 0 && <span><span className="sw" style={{ background: "var(--up)" }} />hand where they actually coordinated</span>}
      </div>
      <div className="small"><Term k="cum_flow">Running chip flow A → B (bb)</Term></div>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={data} syncId="tl" margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="i" {...AXIS} hide />
          <YAxis {...AXIS} width={52} tickFormatter={(v: number) => v.toFixed(0)} />
          <Tooltip content={<TlTooltip />} cursor={{ stroke: "var(--axis)" }} />
          <Line type="linear" dataKey="cum" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
          {evPts.map((p) => <ReferenceDot key={p.hand_id} x={p.i} y={p.cum} {...dot} />)}
        </LineChart>
      </ResponsiveContainer>
      <div className="small"><Term k="suspicion">Rolling hand suspicion (0–1)</Term></div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} syncId="tl" margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="i" {...AXIS} label={{ value: "shared hand #", position: "insideBottomRight", offset: -2,
            fill: "var(--muted)", fontSize: 12 }} />
          <YAxis {...AXIS} width={52} domain={[0, "auto"]} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip content={<TlTooltip />} cursor={{ stroke: "var(--axis)" }} />
          <Line type="linear" dataKey="susp" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
          {evPts.map((p) => <ReferenceDot key={p.hand_id} x={p.i} y={p.susp} {...dot} />)}
        </LineChart>
      </ResponsiveContainer>
      {coordinated.length > 0 && (
        <div aria-label="Hands where they coordinated" style={{ position: "relative", height: 14, margin: "2px 12px 0 52px",
          background: "var(--surface-2)", borderRadius: 3 }}>
          {data.filter((d) => d.coord).map((d) => (
            <span key={d.hand_id} title={`coordinated: ${d.hand_id}`} style={{ position: "absolute", top: 1, bottom: 1, width: 2,
              left: `${((d.i - 1) / Math.max(data.length - 1, 1)) * 100}%`, background: "var(--up)" }} />
          ))}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ PR curves
export function PrChart({ curves }: { curves: { name: string; curve: PrCurve; color: string }[] }) {
  const data = curves[0].curve.recall.map((r, k) => {
    const row: Record<string, number> = { recall: r };
    curves.forEach((c) => { row[c.name] = c.curve.precision[k]; });
    return row;
  });
  return (
    <div>
      <div className="legend">
        {curves.map((c) => (
          <span key={c.name}><span className="key" style={{ background: c.color }} />{c.name} (base rate {fmt.pct(c.curve.base_rate)})</span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" />
          <XAxis dataKey="recall" type="number" domain={[0, 1]} {...AXIS} tickFormatter={(v: number) => v.toFixed(1)}
            label={{ value: "recall", position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 }} />
          <YAxis domain={[0, 1]} {...AXIS} width={40} tickFormatter={(v: number) => v.toFixed(1)} />
          <Tooltip content={({ active, payload }) => active && payload?.length ? (
            <div className="viz-tooltip">
              <div className="muted">recall {Number(payload[0].payload.recall).toFixed(2)}</div>
              {payload.map((p) => <div key={String(p.dataKey)}><span className="v num">{Number(p.value).toFixed(3)}</span>{" "}
                <span className="muted">precision · {String(p.dataKey)}</span></div>)}
            </div>) : null} cursor={{ stroke: "var(--axis)" }} />
          {curves.map((c) => <Line key={c.name} dataKey={c.name} stroke={c.color} strokeWidth={2} dot={false}
            isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------------ horizontal bars (single series)
export function HBars({ rows, format = (v: number) => v.toFixed(3), color = "var(--accent)" }: {
  rows: { label: string; value: number; sub?: string }[]; format?: (v: number) => string; color?: string }) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div className="wf hb">
      {rows.map((r, k) => (
        <div key={r.label + k} style={{ display: "contents" }} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}>
          <div className="wf-label">{r.label}{r.sub && <div className="muted" style={{ fontSize: "0.74rem" }}>{r.sub}</div>}</div>
          <div style={{ height: 12, background: "var(--surface-2)", borderRadius: 3 }} title={`${r.label}: ${format(r.value)}`}>
            <div style={{ height: "100%", width: `${(r.value / max) * 100}%`, background: color, borderRadius: "0 4px 4px 0",
              opacity: hover === k ? 0.75 : 1 }} />
          </div>
          <div className="num">{format(r.value)}</div>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ confusion matrix (sequential shading)
export function Matrix({ rowsLabel, colsLabel, rows, cols, m }: { rowsLabel: string; colsLabel: string; rows: string[];
  cols: string[]; m: number[][] }) {
  const max = Math.max(...m.flat(), 1);
  return (
    <div className="table-wrap" style={{ width: "fit-content", maxWidth: "100%" }}>
      <table className="data matrix">
        <thead>
          <tr><th>{rowsLabel} ↓ / {colsLabel} →</th>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r}>
              <th>{r}</th>
              {m[i].map((v, j) => {
                const a = v / max;
                return <td key={j} title={`${r} → ${cols[j]}: ${v}`} style={{
                  background: `color-mix(in srgb, var(--accent) ${Math.round(a * 85)}%, var(--surface))`,
                  color: a > 0.55 ? "#fff" : "var(--ink)" }}>{v}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
