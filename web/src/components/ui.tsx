import type { ReactNode } from "react";
import { DECOY_LABEL, FAMILY_COLOR, FAMILY_LABEL, fmt } from "../format";
import { GLOSSARY } from "../glossary";
import type { Behavior, Truth } from "../types";

/** A term with a definition on hover and keyboard focus. */
export function Term({ k, children }: { k: keyof typeof GLOSSARY | string; children: ReactNode }) {
  const tip = GLOSSARY[k];
  if (!tip) return <>{children}</>;
  return (
    <span className="term" tabIndex={0} aria-label={`${typeof children === "string" ? children : ""}: ${tip}`}>
      {children}
      <span className="tip" role="tooltip">{tip}</span>
    </span>
  );
}

export function FamilyChip({ family, prob }: { family: Behavior | null; prob?: number }) {
  const f = family ?? "none";
  return (
    <span className="chip">
      <span className="dot" style={{ background: FAMILY_COLOR[f] }} aria-hidden />
      {FAMILY_LABEL[f]}
      {prob !== undefined && <span className="muted num">{fmt.pct(prob, 0)}</span>}
    </span>
  );
}

export function RiskBar({ risk }: { risk: number }) {
  return (
    <span className="riskbar" title={`risk ${risk.toFixed(4)}`}>
      <span className="track"><span className="fill" style={{ width: `${Math.max(risk * 100, 1)}%` }} /></span>
      <span className="num">{fmt.risk(risk)}</span>
    </span>
  );
}

export function TruthLabel({ truth }: { truth: Truth }) {
  if (truth.role === "planted") return <FamilyChip family={truth.family} />;
  if (truth.role === "decoy") return <span className="chip">{DECOY_LABEL[truth.decoy ?? ""] ?? "Decoy"}</span>;
  return <span className="muted">innocent</span>;
}

/** ✓ / ✗ for a pair given whether it is flagged at the current threshold. */
export function Verdict({ truth, flagged }: { truth: Truth; flagged: boolean }) {
  const planted = truth.role === "planted";
  const right = planted === flagged;
  const text = planted ? (flagged ? "caught" : "missed") : flagged ? "false alarm" : "cleared";
  return (
    <span className={right ? "badge-ok" : "badge-bad"} title={text}>
      {right ? "✓" : "✗"} <span className="small">{text}</span>
    </span>
  );
}

export function Tile({ label, value, sub, term }: { label: string; value: ReactNode; sub?: ReactNode; term?: string }) {
  return (
    <div className="card tile">
      <div className="label">{term ? <Term k={term}>{label}</Term> : label}</div>
      <div className="value num">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Status({ error }: { error?: string }) {
  return error ? <p className="badge-bad">Could not load data: {error}</p> : <p className="muted">Loading…</p>;
}

export function ProbBars({ probs }: { probs: Record<string, number> }) {
  return (
    <div style={{ display: "grid", gap: 4, minWidth: 220 }}>
      {Object.entries(probs).map(([f, p]) => (
        <div key={f} className="row small" style={{ gap: 8, flexWrap: "nowrap" }}>
          <span style={{ width: 150 }}>{FAMILY_LABEL[f as Behavior]}</span>
          <span style={{ flex: 1, height: 8, background: "var(--surface-2)", borderRadius: 4, overflow: "hidden" }}>
            <span style={{ display: "block", height: "100%", width: `${p * 100}%`, background: FAMILY_COLOR[f as Behavior],
              borderRadius: "0 4px 4px 0" }} />
          </span>
          <span className="num" style={{ width: 40, textAlign: "right" }}>{fmt.pct(p, 0)}</span>
        </div>
      ))}
    </div>
  );
}
