import type { ReactNode } from "react";
import { avatar, prettyName } from "../avatars";
import { DECOY_LABEL, FAMILY_COLOR, FAMILY_EMOJI, FAMILY_LABEL, fmt } from "../format";
import { GLOSSARY } from "../glossary";
import type { Behavior, Truth } from "../types";

/** A term with a definition on hover and keyboard focus. */
export function Term({ k, children }: { k: string; children: ReactNode }) {
  const tip = GLOSSARY[k];
  if (!tip) return <>{children}</>;
  return (
    <span className="term" tabIndex={0}>
      {children}
      <span className="tip" role="tooltip">{tip}</span>
    </span>
  );
}

export function FamilyChip({ family, prob }: { family: Behavior | null; prob?: number }) {
  const f = family ?? "none";
  return (
    <span className="chip" style={{ borderColor: `color-mix(in srgb, ${FAMILY_COLOR[f]} 45%, transparent)` }}>
      <span className="dot" style={{ background: FAMILY_COLOR[f] }} aria-hidden />
      {FAMILY_EMOJI[f]} {FAMILY_LABEL[f]}
      {prob !== undefined && <span className="muted num">{fmt.pct(prob, 0)}</span>}
    </span>
  );
}

export function RiskBar({ risk }: { risk: number }) {
  const w = Math.max(risk * 100, 2);
  return (
    <span className="riskbar" title={`suspicion ${risk.toFixed(4)}`}>
      <span className="track"><span className="fill" style={{ width: `${w}%`, backgroundSize: `${(100 / w) * 100}% 100%` }} /></span>
      <span className="num">{fmt.risk(risk)}</span>
    </span>
  );
}

export function Who({ name, size = "sm" }: { name: string; size?: "sm" | "md" | "lg" }) {
  return (
    <span className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
      <span className={`avatar ${size === "md" ? "" : size}`}>{avatar(name)}</span>{prettyName(name)}
    </span>
  );
}

export function TruthLabel({ truth }: { truth: Truth }) {
  if (truth.role === "planted") return <FamilyChip family={truth.family} />;
  if (truth.role === "decoy") return <span className="chip">🎭 {DECOY_LABEL[truth.decoy ?? ""] ?? "Decoy"}</span>;
  return <span className="chip">😇 innocent</span>;
}

/** ✓ / ✗ for a pair given whether it is above the investigation line. */
export function Verdict({ truth, flagged }: { truth: Truth; flagged: boolean }) {
  const planted = truth.role === "planted";
  const right = planted === flagged;
  const text = planted ? (flagged ? "caught" : "got away") : flagged ? "wrongly accused" : "left alone";
  return <span className={right ? "badge-ok" : "badge-bad"}>{right ? "✓" : "✗"} <span className="small">{text}</span></span>;
}

export function Status({ error }: { error?: string }) {
  return error ? <p className="badge-bad">Could not load data: {error}</p> : <p className="muted"><span className="spinner" /> Loading…</p>;
}

export function ProbBars({ probs }: { probs: Record<string, number> }) {
  return (
    <div style={{ display: "grid", gap: 6, minWidth: 220 }}>
      {Object.entries(probs).map(([f, p], i) => (
        <div key={f} className="row small" style={{ gap: 8, flexWrap: "nowrap" }}>
          <span style={{ width: 170 }}>{FAMILY_EMOJI[f as Behavior]} {FAMILY_LABEL[f as Behavior]}</span>
          <span style={{ flex: 1, height: 10, background: "var(--surface-2)", borderRadius: 5, overflow: "hidden" }}>
            <span style={{ display: "block", height: "100%", width: `${p * 100}%`, background: FAMILY_COLOR[f as Behavior], borderRadius: 5,
              transformOrigin: "left", animation: `grow .8s ${i * 0.1}s both` }} />
          </span>
          <span className="num" style={{ width: 40, textAlign: "right" }}>{fmt.pct(p, 0)}</span>
        </div>
      ))}
    </div>
  );
}
