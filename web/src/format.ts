import type { Behavior } from "./types";

export const FAMILY_LABEL: Record<Behavior, string> = {
  directed_transfer: "Directed transfer",
  soft_play: "Soft play",
  coordinated_isolation: "Coordinated isolation",
  other_coordination: "Other coordination",
  none: "None",
};

export const FAMILY_EMOJI: Record<Behavior, string> = {
  directed_transfer: "💸",
  soft_play: "🤝",
  coordinated_isolation: "🎯",
  other_coordination: "❓",
  none: "😇",
};

export const FAMILY_BLURB: Record<Behavior, string> = {
  directed_transfer: "one player quietly feeds chips to the other",
  soft_play: "partners go easy on each other",
  coordinated_isolation: "partners squeeze everyone else out of the pot",
  other_coordination: "teamwork of an unknown kind",
  none: "no teamwork",
};

export const FAMILY_COLOR: Record<Behavior, string> = {
  directed_transfer: "var(--fam-dt)",
  soft_play: "var(--fam-sp)",
  coordinated_isolation: "var(--fam-ci)",
  other_coordination: "var(--fam-oc)",
  none: "var(--fam-none)",
};

export const DECOY_LABEL: Record<string, string> = {
  tilt: "Decoy: a tilting player",
  passive_player: "Decoy: a pushover",
  similar_styles: "Decoy: look-alike styles",
  seated_together: "Decoy: always sit together",
};

export const STYLE_LABEL: Record<string, string> = {
  TAG: "tight-aggressive",
  LAG: "loose-aggressive",
  nit: "very tight",
  calling_station: "calling station",
  maniac: "maniac",
};

export function styleLabel(style: string): string {
  const [base, extra] = style.split("+");
  const name = STYLE_LABEL[base] ?? base;
  return extra ? `${name} (${extra === "tilt" ? "tilts after losses" : extra})` : name;
}

export const fmt = {
  pct: (x: number, d = 1) => `${(x * 100).toFixed(d)}%`,
  num: (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : "–"),
  int: (x: number) => x.toLocaleString("en-US"),
  risk: (x: number) => (x >= 0.01 ? x.toFixed(3) : x.toExponential(1)),
  signed: (x: number, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`,
  bb: (x: number) => `${x.toFixed(x >= 10 || x <= -10 ? 0 : 1)} bb`,
};

/** "SIM-P0412" -> "P0412" for compact displays */
export const shortId = (id: string) => id.replace(/^SIM-/, "");

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
