// Frozen JSON formats shared by the Python exporter (suspoker/export_web.py), the game API (P8) and the
// React app (P7). Change a format here and in the exporter together.
//
// Static files under web/public/data/:
//   summary.json            -> Summary
//   alerts.json             -> AlertsFile
//   pairs/<pair_id>.json    -> PairDetail
//   hands/<hand_id>.json    -> HandReplay   (also served by GET /api/sessions/{id}/hands/{hand_id})
//   model_card.json         -> ModelCard
//
// All hands, players and pairs on the dashboard are simulated (SimRoom, IDs start with "SIM-"). The model card
// holds competition aggregates only.

export type Family = "directed_transfer" | "soft_play" | "coordinated_isolation";
export type Behavior = Family | "none" | "other_coordination";
export type Role = "planted" | "decoy" | "innocent";
export type DecoyType = "tilt" | "passive_player" | "similar_styles" | "seated_together";
export type Street = "preflop" | "flop" | "turn" | "river";
export type ActionName = "fold" | "check" | "call" | "bet" | "raise" | "all_in";
export type FamilyProbs = Record<Family, number>;

// ------------------------------------------------------------------ metrics (SimRoom vs planted ground truth)
export interface MetricScores {
  final: number;
  pair_ap: number;
  evidence_map: number;
  behavior_map: number;
  behavior_ap: Record<Family, number>;
}

export interface SimRoomMetrics {
  population: string;
  pairs_scored: number;
  planted_scored: number;
  planted_total: number;
  decoys_scored: number;
  base_rate: number;
  models: { M_full: MetricScores; B0_shared_hands: MetricScores; B1_net_flow: MetricScores };
  break_even: { k: number; tp: number; fp: number; precision: number };
  recall_by_family: { family: Family; n: number; caught: number; median_rank: number; behavior_correct: number }[];
  decoys: { decoy: DecoyType; n: number; best_rank: number; median_rank: number; in_top_k: number }[];
  detection_by_coordinated_hands: { coordinated: "0-2" | "3-9" | "10-19" | "20+"; n: number; caught: number;
    median_rank: number }[];
  families: Family[];
}

export interface CalibrationRow {
  stat: string;
  competition: string; // "p5 / p50 / p95" for player stats and pot size, a single share for hand stats
  simroom: string;
  ok: boolean;
}

// ------------------------------------------------------------------ summary.json
export interface KaggleSubmission {
  n: number;
  commit: string;
  change: string;
  local_window_final: number;
  public: number;
  private: number;
  deployed?: boolean;
}

export interface Summary {
  generated: string;
  code_version: string;
  banner: string;
  citation: string;
  simroom: { tables: number; hands_per_table: number; bots_per_table: number; hands: number; pairs_total: number;
    stakes: string[] };
  metrics: SimRoomMetrics;
  default_threshold: number; // risk of the K-th pair, K = number of planted pairs scored (break-even cut)
  calibration: { ok: boolean; rows: CalibrationRow[] };
  kaggle: { best_private: number; deployed: KaggleSubmission };
  families: Family[];
  detail_pairs: number;
  replays: number;
}

// ------------------------------------------------------------------ alerts.json
export interface PlayerRef {
  player_id: string;
  name: string;
}

export interface Truth {
  role: Role;
  family: Family | null;   // planted pairs
  decoy: DecoyType | null; // decoy pairs
}

export interface Alert {
  rank: number; // 1 = highest risk
  pair_id: string;
  table_id: string;
  players: PlayerRef[]; // always 2: [A, B], A lexically first
  shared_hands: number;
  risk: number;
  behavior: Behavior;
  probs: FamilyProbs;
  truth: Truth;
  detail: boolean; // a pairs/<pair_id>.json page exists
}

export interface AlertsFile {
  threshold: number;
  alerts: Alert[]; // sorted by rank
}

// ------------------------------------------------------------------ pairs/<pair_id>.json
export interface ShapFeature {
  feature: string;
  family: string;
  label: string;  // plain English
  value: number | null;
  median: number; // population median over all scored SimRoom pairs
  shap: number;   // log-odds contribution
}

export interface ShapWaterfall {
  units: "log-odds";
  base: number; // model's average raw score
  raw: number;  // base + all contributions = this pair's raw score
  rest: number; // sum of the contributions not listed
  features: ShapFeature[]; // top 10 by |shap|
}

export interface Timeline {
  a: string; // player_id of player A (lexically first); positive flow = chips from A to B
  b: string;
  hand_ids: string[]; // every shared hand, in time order
  table_hand_no: number[];
  flow_bb: number[];
  cum_flow_bb: number[];
  suspicion: number[]; // rolling mean (25 shared hands) of the hand-suspicion model's probability
}

export interface ReasonTag {
  id: string;
  text: string;
}

export interface EvidenceHand {
  rank: number;
  hand_id: string;
  score: number; // evidence ranker (M4)
  pot_bb: number;
  tags: ReasonTag[];
  coordinated: boolean; // ground truth: the overlay changed a decision in this hand
}

export interface BenignAlternative {
  id: string;
  title: string;
  feature: string;
  value: number;
  percentile: number;
  strength: number; // 90-100: how extreme the trigger is
  text: string;
}

export interface Highlight {
  action_no: number;
  text: string;
}

export interface PairTruth extends Truth {
  intensity?: number;
  windows?: number[][]; // each [start, end): table hand numbers
  giver?: string | null;        // directed_transfer only
  receiver?: string | null;
  coordinated_hands?: string[];
  behavior_correct?: boolean;
}

export interface PairDetail {
  pair_id: string;
  table_id: string;
  rank: number;
  shared_hands: number;
  players: (PlayerRef & { style: string })[]; // always 2: [A, B]
  risk: number;
  behavior: Behavior;
  probs: FamilyProbs;
  novelty_pct: number;
  truth: PairTruth;
  shap: ShapWaterfall;
  timeline: Timeline;
  evidence: EvidenceHand[];
  benign: BenignAlternative[]; // strongest first, at most 3
  replays: string[];           // hand ids with a hands/<hand_id>.json file
  highlights: Record<string, Highlight[]>; // hand_id -> actions where one partner acts on the other
}

// ------------------------------------------------------------------ hands/<hand_id>.json
export interface ReplaySeat {
  seat_no: number; // 0-5
  player_id: string;
  name: string;
  starting_stack: number;
  cards: [string, string]; // e.g. "As", "Td"
  contribution: number;
  net_chips: number;
  folded: boolean;
  showdown: boolean;
}

export interface ReplayAction {
  action_no: number;
  street: Street;
  player_id: string;
  action: ActionName;
  amount: number;    // chips added by this action
  amount_to: number; // player's street commitment after it
  pot_before: number;
  stack_before: number;
  to_call: number;
}

export interface HandReplay {
  hand_id: string;
  table_id: string;
  started_at: string; // ISO 8601, UTC
  small_blind: number;
  big_blind: number;
  button_seat: number; // SB = button + 1, BB = button + 2 (mod 6)
  board: string[];     // only the cards dealt before the hand ended
  final_pot: number;
  seats: ReplaySeat[];
  actions: ReplayAction[];
}

// ------------------------------------------------------------------ model_card.json (aggregates only)
export interface ResultRow {
  model: string;
  note: string;
  view: "labeled" | "mirror" | "window";
  final: number;
  pair_ap: number;
  evidence_map: number;
  behavior_map: number;
}

export interface ImportanceRow {
  feature: string;
  family: string;
  label: string;
  mean_abs_shap: number;
}

export interface PrCurve {
  recall: number[];
  precision: number[];
  base_rate: number;
}

export interface ModelCard {
  generated: string;
  code_version: string;
  model: { pair_features: number; hand_features: number; feature_families: string[]; seeds: number;
    behaviors: Family[]; calibration: { a: number; b: number } };
  data: { source: string; citation: string; use: string };
  validation: { folds: string; views: Record<"labeled" | "mirror" | "window", string> };
  results: ResultRow[];
  repeated_cv: { tag: string; fold_seeds: number[] } & Record<string, { mean: number; sd: number } | unknown>;
  kaggle: { note: string; leaderboard: { teams: number; median: number; p90: number; top: number };
    submissions: KaggleSubmission[] };
  ablation: { note: string; rows: { dropped: string; n_features: number; labeled_ap: number; mirror_ap: number;
    window_ap: number; window_delta: number; mirror_delta: number }[] };
  importance: { population: string; risk: ImportanceRow[];
    risk_by_family: { family: string; mean_abs_shap: number; n_features: number; share: number }[];
    behavior: Record<Family, ImportanceRow[]> };
  errors: {
    operating_point: string;
    counts: Record<string, number>;
    recall_by_family: { behavior_family: Family; positives: number; caught: number; median_rank: number;
      recall: number; missed: number }[];
    decoy_rates: ({ outcome: "TP" | "FP" | "FN" | "TN"; pairs: number } & Record<string, number | string>)[];
    decoy_top_template: { outcome: string; top_template: string; len: number; share: number }[];
    missed_profile: { family: Family; feature: string; caught_median: number; missed_median: number }[];
    evidence_by_family: { behavior_family: Family; pairs: number; evidence_ap5: number; any_planted_in_top5: number;
      planted_per_pair: number; median_candidates: number }[];
    confusion: { view: string; true: string[]; predicted: string[]; matrix: number[][] };
    pr_curves: { labeled: PrCurve; window: PrCurve };
  };
  simroom: (SimRoomMetrics & { calibration?: CalibrationRow[] }) | null; // section 9
  explanations: { evidence_tags: Record<string, string>;
    benign_templates: Record<string, { title: string; trigger: string; tail: "high" | "low" }> };
}
