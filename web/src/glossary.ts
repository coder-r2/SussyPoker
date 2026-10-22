// Plain-English definitions shown on hover / keyboard focus by <Term k="...">.
export const GLOSSARY: Record<string, string> = {
  risk: "Suspicion score from 0 to 1: how likely the detector thinks it is that these two are working together.",
  rank: "Where this pair lands when every pair is sorted by suspicion. #1 is who an investigator checks first.",
  shared: "Hands both players were dealt into. The detector needs at least 38 before it judges a pair.",
  behavior: "The kind of teamwork the detector thinks it sees: chip dumping, soft play or squeezing others out.",
  truth: "The simulator's answer key: we know which bots we told to cheat, and which innocent bots we made look shady on purpose.",
  threshold: "Pairs above this line get investigated. Slide it to see how many cheaters you'd catch and how many innocent pairs you'd bother.",
  evidence_score: "How strongly the detector thinks this hand shows the pair cheating.",
  coordinated: "Answer key: in this hand the partner bot really changed its play to help the other.",
  cum_flow: "Running total of chips that moved from A to B in hands they played together (negative = from B to A).",
  suspicion: "How planted-looking their recent hands are, averaged over the last 25 they played together.",
  intensity: "How often your partner cheats when it gets the chance.",
  benign: "An honest reason that could explain the same pattern. Worth ruling out before accusing anyone.",
  final: "Kaggle's score blends three things: ranking cheaters first (70%), picking the right evidence hands (20%) and naming the scheme (10%).",
  kaggle: "Scored by Kaggle on hidden data after the competition closed.",
};
