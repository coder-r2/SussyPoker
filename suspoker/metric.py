"""Competition metric: Pair AP, Evidence MAP@5 and Behavior MAP.

Ported from the official Kaggle metric notebook
(florianderoofr/slash-poker-competition-metric, Apache 2.0). The scoring logic is kept
identical, including tie-breaking: rows are sorted by pair_id and then stably sorted by
descending score, so tied scores are ranked by ascending pair_id. The only addition is
that components are returned separately.

    final = 0.70 * pair_ap + 0.20 * evidence_map + 0.10 * behavior_map
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ALLOWED_BEHAVIORS = frozenset(
    {"none", "directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"}
)
TARGET_BEHAVIORS = ("directed_transfer", "soft_play", "coordinated_isolation")
EVIDENCE_COLUMNS = tuple(f"evidence_hand_{rank}" for rank in range(1, 6))
NO_EVIDENCE = "NO_EVIDENCE"
SUBMISSION_COLUMNS = ("pair_id", "risk_score", "predicted_behavior", *EVIDENCE_COLUMNS)
WEIGHTS = {"pair_ap": 0.70, "evidence_map": 0.20, "behavior_map": 0.10}


class MetricError(ValueError):
    """The submission or solution is malformed."""


@dataclass(frozen=True)
class Score:
    pair_ap: float
    evidence_map: float
    behavior_map: float
    behavior_ap: dict[str, float]

    @property
    def final(self) -> float:
        return (
            WEIGHTS["pair_ap"] * self.pair_ap
            + WEIGHTS["evidence_map"] * self.evidence_map
            + WEIGHTS["behavior_map"] * self.behavior_map
        )

    def as_dict(self) -> dict:
        return {**asdict(self), "final": self.final}


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AP with stable ordering: ties keep their input order (callers sort by pair_id first)."""
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = y_true[order]
    true_positives = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    return float(np.sum((true_positives / ranks) * ranked) / positives)


def clean_evidence(values: list[object]) -> list[str]:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != NO_EVIDENCE:
            cleaned.append(text)
    return cleaned


def average_precision_at_5(relevant: set[str], submitted: list[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, hand_id in enumerate(submitted[:5], start=1):
        if hand_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), 5)


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> Score:
    """Score a submission against a solution with the same layout.

    The solution's `risk_score` holds binary labels, `predicted_behavior` the true family,
    and `evidence_hand_*` the planted evidence hands.
    """
    for name, frame in (("solution", solution), ("submission", submission)):
        missing = set(SUBMISSION_COLUMNS) - set(frame.columns)
        if missing:
            raise MetricError(f"{name} is missing columns: {sorted(missing)}")
        if frame["pair_id"].duplicated().any():
            raise MetricError(f"{name} has duplicate pair_id values")
    if set(solution["pair_id"].astype(str)) != set(submission["pair_id"].astype(str)):
        raise MetricError("pair_id coverage mismatch between solution and submission")

    truth = solution.set_index("pair_id").sort_index()
    predictions = submission.set_index("pair_id").loc[truth.index]

    risk = pd.to_numeric(predictions["risk_score"], errors="coerce")
    if risk.isna().any() or not risk.between(0, 1).all():
        raise MetricError("risk_score must be numeric and between 0 and 1")
    predicted_behavior = predictions["predicted_behavior"].astype(str).to_numpy()
    invalid = set(predicted_behavior) - ALLOWED_BEHAVIORS
    if invalid:
        raise MetricError(f"invalid predicted_behavior values: {sorted(invalid)}")

    submitted_evidence = [
        clean_evidence(list(row))
        for row in predictions.loc[:, EVIDENCE_COLUMNS].itertuples(index=False, name=None)
    ]
    if any(len(ev) != len(set(ev)) for ev in submitted_evidence):
        raise MetricError("evidence hand IDs must not repeat within a pair")

    y_true = pd.to_numeric(truth["risk_score"], errors="raise").to_numpy(dtype=int)
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise MetricError("solution risk_score must be binary labels")
    risk_values = risk.to_numpy(dtype=float)
    pair_ap = average_precision(y_true, risk_values)

    true_behavior = truth["predicted_behavior"].astype(str).to_numpy()
    behavior_ap = {}
    for behavior in TARGET_BEHAVIORS:
        behavior_truth = (true_behavior == behavior).astype(int)
        behavior_risk = np.where(predicted_behavior == behavior, risk_values, 0.0)
        behavior_ap[behavior] = average_precision(behavior_truth, behavior_risk)
    behavior_map = float(np.mean(list(behavior_ap.values())))

    true_evidence = [
        clean_evidence(list(row))
        for row in truth.loc[:, EVIDENCE_COLUMNS].itertuples(index=False, name=None)
    ]
    evidence_scores = [
        average_precision_at_5(set(true_evidence[i]), submitted_evidence[i])
        for i in np.flatnonzero(y_true == 1)
    ]
    evidence_map = float(np.mean(evidence_scores)) if evidence_scores else 0.0

    return Score(pair_ap=pair_ap, evidence_map=evidence_map, behavior_map=behavior_map,
                 behavior_ap=behavior_ap)


def build_solution(labels: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Build a solution frame from development_labels.csv and development_evidence.csv.

    Used to score out-of-fold predictions on labeled development pairs.
    """
    wide = (
        evidence.sort_values(["pair_id", "evidence_rank"])
        .assign(col=lambda d: "evidence_hand_" + d["evidence_rank"].astype(str))
        .pivot(index="pair_id", columns="col", values="hand_id")
    )
    sol = labels[["pair_id", "label", "behavior_family"]].rename(
        columns={"label": "risk_score", "behavior_family": "predicted_behavior"}
    )
    sol = sol.merge(wide, left_on="pair_id", right_index=True, how="left")
    for col in EVIDENCE_COLUMNS:
        if col not in sol.columns:
            sol[col] = NO_EVIDENCE
        sol[col] = sol[col].fillna(NO_EVIDENCE)
    return sol[list(SUBMISSION_COLUMNS)].reset_index(drop=True)
