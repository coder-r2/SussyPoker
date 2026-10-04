"""Validate a submission file before it is sent to Kaggle (SPEC §5.8).

Errors make the submission invalid or throw away score; warnings are worth a look.

Usage: python -m suspoker.validate_submission artifacts/submission.csv [--no-hand-check]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from suspoker.config import RAW_DIR
from suspoker.metric import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
    SUBMISSION_COLUMNS,
    TARGET_BEHAVIORS,
)


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    message: str


def validate(sub: pd.DataFrame, eval_pairs: pd.DataFrame,
             hands: pl.LazyFrame | None = None, seats: pl.LazyFrame | None = None) -> list[Issue]:
    """Check a submission. Pass `hands` and `seats` to also verify every evidence hand."""
    issues: list[Issue] = []

    def error(msg: str) -> None:
        issues.append(Issue("error", msg))

    if list(sub.columns) != list(SUBMISSION_COLUMNS):
        error(f"columns must be exactly {list(SUBMISSION_COLUMNS)}, got {list(sub.columns)}")
        return issues

    text = sub.drop(columns="risk_score").astype(str).to_numpy()
    if sub.isna().any().any() or (np.char.strip(text.astype("U")) == "").any():
        error("submission has empty cells (use NO_EVIDENCE for unused evidence slots)")
    if sub["pair_id"].duplicated().any():
        error(f"{sub['pair_id'].duplicated().sum()} duplicate pair_id rows")
    expected, got = set(eval_pairs["pair_id"]), set(sub["pair_id"])
    if len(sub) != len(eval_pairs) or expected != got:
        error(f"pair coverage mismatch: {len(expected - got)} missing, {len(got - expected)} extra, "
              f"{len(sub)} rows vs {len(eval_pairs)} expected")

    risk = pd.to_numeric(sub["risk_score"], errors="coerce")
    if risk.isna().any() or not np.isfinite(risk).all() or not risk.between(0, 1).all():
        error("risk_score must be numeric, finite and within [0, 1]")
    elif risk.duplicated().any():
        issues.append(Issue("warning", f"{risk.duplicated().sum()} tied risk scores (ties are broken by pair_id)"))

    behaviors = sub["predicted_behavior"].astype(str)
    if invalid := set(behaviors) - ALLOWED_BEHAVIORS:
        error(f"invalid predicted_behavior values: {sorted(invalid)}")
    if absent := [b for b in TARGET_BEHAVIORS if b not in set(behaviors)]:
        error(f"disclosed families never predicted (their Behavior AP collapses): {absent}")

    evidence = sub[list(EVIDENCE_COLUMNS)].astype(str).to_numpy()
    repeats = np.zeros(len(sub), dtype=bool)
    for i in range(evidence.shape[1]):
        for j in range(i + 1, evidence.shape[1]):
            repeats |= (evidence[:, i] == evidence[:, j]) & (evidence[:, i] != NO_EVIDENCE)
    if repeats.any():
        error(f"{repeats.sum()} rows repeat an evidence hand")

    # Warnings (e.g. tied scores) must not skip the hand check; only errors make it meaningless.
    if hands is not None and seats is not None and not any(i.level == "error" for i in issues):
        issues += _check_evidence_hands(sub, eval_pairs, hands, seats)
    return issues


def _check_evidence_hands(sub: pd.DataFrame, eval_pairs: pd.DataFrame,
                          hands: pl.LazyFrame, seats: pl.LazyFrame) -> list[Issue]:
    ev = (
        pl.from_pandas(sub[["pair_id", *EVIDENCE_COLUMNS]])
        .unpivot(index="pair_id", value_name="hand_id")
        .filter(pl.col("hand_id") != NO_EVIDENCE)
        .join(pl.from_pandas(eval_pairs[["pair_id", "player_1", "player_2"]]), on="pair_id")
    )
    if ev.is_empty():
        return [Issue("warning", "no evidence hands submitted at all")]
    hand_ids = ev.select("hand_id").unique()
    phase = hands.select("hand_id", "phase").join(hand_ids.lazy(), on="hand_id").collect()
    present = seats.select("hand_id", "player_id").join(hand_ids.lazy(), on="hand_id").collect()

    checked = (
        ev.join(phase, on="hand_id", how="left")
        .join(present.rename({"player_id": "player_1"}).with_columns(pl.lit(True).alias("has_1")),
              on=["hand_id", "player_1"], how="left")
        .join(present.rename({"player_id": "player_2"}).with_columns(pl.lit(True).alias("has_2")),
              on=["hand_id", "player_2"], how="left")
    )
    issues = []
    unknown = checked.filter(pl.col("phase").is_null()).height
    wrong_phase = checked.filter(pl.col("phase") == "development").height
    not_shared = checked.filter(pl.col("phase").is_not_null()
                                & (pl.col("has_1").is_null() | pl.col("has_2").is_null())).height
    if unknown:
        issues.append(Issue("error", f"{unknown} evidence hands do not exist"))
    if wrong_phase:
        issues.append(Issue("error", f"{wrong_phase} evidence hands are from the development period"))
    if not_shared:
        issues.append(Issue("error", f"{not_shared} evidence hands do not contain both players of the pair"))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--no-hand-check", action="store_true", help="skip evidence-hand membership checks")
    args = parser.parse_args(argv)

    sub = pd.read_csv(args.submission, dtype={"pair_id": str, "predicted_behavior": str,
                                              **dict.fromkeys(EVIDENCE_COLUMNS, str)},
                      keep_default_na=False)
    eval_pairs = pd.read_csv(RAW_DIR / "evaluation_pairs.csv")
    hands = seats = None
    if not args.no_hand_check:
        hands = pl.scan_parquet(RAW_DIR / "hands.parquet")
        seats = pl.scan_parquet(RAW_DIR / "seats.parquet")

    issues = validate(sub, eval_pairs, hands, seats)
    for issue in issues:
        print(f"[{issue.level.upper()}] {issue.message}")
    n_errors = sum(i.level == "error" for i in issues)
    print("INVALID" if n_errors else "OK", f"({len(sub):,} rows, {n_errors} errors)")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
