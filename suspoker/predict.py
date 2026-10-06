"""S7: score the evaluation pairs with the saved models and write the Kaggle submission.

    python -m suspoker.predict --submission     # -> artifacts/submission.csv (+ artifacts/predictions.parquet)

Uses only `models/` (boosters, M3, meta.json) and the evaluation-period feature tables from S5. Every
evaluation feature and evidence candidate comes from evaluation-period hands that contain both players.
pair_id and hand_id are written as the original strings, in the column order of sample_submission.csv.
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from suspoker.config import ARTIFACTS_DIR, MODELS_DIR, RAW_DIR
from suspoker.metric import EVIDENCE_COLUMNS, NO_EVIDENCE, SUBMISSION_COLUMNS
from suspoker.modeling import KEYS, cross_features, hand_context, keyed, matrix, with_exposure
from suspoker.pipeline import FEATURES_DIR
from suspoker.train import FAMILY_PROBS, assign_behavior, calibrate, novelty_percentile, top5

EVIDENCE_CHUNKS = 8  # evaluation hands are scored in pair chunks to bound memory (3.6M candidate hands)


@dataclass
class Models:
    meta: dict
    m1: list[lgb.Booster]
    m2: list[lgb.Booster]
    m3: object
    m4: list[lgb.Booster]

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "Models":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))

        def boosters(name: str) -> list[lgb.Booster]:
            return [lgb.Booster(model_file=str(directory / f"{name}_s{s}.txt")) for s in meta["seeds"]]

        with open(directory / "m3_novelty.pkl", "rb") as fh:
            m3 = pickle.load(fh)  # noqa: S301 - our own artifact
        models = cls(meta, boosters("m1_risk"), boosters("m2_behavior"), m3, boosters("m4_evidence"))
        for b in models.m1 + models.m2:
            assert b.feature_name() == meta["pair_features"], "pair feature order differs from meta.json"
        for b in models.m4:
            assert b.feature_name() == meta["hand_features"], "hand feature order differs from meta.json"
        return models


def average(boosters: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    return np.mean([b.predict(X) for b in boosters], axis=0)


def score_pairs(models: Models, pairs: pl.DataFrame) -> pl.DataFrame:
    """Risk (raw + calibrated), family probabilities, novelty and predicted behavior for each pair row."""
    meta = models.meta
    X = matrix(pairs, meta["pair_features"])
    risk_raw = average(models.m1, X)
    probs = average(models.m2, X)
    novelty = novelty_percentile(models.m3, pairs)
    behavior = assign_behavior(risk_raw, probs, novelty, meta["behavior_policy"], meta["none_below_raw"])
    return pairs.select("pair_id", *KEYS).with_columns(
        pl.Series("risk_raw", risk_raw), pl.Series("risk", calibrate(risk_raw, meta["calibration"])),
        *[pl.Series(n, probs[:, i]) for i, n in enumerate(FAMILY_PROBS)],
        pl.Series("novelty_pct", novelty), pl.Series("predicted_behavior", behavior))


def score_evidence(models: Models, hands: pl.DataFrame, scored_pairs: pl.DataFrame) -> pl.DataFrame:
    """Top-5 evidence hands per pair (wide), from candidate hands with M4."""
    feats = models.meta["hand_features"]
    pairs = scored_pairs.select("pair_id", *KEYS, *FAMILY_PROBS)
    ids = pairs["pair_id"].unique().sort()
    out = []
    for chunk in np.array_split(ids.to_numpy(), EVIDENCE_CHUNKS):
        p = pairs.filter(pl.col("pair_id").is_in(chunk.tolist()))
        h = hand_context(hands.join(p, on=KEYS, how="inner").sort("pair_id", "hand_idx"))
        out.append(top5(h.select("pair_id", "hand_id", pl.Series("score", average(models.m4, matrix(h, feats))))))
    return pl.concat(out)


def build_submission(scored: pl.DataFrame, evidence: pl.DataFrame, order: pl.Series) -> pd.DataFrame:
    sub = (scored.select("pair_id", pl.col("risk").alias("risk_score"), "predicted_behavior")
           .join(evidence, on="pair_id", how="left")
           .with_columns(pl.col(c).fill_null(NO_EVIDENCE) for c in EVIDENCE_COLUMNS))
    sub = pl.DataFrame({"pair_id": order}).join(sub, on="pair_id", how="left", maintain_order="left")
    return sub.select(SUBMISSION_COLUMNS).to_pandas()


def predict_evaluation(models: Models) -> tuple[pd.DataFrame, pl.DataFrame]:
    evals = keyed(pl.read_csv(RAW_DIR / "evaluation_pairs.csv"))
    pf = with_exposure(pl.read_parquet(FEATURES_DIR / "pair_features.parquet").filter(pl.col("phase") == 1))
    pairs = evals.select("pair_id", *KEYS).join(pf, on=KEYS, how="left", maintain_order="left")
    assert pairs.height == evals.height and pairs["n_shared"].null_count() == 0, "missing evaluation features"
    if models.meta.get("cross_period"):
        # background from the pair's latest 2,000 development hands: same exposure as the training-time
        # background (2,000 evaluation hands) and adjacent in time to the evaluation period (DEC-023)
        w = pl.read_parquet(FEATURES_DIR / "pair_features_windows.parquet")
        w = w.filter(pl.col("window_start") == w["window_start"].max())
        pairs = pairs.join(cross_features(w), on=KEYS, how="left", maintain_order="left")
    scored = score_pairs(models, pairs)
    hands = pl.read_parquet(FEATURES_DIR / "hand_features.parquet").filter(pl.col("phase") == 1)
    evidence = score_evidence(models, hands, scored)
    return build_submission(scored, evidence, evals["pair_id"]), scored


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submission", action="store_true", help="score evaluation_pairs.csv and write the submission")
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "submission.csv")
    args = p.parse_args(argv)
    if not args.submission:
        p.error("nothing to do: pass --submission")
    sub, scored = predict_evaluation(Models.load())
    sub.to_csv(args.out, index=False)
    scored.write_parquet(ARTIFACTS_DIR / "predictions.parquet")
    counts = sub["predicted_behavior"].value_counts().to_dict()
    no_ev = (sub[list(EVIDENCE_COLUMNS)] == NO_EVIDENCE).all(axis=1).sum()
    print(f"wrote {args.out}: {len(sub):,} pairs; behaviors {counts}; pairs without evidence {no_ev:,}")
    print(f"risk: max {sub['risk_score'].max():.4f}, pairs > 0.5: {(sub['risk_score'] > 0.5).sum():,}, "
          f"distinct scores {sub['risk_score'].nunique():,}")


if __name__ == "__main__":
    main()
