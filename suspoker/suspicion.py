"""Hand-suspicion features (DEC-024): find a pair's few planted-looking hands instead of averaging over all.

A colluding pair shares 100-200 hands, but only a handful carry the planted action. Pair features (F1-F8)
average over every shared hand, which dilutes those few. Here a hand-level model H learns what a planted hand
looks like, and each pair gets summaries of its most suspicious hands as extra M1 features:

    H         LightGBM on the h_* hand features. Positives: planted evidence hands of labeled positive pairs.
              Negatives: candidate hands of confirmed-negative pairs. (Positive pairs' other hands are left
              out: some may be unlisted manipulated hands.)
    hs_*      per pair and exposure view: max, mean of top 3 and top 5 hand probabilities, number of hands
              above 0.5, and that number as a share of the pair's candidate hands.

Out of fold: H_f trains on labeled pairs outside fold f and scores the hands of every pair in fold f, so no
pair is scored by a model that saw its table's labels (standard stacking).
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from suspoker.modeling import KEYS, LGB_BINARY, DevData, fit_predict, matrix
from suspoker.pipeline import WINDOW_HANDS

HS_FEATURES = ["hs_max", "hs_top3", "hs_top5", "hs_n50", "hs_frac"]
LGB_HAND = dict(LGB_BINARY, n_estimators=300)


def hand_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("h_")]


def aggregate(scored: pl.DataFrame) -> pl.DataFrame:
    """Per-pair summaries of hand probabilities `p` (rows: player keys + p)."""
    top = pl.col("p").sort(descending=True)
    return scored.group_by(KEYS).agg(
        pl.col("p").max().alias("hs_max"), top.head(3).mean().alias("hs_top3"), top.head(5).mean().alias("hs_top5"),
        (pl.col("p") > 0.5).sum().cast(pl.Float64).alias("hs_n50"), (pl.col("p") > 0.5).mean().alias("hs_frac"))


def labeled_hands(d: DevData, parts: list[Path]) -> pl.DataFrame:
    """Training hands for H: planted hands of positive pairs (y=1) and all hands of negative pairs (y=0)."""
    lab = d.labeled.select("pair_id", *KEYS, "label", "fold")
    planted = d.evidence.select("pair_id", "hand_id", pl.lit(1, pl.Int8).alias("y"))
    h = pl.concat([pl.read_parquet(p).join(lab, on=KEYS, how="inner") for p in parts])
    h = h.join(planted, on=["pair_id", "hand_id"], how="left").with_columns(pl.col("y").fill_null(0))
    return h.filter((pl.col("label") == 0) | (pl.col("y") == 1))


def fit_hand_models(train: pl.DataFrame, cols: list[str], seeds: tuple[int, ...]) -> list[lgb.LGBMClassifier]:
    _, models = fit_predict(LGB_HAND, matrix(train, cols), train["y"].to_numpy(), seeds=seeds, feature_names=cols)
    return models


def predict_hands(models: list, hands: pl.DataFrame, cols: list[str]) -> np.ndarray:
    X = matrix(hands, cols)
    # boosters predict P(planted) directly; going through the sklearn wrapper warns about feature names
    return np.mean([getattr(m, "booster_", m).predict(X) for m in models], axis=0)


def oof_suspicion(d: DevData, parts: list[Path], seeds: tuple[int, ...]) -> dict[str, pl.DataFrame]:
    """hs_* features for every mirror pair: key "full" (whole development period) and "w<start>" per window."""
    train = labeled_hands(d, parts)
    cols = hand_columns(train)
    folds = d.mirror.select(*KEYS, "fold")
    n_folds = int(d.labeled["fold"].max()) + 1
    models = [fit_hand_models(train.filter(pl.col("fold") != f), cols, seeds) for f in range(n_folds)]
    scored = []
    for part in parts:
        h = pl.read_parquet(part).join(folds, on=KEYS, how="inner")
        for f in range(n_folds):
            hf = h.filter(pl.col("fold") == f)
            if hf.height:
                scored.append(hf.select(*KEYS, "pos").with_columns(pl.Series("p", predict_hands(models[f], hf, cols))))
    s = pl.concat(scored)
    out = {"full": aggregate(s)}
    for start in d.windows:
        out[f"w{start}"] = aggregate(s.filter(pl.col("pos").is_between(start, start + WINDOW_HANDS - 1)))
    return out


def attach(d: DevData, hs: dict[str, pl.DataFrame]) -> DevData:
    """Join hs_* onto every view (pairs without candidate hands get nulls)."""
    def j(df: pl.DataFrame, key: str) -> pl.DataFrame:
        return df.join(hs[key], on=KEYS, how="left", maintain_order="left")

    d.labeled = j(d.labeled, "full")
    d.mirror = j(d.mirror, "full")
    d.windows = {w: j(v, f"w{w}") for w, v in d.windows.items()}
    return d
