"""Shared modelling plumbing for P3: datasets, folds, feature lists and the LightGBM helper.

Two validation sets are built from the development period:

* **labeled**: the 1,860 labeled pairs (372 positives, 20% base rate). It is the set the
  labels describe, but its base rate is far higher than the evaluation set's.
* **mirror**: the labeled pairs plus every unlabeled development pair that the organizer's
  evaluation filter would keep (no labeled positive player, enough shared hands). Unlabeled
  pairs count as negatives here. Some are hidden positives, so mirror scores are a pessimistic,
  realistic-prevalence estimate of the leaderboard.

Folds are `StratifiedGroupKFold` over labeled pairs (group = table, stratify = behavior family).
Every pair inherits the fold of its table, so no table is ever on both sides of a split.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from suspoker.config import RAW_DIR, load_config
from suspoker.features import CATALOG

PERIOD_HANDS = {0: 3000, 1: 2000}  # table hands per period (exactly 60% / 40% of 5,000)
EVAL_MIN_SHARED = 38  # smallest shared_hands in evaluation_pairs.csv
EXPOSURE_COUNTS = ("n_shared", "hands_max", "hands_min")
PAIR_FEATURES = [c for c in CATALOG if c not in EXPOSURE_COUNTS] + [
    "n_shared_frac", "hands_max_frac", "hands_min_frac"]
FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")
KEYS = ["player_lo", "player_hi"]

LGB_BINARY = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=20,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=500, verbose=-1)
LGB_MULTI = dict(LGB_BINARY, objective="multiclass", num_class=len(FAMILIES), num_leaves=7,
                 min_child_samples=10, n_estimators=300)


def keyed(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.min_horizontal("player_1", "player_2").alias("player_lo"),
                           pl.max_horizontal("player_1", "player_2").alias("player_hi"))


def with_exposure(pf: pl.DataFrame) -> pl.DataFrame:
    """Raw counts grow with period length (3,000 vs 2,000 hands); use shares of the period instead."""
    period = (pl.col("period_hands").cast(pl.Float64) if "period_hands" in pf.columns
              else pl.col("phase").replace_strict(PERIOD_HANDS, return_dtype=pl.Float64))
    return pf.with_columns((pl.col(c) / period).alias(f"{c}_frac") for c in EXPOSURE_COUNTS)


@dataclass
class DevData:
    labeled: pl.DataFrame   # pair_id, label, behavior_family, table_id, fold, features
    mirror: pl.DataFrame    # labeled + eval-filtered unlabeled pairs (label 0, synthetic pair_id), fold, features
    labels: pl.DataFrame    # development_labels.csv as given (string ids)
    evidence: pl.DataFrame  # development_evidence.csv as given
    windows: dict[int, pl.DataFrame]  # window start -> mirror rows (same order) with 2,000-hand window features

    def views(self) -> list[pl.DataFrame]:
        """Every exposure view of the mirror pairs: the full development period, then each window."""
        return [self.mirror, *self.windows.values()]


def load_dev(pair_features: pl.DataFrame, windows: pl.DataFrame | None = None, n_folds: int | None = None,
             seed: int | None = None) -> DevData:
    cfg = load_config()
    n_folds = n_folds or cfg["cv"]["n_folds"]
    seed = cfg["seed"] if seed is None else seed
    labels = pl.read_csv(RAW_DIR / "development_labels.csv")
    evidence = pl.read_csv(RAW_DIR / "development_evidence.csv")
    # sort: polars group_by output order varies between pipeline runs, and sampling must not depend on it
    dev = with_exposure(pair_features.filter(pl.col("phase") == 0)).sort(KEYS)
    lab = (keyed(labels).select("pair_id", *KEYS, "label", "behavior_family").join(dev, on=KEYS, how="inner")
           .sort("pair_id"))
    assert lab.height == labels.height, "every labeled pair must have development features"

    folds = make_folds(lab, n_folds, seed)
    table_fold = (folds.join(lab.select("pair_id", "table_id"), on="pair_id")
                  .unique("table_id").select("table_id", "fold"))
    lab = lab.join(folds, on="pair_id").sort("pair_id").with_columns(pl.lit(True).alias("labeled"))

    pos = lab.filter(pl.col("label") == 1)
    pos_players = pl.concat([pos["player_lo"], pos["player_hi"]]).unique()
    min_shared = int(np.ceil(EVAL_MIN_SHARED * PERIOD_HANDS[0] / PERIOD_HANDS[1]))  # 57, as for labeled pairs
    unl = (dev.join(lab.select(KEYS), on=KEYS, how="anti")
           .filter(~pl.col("player_lo").is_in(pos_players.implode()) & ~pl.col("player_hi").is_in(pos_players.implode())
                   & (pl.col("n_shared") >= min_shared))
           .join(table_fold, on="table_id", how="left")
           .with_columns(pl.col("fold").fill_null(pl.col("table_id").hash(seed) % n_folds).cast(pl.Int8),
                         pl.lit(0, pl.Int64).alias("label"), pl.lit("none").alias("behavior_family"),
                         pl.lit(False).alias("labeled"))
           .sort(KEYS))
    unl = unl.with_columns(pl.Series("pair_id", synthetic_pair_ids(unl, seed)))
    mirror = pl.concat([lab, unl.select(lab.columns)])
    assert mirror["pair_id"].n_unique() == mirror.height, "synthetic pair_id collision"
    views = {}
    if windows is not None:
        meta = mirror.select("pair_id", *KEYS, "table_id", "fold", "labeled", "label", "behavior_family")
        w = with_exposure(windows)
        for start in w["window_start"].unique().sort().to_list():
            # pairs that share no hand inside a window get null features (LightGBM treats them as missing)
            ws = w.filter(pl.col("window_start") == start).select(*KEYS, "n_shared", *PAIR_FEATURES)
            views[start] = meta.join(ws, on=KEYS, how="left", maintain_order="left")
    return DevData(labeled=lab, mirror=mirror, labels=labels, evidence=evidence, windows=views)


def synthetic_pair_ids(df: pl.DataFrame, seed: int) -> list[str]:
    """Hash-based ids in the official format ("P" + 12 upper-case hex digits).

    The metric breaks score ties by pair_id, so unlabeled pairs need ids that interleave randomly with the
    real ones. A constant prefix (e.g. "~") would rank every labeled pair ahead of every unlabeled pair in
    a tie and inflate the scores of tie-heavy predictions (baselines, the score-0 pile in Behavior MAP).
    """
    h = df.select(pl.concat_str("player_lo", pl.lit("_"), "player_hi").hash(seed)).to_series().to_list()
    return [f"P{x & 0xFFFFFFFFFFFF:012X}" for x in h]


def make_folds(lab: pl.DataFrame, n_folds: int, seed: int) -> pl.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(lab.height, dtype=np.int8)
    for f, (_, test) in enumerate(sgkf.split(np.zeros(lab.height), lab["behavior_family"].to_numpy(),
                                             lab["table_id"].to_numpy())):
        fold[test] = f
    return pl.DataFrame({"pair_id": lab["pair_id"], "fold": fold})


HAND_CONTEXT_RANKS = ("h_pot_bb", "h_netflow_bb", "h_contrib_max_bb", "h_pre_pct_max")
HAND_CONTEXT_BLOCK = ("h_fold_ahead_max", "h_payoff_max", "h_hu_strong_check", "h_open_step_aside", "h_call_max")


def hand_context(hf: pl.DataFrame, pair: str = "pair_id") -> pl.DataFrame:
    """Within-pair context for the evidence ranker: how a hand compares with the pair's other candidates.

    Ranks (0-1) of size/flow within the pair, the number of candidates, and the pair's average of
    behaviour flags in the same time block (coordination is episodic, so evidence clusters in time).
    """
    n = pl.len().over(pair)
    return hf.with_columns(
        n.alias("n_cand"),
        *[(pl.col(c).cast(pl.Float64).rank("average").over(pair) / n).alias(f"r_{c}") for c in HAND_CONTEXT_RANKS],
        *[pl.col(c).cast(pl.Float64).mean().over(pair, "block").alias(f"blk_{c}") for c in HAND_CONTEXT_BLOCK],
    )


def hand_feature_names(hf: pl.DataFrame) -> list[str]:
    return ([c for c in hf.columns if c.startswith("h_")] + ["n_cand"]
            + [f"r_{c}" for c in HAND_CONTEXT_RANKS] + [f"blk_{c}" for c in HAND_CONTEXT_BLOCK])


def matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select(pl.col(features).cast(pl.Float64)).to_numpy()


def fit_predict(params: dict, X: np.ndarray, y: np.ndarray, X_test: np.ndarray | None = None,
                seeds: tuple[int, ...] = (0, 1, 2), sample_weight: np.ndarray | None = None,
                feature_names: list[str] | None = None) -> tuple[np.ndarray | None, list[lgb.LGBMClassifier]]:
    """Seed-averaged LightGBM. Returns test predictions (probabilities; None without X_test) and the models.

    Pass `feature_names` for models that are saved, so the booster files carry real names (SHAP, predict).
    """
    names = feature_names or "auto"
    models = [lgb.LGBMClassifier(**params, random_state=s).fit(X, y, sample_weight=sample_weight, feature_name=names)
              for s in seeds]
    if X_test is None:
        return None, models
    return predict_proba(models, X_test), models


def predict_proba(models: list[lgb.LGBMClassifier], X: np.ndarray) -> np.ndarray:
    """Average class probabilities over seeds; binary models return P(positive) only."""
    p = np.mean([m.predict_proba(X) for m in models], axis=0)
    return p[:, 1] if p.shape[1] == 2 else p
