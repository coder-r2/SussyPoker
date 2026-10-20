"""P5 error analysis on the out-of-fold development predictions (SPEC §6 model card §8). The inputs are local
competition data; every function returns aggregates (counts, rates, medians), which are all the model card shows.

Operating point: the **break-even cut**. Flag the K highest-risk pairs, where K is the number of labeled colluders
(372). At that cut precision equals recall, so one number summarizes it, and each missed colluder (false negative)
is matched by one flagged non-colluder (false positive). Flagged pairs split into *confirmed negatives* (labeled
innocent) and *unlabeled* pairs, which may be colluders the labels do not list.
"""

import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve

from suspoker.config import RAW_DIR
from suspoker.explain import BENIGN, benign_flags, fit_reference
from suspoker.metric import average_precision_at_5
from suspoker.modeling import FAMILIES, KEYS, with_exposure
from suspoker.pipeline import FEATURES_DIR
from suspoker.train import OOF_DIR, WINDOW_VIEW

OUTCOMES = ("TP", "FP", "FN", "TN")
# the pair features that define each family (SPEC §5.3), used to describe what missed colluders lack
FAMILY_KEYS = {
    "directed_transfer": ["fold_lift_max", "ev_gift_lift_max", "pre_fold_ahead_lift_max", "netflow_bb100"],
    "soft_play": ["hu_strong_check_lift", "raise_lift_min", "fold_strong_lift_max", "hu_check_lift"],
    "coordinated_isolation": ["step_aside_lift_max", "vpip_pres_max", "iso_lift", "squeeze_lift_max"],
}


def load_oof() -> pl.DataFrame:
    return pl.read_parquet(OOF_DIR / "pairs.parquet")


def with_window_features(p: pl.DataFrame, start: int = WINDOW_VIEW) -> pl.DataFrame:
    """Attach the evaluation-length window features (the view `risk_window` was scored on)."""
    w = with_exposure(pl.read_parquet(FEATURES_DIR / "pair_features_windows.parquet")
                      .filter(pl.col("window_start") == start)).drop("table_id", "n_shared", strict=False)
    return p.join(w, on=KEYS, how="left", maintain_order="left")


def flag_top_k(p: pl.DataFrame, col: str = "risk_window", k: int | None = None) -> pl.DataFrame:
    """Flag the k highest-risk pairs (default: break-even k = number of positives); ties go to the smaller
    pair_id, as in the official metric. Adds `flagged` and `outcome` (TP/FP/FN/TN)."""
    k = int(p["label"].sum()) if k is None else k
    ranked = p.sort("pair_id").sort(col, descending=True, maintain_order=True).with_row_index("rank", offset=1)
    return ranked.with_columns((pl.col("rank") <= k).alias("flagged")).with_columns(
        pl.when(pl.col("flagged") & (pl.col("label") == 1)).then(pl.lit("TP"))
        .when(pl.col("flagged")).then(pl.lit("FP"))
        .when(pl.col("label") == 1).then(pl.lit("FN")).otherwise(pl.lit("TN")).alias("outcome"))


def outcome_counts(f: pl.DataFrame) -> dict:
    """Counts at the cut, with false positives split by whether the pair is a confirmed negative."""
    c = dict(f.group_by("outcome").len().iter_rows())
    tp, fp, fn = (c.get(o, 0) for o in ("TP", "FP", "FN"))
    fps = f.filter(pl.col("outcome") == "FP")
    neg = f.filter(pl.col("labeled") & (pl.col("label") == 0))
    return {
        "k": tp + fp, **{o: c.get(o, 0) for o in OUTCOMES},
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "fp_confirmed_negative": int(fps["labeled"].sum()), "fp_unlabeled": int((~fps["labeled"]).sum()),
        # where the best-ranked confirmed negative lands: a decoy check (lift features should push them down)
        "best_confirmed_negative_rank": int(neg["rank"].min()) if neg.height else None,
        "confirmed_negatives": neg.height,
    }


def recall_by_family(f: pl.DataFrame) -> pl.DataFrame:
    return (f.filter(pl.col("label") == 1).group_by("behavior_family")
            .agg(pl.len().alias("positives"), (pl.col("outcome") == "TP").sum().alias("caught"),
                 pl.col("rank").median().alias("median_rank"))
            .with_columns((pl.col("caught") / pl.col("positives")).alias("recall"),
                          (pl.col("positives") - pl.col("caught")).alias("missed"))
            .sort("behavior_family"))


def decoy_profile(f: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Benign-alternative ("decoy") templates by outcome: share of pairs where each fires, and the share whose
    *strongest* template it is. Reference population: every pair in `f` (window features)."""
    flags = benign_flags(f, fit_reference(f))
    keys = list(BENIGN)
    strength = np.column_stack([flags[f"strength_{k}"].to_numpy() for k in keys])
    fires = np.column_stack([flags[f"benign_{k}"].to_numpy() for k in keys])
    top = np.where(fires.any(axis=1), np.asarray(keys, dtype=object)[np.where(fires, strength, -1).argmax(axis=1)],
                   "none fits")
    g = pl.concat([f.select("outcome"), flags.drop("pair_id")], how="horizontal").with_columns(
        pl.Series("top_template", top))
    rates = g.group_by("outcome").agg(pl.len().alias("pairs"), *[pl.col(f"benign_{k}").mean().alias(k) for k in keys])
    tops = (g.group_by("outcome", "top_template").len()
            .with_columns((pl.col("len") / pl.col("len").sum().over("outcome")).alias("share")))
    return rates.sort("outcome"), tops.sort("outcome", "share", descending=[False, True])


def missed_profile(f: pl.DataFrame) -> pl.DataFrame:
    """Per family: medians of exposure and the family's key features for caught (TP) vs missed (FN) colluders."""
    rows = []
    for fam in FAMILIES:
        g = f.filter((pl.col("label") == 1) & (pl.col("behavior_family") == fam))
        for feat in ["n_shared_frac", *FAMILY_KEYS[fam]]:
            med = dict(g.group_by("outcome").agg(pl.col(feat).median()).iter_rows())
            rows.append({"family": fam, "feature": feat, "caught_median": med.get("TP"),
                         "missed_median": med.get("FN")})
    return pl.DataFrame(rows)


def behavior_confusion(p: pl.DataFrame, col: str = "predicted_behavior_window") -> tuple[list[str], list[str],
                                                                                         list[list[int]]]:
    """True family (rows) x predicted behavior (columns) for the labeled colluders."""
    pos = p.filter(pl.col("label") == 1)
    pred = [*FAMILIES, *sorted(set(pos[col]) - set(FAMILIES))]
    m = [[int(((pos["behavior_family"] == t) & (pos[col] == q)).sum()) for q in pred] for t in FAMILIES]
    return list(FAMILIES), pred, m


def evidence_by_family(ev: pl.DataFrame | None = None) -> pl.DataFrame:
    """Out-of-fold Evidence AP@5 per family, plus how often at least one planted hand reaches the top 5."""
    ev = pl.read_parquet(OOF_DIR / "evidence.parquet") if ev is None else ev
    planted = pl.read_csv(RAW_DIR / "development_evidence.csv")
    fam = planted.select("pair_id", "behavior_family").unique()
    rel = planted.group_by("pair_id").agg(pl.col("hand_id"))
    top = (ev.sort(["pair_id", "score"], descending=[False, True], maintain_order=True)
           .group_by("pair_id", maintain_order=True).agg(pl.col("hand_id").head(5).alias("top"),
                                                         pl.len().alias("candidates")))
    j = rel.join(top, on="pair_id", how="left").join(fam, on="pair_id")
    pairs = list(zip(j["hand_id"].to_list(), j["top"].to_list(), strict=True))
    ap = [average_precision_at_5(set(r), t or []) for r, t in pairs]
    hit = [bool(set(r) & set(t or [])) for r, t in pairs]
    return (j.with_columns(pl.Series("ap5", ap), pl.Series("hit", hit), pl.col("hand_id").list.len().alias("planted"))
            .group_by("behavior_family")
            .agg(pl.len().alias("pairs"), pl.col("ap5").mean().alias("evidence_ap5"), pl.col("hit").mean().alias(
                "any_planted_in_top5"), pl.col("planted").mean().alias("planted_per_pair"),
                 pl.col("candidates").median().alias("median_candidates"))
            .sort("behavior_family"))


def pr_curve(y: np.ndarray, s: np.ndarray, points: int = 101) -> dict:
    """Interpolated precision on a fixed recall grid (precision at recall >= r), small enough for a chart."""
    prec, rec, _ = precision_recall_curve(y, s)
    grid = np.linspace(0, 1, points)
    interp = [float(prec[rec >= r].max()) for r in grid]  # recall reaches 1 at the lowest threshold
    return {"recall": [round(float(r), 3) for r in grid], "precision": [round(p, 4) for p in interp],
            "base_rate": round(float(np.mean(y)), 5)}
