"""P5 explainability (SPEC §5.7, D5): why a pair scored high, why a hand was picked as evidence, and which
innocent story an investigator should rule out first.

    SHAP     exact TreeSHAP contributions of M1 (risk) and M2 (behavior family), averaged over the seed models.
             LightGBM's `pred_contrib` runs the same algorithm as `shap.TreeExplainer` (checked in the tests)
             without the extra dependency in the API image. Units: log-odds; the contributions of one pair plus
             the base value add up to the model's raw score.
    tags     1-3 reason tags per evidence hand, from the hand features that fired (M4's inputs), ordered by
             how diagnostic each tag is of the pair's predicted family.
    benign   innocent-explanation templates. Each fires when its trigger feature is extreme (top or bottom
             10%) relative to a reference population, stored as percentiles only (models/benign_reference.json).

    python -m suspoker.explain     # evaluation pairs -> artifacts/explanations/ (local) + models/benign_reference.json
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from suspoker.config import ARTIFACTS_DIR, MODELS_DIR
from suspoker.features import CATALOG
from suspoker.modeling import FAMILIES, matrix

EXPLAIN_DIR = ARTIFACTS_DIR / "explanations"
BENIGN_REFERENCE = MODELS_DIR / "benign_reference.json"
TOP_REASONS = 5

# ----------------------------------------------------------------------------- plain-English labels
EXTRA_LABELS: dict[str, tuple[str, str]] = {
    "n_shared_frac": ("F1 exposure", "Share of the period's table hands they played together"),
    "hands_max_frac": ("F7 controls", "Share of the period's hands played by the busier player"),
    "hands_min_frac": ("F7 controls", "Share of the period's hands played by the less active player"),
    "hs_max": ("F9 hand suspicion", "How planted-looking their most suspicious shared hand is"),
    "hs_top3": ("F9 hand suspicion", "Average suspicion of their 3 most suspicious shared hands"),
    "hs_top5": ("F9 hand suspicion", "Average suspicion of their 5 most suspicious shared hands"),
    "hs_n50": ("F9 hand suspicion", "Number of shared hands that look more planted than not"),
    "hs_frac": ("F9 hand suspicion", "Share of their contested hands that look more planted than not"),
    "hs_max_rel": ("F9 hand suspicion", "Their most suspicious hand vs each player's other pairs (lift)"),
    "hs_top3_rel": ("F9 hand suspicion", "Their 3 most suspicious hands vs each player's other pairs (lift)"),
    "hs_base_max": ("F9 hand suspicion", "How suspicious the more suspicious player's hands look with everyone else"),
}


def feature_info(name: str) -> tuple[str, str]:
    """(family, plain-English label) for any M1/M2 input, including the cross-period `x_*` background."""
    if name.startswith("x_"):
        fam, label = feature_info(name[2:])
        return fam, f"Other period: {label[0].lower()}{label[1:]}"
    return CATALOG.get(name) or EXTRA_LABELS.get(name) or ("other", name)


# ----------------------------------------------------------------------------- SHAP
def contributions(boosters: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    """Seed-averaged TreeSHAP. Binary: (n, n_features + 1); multiclass: (n, n_classes, n_features + 1).
    The last column is the base value (the model's average raw score)."""
    c = np.mean([b.predict(X, pred_contrib=True) for b in boosters], axis=0)
    k = boosters[0].num_model_per_iteration()
    return c if k == 1 else c.reshape(len(X), k, -1)


def reasons(ids: pl.Series, contrib: np.ndarray, X: np.ndarray, features: list[str], k: int = TOP_REASONS
            ) -> pl.DataFrame:
    """Long table of each row's top-k features pushing the score up (`for`) and down (`against`)."""
    phi = contrib[:, :len(features)]
    up = np.argsort(-phi, axis=1)[:, :k]
    down = np.argsort(phi, axis=1)[:, :k]
    rows = []
    for direction, idx in (("for", up), ("against", down)):
        r = np.repeat(np.arange(len(phi)), k)
        f = idx.ravel()
        rows.append(pl.DataFrame({"id": ids.gather(r), "direction": direction, "rank": np.tile(np.arange(1, k + 1),
                                  len(phi)), "feature": np.asarray(features)[f], "value": X[r, f], "shap": phi[r, f]}))
    out = pl.concat(rows)
    # keep only contributions that really point in their direction
    out = out.filter(pl.when(pl.col("direction") == "for").then(pl.col("shap") > 0).otherwise(pl.col("shap") < 0))
    info = pl.DataFrame([(f, *feature_info(f)) for f in features], schema=["feature", "family", "label"], orient="row")
    return out.join(info, on="feature", how="left").sort("id", "direction", "rank")


def global_importance(contrib: np.ndarray, features: list[str]) -> pl.DataFrame:
    """Mean |SHAP| per feature (how much it moves scores on average), with its family and label."""
    phi = contrib[:, :len(features)]
    return pl.DataFrame({
        "feature": features, "family": [feature_info(f)[0] for f in features],
        "label": [feature_info(f)[1] for f in features],
        "mean_abs_shap": np.abs(phi).mean(axis=0), "mean_shap": phi.mean(axis=0),
    }).sort("mean_abs_shap", descending=True)


def family_importance(imp: pl.DataFrame) -> pl.DataFrame:
    return (imp.group_by("family").agg(pl.col("mean_abs_shap").sum(), pl.len().alias("n_features"))
            .with_columns((pl.col("mean_abs_shap") / pl.col("mean_abs_shap").sum()).alias("share"))
            .sort("mean_abs_shap", descending=True))


def explain_pairs(m1: list[lgb.Booster], m2: list[lgb.Booster], pairs: pl.DataFrame, features: list[str],
                  k: int = TOP_REASONS) -> dict[str, pl.DataFrame | np.ndarray]:
    """Risk reasons (M1) and behavior reasons (M2, for each pair's most likely family)."""
    X = matrix(pairs, features)
    c1 = contributions(m1, X)
    c2 = contributions(m2, X)
    fam = np.mean([b.predict(X) for b in m2], axis=0).argmax(axis=1)
    c2_fam = c2[np.arange(len(X)), fam]
    risk = reasons(pairs["pair_id"], c1, X, features, k)
    beh = reasons(pairs["pair_id"], c2_fam, X, features, k).with_columns(
        pl.col("id").replace_strict(dict(zip(pairs["pair_id"], np.asarray(FAMILIES)[fam], strict=True)))
        .alias("family_explained"))
    return {"risk": risk, "behavior": beh, "risk_contrib": c1, "behavior_contrib": c2}


# ----------------------------------------------------------------------------- evidence reason tags
@dataclass(frozen=True)
class Tag:
    fires: pl.Expr
    text: str  # str.format template over the hand's feature values


def _n(c: str) -> pl.Expr:
    return pl.col(c).cast(pl.Float64).fill_null(0)


TAGS: dict[str, Tag] = {
    "ev_gift": Tag(_n("h_ev_gift_bb_max") >= 1,
                   "Gave up {h_ev_gift_bb_max:.1f} bb of expected value against the partner"),
    "fold_ahead": Tag(_n("h_fold_ahead_max") > 0, "Folded the better hand to the partner's bet after the flop"),
    "pre_fold_ahead": Tag(_n("h_pre_fold_ahead_max") > 0,
                          "Folded a clearly stronger starting hand to the partner's raise"),
    "costly_fold": Tag(_n("h_costly_fold_max") > 0, "Folded to the partner when calling was worth at least 1 bb"),
    "payoff": Tag(_n("h_payoff_max") > 0, "Paid the partner off while behind ({h_payoff_bb_max:.1f} bb)"),
    "costly_call": Tag(_n("h_costly_call_max") > 0, "Called the partner when the call lost at least 1 bb"),
    "junk_call": Tag(_n("h_pre_junk_call_max") > 0, "Called the partner's raise with a junk starting hand"),
    "strong_check": Tag((_n("h_hu_strong_check") > 0) | (_n("h_strong_check_partner") > 0),
                        "Checked a strong hand (two pair or better) with the partner still in"),
    "open_step_aside": Tag(_n("h_open_step_aside") > 0,
                           "One partner opened and the other folded behind them (isolation step)"),
    "squeeze": Tag(_n("h_squeeze_max") > 0, "Re-raised the partner's bet in a pot with other players in it"),
    "weak_open": Tag(_n("h_weak_open") > 0, "Opened with a bottom-half starting hand with the partner at the table"),
    "big_flow": Tag((_n("r_h_netflow_bb") >= 0.9) & (_n("h_netflow_bb") >= 5),
                    "One of their largest chip transfers ({h_netflow_bb:.0f} bb)"),
    "big_pot": Tag(_n("r_h_pot_bb") >= 0.9, "One of the biggest pots they shared ({h_pot_bb:.0f} bb)"),
    "contested": Tag(pl.lit(True), "Both players were involved in this pot ({h_pot_bb:.0f} bb)"),
}
# Behavior tags per family, most diagnostic first: ordered by lift (how much more often the tag fires on planted
# hands than on the pair's other candidate hands), measured on the labeled development colluders (notebook 04 §4).
# Tags with lift below ~1.5 for a family are left out: they fire as often on ordinary hands.
TAG_ORDER: dict[str, list[str]] = {
    "directed_transfer": ["fold_ahead", "strong_check", "junk_call", "costly_fold", "pre_fold_ahead", "ev_gift",
                          "payoff", "costly_call", "squeeze", "weak_open"],
    "soft_play": ["fold_ahead", "pre_fold_ahead", "costly_call", "strong_check", "payoff", "ev_gift", "costly_fold",
                  "junk_call"],
    "coordinated_isolation": ["weak_open", "junk_call", "pre_fold_ahead", "squeeze", "costly_fold", "open_step_aside",
                              "ev_gift"],
}
CONTEXT_TAGS = ["big_flow", "big_pot"]
MAX_TAGS = 3


def tag_flags(hands: pl.DataFrame) -> pl.DataFrame:
    """One boolean column `tag_<id>` per reason tag."""
    return hands.with_columns(t.fires.alias(f"tag_{k}") for k, t in TAGS.items())


def reason_tags(hands: pl.DataFrame, family: str | None = None, max_tags: int = MAX_TAGS) -> list[list[dict]]:
    """1-`max_tags` tags per hand row. `family` is a column name holding each hand's pair family, or None."""
    flags = tag_flags(hands)
    out = []
    for r in flags.iter_rows(named=True):
        order = TAG_ORDER.get(r.get(family) if family else None, TAG_ORDER["directed_transfer"])
        fired = [k for k in order if r[f"tag_{k}"]][:max_tags]
        fired += [k for k in CONTEXT_TAGS if r[f"tag_{k}"]][:max(0, max_tags - len(fired))]
        fired = fired or ["contested"]
        vals = {k: (v if v is not None else 0) for k, v in r.items()}
        out.append([{"id": k, "text": TAGS[k].text.format(**vals)} for k in fired])
    return out


# ----------------------------------------------------------------------------- benign alternatives
@dataclass(frozen=True)
class Benign:
    title: str
    feature: str
    tail: str   # "high": fires in the top 10%; "low": fires in the bottom 10%
    text: str   # template over {value} and {pct} (percentile of the value in the reference population)


BENIGN: dict[str, Benign] = {
    "passive_player": Benign(
        "A passive player folds to everyone", "fold_rate_max", "high",
        "One player folds to {value:.0%} of bets from anyone (more than {pct:.0f}% of pairs). Folding to the partner "
        "may be that player's style, not a favour: check that the folds against the partner exceed their usual rate."),
    "losing_player": Benign(
        "A losing player loses to the whole table", "net_bb100_min", "low",
        "One player's overall result is {value:+.0f} bb per 100 hands (worse than {inv:.0f}% of pairs). Chips "
        "flowing to the partner may be part of losing to everyone."),
    "tilt": Benign(
        "Tilt after big losses", "tilt_vpip_max", "high",
        "One player enters {value:+.0%} more pots right after a big loss (more than {pct:.0f}% of pairs). Reckless "
        "play while tilting can look like chip dumping."),
    "aggressive_player": Benign(
        "A hyper-aggressive player", "af_max", "high",
        "One player bets and raises {value:.1f}x as often as they call (more than {pct:.0f}% of pairs). Folding to an "
        "aggressive player is normal; the question is whether the folds are specific to the partner."),
    "similar_styles": Benign(
        "Similar playing styles", "style_dist", "low",
        "Their overall styles are closer than {inv:.0f}% of pairs (distance {value:.2f}). Two players with the same "
        "habits make the same choices without any agreement."),
    "schedule_overlap": Benign(
        "They simply play together a lot", "n_shared_frac", "high",
        "They played {value:.0%} of the table's hands together (more than {pct:.0f}% of pairs). Regulars on the same "
        "schedule meet often, so rare interactions pile up by chance."),
    "small_sample": Benign(
        "Too few hands to be sure", "n_shared_frac", "low",
        "They played only {value:.0%} of the table's hands together (fewer than {inv:.0f}% of pairs). A few big pots "
        "can swing every rate; treat the scores as weak evidence."),
    "neighbours": Benign(
        "Seated side by side", "adjacent_rate", "high",
        "They sat next to each other in {value:.0%} of shared hands (more than {pct:.0f}% of pairs). Position alone "
        "creates more steals and folds between neighbours."),
}
BENIGN_PCT = 90.0
QUANTILES = np.linspace(0, 100, 101)


def fit_reference(pairs: pl.DataFrame) -> dict[str, list[float]]:
    """Percentiles (0, 1, ..., 100) of every trigger feature in a population of pairs: aggregates only."""
    feats = sorted({b.feature for b in BENIGN.values()})
    return {f: np.nanpercentile(pairs[f].cast(pl.Float64).to_numpy(), QUANTILES).tolist() for f in feats}


def percentile(ref: dict[str, list[float]], feature: str, values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value within the reference grid (ties at the ends stay fair)."""
    q = np.asarray(ref[feature])
    lo = np.searchsorted(q, values, side="left")
    hi = np.searchsorted(q, values, side="right")
    return np.clip((lo + hi) / 2 / (len(q) - 1) * 100, 0, 100)


def benign_flags(pairs: pl.DataFrame, ref: dict[str, list[float]]) -> pl.DataFrame:
    """Per pair and template: the strength (how extreme the trigger is, 0-100) and whether it fires."""
    cols = []
    for k, b in BENIGN.items():
        v = pairs[b.feature].cast(pl.Float64).fill_null(np.nan).to_numpy()
        pct = percentile(ref, b.feature, v)
        strength = np.where(np.isnan(v), 0.0, pct if b.tail == "high" else 100 - pct)
        cols += [pl.Series(f"benign_{k}", strength >= BENIGN_PCT), pl.Series(f"strength_{k}", strength)]
    return pairs.select("pair_id").with_columns(cols)


def benign_alternatives(row: dict, ref: dict[str, list[float]]) -> list[dict]:
    """Every template that fires for one pair (feature values in `row`), strongest first."""
    out = []
    for k, b in BENIGN.items():
        v = row.get(b.feature)
        if v is None or np.isnan(v):
            continue
        pct = float(percentile(ref, b.feature, np.array([v]))[0])
        strength = pct if b.tail == "high" else 100 - pct
        if strength >= BENIGN_PCT:
            out.append({"id": k, "title": b.title, "feature": b.feature, "value": float(v),
                        "percentile": round(pct, 1), "strength": round(strength, 1),
                        "text": b.text.format(value=v, pct=pct, inv=100 - pct)})
    return sorted(out, key=lambda x: -x["strength"])


def load_reference(path: Path = BENIGN_REFERENCE) -> dict[str, list[float]]:
    return json.loads(path.read_text(encoding="utf-8"))["percentiles"]


# ----------------------------------------------------------------------------- CLI: evaluation pairs (local)
def main() -> None:
    import time

    from suspoker.predict import Models, evaluation_candidates, evaluation_pairs, score_pairs

    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:5.0f}s] {msg}", flush=True)

    models = Models.load()
    feats = models.meta["pair_features"]
    pairs = evaluation_pairs(models)
    scored = score_pairs(models, pairs)
    log(f"{pairs.height:,} evaluation pairs scored")
    ex = explain_pairs(models.m1, models.m2, pairs, feats)
    # MA5 check: the contributions reproduce each seed-averaged raw score (log-odds) exactly
    raw = np.mean([b.predict(matrix(pairs, feats), raw_score=True) for b in models.m1], axis=0)
    assert np.allclose(ex["risk_contrib"].sum(axis=1), raw, atol=1e-6), "SHAP does not add up to the raw score"
    log("SHAP for M1 and M2 done (contributions add up to the raw scores)")

    ref = fit_reference(pairs)
    BENIGN_REFERENCE.write_text(json.dumps({
        "population": f"{pairs.height} evaluation pairs (percentiles only, no pair data)",
        "percentiles": ref}, indent=1), encoding="utf-8")
    benign = benign_flags(pairs, ref)

    ev = evaluation_candidates(models, scored)
    ev = ev.join(scored.select("pair_id", "predicted_behavior"), on="pair_id", how="left")
    tags = reason_tags(ev, family="predicted_behavior")
    ev_tags = ev.select("pair_id", "hand_id", "score").with_columns(
        pl.Series("tags", [[t["id"] for t in ts] for ts in tags], dtype=pl.List(pl.String)))
    log(f"reason tags for {ev_tags.height:,} evidence hands")

    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    ex["risk"].write_parquet(EXPLAIN_DIR / "risk_reasons.parquet")
    ex["behavior"].write_parquet(EXPLAIN_DIR / "behavior_reasons.parquet")
    imp = global_importance(ex["risk_contrib"], feats)
    imp.write_parquet(EXPLAIN_DIR / "importance_m1.parquet")
    for i, fam in enumerate(FAMILIES):
        imp_fam = global_importance(ex["behavior_contrib"][:, i], feats)
        imp_fam.write_parquet(EXPLAIN_DIR / f"importance_m2_{fam}.parquet")
    benign.write_parquet(EXPLAIN_DIR / "benign.parquet")
    ev_tags.write_parquet(EXPLAIN_DIR / "evidence_tags.parquet")
    n_expl = ex["risk"].filter(pl.col("direction") == "for")["id"].n_unique()
    log(f"wrote {EXPLAIN_DIR}: risk reasons for {n_expl:,}/{pairs.height:,} pairs; top features: "
        + ", ".join(imp["feature"].head(5)))


if __name__ == "__main__":
    main()
