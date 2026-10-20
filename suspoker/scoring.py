"""Score any hands in the competition schema with the Kaggle-trained models: the SimRoom now, live games in P8.

The path is exactly the competition's: encode -> exact-odds equity -> S2-S5 features -> hand suspicion (H) ->
M1 risk, M2 behavior, M3 novelty -> M4 evidence -> explanations (SHAP reasons, reason tags, benign templates).
Nothing here reads competition data; it only uses `models/`.

    world = score_tables(hands, seats, actions, Models.load(), pair_ids=simroom.pair_id)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from suspoker.config import load_config
from suspoker.equity import decision_equity
from suspoker.explain import BENIGN, benign_alternatives, explain_pairs, load_reference, reason_tags
from suspoker.features import build_chunk
from suspoker.ingest import Tables, encode
from suspoker.interactions import enrich_actions, hand_context, hand_players, response_events, street_ranks
from suspoker.modeling import EVAL_MIN_SHARED, KEYS, with_exposure
from suspoker.pipeline import with_string_ids
from suspoker.predict import Models, hand_probabilities, score_evidence, score_pairs
from suspoker.suspicion import pair_suspicion


@dataclass
class World:
    tables: Tables
    pairs: pl.DataFrame          # one row per scored pair: ids, features, risk, family probs, behavior
    hands: pl.DataFrame          # candidate hands per pair (hand features, string ids) with H's probability `p`
    evidence: pl.DataFrame       # long: pair_id, rank, hand_id, score, tags (list of {id, text})
    risk_reasons: pl.DataFrame   # SHAP reasons for the risk score (explain.reasons format)
    behavior_reasons: pl.DataFrame
    risk_contrib: np.ndarray     # (pairs, features + 1) seed-averaged TreeSHAP of M1, last column = base value
    features: list[str]


def default_pair_id(a: str, b: str) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}|{hi}"


BENIGN_SCHEMA = pl.List(pl.Struct({"id": pl.String, "title": pl.String, "feature": pl.String, "value": pl.Float64,
                                  "percentile": pl.Float64, "strength": pl.Float64, "text": pl.String}))


def attach_benign(pairs: pl.DataFrame, ref: dict[str, list[float]], top: int = 3) -> pl.DataFrame:
    """Column `benign`: the strongest innocent-explanation templates that fire for each pair."""
    triggers = sorted({b.feature for b in BENIGN.values()})
    benign = [benign_alternatives(r, ref)[:top] for r in pairs.select(triggers).iter_rows(named=True)]
    return pairs.with_columns(pl.Series("benign", benign, dtype=BENIGN_SCHEMA))


def equity_table(t: Tables, seed: int, workers: int | None = None) -> pl.DataFrame:
    block = load_config()["features"]["block_hands"]
    ctx = hand_context(t, block)
    ev = response_events(enrich_actions(t, ctx, hand_players(t, ctx), street_ranks(t)))
    return decision_equity(ev.select("hand_idx", "action_no", "street", "x", "y"), t.seats, t.hands, seed=seed,
                           workers=workers)


def score_tables(hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame, models: Models,
                 pair_ids: Callable[[str, str], str] = default_pair_id, min_shared: int = EVAL_MIN_SHARED,
                 keep: int = 5, workers: int | None = None, benign_ref: dict | None = None,
                 equity: pl.DataFrame | None = None, exposure_scale: dict[str, float] | None = None,
                 neutral: list[str] | None = None, novelty: bool = True) -> World:
    """Score every pair that shared at least `min_shared` hands (the competition's evaluation filter).
    `benign_ref`: percentiles for the innocent-explanation templates (default: models/benign_reference.json).
    `equity`: precomputed decision equity keyed by (hand_id, action_no), e.g. a live session's cache.
    `exposure_scale`: divides the exposure shares (`n_shared_frac`, `hands_max_frac`, `hands_min_frac`). At a
    six-seat game table everyone plays every hand; as six seats of a virtual 30-player pool a pair shares 1/29 of
    the hands and a player plays 1/5 of them, so a live session uses {n_shared: 29, hands_*: 5} (DEC-030).
    `neutral`: features that cannot be measured in this setting, set to 0 = "no effect" (a live table where nobody
    ever leaves cannot measure F8 presence: there are no hands without the partner).
    `novelty`: run M3 (False for a single small table)."""
    cfg = load_config()
    phases = hands["phase"].unique().to_list()
    if len(phases) != 1:
        raise ValueError("score one period at a time")
    t = encode(hands, seats, actions)
    phase = int(t.hands["phase"][0])
    if equity is None:
        eq = equity_table(t, cfg["seed"], workers)
    else:
        eq = equity.join(t.hand_ids, on="hand_id").select("hand_idx", "action_no", "eq")
    pf, hf, _ = build_chunk(t, k=cfg["features"]["shrinkage_k"], block_hands=cfg["features"]["block_hands"],
                            hand_pairs=None, dev_min_shared=min_shared, equity=eq, min_shared_phase=phase)
    # exposure features are shares of the period: here a period is each table's own number of hands
    period = t.hands.group_by("table_idx").agg(pl.len().cast(pl.Int32).alias("period_hands"))
    pf = with_exposure(pf.filter(pl.col("n_shared") >= min_shared).join(period, on="table_idx"))
    if exposure_scale:
        pf = pf.with_columns(pl.col(f"{c}_frac") / s for c, s in exposure_scale.items())
    if neutral:
        pf = pf.with_columns(pl.lit(0.0).alias(f) for f in neutral)
    pairs = with_string_ids(pf, t)
    pairs = pairs.with_columns(pl.struct(KEYS).map_elements(lambda r: pair_ids(r["player_lo"], r["player_hi"]),
                                                            return_dtype=pl.String).alias("pair_id"))
    hf = with_string_ids(hf, t).join(pairs.select(*KEYS, "pair_id"), on=KEYS, how="inner")
    meta = models.meta
    hf = hf.with_columns(pl.Series("p", hand_probabilities(meta, hf)))
    if meta.get("hand_suspicion"):
        pairs = pairs.join(pair_suspicion(hf.select(*KEYS, "p"), rel=bool(meta.get("hs_relative"))), on=KEYS,
                           how="left")
    pairs = pairs.sort("table_id", "pair_id")

    scored = score_pairs(models, pairs, novelty=novelty)
    pairs = pairs.join(scored.drop(KEYS), on="pair_id", how="left", maintain_order="left")

    cand = score_evidence(models, hf.drop("pair_id"), scored, keep=keep)
    cand = cand.with_columns(pl.int_range(1, pl.len() + 1).over("pair_id").alias("rank")).join(
        pairs.select("pair_id", "predicted_behavior"), on="pair_id", how="left")
    tags = reason_tags(cand, family="predicted_behavior")
    evidence = cand.select("pair_id", "rank", "hand_id", "score").with_columns(
        pl.Series("tags", tags, dtype=pl.List(pl.Struct({"id": pl.String, "text": pl.String}))))

    feats = meta["pair_features"]
    ex = explain_pairs(models.m1, models.m2, pairs, feats)
    pairs = attach_benign(pairs, benign_ref if benign_ref is not None else load_reference())
    return World(tables=t, pairs=pairs, hands=hf, evidence=evidence, risk_reasons=ex["risk"],
                 behavior_reasons=ex["behavior"], risk_contrib=ex["risk_contrib"], features=feats)
