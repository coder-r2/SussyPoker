"""SimRoom metrics: the Kaggle-trained models scored against our planted ground truth (domain-shift test).

The official metric is used unchanged for Pair AP and Behavior MAP. For Evidence MAP@5 every hand where the
pair's overlay actually changed a decision counts as relevant (the competition lists up to 5 per pair; we know
all of them), with the same AP@5 formula.
"""

import numpy as np
import pandas as pd
import polars as pl

from suspoker.metric import EVIDENCE_COLUMNS, NO_EVIDENCE, TARGET_BEHAVIORS, WEIGHTS, average_precision_at_5, score

ROLES = ("planted", "decoy", "innocent")


def truth_table(truth: dict) -> pl.DataFrame:
    """One row per planted or decoy pair: role, family / decoy type, coordinated hands."""
    rows = []
    for t in truth["tables"]:
        rows += [{"pair_id": p["pair_id"], "role": "planted", "family": p["family"], "decoy": None,
                  "coordinated_hands": p["coordinated_hands"]} for p in t["planted"]]
        rows += [{"pair_id": d["pair_id"], "role": "decoy", "family": None, "decoy": d["type"],
                  "coordinated_hands": []} for d in t["decoys"]]
    return pl.DataFrame(rows, schema={"pair_id": pl.String, "role": pl.String, "family": pl.String,
                                      "decoy": pl.String, "coordinated_hands": pl.List(pl.String)})


def with_truth(pairs: pl.DataFrame, truth: dict) -> pl.DataFrame:
    return pairs.join(truth_table(truth), on="pair_id", how="left").with_columns(
        pl.col("role").fill_null("innocent"), pl.col("coordinated_hands").fill_null([]),
        (pl.col("role") == "planted").fill_null(False).cast(pl.Int8).alias("label"))


def official(p: pl.DataFrame, risk: str, behavior: str, evidence: dict[str, list[str]]) -> dict:
    """Pair AP / Behavior MAP with the official code, Evidence MAP@5 against all coordinated hands."""
    sol = pd.DataFrame({"pair_id": p["pair_id"], "risk_score": p["label"],
                        "predicted_behavior": p["family"].fill_null("none")})
    sub = pd.DataFrame({"pair_id": p["pair_id"], "risk_score": p[risk], "predicted_behavior": p[behavior]})
    for c in EVIDENCE_COLUMNS:
        sol[c] = NO_EVIDENCE
        sub[c] = NO_EVIDENCE
    s = score(sol, sub)
    planted = p.filter(pl.col("label") == 1)
    ev = [average_precision_at_5(set(r["coordinated_hands"]), evidence.get(r["pair_id"], []))
          for r in planted.iter_rows(named=True)]
    ev_map = float(np.mean(ev)) if ev else 0.0
    final = WEIGHTS["pair_ap"] * s.pair_ap + WEIGHTS["evidence_map"] * ev_map + WEIGHTS["behavior_map"] * s.behavior_map
    return {"final": final, "pair_ap": s.pair_ap, "evidence_map": ev_map, "behavior_map": s.behavior_map,
            "behavior_ap": s.behavior_ap}


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def simroom_metrics(pairs: pl.DataFrame, evidence: pl.DataFrame, hands: pl.DataFrame, truth: dict) -> dict:
    """pairs: scored World.pairs; evidence: World.evidence; hands: World.hands (candidate hand features)."""
    p = with_truth(pairs, truth)
    ev = {k[0]: g.sort("rank")["hand_id"].to_list() for k, g in evidence.group_by("pair_id")}

    def biggest(col: str) -> dict[str, list[str]]:
        top = hands.sort(["pair_id", col], descending=[False, True]).group_by("pair_id", maintain_order=True).head(5)
        return {k[0]: g["hand_id"].to_list() for k, g in top.group_by("pair_id")}

    b0 = p.with_columns(pl.Series("s", rank01(p["n_shared"].to_numpy())), pl.lit("directed_transfer").alias("b"))
    b1 = p.with_columns(pl.Series("s", rank01(p["netflow_bb100"].abs().fill_null(0).to_numpy())),
                        pl.lit("directed_transfer").alias("b"))
    models = {
        "M_full": official(p, "risk", "predicted_behavior", ev),
        "B0_shared_hands": official(b0, "s", "b", biggest("h_pot_bb")),
        "B1_net_flow": official(b1, "s", "b", biggest("h_netflow_bb")),
    }

    ranked = p.sort(["risk", "pair_id"], descending=[True, False]).with_row_index("rank", offset=1)
    k = int(p["label"].sum())
    top = ranked.filter(pl.col("rank") <= k)
    tp = int(top["label"].sum())
    recall = (ranked.filter(pl.col("role") == "planted").group_by("family").agg(
        pl.len().alias("n"), (pl.col("rank") <= k).sum().alias("caught"), pl.col("rank").median().alias("median_rank"),
        (pl.col("predicted_behavior") == pl.col("family")).mean().alias("behavior_correct")).sort("family"))
    decoys = (ranked.filter(pl.col("role") == "decoy").group_by("decoy").agg(
        pl.len().alias("n"), pl.col("rank").min().alias("best_rank"), pl.col("rank").median().alias("median_rank"),
        (pl.col("rank") <= k).sum().alias("in_top_k")).sort("decoy"))
    n_coord = ranked.filter(pl.col("role") == "planted").with_columns(
        pl.col("coordinated_hands").list.len().alias("n_coord"))
    buckets = (n_coord.with_columns(pl.col("n_coord").cut([2, 9, 19], labels=["0-2", "3-9", "10-19", "20+"])
                                    .cast(pl.String).alias("coordinated"))
               .group_by("coordinated").agg(pl.len().alias("n"), (pl.col("rank") <= k).sum().alias("caught"),
                                            pl.col("rank").median().alias("median_rank")).sort("coordinated"))
    planted_all = [p_ for t in truth["tables"] for p_ in t["planted"]]
    return {
        "population": "every SimRoom pair with at least 38 shared hands (the competition's evaluation filter)",
        "pairs_scored": p.height, "planted_scored": k, "planted_total": len(planted_all),
        "decoys_scored": int((p["role"] == "decoy").sum()), "base_rate": k / p.height,
        "models": models,
        "break_even": {"k": k, "tp": tp, "fp": k - tp, "precision": tp / k if k else 0.0},
        "recall_by_family": recall.to_dicts(), "decoys": decoys.to_dicts(),
        "detection_by_coordinated_hands": buckets.to_dicts(),
        "families": list(TARGET_BEHAVIORS),
    }
