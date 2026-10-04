import numpy as np
import polars as pl
import pytest

from suspoker.config import has_raw_data
from suspoker.metric import NO_EVIDENCE
from suspoker.modeling import FAMILIES, hand_context, make_folds, with_exposure
from suspoker.train import assign_behavior, calibrate, platt, top5

POLICY = {"other_coordination": {"novelty_percentile": 99, "max_family_prob": 0.5, "min_risk_quantile": 0.5}}


def test_folds_never_split_a_table_and_keep_every_family_in_every_fold():
    rng = np.random.default_rng(0)
    fams = ["none"] * 400 + [f for f in FAMILIES for _ in range(60)]
    lab = pl.DataFrame({"pair_id": [f"P{i}" for i in range(len(fams))], "behavior_family": fams,
                        "table_id": [f"T{t}" for t in rng.integers(0, 80, len(fams))]})
    f = make_folds(lab, 5, seed=42).join(lab, on="pair_id")
    assert f.group_by("table_id").agg(pl.col("fold").n_unique())["fold"].max() == 1
    assert (f.group_by("fold", "behavior_family").len().group_by("fold").len()["len"] == 4).all()


def test_exposure_counts_become_shares_of_the_period():
    pf = pl.DataFrame({"phase": [0, 1], "n_shared": [300, 200], "hands_max": [3000, 1000], "hands_min": [30, 20]},
                      schema_overrides={"phase": pl.Int8})
    out = with_exposure(pf)
    assert out["n_shared_frac"].to_list() == [0.1, 0.1]  # 300 of 3,000 dev hands == 200 of 2,000 eval hands
    win = with_exposure(pf.with_columns(pl.lit(2000).alias("period_hands")))
    assert win["n_shared_frac"].to_list() == [0.15, 0.1]


def test_assign_behavior_argmax_none_and_other_coordination():
    risk = np.array([0.9, 0.8, 0.01, 0.85])  # median 0.825
    probs = np.array([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.4, 0.3, 0.3]])
    assert list(assign_behavior(risk, probs, None, POLICY)) == ["soft_play", "directed_transfer",
                                                               "coordinated_isolation", "directed_transfer"]
    assert assign_behavior(risk, probs, None, POLICY, none_below=0.05)[2] == "none"
    novelty = np.array([50.0, 50.0, 99.5, 99.5])  # the last pair is novel, unsure of family and risky
    assert list(assign_behavior(risk, probs, novelty, POLICY))[2:] == ["coordinated_isolation", "other_coordination"]


def test_top5_ranks_by_score_and_pads_with_no_evidence():
    scored = pl.DataFrame({"pair_id": ["A"] * 7 + ["B"] * 2, "hand_id": [f"h{i}" for i in range(9)],
                           "score": [0.1, 0.9, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]})
    wide = top5(scored).sort("pair_id")
    assert wide.row(0) == ("A", "h1", "h4", "h6", "h2", "h3")
    assert wide.row(1) == ("B", "h8", "h7", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE)


def test_calibration_preserves_ranking():
    rng = np.random.default_rng(1)
    raw = rng.random(500)
    y = (rng.random(500) < raw * 0.2).astype(int)
    cal = calibrate(raw, platt(raw, y))
    assert (np.argsort(cal) == np.argsort(raw)).all() and cal.max() < 1


def test_hand_context_is_computed_within_each_pair():
    hf = pl.DataFrame({"pair_id": ["A", "A", "B"], "block": [0, 0, 0], "h_pot_bb": [1.0, 3.0, 2.0],
                       "h_netflow_bb": [0.0, 1.0, 1.0], "h_contrib_max_bb": [1.0, 1.0, 1.0],
                       "h_pre_pct_max": [0.5, 0.5, 0.5], "h_fold_ahead_max": [1, 0, 0], "h_payoff_max": [0, 0, 0],
                       "h_hu_strong_check": [0, 0, 0], "h_open_step_aside": [False, True, False],
                       "h_call_max": [0, 0, 0]})
    out = hand_context(hf)
    assert out["n_cand"].to_list() == [2, 2, 1]
    assert out["r_h_pot_bb"].to_list() == [0.5, 1.0, 1.0]
    assert out["blk_h_fold_ahead_max"].to_list() == [0.5, 0.5, 0.0]


@pytest.mark.data
@pytest.mark.skipif(not has_raw_data(), reason="competition data not available")
def test_mirror_set_follows_the_evaluation_filter():
    from suspoker.modeling import load_dev
    from suspoker.pipeline import FEATURES_DIR

    path = FEATURES_DIR / "pair_features.parquet"
    if not path.exists():
        pytest.skip("run the feature pipeline first")
    d = load_dev(pl.read_parquet(path))
    unl = d.mirror.filter(~pl.col("labeled"))
    pos = d.labeled.filter(pl.col("label") == 1)
    players = set(pos["player_lo"]) | set(pos["player_hi"])
    assert unl["n_shared"].min() >= 57
    assert not (set(unl["player_lo"]) | set(unl["player_hi"])) & players
    assert d.mirror["pair_id"].n_unique() == d.mirror.height


@pytest.mark.data
@pytest.mark.skipif(not has_raw_data(), reason="competition data not available")
def test_load_dev_does_not_depend_on_input_row_order():
    # polars group_by output order changes between pipeline runs; training samples must not change with it
    from suspoker.modeling import load_dev
    from suspoker.pipeline import FEATURES_DIR

    path = FEATURES_DIR / "pair_features.parquet"
    if not path.exists():
        pytest.skip("run the feature pipeline first")
    pf = pl.read_parquet(path).filter(pl.col("phase") == 0)
    a, b = load_dev(pf), load_dev(pf.sample(fraction=1.0, shuffle=True, seed=1))
    assert a.mirror.equals(b.mirror)
    assert a.mirror.sample(500, seed=3).equals(b.mirror.sample(500, seed=3))
