import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from suspoker.error_analysis import flag_top_k, outcome_counts, pr_curve
from suspoker.explain import (
    BENIGN,
    benign_alternatives,
    contributions,
    feature_info,
    fit_reference,
    percentile,
    reason_tags,
    reasons,
)
from suspoker.export_web import LeakError, leak_check, write_json
from suspoker.modeling import CROSS_FEATURES, PAIR_FEATURES
from suspoker.suspicion import HS_FEATURES, HS_REL_FEATURES


def _toy_model(objective: str = "binary", n_class: int = 1):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) if n_class == 1 else (X[:, 0] > 0).astype(int) + (X[:, 1] > 0)
    params = dict(objective=objective, n_estimators=30, num_leaves=7, verbose=-1)
    if n_class > 1:
        params["num_class"] = n_class
    models = [lgb.LGBMClassifier(**params, random_state=s).fit(X, y).booster_ for s in (0, 1)]
    return models, X


def test_contributions_match_shap_tree_explainer_and_add_up_to_the_raw_score():
    shap = pytest.importorskip("shap")
    models, X = _toy_model()
    c = contributions(models, X)
    assert c.shape == (400, 5)
    raw = np.mean([m.predict(X, raw_score=True) for m in models], axis=0)
    assert np.allclose(c.sum(axis=1), raw, atol=1e-6)
    ref = np.mean([np.asarray(shap.TreeExplainer(m).shap_values(X)) for m in models], axis=0)
    assert np.allclose(c[:, :4], ref.reshape(400, 4), atol=1e-6)


def test_multiclass_contributions_have_one_block_per_class():
    models, X = _toy_model("multiclass", 3)
    assert contributions(models, X).shape == (400, 3, 5)


def test_reasons_split_for_and_against_with_labels():
    X = np.array([[1.0, 2.0, 3.0]])
    contrib = np.array([[0.5, -0.2, 0.1, -1.0]])  # last column is the base value
    r = reasons(pl.Series(["p"]), contrib, X, ["hs_max", "n_shared_frac", "x_vpip_pres_max"], k=2)
    up = r.filter(pl.col("direction") == "for")
    assert up["feature"].to_list() == ["hs_max", "x_vpip_pres_max"]
    assert r.filter(pl.col("direction") == "against")["feature"].to_list() == ["n_shared_frac"]
    assert up["label"][1].startswith("Other period:")


def test_every_model_input_has_a_plain_english_label():
    for f in [*PAIR_FEATURES, *HS_FEATURES, *HS_REL_FEATURES, *CROSS_FEATURES]:
        fam, label = feature_info(f)
        assert fam != "other" and label != f, f


def _hands(**cols) -> pl.DataFrame:
    base = {c: [0] for c in ("h_ev_gift_bb_max", "h_fold_ahead_max", "h_pre_fold_ahead_max", "h_costly_fold_max",
                             "h_payoff_max", "h_payoff_bb_max", "h_costly_call_max", "h_pre_junk_call_max",
                             "h_hu_strong_check", "h_strong_check_partner", "h_open_step_aside", "h_squeeze_max",
                             "h_weak_open", "r_h_netflow_bb", "h_netflow_bb", "r_h_pot_bb", "h_pot_bb")}
    return pl.DataFrame(base | {k: [v] for k, v in cols.items()})


def test_reason_tags_follow_the_family_and_fall_back_to_contested():
    h = _hands(h_fold_ahead_max=1, h_open_step_aside=1, h_ev_gift_bb_max=2.5, h_weak_open=1, h_pot_bb=40)
    dt = reason_tags(h.with_columns(pl.lit("directed_transfer").alias("fam")), family="fam")[0]
    assert [t["id"] for t in dt] == ["fold_ahead", "ev_gift", "weak_open"]
    assert dt[1]["text"].startswith("Gave up 2.5 bb")
    iso = reason_tags(h.with_columns(pl.lit("coordinated_isolation").alias("fam")), family="fam")[0]
    assert [t["id"] for t in iso] == ["weak_open", "open_step_aside", "ev_gift"]
    assert [t["id"] for t in reason_tags(_hands(h_pot_bb=12))[0]] == ["contested"]
    big = reason_tags(_hands(r_h_pot_bb=0.95, h_pot_bb=80))[0]
    assert [t["id"] for t in big] == ["big_pot"] and "80 bb" in big[0]["text"]


def test_benign_templates_fire_on_extreme_values_only():
    assert len(BENIGN) >= 5
    pop = pl.DataFrame({f: np.linspace(0, 1, 1001) for f in sorted({b.feature for b in BENIGN.values()})})
    ref = fit_reference(pop)
    assert percentile(ref, "af_max", np.array([0.5]))[0] == pytest.approx(50, abs=1)
    typical = {f: 0.5 for f in pop.columns}
    assert benign_alternatives(typical, ref) == []
    fired = benign_alternatives(typical | {"tilt_vpip_max": 0.999, "style_dist": 0.001}, ref)
    assert {x["id"] for x in fired} == {"tilt", "similar_styles"}
    assert all(x["strength"] >= 90 and x["text"] for x in fired)


def test_leak_check_refuses_competition_ids(tmp_path):
    leak_check('{"pair_ap": 0.83, "note": "P3 model", "hash": "PFF"}')
    # made-up ids, assembled at runtime so the repository's own leak scan never matches this file
    hexes = "0123456789ABCDEF"
    fakes = ["P" + hexes[:12], "U" + hexes[4:16], "T" + hexes[8:16], "H" + hexes[1:15]]
    for bad in fakes:
        with pytest.raises(LeakError):
            leak_check(f'{{"x": "{bad}"}}')
    with pytest.raises(LeakError):
        write_json({"pair": fakes[0]}, tmp_path / "card.json")
    assert not (tmp_path / "card.json").exists()


def test_break_even_cut_counts_and_pr_curve():
    p = pl.DataFrame({"pair_id": [f"p{i}" for i in range(6)], "label": [1, 0, 1, 0, 0, 1],
                      "labeled": [True, True, True, False, True, True], "risk_window": [0.9, 0.8, 0.7, 0.6, 0.1, 0.2]})
    f = flag_top_k(p)
    c = outcome_counts(f)
    assert (c["k"], c["TP"], c["FP"], c["FN"]) == (3, 2, 1, 1)
    assert c["fp_confirmed_negative"] == 1 and c["best_confirmed_negative_rank"] == 2
    curve = pr_curve(p["label"].to_numpy(), p["risk_window"].to_numpy(), points=11)
    assert curve["precision"][0] == 1.0 and curve["precision"][-1] == 0.6  # 3 colluders in the top 5
    assert all(a >= b for a, b in zip(curve["precision"], curve["precision"][1:], strict=False))


def test_relative_hand_suspicion_compares_with_each_players_other_pairs():
    from suspoker.suspicion import relative

    # player "s" (a calling station) looks suspicious with everyone; pair (a, b) is suspicious only together
    hs = pl.DataFrame({"player_lo": ["a", "a", "b", "c", "c", "c"], "player_hi": ["b", "x", "y", "s", "t", "u"],
                       "hs_max": [0.9, 0.1, 0.1, 0.9, 0.1, 0.1], "hs_top3": [0.8, 0.1, 0.1, 0.8, 0.1, 0.1],
                       "hs_top5": 0.0, "hs_n50": 0.0, "hs_frac": 0.0})
    hs = pl.concat([hs, pl.DataFrame({"player_lo": ["s", "s"], "player_hi": ["v", "w"], "hs_max": [0.9, 0.9],
                                      "hs_top3": [0.8, 0.8], "hs_top5": 0.0, "hs_n50": 0.0, "hs_frac": 0.0})])
    r = {(a, b): x for a, b, x in relative(hs).select("player_lo", "player_hi", "hs_top3_rel").iter_rows()}
    assert r[("a", "b")] == pytest.approx(0.7)       # 0.8 vs a baseline of 0.1 for both players
    assert r[("c", "s")] == pytest.approx(0.0)       # 0.8 vs s's other pairs, which are all 0.8
    assert relative(hs)["hs_base_max"].null_count() == 0
