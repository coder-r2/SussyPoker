import polars as pl
import pytest

from suspoker.ingest import encode
from suspoker.interactions import (
    enrich_actions,
    flows,
    hand_context,
    hand_players,
    hu_streets,
    response_events,
    seat_pairs,
    street_ranks,
)
from tests.fixtures import two_hands


@pytest.fixture(scope="module")
def built():
    t = encode(*two_hands())
    pid = dict(zip(t.player_ids["player_id"], t.player_ids["player_idx"], strict=True))
    ctx = hand_context(t, block_hands=500)
    hp = hand_players(t, ctx)
    ranks = street_ranks(t)
    ea = enrich_actions(t, ctx, hp, ranks)
    return t, pid, ctx, hp, ranks, ea


def test_encode_orders_hands_by_time_and_maps_back(built):
    t, *_ = built
    assert t.hand_ids.sort("hand_idx")["hand_id"].to_list() == ["H1", "H2"]
    assert t.hands["phase"].to_list() == [0, 1]
    assert t.hands.filter(pl.col("hand_idx") == 0)[["b1", "b4"]].row(0) == (12 * 4 + 2, -1)  # Ah, no turn


def test_vpip_and_preflop_strength(built):
    _, pid, _, hp, *_ = built
    h1 = hp.filter(pl.col("hand_idx") == 0)
    vpip = dict(zip(h1["player_idx"], h1["vpip"], strict=True))
    assert vpip[pid["UA"]] and vpip[pid["UB"]] and not vpip[pid["UC"]]  # C only posted the blind
    pre = dict(zip(h1["player_idx"], h1["pre_pct"], strict=True))
    assert pre[pid["UB"]] > 0.9 > pre[pid["UA"]]  # AQs is a top hand, 76s is not


def test_ranks_only_for_postflop_actors(built):
    _, pid, _, _, ranks, _ = built
    flop = ranks.filter(pl.col("hand_idx") == 0)
    r = dict(zip(flop["player_idx"], flop["rank"], strict=True))
    assert set(r) == {pid["UA"], pid["UB"]}
    assert r[pid["UB"]] < r[pid["UA"]]  # pair of aces beats seven-high (lower rank = better)


def test_last_aggressor_and_responses(built):
    _, pid, _, _, _, ea = built
    ev = response_events(ea)
    h1 = ev.filter(pl.col("hand_idx") == 0).sort("street", "x")
    rows = {(r["street"], r["x"]): r for r in h1.iter_rows(named=True)}
    # preflop: B calls A's raise, C folds to it -> both respond to A
    assert rows[(0, pid["UB"])]["y"] == pid["UA"] and rows[(0, pid["UB"])]["call"]
    assert rows[(0, pid["UC"])]["fold"]
    # flop: B folds to A's bet while ahead
    flop = rows[(1, pid["UB"])]
    assert flop["y"] == pid["UA"] and flop["fold"] and flop["ahead"] and flop["fold_ahead"]
    # B's flop check was unopened, not a response
    assert ea.filter((pl.col("hand_idx") == 0) & (pl.col("street") == 1) & (pl.col("action_no") == 3))["unopened"][0]


def test_all_in_that_only_calls_is_not_aggression(built):
    _, pid, _, _, _, ea = built
    b = ea.filter((pl.col("hand_idx") == 1) & (pl.col("player_idx") == pid["UB"])).row(0, named=True)
    assert not b["aggressive"] and b["resp"] == 1
    c = ea.filter((pl.col("hand_idx") == 1) & (pl.col("player_idx") == pid["UC"])).row(0, named=True)
    assert c["last_aggr"] == pid["UA"]  # C still faces A's raise, not B's all-in call


def test_pairwise_flows(built):
    t, pid, ctx, *_ = built
    f = flows(t, ctx).filter(pl.col("hand_idx") == 0)
    got = {(r["a"], r["b"]): r["flow_bb"] for r in f.iter_rows(named=True)}
    assert got == {(pid["UB"], pid["UA"]): 3.0, (pid["UC"], pid["UA"]): 1.0}


def test_heads_up_street(built):
    _, pid, _, _, _, ea = built
    hu = hu_streets(ea)
    assert hu.height == 1
    row = hu.row(0, named=True)
    assert {row["lo"], row["hi"]} == {pid["UA"], pid["UB"]} and row["bet_made"] and not row["checked_through"]


def test_seat_pairs(built):
    _, _, _, hp, *_ = built
    sp = seat_pairs(hp)
    assert sp.height == 6  # 3 pairs per hand x 2 hands
    # seats 0-1 and 1-2 are neighbours; 0-2 is not (six seats around the table, 5 wraps to 0)
    assert sp["adjacent"].sum() == 4


def test_preflop_fold_ahead_and_junk_call_use_hole_card_edge():
    from suspoker.interactions import response_events

    base = {"hand_idx": 1, "table_idx": 0, "phase": 0, "block": 0, "street": 0, "facing": True, "last_aggr": 9,
            "ahead": False, "behind": False, "strong": False, "amount_bb": 1.0, "players_active": 3,
            "aggr_rank": None, "rank": None, "action_no": 3, "pot_before": 30, "to_call": 10, "bb": 2, "eq": None}
    rows = [
        {**base, "player_idx": 1, "resp": 0, "pre_pct": 0.80, "aggr_pre_pct": 0.40},  # folds a much better hand
        {**base, "player_idx": 2, "resp": 0, "pre_pct": 0.45, "aggr_pre_pct": 0.40},  # folds, only slightly ahead
        {**base, "player_idx": 3, "resp": 1, "pre_pct": 0.10, "aggr_pre_pct": 0.60},  # calls with junk, behind
        {**base, "player_idx": 4, "resp": 1, "pre_pct": 0.90, "aggr_pre_pct": 0.60},  # calls with a strong hand
    ]
    ev = response_events(pl.DataFrame(rows, schema_overrides={"aggr_rank": pl.Int32, "rank": pl.Int32,
                                                              "eq": pl.Float64}))
    ev = ev.sort("x")
    assert ev["pre_fold_ahead"].to_list() == [True, False, False, False]
    assert ev["pre_junk_call"].to_list() == [False, False, True, False]
    assert ev["edge"].to_list() == pytest.approx([0.40, 0.05, -0.50, 0.30])


def test_expected_value_cost_of_folds_and_calls():
    from suspoker.interactions import response_events

    base = {"hand_idx": 1, "table_idx": 0, "phase": 0, "block": 0, "street": 2, "facing": True, "last_aggr": 9,
            "ahead": False, "behind": False, "strong": False, "amount_bb": 5.0, "players_active": 2,
            "aggr_rank": 100, "rank": 200, "pre_pct": 0.5, "aggr_pre_pct": 0.5, "action_no": 7,
            "pot_before": 30, "to_call": 10, "bb": 2}
    rows = [
        {**base, "player_idx": 1, "resp": 0, "eq": 0.8},  # folds an 80% hand: calling was worth 0.8*40-10 = 22
        {**base, "player_idx": 2, "resp": 1, "eq": 0.1},  # calls with 10%: 0.1*40-10 = -6 chips
        {**base, "player_idx": 3, "resp": 0, "eq": 0.2},  # folds a hand that should fold: no cost
        {**base, "player_idx": 4, "resp": 1, "eq": None},  # equity unknown: no cost, no flag
    ]
    ev = response_events(pl.DataFrame(rows, schema_overrides={"aggr_rank": pl.Int32, "rank": pl.Int32,
                                                              "eq": pl.Float64})).sort("x")
    assert ev["fold_cost_bb"].to_list() == pytest.approx([11.0, 0.0, 0.0, 0.0])
    assert ev["call_cost_bb"].to_list() == pytest.approx([0.0, 3.0, 0.0, 0.0])
    assert ev["costly_fold"].to_list() == [True, False, False, False]
    assert ev["costly_call"].to_list() == [False, True, False, False]
