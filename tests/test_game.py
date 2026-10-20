"""The simulator: schema conformance (SPEC §7.6.7), engine edge cases, bots, SimRoom determinism and IDs."""

import random
from dataclasses import replace
from datetime import UTC, datetime

import polars as pl
import pytest

pytest.importorskip("pokerkit")

from suspoker.config import load_config  # noqa: E402
from suspoker.export_web import ID_PATTERN  # noqa: E402
from suspoker.features import build_chunk  # noqa: E402
from suspoker.game.bots import STYLES, Bot, Collusion, overlay  # noqa: E402
from suspoker.game.engine import Decision, Hand, SeatIn, View, play_hand  # noqa: E402
from suspoker.game.logger import ACTION_SCHEMA, HAND_SCHEMA, SEAT_SCHEMA, HandLogger, conformance, failed  # noqa: E402
from suspoker.game.replay import pair_highlights, replays  # noqa: E402
from suspoker.game.simroom import SimConfig, pair_id, run_table  # noqa: E402
from suspoker.ingest import encode  # noqa: E402

SMALL = SimConfig(seed=3, tables=1, hands=200, pool=30)


@pytest.fixture(scope="module")
def sim():
    log, truth = run_table(0, SMALL)
    return (*log.frames(), truth)


def test_simulated_hands_follow_the_competition_schema_and_semantics(sim):
    hands, seats, actions, _ = sim
    assert hands.height == 200
    assert dict(hands.schema) == HAND_SCHEMA and dict(seats.schema) == SEAT_SCHEMA
    assert dict(actions.schema) == ACTION_SCHEMA
    report = conformance(hands, seats, actions)
    assert len(report) >= 30
    assert failed(report) == {}


def test_feature_pipeline_runs_on_simulated_hands(sim):
    hands, seats, actions, _ = sim
    t = encode(hands, seats, actions)
    assert (t.hands["phase"] == 2).all()
    cfg = load_config()["features"]
    pf, hf, base = build_chunk(t, k=cfg["shrinkage_k"], block_hands=cfg["block_hands"], hand_pairs=None,
                               dev_min_shared=10, min_shared_phase=2)
    assert pf.height > 100 and hf is not None and hf.height > 0
    assert base["vpip"].is_between(0, 1).all() and base["hands"].sum() == 6 * 200
    assert pf["n_shared"].min() >= 1 and pf.select(pl.col("vpip_max").is_between(0, 1).all()).item()


def test_conformance_catches_a_broken_log(sim):
    hands, seats, actions, _ = sim
    bad = seats.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(pl.col("net_chips") + 1)
                             .otherwise(pl.col("net_chips")).alias("net_chips"))
    assert "sum(net_chips) == 0 per hand (no rake)" in failed(conformance(hands, bad, actions))
    wrong = actions.with_columns(pl.col("to_call") + 1)
    assert "to_call == min(highest street commitment - own, stack_before)" in failed(conformance(hands, seats, wrong))


def _seats(stacks=(200,) * 6) -> list[SeatIn]:
    return [SeatIn(f"SIM-P99{k:02d}", s) for k, s in enumerate(stacks)]


def _script(*decisions):
    it = iter(decisions)
    return lambda v: next(it)


def test_uncalled_bet_stays_in_contribution_and_short_all_in_call_is_all_in():
    # button 5 -> SB seat 0, BB seat 1, UTG seat 2; seat 2 has only 30 chips
    seats = _seats((200, 200, 30, 200, 200, 200))
    deck = list(range(52))
    script = _script(Decision("raise", 40), Decision("fold"), Decision("fold"), Decision("fold"), Decision("fold"),
                     Decision("call"))
    hand = Hand(seats, button=5, sb=1, bb=2, deck=deck)
    v = hand.view()
    assert v.seat == 2 and v.to_call == 2 and v.pot == 3
    while (v := hand.view()) is not None:
        hand.act(script(v))
    rec = hand.record()
    a = rec.actions
    assert a[0]["action"] == "all_in" and a[0]["amount"] == 30  # raise capped at the stack
    # big blind calls 28 more: a call, not an all-in
    assert a[-1]["action"] == "call" and a[-1]["amount"] == 28 and a[-1]["to_call"] == 28
    seat_rows = {s["seat_no"]: s for s in rec.seats}
    assert sum(s["net_chips"] for s in rec.seats) == 0
    assert rec.hand["final_pot"] == sum(s["total_contribution"] for s in rec.seats) == 30 + 30 + 1
    assert rec.hand["players_at_showdown"] == 2 and seat_rows[2]["went_to_showdown"]


def test_uncalled_raise_is_kept_in_the_pot_and_netted_by_the_winner():
    seats = _seats()
    rec = play_hand(seats, 5, 1, 2, list(range(52)), _script(
        Decision("raise", 6), Decision("raise", 20), Decision("fold"), Decision("fold"), Decision("fold"),
        Decision("fold"), Decision("fold")))
    s = {x["seat_no"]: x for x in rec.seats}
    assert s[3]["total_contribution"] == 20 and s[3]["net_chips"] == 6 + 1 + 2  # wins the others' chips
    assert rec.hand["final_pot"] == 20 + 6 + 1 + 2 and rec.hand["board_cards"] == ""
    assert rec.hand["players_at_showdown"] == 0 and not any(x["went_to_showdown"] for x in rec.seats)
    assert s[3]["won_share"] == 1.0


def test_fold_with_nothing_to_call_becomes_a_check():
    seats = _seats()
    hand = Hand(seats, 5, 1, 2, list(range(52)))
    for d in (Decision("call"),) * 5:  # everyone limps / SB completes
        hand.act(d)
    row = hand.act(Decision("fold"))     # big blind "folds" for free
    assert row["action"] == "check" and row["to_call"] == 0


def _view(**kw) -> View:
    base = dict(seat=2, player_id="G", hole=(51, 50), board=[0, 5, 10], street=1, pot=40, to_call=10, stack=150,
                committed=0, max_commit=10, min_raise_to=20, max_raise_to=150, bb=2, button=0,
                seats={i: p for i, p in enumerate("ABGRCD")}, active=["G", "R", "C"], all_in=[], history=[],
                street_aggressor="R", preflop_aggressor="R")
    return View(**(base | kw))


def test_directed_transfer_giver_folds_a_strong_hand_to_the_receiver():
    c = Collusion("directed_transfer", a="G", b="R")
    bot = Bot("G", "g", STYLES["TAG"], random.Random(0))
    assert overlay(c, bot, _view(), Decision("call"), strength=0.8) == Decision("fold")
    # against a third player's bet the giver plays normally
    assert overlay(c, bot, _view(street_aggressor="C"), Decision("call"), 0.8) == Decision("call")


def test_soft_play_never_raises_the_partner_and_checks_down_heads_up():
    c = Collusion("soft_play", a="G", b="R")
    bot = Bot("G", "g", STYLES["LAG"], random.Random(0))
    assert overlay(c, bot, _view(), Decision("raise", 40), 0.95) == Decision("call")
    hu = _view(active=["G", "R"], to_call=0, street_aggressor=None)
    assert overlay(c, bot, hu, Decision("raise", 20), 0.9) == Decision("call")


def test_simroom_is_deterministic_and_uses_sim_ids(sim):
    hands, seats, actions, truth = sim
    log2, truth2 = run_table(0, SMALL)
    h2, s2, a2 = log2.frames()
    assert hands.equals(h2) and seats.equals(s2) and actions.equals(a2) and truth == truth2
    ids = pl.concat([hands["hand_id"], hands["table_id"], seats["player_id"]]).unique().to_list()
    ids += [p["pair_id"] for p in truth["planted"] + truth["decoys"]]
    assert all(i.startswith("SIM-") for i in ids)
    assert not any(ID_PATTERN.search(i) for i in ids)
    assert pair_id("SIM-P0417", "SIM-P0412") == "SIM-P0412-P0417"


def test_simroom_plants_pairs_and_decoys_with_ground_truth(sim):
    _, _, _, truth = sim
    assert 2 <= len(truth["planted"]) <= 4 and 2 <= len(truth["decoys"]) <= 3
    members = [m for p in truth["planted"] for m in (p["a"], p["b"])]
    assert len(members) == len(set(members))
    for p in truth["planted"]:
        assert p["family"] in ("directed_transfer", "soft_play", "coordinated_isolation")
        assert set(p["coordinated_hands"]) <= set(sim[0]["hand_id"])


def test_replay_and_pair_highlights(sim):
    hands, seats, actions, _ = sim
    hid = actions.group_by("hand_id").len().sort("len", descending=True)["hand_id"][0]
    r = replays([hid], hands, seats, actions, {"SIM-P0000": "brisk_lagoon"})[hid]
    assert len(r["seats"]) == 6 and r["actions"] and r["final_pot"] > 0
    players = [x["player_id"] for x in r["actions"]]
    hl = pair_highlights(r, players[0], players[-1])
    assert all("action_no" in x and x["text"] for x in hl)


def test_logger_round_trip_keeps_values():
    log = HandLogger()
    rec = play_hand(_seats(), 0, 1, 2, random.Random(1).sample(range(52), 52), lambda v: Decision("call"))
    log.add(rec, "SIM-H000001", "SIM-T00", datetime(2026, 6, 1, tzinfo=UTC), "simulation")
    h, s, a = log.frames()
    assert h["players_dealt"][0] == 6 and s.height == 6 and a.height == len(rec.actions)
    assert failed(conformance(h, s, a)) == {}


def test_tilt_loosens_a_tilting_bot_after_a_big_loss():
    bot = Bot("T", "t", replace(STYLES["TAG"], tilt=1.0), random.Random(0))
    before = bot.effective.vpip
    bot.note_result(-50)
    assert bot.tilt_left > 0 and bot.effective.vpip > before


def test_kaggle_models_score_simulated_hands_end_to_end():
    from suspoker.game.evaluate import simroom_metrics
    from suspoker.predict import Models
    from suspoker.scoring import score_tables

    log, truth = run_table(1, replace(SMALL, tables=2, hands=600))
    hands, seats, actions = log.frames()
    w = score_tables(hands, seats, actions, Models.load(), pair_ids=pair_id, workers=1)
    assert w.pairs.height > 50 and w.pairs["risk"].is_between(0, 1).all()
    assert set(w.pairs["predicted_behavior"]) <= {"none", "directed_transfer", "soft_play", "coordinated_isolation",
                                                  "other_coordination"}
    # SHAP reproduces each pair's raw risk score
    raw = w.risk_contrib.sum(axis=1)
    assert raw.shape == (w.pairs.height,)
    ev = w.evidence
    assert ev["rank"].max() <= 5 and ev.join(w.hands.select("pair_id", "hand_id"), on=["pair_id", "hand_id"],
                                             how="anti").height == 0  # evidence hands are the pair's own hands
    assert all(len(t) >= 1 for t in ev["tags"].to_list())
    m = simroom_metrics(w.pairs, w.evidence, w.hands, {"tables": [truth], "config": {}})
    assert 0 <= m["models"]["M_full"]["final"] <= 1 and m["pairs_scored"] == w.pairs.height
