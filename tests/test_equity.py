import numpy as np
import polars as pl
import pytest

from suspoker.cards import card_to_int as c
from suspoker.equity import class_index, decision_equity, hu_preflop_table


def frames(xcards, ycards, board, street):
    seats = pl.DataFrame({"hand_idx": [1, 1], "player_idx": [10, 20], "c1": [c(xcards[0]), c(ycards[0])],
                          "c2": [c(xcards[1]), c(ycards[1])]})
    b = [c(x) for x in board] + [-1] * (5 - len(board))
    hands = pl.DataFrame({"hand_idx": [1], **{f"b{i + 1}": [b[i]] for i in range(5)}})
    events = pl.DataFrame({"hand_idx": [1], "action_no": [5], "street": [street], "x": [10], "y": [20]},
                          schema_overrides={"street": pl.Int8, "action_no": pl.Int16})
    return events, seats, hands


def test_turn_equity_is_exact():
    # AA vs KK on a blank turn: KK needs one of the 2 remaining kings among 44 unseen rivers
    ev, seats, hands = frames(("Ah", "As"), ("Kh", "Ks"), ["2c", "7d", "9c", "Jd", "3h"], street=2)
    eq = decision_equity(ev, seats, hands, workers=1)["eq"][0]
    assert eq == pytest.approx(42 / 44)


def test_river_equity_is_the_showdown_result():
    ev, seats, hands = frames(("Kh", "Ks"), ("Ah", "As"), ["2c", "7d", "9c", "Jd", "Kd"], street=3)
    assert decision_equity(ev, seats, hands, workers=1)["eq"][0] == 1.0  # set of kings beats aces


def test_preflop_table_is_consistent():
    t = hu_preflop_table()
    aa = class_index(np.array([c("Ah")]), np.array([c("As")]))[0]
    kk = class_index(np.array([c("Kh")]), np.array([c("Ks")]))[0]
    seven_two = class_index(np.array([c("7h")]), np.array([c("2c")]))[0]
    assert t.shape == (169, 169) and np.allclose(t + t.T, 1, atol=1e-6)
    assert 0.79 < t[aa, kk] < 0.85 and t[aa, seven_two] > 0.85
