"""Tiny hand-crafted hands in the competition schema, with known right answers."""

from datetime import UTC, datetime, timedelta

import polars as pl

T0 = datetime(2026, 1, 1, tzinfo=UTC)

HAND_COLS = ["hand_id", "table_id", "started_at", "phase", "button_seat", "small_blind", "big_blind",
             "board_cards", "final_pot", "players_dealt", "players_at_showdown"]
SEAT_COLS = ["hand_id", "player_id", "seat_no", "starting_stack", "hole_card_1", "hole_card_2",
             "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share"]
ACTION_COLS = ["hand_id", "action_no", "street", "player_id", "action", "amount", "amount_to", "pot_before",
               "stack_before", "to_call", "players_active"]


def two_hands() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Hand 1 (bb=2; A button seat 0, B small blind seat 1, C big blind seat 2):
      preflop: A raises to 6; B calls 5; C folds.
      flop Ah Kd 2c: B checks; A bets 6; B folds while holding a pair of aces vs A's 7-high
      -> B "folded while ahead" of A. A wins 20 (B put in 6, C 2).
    Hand 2 (bb=2, one minute later): A raises to 8; B (7 chips behind after the SB) goes all-in for 7,
      which does not exceed to_call -> an all-in *call*, not aggression. C folds. B wins at showdown.
    """
    hands = pl.DataFrame([
        ("H1", "T1", T0, "development", 0, 1, 2, "Ah Kd 2c", 20, 3, 0),
        ("H2", "T1", T0 + timedelta(minutes=1), "evaluation", 0, 1, 2, "2s 3s 4d 9h Jc", 18, 3, 2),
    ], schema=HAND_COLS, orient="row")
    seats = pl.DataFrame([
        ("H1", "UA", 0, 200, "7c", "6c", 12, 8, False, False, 1.0),
        ("H1", "UB", 1, 200, "As", "Qs", 6, -6, True, False, 0.0),
        ("H1", "UC", 2, 200, "9d", "3h", 2, -2, True, False, 0.0),
        ("H2", "UA", 0, 200, "Kc", "Qc", 8, -8, False, True, 0.0),
        ("H2", "UB", 1, 8, "Ad", "Ah", 8, 10, False, True, 1.0),
        ("H2", "UC", 2, 200, "5h", "6h", 2, -2, True, False, 0.0),
    ], schema=SEAT_COLS, orient="row")
    actions = pl.DataFrame([
        ("H1", 0, "preflop", "UA", "raise", 6, 6, 3, 200, 2, 3),
        ("H1", 1, "preflop", "UB", "call", 5, 6, 9, 199, 5, 3),
        ("H1", 2, "preflop", "UC", "fold", 0, 2, 14, 198, 4, 3),
        ("H1", 3, "flop", "UB", "check", 0, 0, 14, 194, 0, 2),
        ("H1", 4, "flop", "UA", "bet", 6, 6, 14, 194, 0, 2),
        ("H1", 5, "flop", "UB", "fold", 0, 0, 20, 194, 6, 2),
        ("H2", 0, "preflop", "UA", "raise", 8, 8, 3, 200, 2, 3),
        ("H2", 1, "preflop", "UB", "all_in", 7, 8, 11, 7, 7, 3),
        ("H2", 2, "preflop", "UC", "fold", 0, 2, 18, 198, 6, 3),
    ], schema=ACTION_COLS, orient="row")
    return hands, seats, actions
