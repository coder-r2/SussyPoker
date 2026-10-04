import polars as pl

from suspoker.handview import format_hand

HANDS = pl.DataFrame({"hand_id": ["H1"], "table_id": ["T1"], "phase": ["development"], "button_seat": [0],
                      "small_blind": [1], "big_blind": [2], "board_cards": ["Ah Kd 2c"], "final_pot": [12]})
SEATS = pl.DataFrame({
    "hand_id": ["H1"] * 3, "player_id": ["UAAAAAAAAAA", "UBBBBBBBBBB", "UCCCCCCCCCC"], "seat_no": [0, 1, 2],
    "starting_stack": [200, 200, 200], "hole_card_1": ["As", "7c", "2d"], "hole_card_2": ["Qs", "6c", "3d"],
    "total_contribution": [6, 6, 0], "net_chips": [6, -6, 0], "folded": [False, True, True],
    "went_to_showdown": [False, False, False], "won_share": [1.0, 0.0, 0.0],
})
ACTIONS = pl.DataFrame({
    "hand_id": ["H1"] * 4, "action_no": [0, 1, 2, 3], "street": ["preflop", "preflop", "flop", "flop"],
    "player_id": ["UCCCCCCCCCC", "UAAAAAAAAAA", "UAAAAAAAAAA", "UBBBBBBBBBB"],
    "action": ["fold", "call", "bet", "fold"], "amount": [0, 2, 4, 0], "amount_to": [0, 2, 4, 0],
    "pot_before": [3, 3, 5, 9], "stack_before": [200, 200, 198, 198], "to_call": [2, 2, 0, 4],
    "players_active": [3, 2, 2, 2],
})


def test_format_hand_renders_streets_board_and_tags():
    text = format_hand("H1", HANDS, SEATS, ACTIONS, highlight={"UAAAAAAAAAA": "P1"})
    assert "blinds 1/2" in text and "final pot 12 (6.0 bb)" in text
    assert "-- PREFLOP" in text and "-- FLOP Ah Kd 2c" in text
    assert "UAAAAAA[P1]" in text
    assert "bet to 4 (+4)" in text and "fold  [facing 4]" in text
    assert "WIN" in text
