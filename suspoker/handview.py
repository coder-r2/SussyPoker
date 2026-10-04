"""Plain-text hand histories for EDA, debugging and case reviews."""

import polars as pl

STREET_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def _short(player_id: str) -> str:
    return player_id[:7]


def format_hand(hand_id: str, hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame,
                highlight: dict[str, str] | None = None) -> str:
    """Render one hand. `highlight` maps player_id -> short tag (e.g. {"U1..": "P1"})."""
    highlight = highlight or {}
    h = hands.filter(pl.col("hand_id") == hand_id).row(0, named=True)
    s = seats.filter(pl.col("hand_id") == hand_id).sort("seat_no")
    a = actions.filter(pl.col("hand_id") == hand_id).sort("action_no")
    board = h["board_cards"].split() if h["board_cards"] else []
    bb = h["big_blind"]

    def who(pid: str) -> str:
        tag = highlight.get(pid)
        return f"{_short(pid)}[{tag}]" if tag else _short(pid)

    lines = [
        f"Hand {hand_id} | table {h['table_id']} | {h['phase']} | blinds {h['small_blind']}/{bb} "
        f"| button seat {h['button_seat']} | final pot {h['final_pot']} ({h['final_pot'] / bb:.1f} bb)",
        f"Board: {' '.join(board) or '(none)'}",
        "Seats:",
    ]
    for r in s.iter_rows(named=True):
        result = "WIN" if r["won_share"] > 0 else ("fold" if r["folded"] else "lose")
        sd = " showdown" if r["went_to_showdown"] else ""
        lines.append(
            f"  {r['seat_no']} {who(r['player_id']):<12} {r['hole_card_1']} {r['hole_card_2']}  "
            f"stack {r['starting_stack']:>5}  put in {r['total_contribution']:>5}  "
            f"net {r['net_chips']:>+6} ({r['net_chips'] / bb:+.1f} bb)  {result}{sd}"
        )
    street = None
    for r in a.iter_rows(named=True):
        if r["street"] != street:
            street = r["street"]
            shown = board[: STREET_BOARD[street]]
            lines.append(f"-- {street.upper()} {' '.join(shown)}  (pot {r['pot_before']})")
        if r["action"] in ("fold", "check"):
            what = r["action"]
        elif r["action"] == "call":
            what = f"call {r['amount']}"
        else:
            what = f"{r['action']} to {r['amount_to']} (+{r['amount']})"
        facing = f"  [facing {r['to_call']}]" if r["to_call"] else ""
        lines.append(f"   {who(r['player_id']):<12} {what}{facing}")
    return "\n".join(lines)
