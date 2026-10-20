"""Hand logger: finished hands -> `hands`, `seats` and `actions` tables in the exact competition schema.

The schema and the column semantics are binding (SPEC §7.6.7, docs/data_semantics.md): same names, same dtypes,
same action vocabulary, same amount rules. `conformance()` re-checks the invariants of scripts/check_semantics.py
on any tables, so simulated hands are held to the same rules the competition data was shown to follow.
"""

from datetime import datetime

import polars as pl

from suspoker.cards import CARD_INDEX
from suspoker.game.engine import N_SEATS, HandRecord

UTC_US = pl.Datetime("us", "UTC")
HAND_SCHEMA = {"hand_id": pl.String, "table_id": pl.String, "started_at": UTC_US, "phase": pl.String,
               "button_seat": pl.Int64, "small_blind": pl.Int64, "big_blind": pl.Int64, "board_cards": pl.String,
               "final_pot": pl.Int64, "players_dealt": pl.Int64, "players_at_showdown": pl.Int64}
SEAT_SCHEMA = {"hand_id": pl.String, "player_id": pl.String, "seat_no": pl.Int64, "starting_stack": pl.Int64,
               "hole_card_1": pl.String, "hole_card_2": pl.String, "total_contribution": pl.Int64,
               "net_chips": pl.Int64, "folded": pl.Boolean, "went_to_showdown": pl.Boolean, "won_share": pl.Float64}
ACTION_SCHEMA = {"hand_id": pl.String, "action_no": pl.Int64, "street": pl.String, "player_id": pl.String,
                 "action": pl.String, "amount": pl.Int64, "amount_to": pl.Int64, "pot_before": pl.Int64,
                 "stack_before": pl.Int64, "to_call": pl.Int64, "players_active": pl.Int64}
ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
STREETS = ("preflop", "flop", "turn", "river")


class HandLogger:
    """Collects finished hands; `frames()` returns the three competition-schema tables."""

    def __init__(self) -> None:
        self.hands: list[dict] = []
        self.seats: list[dict] = []
        self.actions: list[dict] = []

    def add(self, rec: HandRecord, hand_id: str, table_id: str, started_at: datetime, phase: str) -> None:
        self.hands.append({"hand_id": hand_id, "table_id": table_id, "started_at": started_at, "phase": phase,
                           **rec.hand})
        self.seats += [{"hand_id": hand_id, **s} for s in rec.seats]
        self.actions += [{"hand_id": hand_id, **a} for a in rec.actions]

    def __len__(self) -> int:
        return len(self.hands)

    def frames(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        return (pl.DataFrame(self.hands, schema=HAND_SCHEMA, orient="row"),
                pl.DataFrame(self.seats, schema=SEAT_SCHEMA, orient="row"),
                pl.DataFrame(self.actions, schema=ACTION_SCHEMA, orient="row"))


def conformance(hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame) -> dict[str, float]:
    """Share of rows (or hands) satisfying each invariant of the competition data. All must be 1.0."""
    out: dict[str, float] = {}

    def check(name: str, ok: pl.Series | bool) -> None:
        out[name] = float(ok) if isinstance(ok, bool) else (float(ok.cast(pl.Float64).mean()) if len(ok) else 1.0)

    for df, schema, name in ((hands, HAND_SCHEMA, "hands"), (seats, SEAT_SCHEMA, "seats"),
                             (actions, ACTION_SCHEMA, "actions")):
        check(f"{name}: column names and dtypes", dict(df.schema) == schema)
    check("action vocabulary", actions["action"].is_in(ACTIONS))
    check("street vocabulary", actions["street"].is_in(STREETS))
    check("hand_id unique", hands["hand_id"].n_unique() == hands.height)

    per_hand = seats.group_by("hand_id").agg(
        pl.col("net_chips").sum().alias("net_sum"), pl.col("total_contribution").sum().alias("contrib_sum"),
        pl.col("won_share").sum().alias("share_sum"), pl.len().alias("n_seats"),
        pl.col("went_to_showdown").sum().alias("n_sd"), pl.col("seat_no").n_unique().alias("n_seat_no"),
    ).join(hands, on="hand_id")
    check("sum(net_chips) == 0 per hand (no rake)", per_hand["net_sum"] == 0)
    check("final_pot == sum(total_contribution)", per_hand["final_pot"] == per_hand["contrib_sum"])
    check("sum(won_share) == 1 (rounding as in the data)", (per_hand["share_sum"] - 1).abs() < 1e-5)
    check("six seat rows, seats 0-5, players_dealt == 6",
          (per_hand["n_seats"] == N_SEATS) & (per_hand["n_seat_no"] == N_SEATS) & (per_hand["players_dealt"] == 6))
    check("players_at_showdown == count(went_to_showdown)", per_hand["n_sd"] == per_hand["players_at_showdown"])
    check("players_at_showdown is 0 or >= 2", per_hand["players_at_showdown"] != 1)
    check("folded players never went to showdown", ~(seats["folded"] & seats["went_to_showdown"]))
    check("won_share > 0 only for non-folded players", ~seats.filter(pl.col("won_share") > 0)["folded"])
    s = seats.join(hands.select("hand_id", "final_pot"), on="hand_id")
    check("won_share * final_pot - total_contribution == net_chips",
          (s["won_share"] * s["final_pot"] - s["total_contribution"] - s["net_chips"]).abs() < 0.01)

    # cards: valid, and no card twice in a hand (hole cards + board)
    board = hands.select("hand_id", pl.col("board_cards").str.split(" ").list.eval(pl.element().filter(
        pl.element() != "")).alias("board"))
    check("board has 0, 3, 4 or 5 cards", board["board"].list.len().is_in([0, 3, 4, 5]))
    dealt = board.explode("board", empty_as_null=True).drop_nulls("board")
    cards = pl.concat([seats.select("hand_id", pl.col("hole_card_1").alias("c")),
                       seats.select("hand_id", pl.col("hole_card_2").alias("c")),
                       dealt.select("hand_id", pl.col("board").alias("c"))])
    check("cards are valid", cards["c"].is_in(list(CARD_INDEX)))
    check("no card dealt twice in a hand", cards.group_by("hand_id").agg(pl.col("c").is_duplicated().any())["c"].not_())

    # blinds are not actions: contribution - sum(amounts) is the posted blind, SB at button+1, BB at button+2
    act_sum = actions.group_by("hand_id", "player_id").agg(pl.col("amount").sum().alias("act_sum"))
    blinds = (seats.join(act_sum, on=["hand_id", "player_id"], how="left").with_columns(pl.col("act_sum").fill_null(0))
              .join(hands.select("hand_id", "small_blind", "big_blind", "button_seat"), on="hand_id")
              .with_columns((pl.col("total_contribution") - pl.col("act_sum")).alias("posted"),
                            ((pl.col("seat_no") - pl.col("button_seat")) % N_SEATS).alias("offset")))
    expected_blind = (pl.when(pl.col("offset") == 1).then(pl.col("small_blind"))
                      .when(pl.col("offset") == 2).then(pl.col("big_blind")).otherwise(0))
    check("posted blind: SB at button+1, BB at button+2, nothing else", blinds["posted"] == blinds.select(
        expected_blind)["small_blind"])

    a = actions.sort("hand_id", "action_no").with_columns(
        pl.col("pot_before").shift(-1).over("hand_id").alias("next_pot"),
        (pl.int_range(pl.len()).over("hand_id")).alias("idx"))
    chain = a.filter(pl.col("next_pot").is_not_null())
    check("pot_before[i+1] == pot_before[i] + amount[i]", chain["next_pot"] == chain["pot_before"] + chain["amount"])
    first = a.filter(pl.col("idx") == 0).join(hands, on="hand_id")
    check("first pot_before == SB + BB", first["pot_before"] == first["small_blind"] + first["big_blind"])
    check("first actor is UTG (button + 3)", first.join(seats.select("hand_id", "player_id", "seat_no"),
                                                         on=["hand_id", "player_id"]).select(
        pl.col("seat_no") == (pl.col("button_seat") + 3) % N_SEATS).to_series())
    check("action_no is 0..n-1 contiguous", a["action_no"] == a["idx"])
    last = a.group_by("hand_id").agg(pl.col("pot_before").last() + pl.col("amount").last()).join(
        hands.select("hand_id", "final_pot"), on="hand_id")
    check("last pot_before + amount == final_pot", last["pot_before"] == last["final_pot"])

    a = a.join(blinds.select("hand_id", "player_id", "posted", "starting_stack"), on=["hand_id", "player_id"]).sort(
        "hand_id", "action_no")  # window expressions below follow action order
    a = a.with_columns(
        (pl.col("amount").cum_sum().over("hand_id", "player_id") - pl.col("amount")).alias("prior_amt"),
        (pl.col("amount").cum_sum().over("hand_id", "player_id", "street") - pl.col("amount")).alias("prior_street"),
    ).with_columns(pl.when(pl.col("street") == "preflop").then(pl.col("posted")).otherwise(0).alias("street_blind"))
    check("stack_before == starting_stack - posted - prior amounts",
          a["stack_before"] == a["starting_stack"] - a["posted"] - a["prior_amt"])
    check("amount_to == street commitment after the action",
          a["amount_to"] == a["street_blind"] + a["prior_street"] + a["amount"])
    for act in ("fold", "check"):
        check(f"{act}: amount == 0", a.filter(pl.col("action") == act)["amount"] == 0)
    calls = a.filter(pl.col("action") == "call")
    check("call: 0 < amount == to_call < stack_before",
          (calls["amount"] == calls["to_call"]) & (calls["amount"] > 0) & (calls["amount"] < calls["stack_before"]))
    check("check: to_call == 0", a.filter(pl.col("action") == "check")["to_call"] == 0)
    alls = a.filter(pl.col("action") == "all_in")
    check("all_in: amount == stack_before", alls["amount"] == alls["stack_before"])
    check("bet only postflop, with nothing to call", a.filter(pl.col("action") == "bet").select(
        (pl.col("street") != "preflop") & (pl.col("to_call") == 0)).to_series())
    # highest commitment before the action on this street (preflop it starts at the big blind)
    earlier = pl.col("amount_to").cum_max().shift(1).over("hand_id", "street").fill_null(0)
    a = a.join(hands.select("hand_id", "big_blind"), on="hand_id").sort("hand_id", "action_no").with_columns(
        (pl.col("street_blind") + pl.col("prior_street")).alias("own"),
        pl.when(pl.col("street") == "preflop").then(pl.max_horizontal(earlier, pl.col("big_blind")))
        .otherwise(earlier).alias("hi"))
    check("to_call == min(highest street commitment - own, stack_before)",
          a.select(pl.col("to_call") == pl.min_horizontal(pl.col("hi") - pl.col("own"), "stack_before")).to_series())
    a = a.with_columns((pl.col("action") == "fold").cast(pl.Int64).cum_sum().over("hand_id").shift(1).over(
        "hand_id").fill_null(0).alias("folds_before"))
    check("players_active == 6 - folds before", a["players_active"] == N_SEATS - a["folds_before"])
    return out


def failed(report: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in report.items() if v < 1.0}
