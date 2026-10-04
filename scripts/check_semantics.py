"""Verify the column semantics of the competition data via invariants (OPEN-5).

Each check prints the share of rows/hands that satisfy a hypothesis. The
conclusions are written up in docs/data_semantics.md.

Usage: python scripts/check_semantics.py [--hands N]
"""

import argparse

import polars as pl

from suspoker.config import RAW_DIR


def main(n_hands: int | None) -> None:
    hands = pl.scan_parquet(RAW_DIR / "hands.parquet")
    if n_hands:
        hands = hands.head(n_hands)
    hands = hands.collect()
    ids = hands.select("hand_id")
    seats = pl.scan_parquet(RAW_DIR / "seats.parquet").join(ids.lazy(), on="hand_id").collect()
    actions = (
        pl.scan_parquet(RAW_DIR / "actions.parquet")
        .join(ids.lazy(), on="hand_id")
        .sort("hand_id", "action_no")
        .collect()
    )
    print(f"hands={hands.height:,} seats={seats.height:,} actions={actions.height:,}\n")

    def report(name: str, ok: pl.Series) -> None:
        print(f"{ok.mean():8.4%}  {name}")

    # Hand-level conservation.
    per_hand = seats.group_by("hand_id").agg(
        pl.col("net_chips").sum().alias("net_sum"),
        pl.col("total_contribution").sum().alias("contrib_sum"),
        pl.col("won_share").sum().alias("share_sum"),
        pl.len().alias("n_seats"),
    ).join(hands, on="hand_id")
    report("sum(net_chips) == 0 per hand (no rake)", per_hand["net_sum"] == 0)
    report("final_pot == sum(total_contribution)", per_hand["final_pot"] == per_hand["contrib_sum"])
    report("sum(won_share) == 1", (per_hand["share_sum"] - 1).abs() < 1e-9)
    odd = per_hand.filter((pl.col("share_sum") - 1).abs() >= 1e-9)
    print("  hands where sum(won_share) != 1:", odd.height, "examples:", odd["share_sum"].head(5).to_list())
    report("players_dealt == number of seat rows", per_hand["players_dealt"] == per_hand["n_seats"])
    print("players_dealt distribution:", hands["players_dealt"].value_counts().sort("players_dealt").rows())

    # Blinds are not actions: contribution - sum(action amounts) is the posted blind.
    act_sum = actions.group_by("hand_id", "player_id").agg(pl.col("amount").sum().alias("act_sum"))
    blinds = (
        seats.join(act_sum, on=["hand_id", "player_id"], how="left")
        .with_columns(pl.col("act_sum").fill_null(0))
        .join(hands.select("hand_id", "small_blind", "big_blind"), on="hand_id")
        .with_columns((pl.col("total_contribution") - pl.col("act_sum")).alias("posted"))
    )
    posted_ok = (
        (blinds["posted"] == 0)
        | (blinds["posted"] == blinds["small_blind"])
        | (blinds["posted"] == blinds["big_blind"])
    )
    report("total_contribution - sum(amount) in {0, SB, BB}", posted_ok)
    per_hand_blinds = blinds.group_by("hand_id").agg(
        (pl.col("posted") == pl.col("small_blind")).sum().alias("n_sb"),
        (pl.col("posted") == pl.col("big_blind")).sum().alias("n_bb"),
    )
    report("exactly one SB and one BB posted per hand",
           (per_hand_blinds["n_sb"] == 1) & (per_hand_blinds["n_bb"] == 1))

    # Seat positions of the blinds relative to the button.
    pos = blinds.join(hands.select("hand_id", "button_seat"), on="hand_id").with_columns(
        ((pl.col("seat_no") - pl.col("button_seat")) % 6).alias("offset")
    )
    print("offset from button of SB poster:",
          pos.filter(pl.col("posted") == pl.col("small_blind"))["offset"].value_counts().sort("offset").rows())
    print("offset from button of BB poster:",
          pos.filter(pl.col("posted") == pl.col("big_blind"))["offset"].value_counts().sort("offset").rows())

    # Pot chain: pot_before[i+1] == pot_before[i] + amount[i]; first pot == SB + BB.
    a = actions.with_columns(
        pl.col("pot_before").shift(-1).over("hand_id").alias("next_pot"),
        pl.col("action_no").rank("ordinal").over("hand_id").alias("idx"),
    )
    chain = a.filter(pl.col("next_pot").is_not_null())
    report("pot_before[i+1] == pot_before[i] + amount[i]",
           chain["next_pot"] == chain["pot_before"] + chain["amount"])
    first = a.filter(pl.col("idx") == 1).join(hands.select("hand_id", "small_blind", "big_blind"), on="hand_id")
    report("first pot_before == SB + BB", first["pot_before"] == first["small_blind"] + first["big_blind"])
    report("action_no is 0..n-1 contiguous", (a["action_no"] == a["idx"] - 1))

    # Per-action commitment semantics.
    a = a.join(blinds.select("hand_id", "player_id", "posted", "starting_stack"), on=["hand_id", "player_id"])
    a = a.with_columns(
        # chips this player put in during earlier actions of the hand
        (pl.col("amount").cum_sum().over("hand_id", "player_id") - pl.col("amount")).alias("prior_amt"),
        # chips this player put in on the same street before this action
        (pl.col("amount").cum_sum().over("hand_id", "player_id", "street") - pl.col("amount")).alias("prior_street"),
    ).with_columns(
        pl.when(pl.col("street") == "preflop").then(pl.col("posted")).otherwise(0).alias("street_blind")
    )
    report("stack_before == starting_stack - posted - prior amounts",
           a["stack_before"] == a["starting_stack"] - a["posted"] - a["prior_amt"])
    report("amount_to == street commitment after action (blind + prior street + amount)",
           a["amount_to"] == a["street_blind"] + a["prior_street"] + a["amount"])
    for act in ["fold", "check"]:
        sub = a.filter(pl.col("action") == act)
        report(f"{act}: amount == 0", sub["amount"] == 0)
    calls = a.filter(pl.col("action") == "call")
    report("call: amount == to_call", calls["amount"] == calls["to_call"])
    checks = a.filter(pl.col("action") == "check")
    report("check: to_call == 0", checks["to_call"] == 0)
    bets = a.filter(pl.col("action") == "bet")
    report("bet: to_call == 0", bets["to_call"] == 0)
    ai = a.filter(pl.col("action") == "all_in")
    report("all_in: amount == stack_before", ai["amount"] == ai["stack_before"])
    report("all_in: amount <= to_call (all-in call) share", ai["amount"] <= ai["to_call"])
    report("to_call <= stack_before", a["to_call"] <= a["stack_before"])

    # to_call = min(highest street commitment so far - own street commitment, stack).
    # Own commitment before the action is amount_to - amount; preflop the BB counts as a commitment.
    a = a.join(hands.select("hand_id", "big_blind"), on="hand_id").with_columns(
        pl.col("amount_to").cum_max().shift(1).over("hand_id", "street").alias("max_prev")
    ).with_columns(
        pl.when(pl.col("street") == "preflop")
        .then(pl.max_horizontal(pl.col("max_prev").fill_null(0), pl.col("big_blind")))
        .otherwise(pl.col("max_prev").fill_null(0))
        .alias("max_commit")
    )
    expected = a.select(
        pl.min_horizontal(pl.col("max_commit") - (pl.col("amount_to") - pl.col("amount")), "stack_before")
    ).to_series()
    report("to_call == min(max street commitment - own commitment, stack_before)", a["to_call"] == expected)

    # players_active == players not folded before this action (including actor).
    a = a.with_columns(
        (pl.col("action") == "fold").cast(pl.Int32).cum_sum().over("hand_id").shift(1).over("hand_id")
        .fill_null(0).alias("folds_before")
    ).join(hands.select("hand_id", "players_dealt"), on="hand_id")
    report("players_active == players_dealt - folds before",
           a["players_active"] == a["players_dealt"] - a["folds_before"])

    # Uncalled bets: is the unmatched portion of the last bet returned?
    last = a.group_by("hand_id").agg(pl.col("action").last().alias("last_action"))
    print("last action of hand:", last["last_action"].value_counts().sort("count", descending=True).rows())

    # Showdown flags vs hands table.
    sd = seats.group_by("hand_id").agg(pl.col("went_to_showdown").sum().alias("n_sd")).join(hands, on="hand_id")
    report("players_at_showdown == count(went_to_showdown)", sd["n_sd"] == sd["players_at_showdown"])
    report("folded players never went to showdown",
           ~(seats["folded"] & seats["went_to_showdown"]))
    winners = seats.filter(pl.col("won_share") > 0)
    report("won_share > 0 only for non-folded players", ~winners["folded"])
    print("board length (cards) distribution:",
          hands.with_columns(pl.col("board_cards").str.split(" ").list.len().alias("n"))
          .with_columns(pl.when(pl.col("board_cards") == "").then(0).otherwise(pl.col("n")).alias("n"))["n"]
          .value_counts().sort("n").rows())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=200_000, help="number of hands to check (0 = all)")
    args = parser.parse_args()
    main(args.hands or None)
