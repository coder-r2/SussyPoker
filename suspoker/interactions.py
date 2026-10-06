"""S2 + S3: hand strength at each decision and who-did-what-to-whom tables.

Everything here works on integer-coded `Tables` (see ingest.py) for any subset of tables, and
produces *event tables* that features.py aggregates. Design choices come from the EDA:

* **Last aggressor** = the player whose bet / raise / aggressive all-in set the current price on the
  *same street*. Preflop, facing only the big blind is "unopened" (blinds are not actions), and an
  all-in that does not exceed `to_call` is a call, not aggression.
* **Response events**: every time player X responds (fold / call / raise) to aggressor Y.
* **Hole cards are known**, so each postflop response knows whether X was ahead of or behind Y.
* **Pairwise chip flow** per hand: flow(A->B) = A's contribution × B's share of the pot, which
  splits multiway pots into pairwise transfers.
"""

import numpy as np
import polars as pl
from phevaluator import evaluate_cards

from suspoker.cards import NO_CARD, hand_class, preflop_percentile
from suspoker.ingest import ALL_IN, BET, CALL, CHECK, FOLD, RAISE, Tables

# phevaluator rank boundaries (1 = royal flush ... 7462 = worst high card)
TWO_PAIR_OR_BETTER = 3325
PAIR_OR_BETTER = 6185
PREFLOP_STRONG = 0.85  # top 15% of starting combos
PRE_AHEAD_MARGIN = 0.10  # preflop "ahead": hand percentile at least 10 points above the aggressor's
JUNK_PCT = 0.35  # preflop "junk": bottom 35% of starting combos


def preflop_lut() -> np.ndarray:
    """52x52 lookup table: preflop strength percentile of every two-card combo."""
    pct = preflop_percentile()
    lut = np.zeros((52, 52), dtype=np.float32)
    for a in range(52):
        for b in range(52):
            if a != b:
                lut[a, b] = pct[hand_class(a, b)]
    return lut


_LUT: np.ndarray | None = None


def hand_context(t: Tables, block_hands: int) -> pl.DataFrame:
    """Per hand: table, phase, big blind, and a time block within (table, phase)."""
    return (
        t.hands.select("hand_idx", "table_idx", "phase", "bb", "final_pot", "started_at",
                       "b1", "b2", "b3", "b4", "b5")
        .sort("table_idx", "started_at", "hand_idx")
        .with_columns(
            (pl.col("hand_idx").rank("ordinal").over("table_idx", "phase") - 1).floordiv(block_hands)
            .cast(pl.Int16).alias("block")
        )
    )


def hand_players(t: Tables, ctx: pl.DataFrame) -> pl.DataFrame:
    """Per (hand, player): seat, results in bb, voluntary participation and preflop strength."""
    global _LUT
    if _LUT is None:
        _LUT = preflop_lut()
    pre = t.actions.filter(pl.col("street") == 0).group_by("hand_idx", "player_idx").agg(
        (pl.col("amount") > 0).any().alias("vpip"),
        (pl.col("action").is_in([BET, RAISE]) | ((pl.col("action") == ALL_IN)
                                                 & (pl.col("amount") > pl.col("to_call")))).any().alias("pfr"),
        # action order, for "stepping aside": folding after someone (e.g. the partner) entered the pot
        pl.col("action_no").filter(pl.col("amount") > 0).min().alias("first_vol_no"),
        pl.col("action_no").max().alias("last_pre_no"),
        pl.col("action_no").filter(pl.col("action") == FOLD).min().alias("fold_pre_no"),
    )
    # earliest voluntary entry by *another* player in the same hand
    entries = pre.group_by("hand_idx").agg(pl.col("first_vol_no").drop_nulls().sort().head(2).alias("_e"))
    pre = pre.join(entries, on="hand_idx", how="left").with_columns(
        pl.when(pl.col("first_vol_no") == pl.col("_e").list.get(0, null_on_oob=True))
        .then(pl.col("_e").list.get(1, null_on_oob=True))
        .otherwise(pl.col("_e").list.get(0, null_on_oob=True)).alias("other_entry_no")
    ).drop("_e").with_columns(
        (pl.col("other_entry_no") < pl.col("last_pre_no")).fill_null(False).alias("sa_opp_any"),
        (pl.col("fold_pre_no") > pl.col("other_entry_no")).fill_null(False).alias("sa_ev_any"),
    )
    post = t.actions.filter(pl.col("street") >= 1).select("hand_idx", "player_idx").unique().with_columns(
        pl.lit(True).alias("saw_flop"))
    s = t.seats.join(ctx.select("hand_idx", "table_idx", "phase", "block", "bb"), on="hand_idx")
    pre_pct = _LUT[s["c1"].to_numpy(), s["c2"].to_numpy()]
    return (
        s.with_columns(pl.Series("pre_pct", pre_pct))
        .join(pre, on=["hand_idx", "player_idx"], how="left")
        .join(post, on=["hand_idx", "player_idx"], how="left")
        .with_columns(
            pl.col("vpip").fill_null(False), pl.col("pfr").fill_null(False), pl.col("saw_flop").fill_null(False),
            pl.col("sa_opp_any").fill_null(False), pl.col("sa_ev_any").fill_null(False),
            # opened: entered the pot first, with a raise
            (pl.col("pfr") & (pl.col("other_entry_no").is_null() | (pl.col("other_entry_no") > pl.col("first_vol_no"))))
            .fill_null(False).alias("open"),
            (pl.col("net") / pl.col("bb")).alias("net_bb"),
            (pl.col("contrib") / pl.col("bb")).alias("contrib_bb"),
        )
    )


def street_ranks(t: Tables) -> pl.DataFrame:
    """phevaluator rank of every (hand, player, street) where the player acted postflop."""
    acted = t.actions.filter(pl.col("street") >= 1).select("hand_idx", "player_idx", "street").unique()
    df = (acted.join(t.seats.select("hand_idx", "player_idx", "c1", "c2"), on=["hand_idx", "player_idx"])
          .join(t.hands.select("hand_idx", "b1", "b2", "b3", "b4", "b5"), on="hand_idx"))
    parts = []
    for street, n_board in ((1, 3), (2, 4), (3, 5)):
        sub = df.filter(pl.col("street") == street)
        if sub.is_empty():
            continue
        cols = ["c1", "c2"] + [f"b{i}" for i in range(1, n_board + 1)]
        arr = sub.select(cols).to_numpy()
        if (arr == NO_CARD).any():
            raise ValueError(f"missing board card for a street-{street} action")
        ranks = np.fromiter((evaluate_cards(*row) for row in arr.tolist()), dtype=np.int16, count=len(arr))
        parts.append(sub.select("hand_idx", "player_idx", "street").with_columns(pl.Series("rank", ranks)))
    if not parts:
        return pl.DataFrame(schema={"hand_idx": pl.Int32, "player_idx": pl.Int32, "street": pl.Int8,
                                    "rank": pl.Int16})
    return pl.concat(parts)


def enrich_actions(t: Tables, ctx: pl.DataFrame, hp: pl.DataFrame, ranks: pl.DataFrame) -> pl.DataFrame:
    """Every action with: aggression, last aggressor, response type, strength vs the aggressor."""
    aggressive = pl.col("action").is_in([BET, RAISE]) | ((pl.col("action") == ALL_IN)
                                                       & (pl.col("amount") > pl.col("to_call")))
    a = (
        t.actions.join(ctx.select("hand_idx", "table_idx", "phase", "block", "bb"), on="hand_idx")
        .sort("hand_idx", "action_no")
        .with_columns(aggressive.alias("aggressive"))
        .with_columns(pl.when("aggressive").then(pl.col("player_idx")).alias("aggr"))
        .with_columns(pl.col("aggr").shift(1).forward_fill().over("hand_idx", "street").alias("last_aggr"))
        .with_columns(
            ((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null()).alias("facing"),
            pl.col("last_aggr").is_null().alias("unopened"),
            pl.when(pl.col("action") == FOLD).then(0)
            .when(pl.col("aggressive")).then(2)
            .when(pl.col("action").is_in([CALL, ALL_IN])).then(1)
            .otherwise(None).cast(pl.Int8).alias("resp"),
            (pl.col("amount") / pl.col("bb")).alias("amount_bb"),
        )
    )
    pre = hp.select("hand_idx", "player_idx", "pre_pct")
    a = (
        a.join(pre, on=["hand_idx", "player_idx"], how="left")
        .join(pre.rename({"player_idx": "last_aggr", "pre_pct": "aggr_pre_pct"}),
              on=["hand_idx", "last_aggr"], how="left")
        .join(ranks, on=["hand_idx", "player_idx", "street"], how="left")
        .join(ranks.rename({"player_idx": "last_aggr", "rank": "aggr_rank"}),
              on=["hand_idx", "last_aggr", "street"], how="left")
    )
    post = pl.col("street") >= 1
    return a.with_columns(
        (post & (pl.col("rank") < pl.col("aggr_rank"))).fill_null(False).alias("ahead"),
        (post & (pl.col("rank") > pl.col("aggr_rank"))).fill_null(False).alias("behind"),
        pl.when(post).then(pl.col("rank") <= TWO_PAIR_OR_BETTER)
        .otherwise(pl.col("pre_pct") >= PREFLOP_STRONG).fill_null(False).alias("strong"),
    )


def response_events(ea: pl.DataFrame) -> pl.DataFrame:
    """X (responder) facing Y (last aggressor): one row per response."""
    return (
        ea.filter(pl.col("facing") & (pl.col("last_aggr") != pl.col("player_idx")))
        .select(
            "hand_idx", "table_idx", "phase", "block", "street",
            pl.col("player_idx").alias("x"), pl.col("last_aggr").alias("y"),
            "resp", "ahead", "behind", "strong", "amount_bb", "players_active", "pre_pct",
            # edge = responder's strength minus the aggressor's: preflop percentile gap, postflop rank gap
            pl.when(pl.col("street") == 0).then(pl.col("pre_pct") - pl.col("aggr_pre_pct"))
            .otherwise((pl.col("aggr_rank") - pl.col("rank")) / 7462.0).alias("edge"),
        )
        .with_columns(
            (pl.col("resp") == 0).alias("fold"),
            (pl.col("resp") == 1).alias("call"),
            (pl.col("resp") == 2).alias("raise_"),
            (pl.col("street") >= 1).alias("post"),
            ((pl.col("resp") == 0) & pl.col("ahead")).alias("fold_ahead"),
            ((pl.col("resp") == 0) & pl.col("strong")).alias("fold_strong"),
            ((pl.col("resp") >= 1) & pl.col("behind")).alias("payoff"),
            pl.when((pl.col("resp") >= 1) & pl.col("behind")).then(pl.col("amount_bb")).otherwise(0.0)
            .alias("payoff_bb"),
            ((pl.col("resp") == 2) & (pl.col("players_active") >= 3)).alias("squeeze"),
            # preflop versions of "fold while ahead" and "pay off while behind" (hole cards are known)
            ((pl.col("street") == 0) & (pl.col("resp") == 0) & (pl.col("edge") > PRE_AHEAD_MARGIN))
            .fill_null(False).alias("pre_fold_ahead"),
            ((pl.col("street") == 0) & (pl.col("resp") >= 1) & (pl.col("pre_pct") < JUNK_PCT) & (pl.col("edge") < 0))
            .fill_null(False).alias("pre_junk_call"),
        )
    )


def strong_checks(ea: pl.DataFrame) -> pl.DataFrame:
    """X checks a strong made hand (two pair or better) on a postflop street where Y also acts.

    Soft play is not only heads-up: partners also slow-play each other in multiway pots.
    """
    checks = ea.filter((pl.col("street") >= 1) & (pl.col("action") == CHECK) & pl.col("strong")).select(
        "hand_idx", "street", pl.col("player_idx").alias("x"))
    actors = ea.filter(pl.col("street") >= 1).select("hand_idx", "street", pl.col("player_idx").alias("y")).unique()
    return checks.join(actors, on=["hand_idx", "street"]).filter(pl.col("x") != pl.col("y"))


def flows(t: Tables, ctx: pl.DataFrame) -> pl.DataFrame:
    """Pairwise chip flow per hand: chips of A that ended up with B, in big blinds."""
    s = t.seats.select("hand_idx", "player_idx", "contrib", "won_share", "sd")
    contributors = s.filter(pl.col("contrib") > 0).select(
        "hand_idx", pl.col("player_idx").alias("a"), "contrib", pl.col("sd").alias("sd_a"))
    winners = s.filter(pl.col("won_share") > 0).select(
        "hand_idx", pl.col("player_idx").alias("b"), "won_share", pl.col("sd").alias("sd_b"))
    return (
        contributors.join(winners, on="hand_idx")
        .filter(pl.col("a") != pl.col("b"))
        .join(ctx.select("hand_idx", "table_idx", "phase", "block", "bb"), on="hand_idx")
        .select("hand_idx", "table_idx", "phase", "block", "a", "b",
                (pl.col("contrib") * pl.col("won_share") / pl.col("bb")).alias("flow_bb"),
                (pl.col("sd_a") & pl.col("sd_b")).alias("at_showdown"))
    )


def hu_streets(ea: pl.DataFrame) -> pl.DataFrame:
    """Postflop streets where exactly two players acted: was it checked through? was a strong hand there?"""
    return (
        ea.filter(pl.col("street") >= 1)
        .group_by("hand_idx", "street", "table_idx", "phase", "block")
        .agg(
            pl.col("player_idx").unique().alias("players"),
            pl.col("aggressive").any().alias("bet_made"),
            pl.col("rank").min().alias("best_rank"),
        )
        .filter(pl.col("players").list.len() == 2)
        .with_columns(
            pl.col("players").list.min().alias("lo"),
            pl.col("players").list.max().alias("hi"),
            (~pl.col("bet_made")).alias("checked_through"),
            (pl.col("best_rank") <= TWO_PAIR_OR_BETTER).fill_null(False).alias("strong_present"),
        )
        .drop("players")
    )


def seat_pairs(hp: pl.DataFrame) -> pl.DataFrame:
    """All unordered pairs of seated players per hand (15 per six-handed hand)."""
    cols = ["hand_idx", "player_idx", "seat_no", "vpip", "pfr", "open", "contrib", "folded", "sd", "net_bb", "pre_pct",
            "first_vol_no", "last_pre_no", "fold_pre_no"]
    left = hp.select(*cols, "table_idx", "phase", "block", "bb")
    right = hp.select(cols)
    p = left.join(right, on="hand_idx", suffix="_r").filter(pl.col("player_idx") < pl.col("player_idx_r"))
    p = p.rename({c: f"{c}_lo" for c in cols[1:]}).rename({f"{c}_r": f"{c}_hi" for c in cols[1:]})
    return p.with_columns(
        ((pl.col("seat_no_lo") - pl.col("seat_no_hi")).abs().is_in([1, 5])).alias("adjacent"),
        # stepping aside: X folds preflop after partner Y has entered the pot (direction 1: X=lo, 2: X=hi)
        (pl.col("first_vol_no_hi") < pl.col("last_pre_no_lo")).fill_null(False).alias("sa_opp_1"),
        (pl.col("fold_pre_no_lo") > pl.col("first_vol_no_hi")).fill_null(False).alias("sa_ev_1"),
        (pl.col("first_vol_no_lo") < pl.col("last_pre_no_hi")).fill_null(False).alias("sa_opp_2"),
        (pl.col("fold_pre_no_hi") > pl.col("first_vol_no_lo")).fill_null(False).alias("sa_ev_2"),
    )


def hand_vpip_summary(hp: pl.DataFrame) -> pl.DataFrame:
    """Per hand: how many players entered voluntarily and how many of those folded."""
    return hp.group_by("hand_idx").agg(
        pl.col("vpip").sum().alias("n_vpip"),
        (pl.col("vpip") & pl.col("folded")).sum().alias("n_vpip_folded"),
    )
