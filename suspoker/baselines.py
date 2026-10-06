"""S4: each player's normal style per period: the yardstick that relationship features are measured against.

Counts are kept next to rates, so features.py can compute "vs everyone except the partner" exactly.
"""

import polars as pl

RESPONSE_COUNTS = ["fac", "fold", "call", "raise_", "fac_post", "fold_ahead", "payoff", "payoff_bb",
                   "fold_strong", "squeeze", "pre_fold_ahead", "pre_junk_call"]
BIG_LOSS_BB = 20.0


def response_counts(ev: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Aggregate response events by `keys` into the counts used for rates and lifts."""
    return ev.group_by(keys).agg(
        pl.len().alias("fac"),
        pl.col("fold").sum(), pl.col("call").sum(), pl.col("raise_").sum(),
        pl.col("post").sum().alias("fac_post"),
        pl.col("fold_ahead").sum(), pl.col("payoff").sum(), pl.col("payoff_bb").sum(),
        pl.col("fold_strong").sum(), pl.col("squeeze").sum(),
        pl.col("pre_fold_ahead").sum(), pl.col("pre_junk_call").sum(),
    )


def player_baselines(hp: pl.DataFrame, ea: pl.DataFrame, ev: pl.DataFrame, hu: pl.DataFrame) -> pl.DataFrame:
    """One row per (player, phase): style rates plus the raw counts behind them."""
    keys = ["player_idx", "phase"]
    hp_sorted = hp.sort("player_idx", "phase", "hand_idx").with_columns(
        (pl.col("net_bb").shift(1).over(keys) <= -BIG_LOSS_BB).fill_null(False).alias("after_big_loss"))
    style = hp_sorted.group_by(keys).agg(
        pl.len().alias("hands"),
        pl.col("vpip").mean(), pl.col("pfr").mean(), pl.col("saw_flop").mean(),
        (pl.col("sd").sum() / pl.col("saw_flop").sum().clip(lower_bound=1)).alias("wtsd"),
        (pl.col("net_bb").sum() / pl.len() * 100).alias("net_bb100"),
        (pl.col("vpip").filter(pl.col("after_big_loss")).mean() - pl.col("vpip").mean()).fill_null(0.0)
        .alias("tilt_vpip"),
        pl.col("sa_opp_any").sum().alias("sa_opp_t"), pl.col("sa_ev_any").sum().alias("sa_ev_t"),
    )
    post = ea.filter(pl.col("street") >= 1)
    aggression = post.group_by(keys).agg(
        (pl.col("aggressive").sum() / (pl.col("resp") == 1).sum().clip(lower_bound=1)).alias("af"),
        pl.col("unopened").sum().alias("unopened_post"),
        (pl.col("unopened") & pl.col("aggressive")).sum().alias("bet_unopened_n"),
    ).with_columns((pl.col("bet_unopened_n") / pl.col("unopened_post").clip(lower_bound=1)).alias("bet_unopened"))
    resp = response_counts(ev.rename({"x": "player_idx"}), keys).with_columns(
        (pl.col("fold") / pl.col("fac")).alias("fold_rate"),
        (pl.col("call") / pl.col("fac")).alias("call_rate"),
        (pl.col("raise_") / pl.col("fac")).alias("raise_rate"),
    )
    hu_long = pl.concat([
        hu.select(pl.col("lo").alias("player_idx"), "phase", "checked_through", "strong_present"),
        hu.select(pl.col("hi").alias("player_idx"), "phase", "checked_through", "strong_present"),
    ])
    hu_counts = hu_long.group_by(keys).agg(
        pl.len().alias("hu_n"), pl.col("checked_through").sum().alias("hu_check"),
        pl.col("strong_present").sum().alias("hu_strong"),
        (pl.col("strong_present") & pl.col("checked_through")).sum().alias("hu_strong_check"),
    )
    return (
        style.join(aggression, on=keys, how="left").join(resp, on=keys, how="left")
        .join(hu_counts, on=keys, how="left").fill_null(0)
    )
