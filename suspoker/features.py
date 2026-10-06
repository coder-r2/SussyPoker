"""S5: pair-level features (families F1-F8) and hand-level features for the evidence ranker.

**The lift principle.** A relationship feature compares how player X treats partner Y with how X
treats everyone else:

    base  = X's rate against everyone except Y (lightly smoothed toward the table-wide rate)
    shrunk = (events vs Y + k * base) / (opportunities vs Y + k)
    lift  = shrunk - base

With few opportunities against Y, `shrunk` stays near `base` and lift stays near 0, so a single odd
hand cannot look like collusion. Weak players, maniacs and tilting players behave the same against
everyone, so their lift is ~0 even though their raw rates are extreme (the EDA's decoys).

Directional features (X->Y and Y->X) are made symmetric with max/min, because the label does not
say which partner is the giver. All money is in big blinds; all counts become rates (EDA §5: the
evaluation period has ~40% less exposure than development).
"""

import numpy as np
import polars as pl

from suspoker.baselines import RESPONSE_COUNTS, player_baselines, response_counts
from suspoker.ingest import Tables
from suspoker.interactions import (
    enrich_actions,
    flows,
    hand_context,
    hand_players,
    hand_vpip_summary,
    hu_streets,
    response_events,
    seat_pairs,
    street_ranks,
    strong_checks,
)

PAIR = ["lo", "hi", "phase"]
GLOBAL_PRIOR = 5.0  # pseudo-opportunities pulling a player's "others" rate toward the table-wide rate

# (feature stem, event count, opportunity count) for partner-conditioned response rates
RESPONSE_RATES = [
    ("fold", "fold", "fac"),
    ("call", "call", "fac"),
    ("raise", "raise_", "fac"),
    ("fold_ahead", "fold_ahead", "fac_post"),
    ("payoff", "payoff", "fac_post"),
    ("fold_strong", "fold_strong", "fac"),
    ("squeeze", "squeeze", "fac"),
    ("pre_fold_ahead", "pre_fold_ahead", "fac"),
    ("junk_call", "pre_junk_call", "fac"),
]

# name -> (family, plain-English label). Directional stems expand to *_max / *_min.
CATALOG: dict[str, tuple[str, str]] = {
    "n_shared": ("F1 exposure", "Hands played together"),
    "contested_rate": ("F1 exposure", "Share of shared hands where both entered the pot voluntarily"),
    "both_contrib_rate": ("F1 exposure", "Share of shared hands where both put chips in (incl. blinds)"),
    "conf_rate": ("F1 exposure", "Confrontations (one facing the other's bet) per shared hand"),
    "hu_rate": ("F1 exposure", "Heads-up postflop streets between them per shared hand"),
    "sd_together_rate": ("F1 exposure", "Share of shared hands where both reached showdown"),
    "adjacent_rate": ("F7 controls", "Share of shared hands seated next to each other"),
    "netflow_bb100": ("F2 chip flow", "Net chips moving one way between them (bb per 100 shared hands)"),
    "grossflow_bb100": ("F2 chip flow", "Total chips moving between them (bb per 100 shared hands)"),
    "flow_asym": ("F2 chip flow", "How one-sided the chip flow is (0 = balanced, 1 = all one way)"),
    "nosd_flow_max_bb100": ("F2 chip flow", "Largest one-way flow won without showdown (bb per 100)"),
    "loss_share_lr_max": ("F2 chip flow", "How over-concentrated one player's losses are on the partner (log ratio)"),
    "loss_share_lr_min": ("F2 chip flow", "Same, for the other direction"),
    "payoff_bb100_max": ("F3 directed transfer",
                         "Chips put in while behind the partner (bb per 100, larger direction)"),
    "payoff_bb100_min": ("F3 directed transfer", "Chips put in while behind the partner (smaller direction)"),
    "hu_check_lift": ("F4 soft play", "Heads-up streets checked through vs each player's usual rate"),
    "hu_strong_check_lift": ("F4 soft play", "Strong hands checked down heads-up vs usual rate"),
    "iso_lift": ("F5 isolation", "Third players pushed out when both partners enter, vs usual rate"),
    "iso_opp_rate": ("F5 isolation", "Multiway pots both partners entered, per shared hand"),
    "step_aside_lift_max": ("F5 isolation", "How much more often a player folds preflop once the partner has "
                            "entered than once anyone else has (larger direction)"),
    "step_aside_lift_min": ("F5 isolation", "Same, smaller direction"),
    "blk_call_lift_max": ("F6 episodic", "Highest call-vs-partner lift in any time block"),
    "blk_raise_lift_min": ("F6 episodic", "Lowest raise-vs-partner lift in any time block"),
    "blk_fold_ahead_lift_max": ("F6 episodic", "Highest fold-while-ahead lift in any time block"),
    "blk_netflow_max_bb100": ("F6 episodic", "Largest net chip flow in any time block (bb per 100)"),
    "blk_call_lift_frac": ("F6 episodic", "Share of time blocks with call-vs-partner lift above 0.1"),
    "style_dist": ("F7 controls", "Distance between the two players' overall styles"),
}
_FAMILY_OF_RATE = {"fold": "F3 directed transfer", "call": "F3 directed transfer", "raise": "F4 soft play",
                   "fold_ahead": "F3 directed transfer", "payoff": "F3 directed transfer",
                   "fold_strong": "F4 soft play", "squeeze": "F5 isolation",
                   "pre_fold_ahead": "F3 directed transfer", "junk_call": "F3 directed transfer"}
_RATE_TEXT = {"fold": "folds", "call": "calls", "raise": "raises", "fold_ahead": "folds while ahead",
              "payoff": "pays off while behind", "fold_strong": "folds a strong hand", "squeeze": "re-raises multiway",
              "pre_fold_ahead": "folds a better starting hand preflop", "junk_call": "calls preflop with a junk hand"}
for _stem, _fam in _FAMILY_OF_RATE.items():
    for _agg, _which in (("max", "the more extreme player"), ("min", "the less extreme player")):
        CATALOG[f"{_stem}_lift_{_agg}"] = (_fam, f"How much more often {_which} {_RATE_TEXT[_stem]} "
                                                 "facing the partner than facing others")
# F8 presence: does a player change how they enter pots when the partner is seated at the table?
WEAK_OPEN_PCT = 0.5   # "weak" opening hand: bottom half of starting hands (combo-weighted percentile)
WEAK_VPIP_PCT = 0.3
PRESENCE = {"vpip": "enters pots voluntarily", "pfr": "raises preflop", "open": "opens the pot with a raise",
            "wopen": "opens with a weak hand", "wvpip": "enters with a weak hand"}
for _stem, _text in PRESENCE.items():
    for _agg, _which in (("max", "the more extreme player"), ("min", "the less extreme player")):
        CATALOG[f"{_stem}_pres_{_agg}"] = ("F8 presence", f"How much more often {_which} {_text} "
                                                          "when the partner is seated than when not")
STYLE = ["vpip", "pfr", "af", "wtsd", "fold_rate", "call_rate", "raise_rate", "bet_unopened", "net_bb100",
         "tilt_vpip", "hands"]
for _s in STYLE:
    CATALOG[f"{_s}_max"] = ("F7 controls", f"Higher of the two players' {_s}")
    CATALOG[f"{_s}_min"] = ("F7 controls", f"Lower of the two players' {_s}")


def _lift(ev_y: pl.Expr, opp_y: pl.Expr, ev_all: pl.Expr, opp_all: pl.Expr, global_rate: float, k: float) -> pl.Expr:
    base = (ev_all - ev_y + GLOBAL_PRIOR * global_rate) / (opp_all - opp_y + GLOBAL_PRIOR)
    return (ev_y + k * base) / (opp_y + k) - base


def _sym(df: pl.DataFrame, a: str, b: str, stem: str) -> pl.DataFrame:
    return df.with_columns(pl.max_horizontal(a, b).alias(f"{stem}_max"),
                           pl.min_horizontal(a, b).alias(f"{stem}_min")).drop(a, b)


def _directional(df: pl.DataFrame, counts: pl.DataFrame, keys_extra: list[str], tot: pl.DataFrame,
                 rates: list[tuple[str, str, str]], global_rates: dict, k: float) -> pl.DataFrame:
    """Attach X->Y counts for both directions and compute symmetric lift features."""
    base_keys = ["phase", *keys_extra]
    for d, (x, y) in {"1": ("lo", "hi"), "2": ("hi", "lo")}.items():
        c = counts.rename({"x": x, "y": y}).rename({n: f"{n}_{d}" for n in RESPONSE_COUNTS})
        df = df.join(c, on=[x, y, *base_keys], how="left")
        t = tot.rename({"player_idx": x}).rename({n: f"{n}_t{d}" for n in RESPONSE_COUNTS})
        df = df.join(t.select(x, "phase", *[f"{n}_t{d}" for n in RESPONSE_COUNTS]), on=[x, "phase"], how="left")
    df = df.with_columns(pl.col("^.*_(1|2|t1|t2)$").fill_null(0))
    for stem, ev, opp in rates:
        g = global_rates[(ev, opp)]
        df = df.with_columns(
            _lift(pl.col(f"{ev}_1"), pl.col(f"{opp}_1"), pl.col(f"{ev}_t1"), pl.col(f"{opp}_t1"), g, k).alias("a_"),
            _lift(pl.col(f"{ev}_2"), pl.col(f"{opp}_2"), pl.col(f"{ev}_t2"), pl.col(f"{opp}_t2"), g, k).alias("b_"),
        )
        df = _sym(df, "a_", "b_", f"{stem}_lift")
    return df


def pair_features(sp: pl.DataFrame, hvs: pl.DataFrame, ev: pl.DataFrame, fl: pl.DataFrame, hu: pl.DataFrame,
                  base: pl.DataFrame, k: float, block_min_hands: int = 10) -> pl.DataFrame:
    """One row per (pair, phase) with features F1-F7 (F8 is added by `presence_features`)."""
    sp = sp.join(hvs, on="hand_idx").with_columns(
        (pl.col("n_vpip") - pl.col("vpip_lo").cast(pl.Int32) - pl.col("vpip_hi").cast(pl.Int32)).alias("third_vpip"),
        (pl.col("n_vpip_folded") - (pl.col("vpip_lo") & pl.col("folded_lo")).cast(pl.Int32)
         - (pl.col("vpip_hi") & pl.col("folded_hi")).cast(pl.Int32)).alias("third_folded"),
    ).with_columns(
        (pl.col("vpip_lo") & pl.col("vpip_hi") & (pl.col("third_vpip") >= 1)).alias("iso_opp"),
    ).with_columns(
        (pl.col("iso_opp") & (pl.col("third_folded") == pl.col("third_vpip"))
         & ~(pl.col("folded_lo") & pl.col("folded_hi"))).alias("iso_succ"),
    )
    sp = sp.rename({"player_idx_lo": "lo", "player_idx_hi": "hi"})
    df = sp.group_by(PAIR).agg(
        pl.col("table_idx").first(),
        pl.len().alias("n_shared"),
        (pl.col("vpip_lo") & pl.col("vpip_hi")).sum().alias("contested"),
        ((pl.col("contrib_lo") > 0) & (pl.col("contrib_hi") > 0)).sum().alias("both_contrib"),
        (pl.col("sd_lo") & pl.col("sd_hi")).sum().alias("sd_together"),
        pl.col("adjacent").sum().alias("adjacent"),
        pl.col("iso_opp").sum(), pl.col("iso_succ").sum(),
    )
    n = pl.col("n_shared")
    df = df.with_columns(
        (pl.col("contested") / n).alias("contested_rate"),
        (pl.col("both_contrib") / n).alias("both_contrib_rate"),
        (pl.col("sd_together") / n).alias("sd_together_rate"),
        (pl.col("adjacent") / n).alias("adjacent_rate"),
        (pl.col("iso_opp") / n).alias("iso_opp_rate"),
    )

    # --- F3/F4/F5 partner-conditioned responses (lift vs the player's behaviour toward others)
    counts = response_counts(ev, ["x", "y", "phase"])
    tot = base.select("player_idx", "phase", *RESPONSE_COUNTS)
    global_rates = {(e, o): float(tot[e].sum() / max(tot[o].sum(), 1)) for _, e, o in RESPONSE_RATES}
    df = _directional(df, counts, [], tot, RESPONSE_RATES, global_rates, k)
    df = df.with_columns(((pl.col("fac_1") + pl.col("fac_2")) / n).alias("conf_rate"),
                         (pl.col("payoff_bb_1") / n * 100).alias("pa"), (pl.col("payoff_bb_2") / n * 100).alias("pb"))
    df = _sym(df, "pa", "pb", "payoff_bb100")

    # --- F2 chip flow
    fdir = fl.group_by("a", "b", "phase").agg(pl.col("flow_bb").sum().alias("flow"),
                                              pl.col("flow_bb").filter(~pl.col("at_showdown")).sum().alias("nosd"))
    lost = fl.group_by(pl.col("a").alias("player_idx"), "phase").agg(pl.col("flow_bb").sum().alias("lost_total"))
    for d, (x, y) in {"1": ("lo", "hi"), "2": ("hi", "lo")}.items():
        df = df.join(fdir.rename({"a": x, "b": y, "flow": f"flow_{d}", "nosd": f"nosd_{d}"}), on=[x, y, "phase"],
                     how="left")
        df = df.join(lost.rename({"player_idx": x, "lost_total": f"lost_{d}"}), on=[x, "phase"], how="left")
        df = df.join(base.select(pl.col("player_idx").alias(x), "phase", pl.col("hands").alias(f"hands_{d}")),
                     on=[x, "phase"], how="left")
    df = df.with_columns(pl.col("^(flow|nosd|lost)_(1|2)$").fill_null(0.0))
    df = df.with_columns(
        ((pl.col("flow_1") - pl.col("flow_2")).abs() / n * 100).alias("netflow_bb100"),
        ((pl.col("flow_1") + pl.col("flow_2")) / n * 100).alias("grossflow_bb100"),
        ((pl.col("flow_1") - pl.col("flow_2")).abs() / (pl.col("flow_1") + pl.col("flow_2") + 1)).alias("flow_asym"),
        (pl.max_horizontal("nosd_1", "nosd_2") / n * 100).alias("nosd_flow_max_bb100"),
        # share of X's losses that went to Y, relative to Y's share of X's opponents (5 per hand)
        ((pl.col("flow_1") + 1).log() - (pl.col("lost_1") * n / (5 * pl.col("hands_1")) + 1).log()).alias("la"),
        ((pl.col("flow_2") + 1).log() - (pl.col("lost_2") * n / (5 * pl.col("hands_2")) + 1).log()).alias("lb"),
    )
    df = _sym(df, "la", "lb", "loss_share_lr")

    # --- F4 heads-up streets (check-through) and F5 isolation, vs each player's own rate
    hup = hu.group_by(PAIR).agg(pl.len().alias("hu_n"), pl.col("checked_through").sum().alias("hu_check"),
                                pl.col("strong_present").sum().alias("hu_strong"),
                                (pl.col("strong_present") & pl.col("checked_through")).sum().alias("hu_strong_check"))
    df = df.join(hup, on=PAIR, how="left").with_columns(pl.col("^hu_.*$").fill_null(0))
    df = df.with_columns((pl.col("hu_n") / n).alias("hu_rate"))
    hb = base.select("player_idx", "phase", "hu_n", "hu_check", "hu_strong", "hu_strong_check")
    iso_long = pl.concat([sp.select(pl.col("lo").alias("player_idx"), "phase", "iso_opp", "iso_succ"),
                          sp.select(pl.col("hi").alias("player_idx"), "phase", "iso_opp", "iso_succ")])
    ib = iso_long.group_by("player_idx", "phase").agg(pl.col("iso_opp").sum().alias("iso_opp_t"),
                                                      pl.col("iso_succ").sum().alias("iso_succ_t"))
    for d, x in {"1": "lo", "2": "hi"}.items():
        df = df.join(hb.rename({"player_idx": x}).rename({c: f"{c}_p{d}" for c in hb.columns[2:]}),
                     on=[x, "phase"], how="left")
        df = df.join(ib.rename({"player_idx": x, "iso_opp_t": f"iso_opp_p{d}", "iso_succ_t": f"iso_succ_p{d}"}),
                     on=[x, "phase"], how="left")
    g_check = float(hb["hu_check"].sum() / max(hb["hu_n"].sum(), 1))
    g_strong = float(hb["hu_strong_check"].sum() / max(hb["hu_strong"].sum(), 1))
    g_iso = float(ib["iso_succ_t"].sum() / max(ib["iso_opp_t"].sum(), 1))

    def pair_lift(ev: str, opp: str, ev_p: str, opp_p: str, g: float) -> pl.Expr:
        # "others" baseline = each player's events excluding this pair, averaged over the two players
        def others(d: str) -> pl.Expr:
            return ((pl.col(f"{ev_p}_p{d}") - pl.col(ev) + GLOBAL_PRIOR * g)
                    / (pl.col(f"{opp_p}_p{d}") - pl.col(opp) + GLOBAL_PRIOR))

        b = (others("1") + others("2")) / 2
        return (pl.col(ev) + k * b) / (pl.col(opp) + k) - b

    df = df.with_columns(
        pair_lift("hu_check", "hu_n", "hu_check", "hu_n", g_check).alias("hu_check_lift"),
        pair_lift("hu_strong_check", "hu_strong", "hu_strong_check", "hu_strong", g_strong)
        .alias("hu_strong_check_lift"),
        pair_lift("iso_succ", "iso_opp", "iso_succ", "iso_opp", g_iso).alias("iso_lift"),
    )

    # --- F5 stepping aside: X folds preflop after partner Y entered, vs after anyone else entered
    sa = sp.group_by(PAIR).agg(*[pl.col(c).sum() for c in ("sa_opp_1", "sa_ev_1", "sa_opp_2", "sa_ev_2")])
    df = df.join(sa, on=PAIR, how="left")
    sab = base.select("player_idx", "phase", "sa_opp_t", "sa_ev_t")
    g_sa = float(sab["sa_ev_t"].sum() / max(sab["sa_opp_t"].sum(), 1))
    for d, x in {"1": "lo", "2": "hi"}.items():
        df = df.join(sab.rename({"player_idx": x, "sa_opp_t": f"sa_opp_t{d}", "sa_ev_t": f"sa_ev_t{d}"}),
                     on=[x, "phase"], how="left")
    df = df.with_columns(pl.col("^sa_.*$").fill_null(0)).with_columns(
        _lift(pl.col("sa_ev_1"), pl.col("sa_opp_1"), pl.col("sa_ev_t1"), pl.col("sa_opp_t1"), g_sa, k).alias("a_"),
        _lift(pl.col("sa_ev_2"), pl.col("sa_opp_2"), pl.col("sa_ev_t2"), pl.col("sa_opp_t2"), g_sa, k).alias("b_"),
    )
    df = _sym(df, "a_", "b_", "step_aside_lift")

    # --- F6 episodic: the same signals per time block, then the most extreme block
    bcounts = response_counts(ev, ["x", "y", "phase", "block"])
    bdf = sp.group_by([*PAIR, "block"]).agg(pl.len().alias("n_b")).filter(pl.col("n_b") >= block_min_hands)
    block_rates = [r for r in RESPONSE_RATES if r[0] in ("call", "raise", "fold_ahead")]
    bdf = _directional(bdf, bcounts, ["block"], tot, block_rates, global_rates, k)
    bflow = fl.group_by("a", "b", "phase", "block").agg(pl.col("flow_bb").sum().alias("f"))
    bdf = (bdf.join(bflow.rename({"a": "lo", "b": "hi", "f": "f1"}), on=[*PAIR, "block"], how="left")
           .join(bflow.rename({"a": "hi", "b": "lo", "f": "f2"}), on=[*PAIR, "block"], how="left")
           .with_columns(((pl.col("f1").fill_null(0) - pl.col("f2").fill_null(0)).abs() / pl.col("n_b") * 100)
                         .alias("netflow_b")))
    blk = bdf.group_by(PAIR).agg(
        pl.col("call_lift_max").max().alias("blk_call_lift_max"),
        pl.col("raise_lift_min").min().alias("blk_raise_lift_min"),
        pl.col("fold_ahead_lift_max").max().alias("blk_fold_ahead_lift_max"),
        pl.col("netflow_b").max().alias("blk_netflow_max_bb100"),
        (pl.col("call_lift_max") > 0.1).mean().alias("blk_call_lift_frac"),
    )
    df = df.join(blk, on=PAIR, how="left")

    # --- F7 controls: both players' overall style
    st = base.select("player_idx", "phase", *STYLE)
    for d, x in {"1": "lo", "2": "hi"}.items():
        df = df.join(st.rename({"player_idx": x}).rename({s: f"{s}_s{d}" for s in STYLE}), on=[x, "phase"],
                     how="left")
    sq = sum((pl.col(f"{s}_s1") - pl.col(f"{s}_s2")) ** 2 for s in ["vpip", "pfr", "wtsd", "fold_rate",
                                                                    "call_rate", "raise_rate", "bet_unopened"])
    df = df.with_columns(sq.sqrt().alias("style_dist"))
    for s in STYLE:
        df = _sym(df, f"{s}_s1", f"{s}_s2", s)

    keep = ["lo", "hi", "phase", "table_idx", *[c for c in feature_names() if c in df.columns]]
    return df.select(keep)


def presence_features(hp: pl.DataFrame, sp: pl.DataFrame, k: float) -> pl.DataFrame:
    """F8: lift of a player's entry habits in hands shared with the partner vs hands without them.

    Unlike F3-F5, which look at responses *to* the partner, this looks at whether the partner's mere
    presence changes play: coordinated isolation shows up as light open-raises while the partner sits
    behind ready to step aside.
    """
    hp = hp.with_columns((pl.col("open") & (pl.col("pre_pct") < WEAK_OPEN_PCT)).alias("wopen"),
                         (pl.col("vpip") & (pl.col("pre_pct") < WEAK_VPIP_PCT)).alias("wvpip"))
    flags = list(PRESENCE)
    tot = hp.group_by("player_idx", "phase").agg(pl.len().alias("n_t"),
                                                 *[pl.col(f).sum().alias(f"{f}_t") for f in flags])
    glob = hp.select(*[pl.col(f).mean() for f in flags]).row(0, named=True)
    x = hp.select("hand_idx", "player_idx", *flags)
    df = sp.select("hand_idx", "phase", "player_idx_lo", "player_idx_hi").rename(
        {"player_idx_lo": "lo", "player_idx_hi": "hi"})
    for side in ("lo", "hi"):
        df = df.join(x.rename({"player_idx": side, **{f: f"{f}_{side}" for f in flags}}), on=["hand_idx", side])
    df = df.group_by(PAIR).agg(pl.len().alias("ns"), *[pl.col(f"{f}_{s}").sum() for f in flags for s in ("lo", "hi")])
    for side in ("lo", "hi"):
        names = {"player_idx": side, "n_t": f"n_t_{side}", **{f"{f}_t": f"{f}_t_{side}" for f in flags}}
        df = df.join(tot.rename(names), on=[side, "phase"])
    for f in flags:
        df = df.with_columns(
            _lift(pl.col(f"{f}_lo"), pl.col("ns"), pl.col(f"{f}_t_lo"), pl.col("n_t_lo"), glob[f], k).alias("a_"),
            _lift(pl.col(f"{f}_hi"), pl.col("ns"), pl.col(f"{f}_t_hi"), pl.col("n_t_hi"), glob[f], k).alias("b_"))
        df = _sym(df, "a_", "b_", f"{f}_pres")
    return df.select(*PAIR, pl.col("^.*_pres_(max|min)$"))


def feature_names() -> list[str]:
    return list(CATALOG)


def hand_features(sp: pl.DataFrame, hvs: pl.DataFrame, ev: pl.DataFrame, fl: pl.DataFrame, hu: pl.DataFrame,
                  ctx: pl.DataFrame, pairs: pl.DataFrame, sc: pl.DataFrame) -> pl.DataFrame:
    """Per (pair, shared hand) features for the evidence ranker (M4), for the requested pairs only.

    Only *candidate* hands are kept: both put chips in, one responded to the other's aggression, or one
    stepped aside (folded preflop after the partner entered). Other shared hands cannot show a
    behaviour-specific action between the two.
    """
    s = (sp.rename({"player_idx_lo": "lo", "player_idx_hi": "hi"})
         .join(pairs.select("lo", "hi", "phase"), on=PAIR, how="semi")
         .join(hvs, on="hand_idx"))
    hc = ctx.select("hand_idx", "final_pot", pl.sum_horizontal([(pl.col(f"b{i}") >= 0).cast(pl.Int8)
                                                                for i in range(1, 6)]).alias("board_n"))
    s = s.join(hc, on="hand_idx").with_columns((pl.col("final_pot") / pl.col("bb")).alias("pot_bb"))

    hcounts = response_counts(ev, ["x", "y", "hand_idx"])
    for d, (x, y) in {"1": ("lo", "hi"), "2": ("hi", "lo")}.items():
        s = s.join(hcounts.rename({"x": x, "y": y}).rename({c: f"{c}_{d}" for c in RESPONSE_COUNTS}),
                   on=[x, y, "hand_idx"], how="left")
        s = s.join(fl.group_by("a", "b", "hand_idx").agg(pl.col("flow_bb").sum().alias(f"flow_{d}"))
                   .rename({"a": x, "b": y}), on=[x, y, "hand_idx"], how="left")
    hh = hu.group_by("lo", "hi", "hand_idx").agg(pl.len().alias("hu_n"),
                                                 pl.col("checked_through").sum().alias("hu_check"),
                                                 (pl.col("strong_present") & pl.col("checked_through")).sum()
                                                 .alias("hu_strong_check"))
    s = s.join(hh, on=["lo", "hi", "hand_idx"], how="left")
    # strength edge at responses between the two (either direction): the most "wrong" fold and call
    pev = ev.select("hand_idx", pl.min_horizontal("x", "y").alias("lo"), pl.max_horizontal("x", "y").alias("hi"),
                    "resp", "edge")
    edges = pev.group_by("lo", "hi", "hand_idx").agg(
        pl.col("edge").filter(pl.col("resp") == 0).max().alias("h_fold_edge_max"),
        pl.col("edge").filter(pl.col("resp") >= 1).min().alias("h_call_edge_min"))
    s = s.join(edges, on=["lo", "hi", "hand_idx"], how="left")
    scp = sc.select("hand_idx", pl.min_horizontal("x", "y").alias("lo"), pl.max_horizontal("x", "y").alias("hi"))
    s = s.join(scp.group_by("lo", "hi", "hand_idx").agg(pl.len().alias("h_strong_check_partner")),
               on=["lo", "hi", "hand_idx"], how="left")
    s = s.with_columns(pl.col("^.*_(1|2)$").fill_null(0), pl.col("^hu_.*$").fill_null(0),
                       pl.col("h_strong_check_partner").fill_null(0))
    s = s.with_columns((pl.col("sa_ev_1").cast(pl.Boolean) | pl.col("sa_ev_2").cast(pl.Boolean)).alias("h_step_aside"))
    s = s.filter(((pl.col("contrib_lo") > 0) & (pl.col("contrib_hi") > 0)) | (pl.col("fac_1") + pl.col("fac_2") > 0)
                 | pl.col("h_step_aside"))

    out = s.with_columns(
        (pl.col("vpip_lo") & pl.col("vpip_hi")).alias("h_both_vpip"),
        (pl.col("sd_lo") & pl.col("sd_hi")).alias("h_sd_together"),
        pl.col("pot_bb").alias("h_pot_bb"),
        pl.col("board_n").alias("h_board_n"),
        pl.col("n_vpip").alias("h_n_vpip"),
        ((pl.col("flow_1") - pl.col("flow_2")).abs()).alias("h_netflow_bb"),
        (pl.max_horizontal("contrib_lo", "contrib_hi") / pl.col("bb")).alias("h_contrib_max_bb"),
        pl.max_horizontal("pre_pct_lo", "pre_pct_hi").alias("h_pre_pct_max"),
        pl.min_horizontal("pre_pct_lo", "pre_pct_hi").alias("h_pre_pct_min"),
        (pl.col("fac_1") + pl.col("fac_2")).alias("h_fac"),
        *[pl.max_horizontal(f"{c}_1", f"{c}_2").alias(f"h_{c}_max") for c in
          ("fold", "call", "raise_", "fold_ahead", "payoff", "payoff_bb", "fold_strong", "squeeze",
           "pre_fold_ahead", "pre_junk_call")],
        pl.col("hu_n").alias("h_hu_n"), pl.col("hu_check").alias("h_hu_check"),
        pl.col("hu_strong_check").alias("h_hu_strong_check"),
        (pl.col("n_vpip") - pl.col("vpip_lo").cast(pl.Int32) - pl.col("vpip_hi").cast(pl.Int32)).alias("h_third_vpip"),
        ((pl.col("open_lo") & (pl.col("pre_pct_lo") < WEAK_OPEN_PCT))
         | (pl.col("open_hi") & (pl.col("pre_pct_hi") < WEAK_OPEN_PCT))).alias("h_weak_open"),
        # isolation step: one partner opens, the other folds preflop behind them
        ((pl.col("open_hi") & pl.col("sa_ev_1").cast(pl.Boolean))
         | (pl.col("open_lo") & pl.col("sa_ev_2").cast(pl.Boolean))).alias("h_open_step_aside"),
        # preflop strength of the partner who opened, and of the partner who stepped aside (null if none)
        pl.when(pl.col("open_lo")).then(pl.col("pre_pct_lo")).when(pl.col("open_hi")).then(pl.col("pre_pct_hi"))
        .alias("h_open_pct"),
        pl.when(pl.col("sa_ev_1").cast(pl.Boolean)).then(pl.col("pre_pct_lo"))
        .when(pl.col("sa_ev_2").cast(pl.Boolean)).then(pl.col("pre_pct_hi")).alias("h_step_aside_pct"),
    )
    return out.select("lo", "hi", "phase", "hand_idx", "block", pl.col("^h_.*$"))


def build_chunk(t: Tables, k: float, block_hands: int, hand_pairs: pl.DataFrame | None
                ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame]:
    """Run S2-S5 for one chunk of tables. Returns (pair features, hand features, player baselines)."""
    ctx = hand_context(t, block_hands)
    hp = hand_players(t, ctx)
    ranks = street_ranks(t)
    ea = enrich_actions(t, ctx, hp, ranks)
    ev = response_events(ea)
    fl = flows(t, ctx)
    hu = hu_streets(ea)
    sp = seat_pairs(hp)
    hvs = hand_vpip_summary(hp)
    base = player_baselines(hp, ea, ev, hu)
    pf = pair_features(sp, hvs, ev, fl, hu, base, k).join(presence_features(hp, sp, k), on=PAIR, how="left")
    hf = hand_features(sp, hvs, ev, fl, hu, ctx, hand_pairs, strong_checks(ea)) if hand_pairs is not None else None
    return pf, hf, base


def catalog_markdown(stats: pl.DataFrame | None = None) -> str:
    lines = ["# Feature catalog", "", "Generated by `python -m suspoker.pipeline --stage S5`. Do not edit by hand.",
             "", "| Feature | Family | Meaning |", "|---|---|---|"]
    for name, (family, label) in sorted(CATALOG.items(), key=lambda kv: (kv[1][0], kv[0])):
        lines.append(f"| `{name}` | {family} | {label} |")
    return "\n".join(lines) + "\n"


__all__ = ["build_chunk", "pair_features", "hand_features", "feature_names", "CATALOG", "np"]
