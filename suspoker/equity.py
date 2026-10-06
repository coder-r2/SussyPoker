"""Exact-odds view of each decision (DEC-025): the responder's chance to beat the aggressor at the moment
they act, using both players' hole cards (the auditor's view; players cannot see each other's cards).

    preflop   lookup in a 169 x 169 heads-up equity table (hand classes; suit overlaps ignored)
    flop      Monte Carlo over FLOP_SAMPLES random turn+river runouts
    turn      exact over every river card
    river     exact (current hands)

Equity is heads-up against the aggressor only, which is the relationship being judged ("did X give up a
winning hand to Y?"). Cards of other players are removed from the deck but their hands are not raced.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from importlib import resources

import numpy as np
import polars as pl
from phevaluator import evaluate_cards

from suspoker.cards import NO_CARD

FLOP_SAMPLES = 100
N_CLASSES = 169


def class_index(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """0..168: pairs 13*r + r on the diagonal; suited above it (hi*13+lo), offsuit below (lo*13+hi)."""
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = np.maximum(r1, r2), np.minimum(r1, r2)
    suited = (c1 % 4) == (c2 % 4)
    return np.where(hi == lo, hi * 13 + hi, np.where(suited, hi * 13 + lo, lo * 13 + hi))


@cache
def hu_preflop_table() -> np.ndarray:
    """169 x 169 heads-up equity of class i vs class j, indexed by `class_index` (scripts/make_hu_preflop_table.py)."""
    path = resources.files("suspoker.resources").joinpath("hu_preflop_equity.npy")
    with path.open("rb") as fh:
        return np.load(fh)


def _score(x: list[int], y: list[int], board: list[int]) -> float:
    rx, ry = evaluate_cards(*x, *board), evaluate_cards(*y, *board)
    return 1.0 if rx < ry else (0.5 if rx == ry else 0.0)


def _postflop_block(args: tuple[np.ndarray, int]) -> np.ndarray:
    """rows: xc1 xc2 yc1 yc2 b1..b5 street dead0..dead7 -> equity of x vs y (worker for a process pool)."""
    rows, seed = args
    rng = np.random.default_rng(seed)
    out = np.empty(len(rows))
    for i, r in enumerate(rows.tolist()):
        x, y, street = r[0:2], r[2:4], r[9]
        n_board = {1: 3, 2: 4, 3: 5}[street]
        board = r[4:4 + n_board]
        if street == 3:
            out[i] = _score(x, y, board)
            continue
        # dead: both hands, the board so far and other players' hole cards. The future board cards are NOT
        # dead: at decision time they are still unknown, so they stay in the deck.
        used = set(x + y + board + r[10:]) - {NO_CARD}
        deck = [c for c in range(52) if c not in used]
        if street == 2:
            out[i] = np.mean([_score(x, y, board + [c]) for c in deck])
        else:
            picks = rng.random((FLOP_SAMPLES, len(deck))).argsort(axis=1)[:, :2]
            out[i] = np.mean([_score(x, y, board + [deck[a], deck[b]]) for a, b in picks.tolist()])
    return out


def decision_equity(events: pl.DataFrame, seats: pl.DataFrame, hands: pl.DataFrame, seed: int = 0,
                    workers: int | None = None) -> pl.DataFrame:
    """events: hand_idx, action_no, street, x (responder), y (aggressor) -> same keys + eq."""
    cards = seats.select("hand_idx", "player_idx", "c1", "c2")
    e = (events.join(cards.rename({"player_idx": "x", "c1": "xc1", "c2": "xc2"}), on=["hand_idx", "x"])
         .join(cards.rename({"player_idx": "y", "c1": "yc1", "c2": "yc2"}), on=["hand_idx", "y"]))
    pre = e.filter(pl.col("street") == 0)
    table = hu_preflop_table()
    ix = class_index(pre["xc1"].to_numpy(), pre["xc2"].to_numpy())
    iy = class_index(pre["yc1"].to_numpy(), pre["yc2"].to_numpy())
    out = [pre.select("hand_idx", "action_no").with_columns(pl.Series("eq", table[ix, iy].astype(np.float64)))]

    post = e.filter(pl.col("street") >= 1)
    if post.height:
        # other players' hole cards are dead too: up to 4 more players at a six-max table
        others = (cards.join(post.select("hand_idx", "x", "y").unique(), on="hand_idx")
                  .filter((pl.col("player_idx") != pl.col("x")) & (pl.col("player_idx") != pl.col("y")))
                  .group_by("hand_idx", "x", "y")
                  .agg(pl.concat_list("c1", "c2").flatten().alias("dead")))
        post = post.join(hands.select("hand_idx", "b1", "b2", "b3", "b4", "b5"), on="hand_idx").join(
            others, on=["hand_idx", "x", "y"], how="left")
        post = post.with_columns(pl.col("dead").fill_null([]).list.concat(pl.lit([NO_CARD] * 8))
                                 .list.head(8).alias("dead"))
        dead = np.array(post["dead"].to_list(), dtype=np.int64)
        base = post.select("xc1", "xc2", "yc1", "yc2", "b1", "b2", "b3", "b4", "b5", "street").to_numpy()
        rows = np.hstack([base.astype(np.int64), dead])
        workers = workers or max(1, (os.cpu_count() or 2) - 2)
        blocks = np.array_split(rows, max(1, min(len(rows) // 2000, workers * 8)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            eq = np.concatenate(list(pool.map(_postflop_block, [(b, seed + i) for i, b in enumerate(blocks)])))
        out.append(post.select("hand_idx", "action_no").with_columns(pl.Series("eq", eq)))
    return pl.concat(out)
