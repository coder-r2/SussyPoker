"""Card encoding and preflop hand classes.

Cards are integers 0..51 = rank_index * 4 + suit_index, with ranks "23456789TJQKA" and suits "cdhs".
This matches phevaluator's encoding, so cards can be passed to it directly. -1 means "no card".
"""

import json
from functools import cache
from importlib import resources

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD_STRINGS = [r + s for r in RANKS for s in SUITS]
CARD_INDEX = {c: i for i, c in enumerate(CARD_STRINGS)}
NO_CARD = -1


def card_to_int(card: str) -> int:
    return CARD_INDEX[card]


def rank_of(card: int) -> int:
    return card // 4


def suit_of(card: int) -> int:
    return card % 4


def hand_class(c1: int, c2: int) -> str:
    """Preflop class of two hole cards, e.g. 'AA', 'AKs', 'T9o' (169 classes)."""
    r1, r2 = rank_of(c1), rank_of(c2)
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if suit_of(c1) == suit_of(c2) else "o")


def all_hand_classes() -> list[str]:
    classes = []
    for hi in range(12, -1, -1):
        for lo in range(hi, -1, -1):
            if hi == lo:
                classes.append(RANKS[hi] * 2)
            else:
                classes += [RANKS[hi] + RANKS[lo] + "s", RANKS[hi] + RANKS[lo] + "o"]
    return classes


@cache
def preflop_equity() -> dict[str, float]:
    """Equity of each class vs one random hand (precomputed by scripts/make_preflop_table.py)."""
    text = resources.files("suspoker.resources").joinpath("preflop_equity.json").read_text("utf-8")
    return json.loads(text)


@cache
def preflop_percentile() -> dict[str, float]:
    """Strength percentile of each class in [0, 1] (1 = best), weighted by combo counts.

    A class's percentile is the share of all 1,326 starting combos that are weaker than it,
    so the top ~2% of combos (AA, KK, QQ...) score close to 1.
    """
    eq = preflop_equity()
    combos = {c: 6 if len(c) == 2 else (4 if c.endswith("s") else 12) for c in eq}
    ordered = sorted(eq, key=eq.get)
    out, below = {}, 0
    for c in ordered:
        out[c] = (below + combos[c] / 2) / 1326
        below += combos[c]
    return out
