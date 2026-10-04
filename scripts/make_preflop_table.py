"""Precompute each preflop class's equity vs one random hand (Monte Carlo, fixed seed).

This is pure poker math (no competition data). Output: suspoker/resources/preflop_equity.json
Usage: python scripts/make_preflop_table.py [--trials 20000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from phevaluator import evaluate_cards

from suspoker.cards import RANKS, all_hand_classes

OUT = Path(__file__).resolve().parent.parent / "suspoker" / "resources" / "preflop_equity.json"


def representative(cls: str) -> tuple[int, int]:
    hi, lo = RANKS.index(cls[0]), RANKS.index(cls[1])
    if len(cls) == 2:
        return hi * 4, hi * 4 + 1
    return hi * 4, lo * 4 + (0 if cls[2] == "s" else 1)


def equity(c1: int, c2: int, trials: int, rng: np.random.Generator) -> float:
    deck = np.array([c for c in range(52) if c not in (c1, c2)])
    draws = np.argsort(rng.random((trials, len(deck))), axis=1)[:, :7]
    score = 0.0
    for row in deck[draws].tolist():
        opp, board = row[:2], row[2:]
        mine, theirs = evaluate_cards(c1, c2, *board), evaluate_cards(*opp, *board)
        score += 1.0 if mine < theirs else (0.5 if mine == theirs else 0.0)
    return score / trials


def main(trials: int) -> None:
    rng = np.random.default_rng(7)
    table = {cls: round(equity(*representative(cls), trials, rng), 4) for cls in all_hand_classes()}
    OUT.write_text(json.dumps(table, indent=0), encoding="utf-8")
    best = sorted(table, key=table.get, reverse=True)
    print(f"wrote {len(table)} classes; strongest {best[:5]}, weakest {best[-3:]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20_000)
    main(p.parse_args().trials)
