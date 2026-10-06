"""Precompute heads-up preflop equity for every pair of hand classes (Monte Carlo, fixed seed).

Pure poker math (no competition data). Output: suspoker/resources/hu_preflop_equity.npy (169 x 169 float32,
indexed by suspoker.equity.class_index). Usage: python scripts/make_hu_preflop_table.py [--trials 1500]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from phevaluator import evaluate_cards

from suspoker.equity import class_index

OUT = Path(__file__).resolve().parent.parent / "suspoker" / "resources" / "hu_preflop_equity.npy"


_c1, _c2 = np.triu_indices(52, k=1)
_cls = class_index(_c1, _c2)
COMBOS = {i: list(zip(_c1[_cls == i].tolist(), _c2[_cls == i].tolist(), strict=True)) for i in range(169)}


def combos_of(idx: int) -> list[tuple[int, int]]:
    return COMBOS[idx]


def row(args: tuple[int, int]) -> np.ndarray:
    i, trials = args
    rng = np.random.default_rng(1000 + i)
    xs, res = combos_of(i), np.full(169, np.nan)
    for j in range(169):
        ys = combos_of(j)
        pairs = [(x, y) for x in xs for y in ys if not set(x) & set(y)]
        score = 0.0
        for _ in range(trials):
            x, y = pairs[rng.integers(len(pairs))]  # random suit assignment that doesn't collide
            deck = [c for c in range(52) if c not in (*x, *y)]
            board = [deck[k] for k in rng.choice(len(deck), 5, replace=False)]
            rx, ry = evaluate_cards(*x, *board), evaluate_cards(*y, *board)
            score += 1.0 if rx < ry else (0.5 if rx == ry else 0.0)
        res[j] = score / trials
    return res


def main(trials: int) -> None:
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        table = np.vstack(list(pool.map(row, [(i, trials) for i in range(169)])))
    table = (table + (1 - table.T)) / 2  # equity(i vs j) + equity(j vs i) = 1: averaging halves the MC noise
    np.save(OUT, table.astype(np.float32))
    aa, kk = class_index(np.array([48]), np.array([49]))[0], class_index(np.array([44]), np.array([45]))[0]
    print(f"wrote {OUT.name}: AA vs KK {table[aa, kk]:.3f} (true ~0.82); symmetry error "
          f"{np.nanmax(np.abs(table + table.T - 1)):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1500)
    main(p.parse_args().trials)
