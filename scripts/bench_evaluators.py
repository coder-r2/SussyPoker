"""Benchmark 7-card hand evaluators (OPEN-2).

Usage: python scripts/bench_evaluators.py [--n 200000]
"""

import argparse
import time

import numpy as np


def random_hands(n: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n, 52)), axis=1)[:, :k].astype(int)


def bench_phevaluator(hands: np.ndarray) -> tuple[float, list[int]]:
    from phevaluator import evaluate_cards

    rows = hands.tolist()
    t = time.perf_counter()
    out = [evaluate_cards(*r) for r in rows]
    return time.perf_counter() - t, out


def bench_treys(hands: np.ndarray) -> tuple[float, list[int]]:
    from treys import Card, Evaluator

    ev = Evaluator()
    ranks, suits = "23456789TJQKA", "cdhs"
    lookup = [Card.new(ranks[i // 4] + suits[i % 4]) for i in range(52)]
    rows = [[lookup[c] for c in r] for r in hands.tolist()]
    t = time.perf_counter()
    out = [ev.evaluate(r[:2], r[2:]) for r in rows]
    return time.perf_counter() - t, out


def main(n: int) -> None:
    hands = random_hands(n, 7)
    t_ph, r_ph = bench_phevaluator(hands)
    t_tr, r_tr = bench_treys(hands)
    agree = np.mean(np.array(r_ph) == np.array(r_tr))
    print(f"phevaluator: {n / t_ph / 1e6:6.2f} M evals/s")
    print(f"treys:       {n / t_tr / 1e6:6.2f} M evals/s")
    print(f"identical ranks: {agree:.2%} (both use the 1 = best ... 7462 = worst scale)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    main(p.parse_args().n)
