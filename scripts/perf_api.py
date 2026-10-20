"""Performance check of the game API against SPEC §7.9 targets (run against a live server).

    uvicorn api.main:app --port 8000      # ideally limited to 2 CPUs, like a Hugging Face CPU Basic Space
    python scripts/perf_api.py [--base http://localhost:8000]

Targets: one action incl. bot responses p95 < 300 ms; detector rescoring after a hand (15 pairs, <= 500 hands)
p95 < 1.5 s; auto-play of 50 hands < 15 s.
"""

import argparse
import random
import statistics
import time

import httpx


def p95(xs: list[float]) -> float:
    return sorted(xs)[max(0, int(0.95 * len(xs)) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base, timeout=120)
    rng = random.Random(0)
    sid = c.post("/api/sessions", json={"strategy": "soft_play", "intensity": 0.6}).json()["session_id"]

    def play(n_actions: int, plain: list, rescored: list) -> None:
        for _ in range(n_actions):
            legal = c.get(f"/api/sessions/{sid}").json()["state"]["legal"]
            choice = rng.choice(["call", "call", "fold", "raise"])
            body = {"action": "check" if legal["check"] else "call"}
            if choice == "fold" and legal["fold"]:
                body = {"action": "fold"}
            elif choice == "raise" and legal["raise"]:
                body = {"action": "raise", "amount": legal["raise"]["min_to"]}
            t0 = time.perf_counter()
            r = c.post(f"/api/sessions/{sid}/action", json=body).json()
            dt = time.perf_counter() - t0
            (rescored if r["hand_complete"] and r["detector"]["status"] == "scored" else plain).append(dt)

    t0 = time.perf_counter()
    c.post(f"/api/sessions/{sid}/autoplay", json={"hands": 50, "user_mode": "colluding"})
    auto = time.perf_counter() - t0
    plain, rescored = [], []
    play(60, plain, rescored)                      # ~50-110 hands
    for _ in range(7):                             # fast-forward to ~450 hands
        c.post(f"/api/sessions/{sid}/autoplay", json={"hands": 50, "user_mode": "colluding"})
    late_plain, late_rescored = [], []
    play(60, late_plain, late_rescored)            # up to 500 hands
    hands = c.get(f"/api/sessions/{sid}").json()["state"]["hands_played"]
    print(f"session reached {hands} hands")
    print(f"action, no rescore      : p95 {1000 * p95(plain + late_plain):.0f} ms  (target < 300 ms; "
          f"median {1000 * statistics.median(plain + late_plain):.0f} ms, n={len(plain + late_plain)})")
    print(f"action + rescore ~100 h : p95 {p95(rescored):.2f} s")
    print(f"action + rescore ~450+ h: p95 {p95(late_rescored):.2f} s  (target < 1.5 s; n={len(late_rescored)})")
    print(f"auto-play 50 hands      : {auto:.1f} s  (target < 15 s)")


if __name__ == "__main__":
    main()
