"""Live-game detection rates (SPEC §7.6.6): bot-only Beat the Detector sessions, scored as the game scores them.

For each strategy x intensity, N sessions are auto-played (the user's seat is played by bot logic: colluding with
the partner, or honest for the control). At each checkpoint we record whether the (you, partner) pair is the
detector's #1 pair, and whether its risk is above the review threshold.

    python scripts/simulate_sessions.py [--sessions 8] [--hands 300]   # -> artifacts/game_sessions.json

The collusion is written by us, not by the competition's generator: any gap is a domain-shift finding.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from suspoker.config import ARTIFACTS_DIR  # noqa: E402

CHECKPOINTS = (38, 50, 100, 150, 200, 300, 400, 500)
CONFIGS = [(f, i) for f in ("directed_transfer", "soft_play", "coordinated_isolation") for i in (0.25, 0.5, 1.0)]
CONFIGS += [("honest", 0.0)]
OUT = ARTIFACTS_DIR / "game_sessions.json"
_MODELS = None


def run(args: tuple[str, float, int, int]) -> dict:
    global _MODELS
    from suspoker.game.session import Session
    from suspoker.predict import Models

    strategy, intensity, seed, max_hands = args
    _MODELS = _MODELS or Models.load()
    s = Session(strategy, intensity, seed=seed, models=_MODELS)
    mode = "honest" if strategy == "honest" else "colluding"
    rows = []
    for cp in [c for c in CHECKPOINTS if c <= max_hands]:
        s.autoplay(cp - s.hands_played, mode)
        mine = next(p for p in s.detector.pairs if p["you_and_partner"])
        rows.append({"hands": s.hands_played, "rank": mine["rank"], "risk": mine["risk"],
                     "flagged": mine["flagged"], "behavior": mine["behavior"]})
    return {"strategy": strategy, "intensity": intensity, "seed": seed, "checkpoints": rows,
            "coordinated_hands": len(s.coordinated)}


def summarize(results: list[dict]) -> list[dict]:
    out = []
    for strategy, intensity in CONFIGS:
        rs = [r for r in results if r["strategy"] == strategy and r["intensity"] == intensity]
        for k in range(len(rs[0]["checkpoints"]) if rs else 0):
            cps = [r["checkpoints"][k] for r in rs]
            n = len(cps)
            out.append({"strategy": strategy, "intensity": intensity, "hands": cps[0]["hands"], "sessions": n,
                        "top1": sum(c["rank"] == 1 for c in cps) / n,
                        "top1_flagged": sum(c["rank"] == 1 and c["flagged"] for c in cps) / n,
                        "median_rank": sorted(c["rank"] for c in cps)[n // 2],
                        "family_right": (sum(c["behavior"] == strategy for c in cps) / n
                                         if strategy != "honest" else None)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    jobs = [(st, it, 1000 + k, args.hands) for st, it in CONFIGS for k in range(args.sessions)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs))
    summary = summarize(results)
    OUT.write_text(json.dumps({"sessions_per_config": args.sessions, "max_hands": args.hands,
                               "chance_top1": 1 / 15, "summary": summary, "sessions": results}, indent=1),
                   encoding="utf-8")
    print(f"{len(jobs)} sessions in {time.perf_counter() - t0:.0f}s -> {OUT}")
    for row in summary:
        if row["hands"] in (50, 100, 300) or row["hands"] == args.hands:
            print(f"  {row['strategy']:22s} {row['intensity']:.2f} @{row['hands']:>3}: #1 {row['top1']:.0%}"
                  f" (#1 and flagged {row['top1_flagged']:.0%}), median rank {row['median_rank']}")


if __name__ == "__main__":
    main()
