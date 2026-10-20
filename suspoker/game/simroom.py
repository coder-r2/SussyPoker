"""The SimRoom: a simulated card room in the competition's shape, with planted colluders and decoys.

    python -m suspoker.game.simroom            # -> artifacts/simroom/{hands,seats,actions}.parquet + ground_truth.json

Mirrors the competition at a smaller scale (SPEC §7.8): each table has a fixed stake and a persistent pool of 30
bots, six seated per hand. Seating follows what the competition data shows: players sit in sessions (they keep
their seat for about 21 hands on average, then leave), some sit more than others, the button is random, stacks
carry over and are topped up with a fresh buy-in below 40 bb or above 250 bb.

Planted per table:
    colluders   2-4 pairs across the three families, each episodic: active in 1-2 windows, coordinating in a
                share of eligible hands set by its intensity
    decoys      2-3 innocent pairs built to look suspicious: a tilting player and the aggressive player who
                sits with them, a passive player and a maniac who sits with them, two near-identical styles,
                and two players who sit together unusually often

Everything is deterministic from `simroom.seed` in config.yaml. IDs use the SIM- format (SIM-T03, SIM-P0412,
SIM-H030001) and bots have names like brisk_lagoon, so nothing can be mistaken for a competition ID.
"""

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from suspoker.config import ARTIFACTS_DIR, load_config
from suspoker.game.bots import FAMILIES, STYLES, Bot, Collusion, TablePolicy, jitter, tilt_decoy
from suspoker.game.engine import N_SEATS, SeatIn, play_hand
from suspoker.game.logger import HandLogger, conformance, failed

SIM_DIR = ARTIFACTS_DIR / "simroom"
STAKES = ((1, 2), (2, 4), (5, 10))
STAKE_WEIGHTS = (0.55, 0.35, 0.10)          # competition: 218 / 143 / 39 tables
STYLE_WEIGHTS = {"TAG": 0.30, "LAG": 0.20, "nit": 0.20, "calling_station": 0.20, "maniac": 0.10}
BUY_INS_BB = ((80, 100, 120, 150), (0.55, 0.20, 0.12, 0.13))  # fresh buy-ins seen in the data
MIN_STACK_BB, MAX_STACK_BB, REBUY_RATE = 40, 250, 0.014
SESSION_MEAN = 21                            # hands per seating session (competition median 14, mean 21)
BUDDY_BOOST = {"decoy": 12.0, "planted": 3.0}  # seating weight multiplier when the partner is already seated
DECOYS = ("tilt", "passive_player", "similar_styles", "seated_together")
START = datetime(2026, 6, 1, tzinfo=UTC)

ADJECTIVES = ("brisk", "calm", "dusky", "eager", "fuzzy", "gentle", "hazy", "icy", "jolly", "keen", "lucky", "mellow",
              "nimble", "odd", "plucky", "quiet", "rusty", "sly", "tidy", "vivid", "wily", "zesty", "amber", "bold")
NOUNS = ("lagoon", "badger", "cactus", "dune", "ember", "falcon", "glacier", "heron", "island", "jackal", "kelp",
         "lynx", "meadow", "newt", "orchid", "pebble", "quartz", "raven", "spruce", "tundra", "otter", "willow",
         "yak", "zephyr")


@dataclass(frozen=True)
class SimConfig:
    seed: int = 7
    tables: int = 12
    hands: int = 2500
    pool: int = 30
    pairs: tuple[int, int] = (2, 4)
    decoys: tuple[int, int] = (2, 3)
    intensity: tuple[float, float] = (0.35, 0.85)

    @classmethod
    def from_config(cls) -> "SimConfig":
        c = load_config().get("simroom", {})
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in c.items()})


def bot_names(rng: random.Random, n: int) -> list[str]:
    names = [f"{a}_{b}" for a in ADJECTIVES for b in NOUNS]
    return rng.sample(names, n)


def pair_id(a: str, b: str) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}-{hi[4:]}"  # SIM-P0412-P0417


def design_table(t: int, cfg: SimConfig, names: list[str]) -> dict:
    """Bots, stake, planted pairs and decoys of table `t` (pure function of the seed)."""
    rng = random.Random(cfg.seed * 1000 + t)
    stake = rng.choices(STAKES, STAKE_WEIGHTS)[0]
    ids = [f"SIM-P{t:02d}{k:02d}" for k in range(cfg.pool)]
    style_names = rng.choices(list(STYLE_WEIGHTS), list(STYLE_WEIGHTS.values()), k=cfg.pool)
    styles = {pid: jitter(STYLES[s], rng) for pid, s in zip(ids, style_names, strict=True)}
    free = list(ids)
    rng.shuffle(free)
    pairs, decoys, buddies = [], [], {}

    def take(want: str | None = None) -> str:
        for k, pid in enumerate(free):
            if want is None or styles[pid].name.startswith(want):
                return free.pop(k)
        return free.pop()

    for kind in rng.sample(DECOYS, rng.randint(*cfg.decoys)):
        if kind == "tilt":
            a, b = take("TAG"), take("LAG")
            styles[a] = tilt_decoy(styles[a])
        elif kind == "passive_player":
            a, b = take("nit"), take("maniac")
            styles[a] = replace(styles[a], name="nit+passive", call_down=1.6, agg=0.3, bluff=0.03)
        elif kind == "similar_styles":
            a, b = take("TAG"), take()
            styles[b] = replace(styles[a])
        else:
            a, b = take(), take()
        decoys.append({"type": kind, "a": a, "b": b})
        if kind != "similar_styles":
            buddies[a], buddies[b] = (b, "decoy"), (a, "decoy")

    for k in range(rng.randint(*cfg.pairs)):
        family = FAMILIES[(t + k) % len(FAMILIES)]
        a, b = take(), take()
        n_win = rng.choice((1, 1, 2))
        cuts = sorted(rng.sample(range(0, cfg.hands, 50), 2 * n_win))
        windows = tuple((cuts[2 * i], cuts[2 * i + 1]) for i in range(n_win))
        if sum(hi - lo for lo, hi in windows) < 0.3 * cfg.hands:   # at least 30% of the timeline
            start = rng.randrange(0, int(0.4 * cfg.hands))
            windows = ((start, start + int(rng.uniform(0.4, 0.6) * cfg.hands)),)
        pairs.append(Collusion(family, a, b, round(rng.uniform(*cfg.intensity), 2), windows))
        buddies[a], buddies[b] = (b, "planted"), (a, "planted")

    return {"table": t, "table_id": f"SIM-T{t:02d}", "stake": stake, "ids": ids,
            "names": dict(zip(ids, names, strict=True)), "styles": styles, "pairs": pairs, "decoys": decoys,
            "buddies": buddies, "propensity": {pid: rng.lognormvariate(0, 0.35) for pid in ids}}


def run_table(t: int, cfg: SimConfig) -> tuple[HandLogger, dict]:
    names = bot_names(random.Random(cfg.seed), cfg.tables * cfg.pool)[t * cfg.pool:(t + 1) * cfg.pool]
    d = design_table(t, cfg, names)
    rng = random.Random(cfg.seed * 7919 + t)
    sb, bb = d["stake"]
    bots = {pid: Bot(pid, d["names"][pid], d["styles"][pid], random.Random(rng.random())) for pid in d["ids"]}
    policy = TablePolicy(bots, d["pairs"], random.Random(rng.random()))

    def buy_in() -> int:
        return bb * rng.choices(*BUY_INS_BB)[0]

    stacks = {pid: int(bb * rng.uniform(60, 200)) for pid in d["ids"]}
    seat: list[str | None] = [None] * N_SEATS
    left = [0] * N_SEATS
    log, when = HandLogger(), START + timedelta(minutes=17 * t)
    coordinated: dict[int, list[str]] = {i: [] for i in range(len(d["pairs"]))}
    enabled = dict.fromkeys(coordinated, 0)
    for n in range(cfg.hands):
        for s in range(N_SEATS):           # refill empty seats: weighted by propensity and seated partners
            if left[s] > 0 and seat[s] is not None:
                continue
            standing = [p for p in d["ids"] if p not in seat]
            weights = []
            for p in standing:
                w = d["propensity"][p]
                if p in d["buddies"] and d["buddies"][p][0] in seat:
                    w *= BUDDY_BOOST[d["buddies"][p][1]]
                weights.append(w)
            seat[s] = rng.choices(standing, weights)[0]
            left[s] = max(1, int(rng.expovariate(1 / SESSION_MEAN)))
        for p in seat:
            if stacks[p] < MIN_STACK_BB * bb or stacks[p] > MAX_STACK_BB * bb or rng.random() < REBUY_RATE:
                stacks[p] = buy_in()
        policy.new_hand(n, seat)
        rec = play_hand([SeatIn(p, stacks[p]) for p in seat], rng.randrange(N_SEATS), sb, bb,
                        rng.sample(range(52), 52), policy)
        policy.end_hand(rec.net, bb)
        for p, x in rec.net.items():
            stacks[p] += x
        hand_id = f"SIM-H{t:02d}{n:04d}"
        for i in policy.memo.coordinating:
            enabled[i] += 1
        for i in policy.memo.changed:
            coordinated[i].append(hand_id)
        log.add(rec, hand_id, d["table_id"], when, "simulation")
        when += timedelta(seconds=max(20.0, rng.lognormvariate(4.3, 0.3)))
        left = [x - 1 for x in left]
    truth = {
        "table_id": d["table_id"], "small_blind": sb, "big_blind": bb,
        "bots": [{"player_id": p, "name": d["names"][p], "style": d["styles"][p].name,
                  "params": {k: round(v, 3) for k, v in asdict(d["styles"][p]).items() if k != "name"}}
                 for p in d["ids"]],
        "planted": [{"pair_id": pair_id(c.a, c.b), "family": c.family, "a": c.a, "b": c.b,
                     "giver": c.a if c.family == "directed_transfer" else None,
                     "receiver": c.b if c.family == "directed_transfer" else None,
                     "intensity": c.intensity,
                     "windows": [[lo, min(hi, cfg.hands)] for lo, hi in c.windows],
                     "hands_enabled": enabled[i], "coordinated_hands": coordinated[i]}
                    for i, c in enumerate(d["pairs"])],
        "decoys": [{"pair_id": pair_id(x["a"], x["b"]), **x} for x in d["decoys"]],
    }
    return log, truth


def _run(args: tuple[int, SimConfig]) -> tuple[tuple[pl.DataFrame, ...], dict]:
    log, truth = run_table(*args)
    return log.frames(), truth


def generate(cfg: SimConfig, out_dir: Path = SIM_DIR, workers: int | None = None) -> dict:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run, [(t, cfg) for t in range(cfg.tables)]))
    hands, seats, actions = (pl.concat([r[0][k] for r in results]) for k in range(3))
    report = conformance(hands, seats, actions)
    bad = failed(report)
    if bad:
        raise AssertionError(f"SimRoom hands break the competition semantics: {bad}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("hands", hands), ("seats", seats), ("actions", actions)):
        df.write_parquet(out_dir / f"{name}.parquet")
    truth = {"config": asdict(cfg), "tables": [r[1] for r in results]}
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"SimRoom: {cfg.tables} tables x {cfg.hands} hands = {hands.height:,} hands, {actions.height:,} actions "
          f"({time.perf_counter() - t0:.0f}s); {len(report)} conformance checks passed -> {out_dir}")
    return truth


def load(out_dir: Path = SIM_DIR) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    frames = [pl.read_parquet(out_dir / f"{n}.parquet") for n in ("hands", "seats", "actions")]
    return *frames, json.loads((out_dir / "ground_truth.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=int, help="override the number of tables (quick runs)")
    ap.add_argument("--hands", type=int, help="override hands per table (quick runs)")
    ap.add_argument("--out", type=Path, default=SIM_DIR)
    args = ap.parse_args(argv)
    cfg = SimConfig.from_config()
    cfg = replace(cfg, tables=args.tables or cfg.tables, hands=args.hands or cfg.hands)
    generate(cfg, args.out)


if __name__ == "__main__":
    main()
