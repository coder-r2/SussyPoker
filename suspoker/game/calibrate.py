"""Calibration: do the SimRoom's players and pots look like the competition's? (aggregates only)

    python -m suspoker.game.calibrate --reference   # competition aggregates -> suspoker/resources/sim_reference.json
    python -m suspoker.game.calibrate               # SimRoom vs reference -> artifacts/simroom/calibration.md

Both sides go through the same code: player styles come from `suspoker.baselines` (the yardstick the model
itself uses), hand stats from the hands table. The reference file holds only quantiles and shares, never a
hand, player or table, so it can be committed (SPEC C5).
"""

import argparse
import json
from importlib import resources

import polars as pl

from suspoker.config import INTERIM_DIR, ROOT, load_config
from suspoker.features import build_chunk
from suspoker.ingest import Tables, encode

REFERENCE = ROOT / "suspoker" / "resources" / "sim_reference.json"
STYLE_STATS = ["vpip", "pfr", "af", "wtsd", "fold_rate", "call_rate", "raise_rate", "bet_unopened", "net_bb100"]
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MIN_HANDS = 200


def style_quantiles(base: pl.DataFrame) -> dict[str, list[float]]:
    b = base.filter(pl.col("hands") >= MIN_HANDS)
    return {c: [round(float(b[c].quantile(q)), 4) for q in QUANTILES] for c in STYLE_STATS}


def hand_stats(hands: pl.DataFrame, actions: pl.DataFrame) -> dict:
    """hands/actions: integer-coded (Tables)."""
    pot_bb = hands["final_pot"] / hands["bb"]
    pre = actions.filter(pl.col("street") == 0)
    return {
        "pot_bb": [round(float(pot_bb.quantile(q)), 2) for q in QUANTILES],
        "saw_flop": round(float((hands["b1"] >= 0).mean()), 4),
        "saw_river": round(float((hands["b5"] >= 0).mean()), 4),
        "showdown": round(float((hands["players_at_showdown"] > 0).mean()), 4),
        "preflop_fold_share": round(float((pre["action"] == 0).mean()), 4),
        "actions_per_hand": round(actions.height / hands.height, 2),
    }


def stats(t: Tables) -> dict:
    cfg = load_config()["features"]
    _, _, base = build_chunk(t, k=cfg["shrinkage_k"], block_hands=cfg["block_hands"], hand_pairs=None)
    return {"players": int((base["hands"] >= MIN_HANDS).sum()), "hands": t.hands.height,
            "style": style_quantiles(base), **hand_stats(t.hands, t.actions)}


def competition_reference() -> dict:
    """Development-period aggregates of the competition data (needs data/interim and the S5 baselines)."""
    from suspoker.pipeline import FEATURES_DIR

    base = pl.read_parquet(FEATURES_DIR / "player_baselines.parquet").filter(pl.col("phase") == 0)
    t = Tables.load(INTERIM_DIR)
    dev = t.hands.filter(pl.col("phase") == 0)
    acts = t.actions.join(dev.select("hand_idx"), on="hand_idx", how="semi")
    return {"source": "competition development period (aggregates only)", "players": int(
        (base["hands"] >= MIN_HANDS).sum()), "hands": dev.height, "style": style_quantiles(base),
        **hand_stats(dev, acts)}


def load_reference() -> dict:
    return json.loads(resources.files("suspoker.resources").joinpath("sim_reference.json").read_text("utf-8"))


def compare(sim: dict, ref: dict) -> tuple[list[dict], bool]:
    """Per statistic: the SimRoom median must fall inside the competition's 5-95% range."""
    rows = []
    for c in STYLE_STATS:
        r, s = ref["style"][c], sim["style"][c]
        rows.append({"stat": c, "competition": f"{r[0]:.3g} / {r[2]:.3g} / {r[4]:.3g}",
                     "simroom": f"{s[0]:.3g} / {s[2]:.3g} / {s[4]:.3g}", "ok": r[0] <= s[2] <= r[4]})
    r, s = ref["pot_bb"], sim["pot_bb"]
    rows.append({"stat": "pot size (bb)", "competition": f"{r[0]:.3g} / {r[2]:.3g} / {r[4]:.3g}",
                 "simroom": f"{s[0]:.3g} / {s[2]:.3g} / {s[4]:.3g}", "ok": r[1] <= s[2] <= r[3]})
    for c in ("saw_flop", "saw_river", "showdown", "preflop_fold_share"):
        rows.append({"stat": c, "competition": f"{ref[c]:.3f}", "simroom": f"{sim[c]:.3f}",
                     "ok": abs(sim[c] - ref[c]) <= 0.35 * ref[c]})
    return rows, all(r["ok"] for r in rows)


def report_markdown(rows: list[dict], sim: dict, ref: dict) -> str:
    lines = ["# SimRoom calibration", "",
             f"SimRoom: {sim['hands']:,} hands, {sim['players']} players with >= {MIN_HANDS} hands. "
             f"Competition: {ref['hands']:,} development hands, {ref['players']:,} players. Aggregates only.", "",
             "Player stats pass when the SimRoom median lies inside the competition's 5-95% range; pot size when "
             "it lies inside the interquartile range; hand shares when within 35% of the competition value.", "",
             "| Statistic | Competition | SimRoom | OK |", "|---|---|---|---|"]
    lines += [f"| {r['stat']} | {r['competition']} | {r['simroom']} | "
              f"{'yes' if r['ok'] else 'NO'} |" for r in rows]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", action="store_true", help="recompute the competition reference (needs data)")
    args = ap.parse_args(argv)
    if args.reference:
        ref = competition_reference()
        REFERENCE.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {REFERENCE}")
        return
    from suspoker.game.simroom import SIM_DIR, load

    hands, seats, actions, _ = load()
    sim = stats(encode(hands, seats, actions))
    rows, ok = compare(sim, load_reference())
    text = report_markdown(rows, sim, load_reference())
    (SIM_DIR / "calibration.md").write_text(text, encoding="utf-8")
    (SIM_DIR / "calibration.json").write_text(json.dumps({"simroom": sim, "rows": rows, "ok": ok}, indent=1),
                                              encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
