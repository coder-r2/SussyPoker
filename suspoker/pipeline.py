"""Pipeline entry point: python -m suspoker.pipeline --all | --stage S1 | S5 | S5W | S6

S1   ingest raw competition files -> data/interim (integer-coded)
S5   strength, interactions, baselines, pair + hand features -> artifacts/features
     (S2-S4 run inside S5, chunked by table so memory stays bounded)
S5W  development pair features recomputed on 2,000-hand windows (the evaluation period's length), used to
     train and validate at evaluation-like exposure -> artifacts/features/pair_features_windows.parquet
S6   models: OOF validation of B0/B1/M1-M4 and the final fit (suspoker.train) -> artifacts/results.json, models/
"""

import argparse
import time
from pathlib import Path

import polars as pl

from suspoker.config import ARTIFACTS_DIR, INTERIM_DIR, RAW_DIR, ROOT, load_config
from suspoker.features import build_chunk, catalog_markdown
from suspoker.ingest import PHASES, Tables, ingest_raw

FEATURES_DIR = ARTIFACTS_DIR / "features"
CHUNK_TABLES = 40
WINDOW_HANDS = 2000          # = evaluation period length per table
WINDOW_STARTS = (0, 1000)    # two overlapping windows inside the 3,000-hand development period


def requested_hand_pairs(t: Tables) -> pl.DataFrame:
    """Pairs that need hand-level features: labeled development pairs and all evaluation pairs."""
    ids = t.player_ids
    labels = pl.read_csv(RAW_DIR / "development_labels.csv").select("player_1", "player_2").with_columns(
        pl.lit(PHASES["development"], dtype=pl.Int8).alias("phase"))
    evals = pl.read_csv(RAW_DIR / "evaluation_pairs.csv").select("player_1", "player_2").with_columns(
        pl.lit(PHASES["evaluation"], dtype=pl.Int8).alias("phase"))
    pairs = pl.concat([labels, evals])
    pairs = (pairs.join(ids.rename({"player_id": "player_1", "player_idx": "p1"}), on="player_1")
             .join(ids.rename({"player_id": "player_2", "player_idx": "p2"}), on="player_2"))
    return pairs.select(pl.min_horizontal("p1", "p2").alias("lo"), pl.max_horizontal("p1", "p2").alias("hi"), "phase")


def with_string_ids(df: pl.DataFrame, t: Tables) -> pl.DataFrame:
    """Replace internal integer codes with the original player / table / hand IDs (strings)."""
    ids = t.player_ids
    out = (df.join(ids.rename({"player_idx": "lo", "player_id": "player_lo"}), on="lo")
           .join(ids.rename({"player_idx": "hi", "player_id": "player_hi"}), on="hi"))
    if "table_idx" in out.columns:
        out = out.join(t.table_ids, on="table_idx")
    if "hand_idx" in out.columns:
        out = out.join(t.hand_ids, on="hand_idx")
    return out


def stage_s5(t: Tables, out_dir: Path = FEATURES_DIR, chunk_tables: int = CHUNK_TABLES) -> None:
    cfg = load_config()["features"]
    hand_pairs = requested_hand_pairs(t)
    tables = t.table_ids["table_idx"].to_list()
    pfs, hfs, bases = [], [], []
    t0 = time.perf_counter()
    for i in range(0, len(tables), chunk_tables):
        chunk = t.subset_tables(tables[i:i + chunk_tables])
        pf, hf, base = build_chunk(chunk, k=cfg["shrinkage_k"], block_hands=cfg["block_hands"], hand_pairs=hand_pairs)
        pfs.append(pf)
        hfs.append(hf)
        bases.append(base)
        print(f"  tables {i:>3}-{i + chunk_tables - 1:<3} done  ({time.perf_counter() - t0:5.0f}s)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = with_string_ids(pl.concat(pfs), t)
    pair.write_parquet(out_dir / "pair_features.parquet")
    with_string_ids(pl.concat(hfs), t).write_parquet(out_dir / "hand_features.parquet")
    pl.concat(bases).join(t.player_ids, on="player_idx").write_parquet(out_dir / "player_baselines.parquet")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "features.md").write_text(catalog_markdown(), encoding="utf-8")
    print(f"S5 done: {pair.height:,} pair rows -> {out_dir}")


def window_tables(t: Tables, start: int, length: int = WINDOW_HANDS) -> Tables:
    """Development hands number start..start+length-1 of each table (in started_at order)."""
    dev = (t.hands.filter(pl.col("phase") == 0).sort("table_idx", "started_at", "hand_idx")
           .with_columns(pl.int_range(pl.len()).over("table_idx").alias("pos")))
    return t.subset_hands(dev.filter(pl.col("pos").is_between(start, start + length - 1)).select("hand_idx"))


def stage_windows(t: Tables, out_dir: Path = FEATURES_DIR, chunk_tables: int = CHUNK_TABLES) -> None:
    cfg = load_config()["features"]
    out, t0 = [], time.perf_counter()
    for start in WINDOW_STARTS:
        w = window_tables(t, start)
        tables = w.table_ids["table_idx"].to_list()
        for i in range(0, len(tables), chunk_tables):
            pf, _, _ = build_chunk(w.subset_tables(tables[i:i + chunk_tables]), k=cfg["shrinkage_k"],
                                   block_hands=cfg["block_hands"], hand_pairs=None)
            out.append(pf.with_columns(pl.lit(start, pl.Int16).alias("window_start"),
                                       pl.lit(WINDOW_HANDS, pl.Int16).alias("period_hands")))
        print(f"  window {start}-{start + WINDOW_HANDS - 1} done  ({time.perf_counter() - t0:5.0f}s)", flush=True)
    pair = with_string_ids(pl.concat(out), t)
    pair.write_parquet(out_dir / "pair_features_windows.parquet")
    print(f"S5W done: {pair.height:,} window pair rows -> {out_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--stage", choices=["S1", "S5", "S5W", "S6"])
    args = p.parse_args(argv)
    if args.all or args.stage == "S1":
        print("S1 ingest ...")
        t = ingest_raw(RAW_DIR, INTERIM_DIR)
    elif args.stage in ("S5", "S5W"):
        t = Tables.load(INTERIM_DIR)
    if args.all or args.stage == "S5":
        print("S5 features ...")
        stage_s5(t)
    if args.all or args.stage == "S5W":
        print("S5W windowed development features ...")
        stage_windows(t)
    if args.all or args.stage == "S6":
        from suspoker.train import main as train_main  # imported late: train imports this module

        print("S6 models ...")
        train_main()


if __name__ == "__main__":
    main()
