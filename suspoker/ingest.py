"""S1 ingest: competition-schema tables -> compact integer-coded tables.

Works on any data in the competition schema (Kaggle files, the SimRoom, a live game session).
Integer codes are internal only; the *_ids mapping tables translate back to the original strings
(submissions must always use the original pair/hand IDs).

Hand order: hands are numbered by (table_id, started_at), so within a table `hand_idx` follows
gameplay time. It never depends on hand_id strings or file order (SPEC C1).
"""

from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

from suspoker.cards import CARD_INDEX, NO_CARD

PHASES = {"development": 0, "evaluation": 1, "simulation": 2}
STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
ACTIONS = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4, "all_in": 5}
FOLD, CHECK, CALL, BET, RAISE, ALL_IN = range(6)


@dataclass
class Tables:
    hands: pl.DataFrame
    seats: pl.DataFrame
    actions: pl.DataFrame
    hand_ids: pl.DataFrame    # hand_idx -> hand_id
    player_ids: pl.DataFrame  # player_idx -> player_id
    table_ids: pl.DataFrame   # table_idx -> table_id

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for f in fields(self):
            getattr(self, f.name).write_parquet(directory / f"{f.name}.parquet")

    @classmethod
    def load(cls, directory: Path) -> "Tables":
        return cls(**{f.name: pl.read_parquet(directory / f"{f.name}.parquet") for f in fields(cls)})

    def subset_tables(self, table_idx: list[int]) -> "Tables":
        """Restrict to some tables (hand_idx ranges are contiguous per table)."""
        return self.subset_hands(self.hands.filter(pl.col("table_idx").is_in(table_idx)).select("hand_idx"))

    def subset_hands(self, keep: pl.DataFrame) -> "Tables":
        """Restrict to the hands listed in `keep` (a frame with a hand_idx column)."""
        return Tables(
            hands=self.hands.join(keep, on="hand_idx", how="semi"),
            seats=self.seats.join(keep, on="hand_idx", how="semi"),
            actions=self.actions.join(keep, on="hand_idx", how="semi"),
            hand_ids=self.hand_ids.join(keep, on="hand_idx", how="semi"),
            player_ids=self.player_ids,
            table_ids=self.table_ids,
        )


def _card(col: str | pl.Expr) -> pl.Expr:
    expr = pl.col(col) if isinstance(col, str) else col
    return expr.replace_strict(CARD_INDEX, default=NO_CARD, return_dtype=pl.Int8)


def encode(hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame) -> Tables:
    """Encode competition-schema DataFrames (string IDs, string cards) to integer-coded Tables."""
    table_ids = (hands.select("table_id").unique().sort("table_id")
                 .with_row_index("table_idx").with_columns(pl.col("table_idx").cast(pl.Int32)))
    hand_ids = (hands.select("hand_id", "table_id", "started_at")
                .sort("table_id", "started_at", "hand_id")
                .with_row_index("hand_idx").with_columns(pl.col("hand_idx").cast(pl.Int32))
                .select("hand_idx", "hand_id"))
    player_ids = (seats.select("player_id").unique().sort("player_id")
                  .with_row_index("player_idx").with_columns(pl.col("player_idx").cast(pl.Int32)))

    board = pl.col("board_cards").str.split(" ")
    h = (
        hands.join(hand_ids, on="hand_id").join(table_ids, on="table_id")
        .with_columns(
            pl.col("phase").replace_strict(PHASES, return_dtype=pl.Int8),
            *[_card(board.list.get(i, null_on_oob=True)).alias(f"b{i + 1}") for i in range(5)],
        )
        .select(
            "hand_idx", "table_idx", "started_at", "phase",
            pl.col("button_seat").cast(pl.Int8),
            pl.col("small_blind").cast(pl.Int32).alias("sb"),
            pl.col("big_blind").cast(pl.Int32).alias("bb"),
            "b1", "b2", "b3", "b4", "b5",
            pl.col("final_pot").cast(pl.Int32),
            pl.col("players_at_showdown").cast(pl.Int8),
        )
        .sort("hand_idx")
    )
    s = (
        seats.join(hand_ids, on="hand_id").join(player_ids, on="player_id")
        .select(
            "hand_idx", "player_idx",
            pl.col("seat_no").cast(pl.Int8),
            pl.col("starting_stack").cast(pl.Int32),
            _card("hole_card_1").alias("c1"), _card("hole_card_2").alias("c2"),
            pl.col("total_contribution").cast(pl.Int32).alias("contrib"),
            pl.col("net_chips").cast(pl.Int32).alias("net"),
            "folded",
            pl.col("went_to_showdown").alias("sd"),
            pl.col("won_share").cast(pl.Float64),
        )
        .sort("hand_idx", "seat_no")
    )
    a = (
        actions.join(hand_ids, on="hand_id").join(player_ids, on="player_id")
        .select(
            "hand_idx",
            pl.col("action_no").cast(pl.Int16),
            pl.col("street").replace_strict(STREETS, return_dtype=pl.Int8),
            "player_idx",
            pl.col("action").replace_strict(ACTIONS, return_dtype=pl.Int8),
            *[pl.col(c).cast(pl.Int32) for c in ("amount", "amount_to", "pot_before", "stack_before", "to_call")],
            pl.col("players_active").cast(pl.Int8),
        )
        .sort("hand_idx", "action_no")
    )
    return Tables(hands=h, seats=s, actions=a, hand_ids=hand_ids, player_ids=player_ids,
                  table_ids=table_ids.select("table_idx", "table_id"))


def ingest_raw(raw_dir: Path, out_dir: Path) -> Tables:
    """Encode the competition files in `raw_dir` and save them to `out_dir`."""
    t = encode(pl.read_parquet(raw_dir / "hands.parquet"),
               pl.read_parquet(raw_dir / "seats.parquet"),
               pl.read_parquet(raw_dir / "actions.parquet"))
    t.save(out_dir)
    return t
