"""P2 checks on the feature tables (run after `python -m suspoker.pipeline --stage S5`).

1. Exposure: our evaluation-period shared-hand counts must equal evaluation_pairs.csv exactly.
2. Evidence coverage: planted development evidence hands must survive the hand-candidate filter.
3. Univariate signal: AP of each feature alone on labeled development pairs (overall and per family).
Writes artifacts/features/univariate.csv and prints a summary.
"""

import numpy as np
import polars as pl

from suspoker.config import RAW_DIR
from suspoker.features import CATALOG
from suspoker.metric import TARGET_BEHAVIORS, average_precision
from suspoker.pipeline import FEATURES_DIR


def keyed(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.min_horizontal("player_1", "player_2").alias("player_lo"),
                           pl.max_horizontal("player_1", "player_2").alias("player_hi"))


def main() -> None:
    pf = pl.read_parquet(FEATURES_DIR / "pair_features.parquet")
    hf = pl.read_parquet(FEATURES_DIR / "hand_features.parquet")
    evals = keyed(pl.read_csv(RAW_DIR / "evaluation_pairs.csv"))
    labels = keyed(pl.read_csv(RAW_DIR / "development_labels.csv"))
    evidence = pl.read_csv(RAW_DIR / "development_evidence.csv")

    # 1. exposure equality
    e = evals.join(pf.filter(pl.col("phase") == 1), on=["player_lo", "player_hi"], how="left")
    missing = e["n_shared"].null_count()
    mismatch = e.filter(pl.col("n_shared") != pl.col("shared_hands")).height
    print(f"[1] evaluation pairs: {evals.height:,}; missing features: {missing}; shared_hands mismatches: {mismatch}")

    # 2. evidence coverage
    lab_hands = hf.filter(pl.col("phase") == 0).join(labels.select("pair_id", "player_lo", "player_hi"),
                                                     on=["player_lo", "player_hi"])
    covered = evidence.join(lab_hands.select("pair_id", "hand_id"), on=["pair_id", "hand_id"], how="semi").height
    per_pair = lab_hands.group_by("pair_id").len()
    print(f"[2] evidence hands kept by the candidate filter: {covered:,} / {evidence.height:,} "
          f"({covered / evidence.height:.1%}); median candidate hands per labeled pair: {per_pair['len'].median()}")
    ev_hands = hf.filter(pl.col("phase") == 1).group_by("player_lo", "player_hi").len()
    print(f"    evaluation pairs with >=1 candidate hand: {ev_hands.height:,} / {evals.height:,}; "
          f"median candidates per pair: {ev_hands['len'].median()}")

    # 3. univariate AP on labeled development pairs
    d = labels.join(pf.filter(pl.col("phase") == 0), on=["player_lo", "player_hi"], how="inner")
    print(f"[3] labeled development pairs with features: {d.height:,} / {labels.height:,}")
    y = d["label"].to_numpy()
    rows = []
    for name in [c for c in CATALOG if c in d.columns]:
        x = d[name].fill_null(d[name].median()).to_numpy().astype(float)
        row = {"feature": name, "family": CATALOG[name][0]}
        ap_pos, ap_neg = average_precision(y, x), average_precision(y, -x)
        row["ap"], row["direction"] = (ap_pos, "+") if ap_pos >= ap_neg else (ap_neg, "-")
        for fam in TARGET_BEHAVIORS:
            yf = (d["behavior_family"] == fam).to_numpy().astype(int)
            row[f"ap_{fam}"] = max(average_precision(yf, x), average_precision(yf, -x))
        rows.append(row)
    uni = pl.DataFrame(rows).sort("ap", descending=True)
    uni.write_csv(FEATURES_DIR / "univariate.csv")
    base = y.mean()
    rates = {f: (d["behavior_family"] == f).mean() for f in TARGET_BEHAVIORS}
    print(f"    base rate {base:.3f}; family base rates " + ", ".join(f"{k} {v:.3f}" for k, v in rates.items()))
    with pl.Config(tbl_rows=25, tbl_width_chars=200, fmt_str_lengths=40, float_precision=3):
        print(uni.head(25))
        print("best feature per family (one-vs-rest AP among labeled pairs):")
        for fam in TARGET_BEHAVIORS:
            top = uni.sort(f"ap_{fam}", descending=True).head(3)
            best = "; ".join(f"{r['feature']} {r[f'ap_{fam}']:.3f}" for r in top.iter_rows(named=True))
            print(f"  {fam:<22} {best}")
        print("best per feature family (overall AP):")
        print(uni.group_by("family").agg(pl.col("feature").first(), pl.col("ap").max()).sort("ap", descending=True))
    _ = np


if __name__ == "__main__":
    main()
