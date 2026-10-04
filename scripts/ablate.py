"""Feature-family ablation for M1 (SPEC 5.5): drop one family at a time, retrain out of fold, compare Pair AP.

    python scripts/ablate.py            # ~15-20 min; writes artifacts/ablation.csv

Uses the same folds, views and training recipe as `suspoker.train` (M1 only; Pair AP is 70% of the score).
"""

import time

import polars as pl

from suspoker import train
from suspoker.config import ARTIFACTS_DIR
from suspoker.features import CATALOG
from suspoker.metric import average_precision
from suspoker.modeling import PAIR_FEATURES, load_dev
from suspoker.pipeline import FEATURES_DIR

EXPOSURE_SHARES = {"n_shared_frac": "F1 exposure", "hands_max_frac": "F7 controls", "hands_min_frac": "F7 controls"}


def family_of(feature: str) -> str:
    return CATALOG[feature][0] if feature in CATALOG else EXPOSURE_SHARES[feature]


def main() -> None:
    t0 = time.perf_counter()
    d = load_dev(pl.read_parquet(FEATURES_DIR / "pair_features.parquet"),
                 pl.read_parquet(FEATURES_DIR / "pair_features_windows.parquet"))
    y, lab = d.mirror["label"].to_numpy(), d.mirror["labeled"].to_numpy()
    vi = 1 + list(d.windows).index(train.WINDOW_VIEW)
    families = sorted({family_of(f) for f in PAIR_FEATURES})
    rows = []
    for dropped in ["(none)", *families]:
        feats = [f for f in PAIR_FEATURES if family_of(f) != dropped]
        train.PAIR_FEATURES = feats  # oof_views/risk_training_set read the module-level list
        r = train.oof_risk(d)
        rows.append({"dropped": dropped, "n_features": len(feats),
                     "labeled_ap": average_precision(y[lab], r[0][lab]),
                     "mirror_ap": average_precision(y, r[0]), "window_ap": average_precision(y, r[vi])})
        print(f"[{time.perf_counter() - t0:5.0f}s] drop {dropped:<22} {len(feats):>2} features  "
              f"window AP {rows[-1]['window_ap']:.4f}  mirror AP {rows[-1]['mirror_ap']:.4f}", flush=True)
    out = pl.DataFrame(rows)
    full = out.row(0, named=True)
    out = out.with_columns((pl.col("window_ap") - full["window_ap"]).alias("window_delta"),
                           (pl.col("mirror_ap") - full["mirror_ap"]).alias("mirror_delta"))
    out.write_csv(ARTIFACTS_DIR / "ablation.csv")
    print(out)


if __name__ == "__main__":
    main()
