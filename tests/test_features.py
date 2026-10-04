import polars as pl
import pytest

from suspoker.config import RAW_DIR, has_raw_data
from suspoker.features import CATALOG, GLOBAL_PRIOR, _lift, build_chunk
from suspoker.ingest import encode
from tests.fixtures import two_hands


def lift(ev_y, opp_y, ev_all, opp_all, g=0.5, k=20.0):
    df = pl.DataFrame({"ev_y": [ev_y], "opp_y": [opp_y], "ev_all": [ev_all], "opp_all": [opp_all]})
    return df.select(_lift(pl.col("ev_y"), pl.col("opp_y"), pl.col("ev_all"), pl.col("opp_all"), g, k)).item()


class TestLift:
    def test_no_difference_means_zero_lift(self):
        # 30% vs partner and 30% vs everyone else -> lift ~ 0 (only the tiny global prior moves it)
        assert abs(lift(30, 100, 330, 1100, g=0.3)) < 1e-9

    def test_few_opportunities_are_shrunk_toward_baseline(self):
        # 1 of 2 vs partner (50%) against a 20% habit: raw gap 0.30, shrunk to under 0.03
        small = lift(1, 2, 201, 1002, g=0.2)
        big = lift(20, 40, 220, 1040, g=0.2)  # the same 50%, but on 40 opportunities
        assert 0 < small < 0.03 < big

    def test_matches_formula(self):
        ev_y, opp_y, ev_all, opp_all, g, k = 12, 30, 112, 530, 0.2, 20.0
        base = (ev_all - ev_y + GLOBAL_PRIOR * g) / (opp_all - opp_y + GLOBAL_PRIOR)
        assert lift(ev_y, opp_y, ev_all, opp_all, g, k) == pytest.approx((ev_y + k * base) / (opp_y + k) - base)

    def test_less_than_usual_is_negative(self):
        assert lift(0, 30, 300, 1030, g=0.3) < -0.1


def test_build_chunk_on_fixture_is_symmetric_and_complete():
    t = encode(*two_hands())
    pairs = pl.DataFrame({"lo": [0], "hi": [1], "phase": [0]},
                         schema={"lo": pl.Int32, "hi": pl.Int32, "phase": pl.Int8})
    pf, hf, base = build_chunk(t, k=20, block_hands=500, hand_pairs=pairs)
    assert set(pf.columns) >= {"lo", "hi", "phase", "n_shared"}
    assert (pf["lo"] < pf["hi"]).all()
    for name in CATALOG:
        if name.endswith("_max") and name.replace("_max", "_min") in pf.columns:
            assert (pf[name].fill_null(0) >= pf[name.replace("_max", "_min")].fill_null(0)).all(), name
    # hand 1 (development): A and B both put chips in -> a candidate hand for the evidence ranker
    assert hf.height == 1 and hf["h_fold_ahead_max"][0] == 1
    assert base.filter(pl.col("phase") == 0).height == 3


@pytest.mark.data
@pytest.mark.skipif(not has_raw_data(), reason="competition data not available")
def test_evaluation_shared_hands_match_organizer_counts():
    from suspoker.pipeline import FEATURES_DIR

    path = FEATURES_DIR / "pair_features.parquet"
    if not path.exists():
        pytest.skip("run `python -m suspoker.pipeline --stage S5` first")
    pf = pl.read_parquet(path, columns=["player_lo", "player_hi", "phase", "n_shared"]).filter(pl.col("phase") == 1)
    ev = pl.read_csv(RAW_DIR / "evaluation_pairs.csv").with_columns(
        pl.min_horizontal("player_1", "player_2").alias("player_lo"),
        pl.max_horizontal("player_1", "player_2").alias("player_hi"))
    joined = ev.join(pf, on=["player_lo", "player_hi"], how="left")
    assert joined["n_shared"].null_count() == 0
    assert (joined["n_shared"] == joined["shared_hands"]).all()
