import polars as pl

from suspoker.metric import NO_EVIDENCE, SUBMISSION_COLUMNS
from suspoker.predict import build_submission


def test_build_submission_keeps_official_order_and_fills_missing_evidence():
    scored = pl.DataFrame({"pair_id": ["b", "a"], "risk": [0.9, 0.1],
                           "predicted_behavior": ["soft_play", "directed_transfer"]})
    evidence = pl.DataFrame({"pair_id": ["b"], **{f"evidence_hand_{r}": [f"h{r}"] for r in range(1, 6)}})
    sub = build_submission(scored, evidence, pl.Series(["a", "b"]))
    assert list(sub.columns) == list(SUBMISSION_COLUMNS)
    assert sub["pair_id"].tolist() == ["a", "b"]
    assert sub.iloc[0, 3:].tolist() == [NO_EVIDENCE] * 5
    assert sub.iloc[1]["risk_score"] == 0.9 and sub.iloc[1]["evidence_hand_5"] == "h5"
