import pandas as pd
import polars as pl
import pytest

from suspoker.config import RAW_DIR, has_raw_data
from suspoker.metric import EVIDENCE_COLUMNS, NO_EVIDENCE
from suspoker.validate_submission import validate

EVAL_PAIRS = pd.DataFrame({
    "pair_id": ["P1", "P2", "P3"],
    "player_1": ["A", "C", "E"],
    "player_2": ["B", "D", "F"],
    "shared_hands": [40, 50, 60],
})
HANDS = pl.LazyFrame({"hand_id": ["H1", "H2", "H3", "H9"],
                      "phase": ["evaluation", "evaluation", "evaluation", "development"]})
SEATS = pl.LazyFrame({
    "hand_id": ["H1", "H1", "H2", "H2", "H3", "H3", "H9", "H9"],
    "player_id": ["A", "B", "C", "D", "E", "X", "A", "B"],
})


def good_submission() -> pd.DataFrame:
    sub = pd.DataFrame({
        "pair_id": ["P1", "P2", "P3"],
        "risk_score": [0.9, 0.5, 0.1],
        "predicted_behavior": ["directed_transfer", "soft_play", "coordinated_isolation"],
    })
    for col in EVIDENCE_COLUMNS:
        sub[col] = NO_EVIDENCE
    sub.loc[0, "evidence_hand_1"] = "H1"
    sub.loc[1, "evidence_hand_1"] = "H2"
    return sub


def errors(sub, **kw):
    return [i.message for i in validate(sub, EVAL_PAIRS, **kw) if i.level == "error"]


def test_good_submission_passes():
    assert errors(good_submission(), hands=HANDS, seats=SEATS) == []


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: d[["risk_score", "pair_id", *d.columns[2:]]], "columns must be exactly"),
    (lambda d: d.iloc[:2], "coverage mismatch"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate pair_id"),
    (lambda d: d.assign(risk_score=[0.9, 1.2, 0.1]), "within [0, 1]"),
    (lambda d: d.assign(risk_score=[0.9, float("nan"), 0.1]), "empty cells"),
    (lambda d: d.assign(predicted_behavior=["none", "soft_play", "bogus"]), "invalid predicted_behavior"),
    (lambda d: d.assign(predicted_behavior=["soft_play"] * 3), "never predicted"),
    (lambda d: d.assign(evidence_hand_2=["H1", NO_EVIDENCE, NO_EVIDENCE]), "repeat an evidence hand"),
    (lambda d: d.assign(evidence_hand_3=["", NO_EVIDENCE, NO_EVIDENCE]), "empty cells"),
])
def test_broken_submissions_fail(mutate, expected):
    assert any(expected in e for e in errors(mutate(good_submission())))


@pytest.mark.parametrize("hand,expected", [
    ("H404", "do not exist"),
    ("H9", "development period"),
    ("H2", "do not contain both players"),
    ("H3", "do not contain both players"),  # contains E but not F
])
def test_bad_evidence_hands(hand, expected):
    sub = good_submission()
    sub.loc[2, "evidence_hand_1"] = hand
    assert any(expected in e for e in errors(sub, hands=HANDS, seats=SEATS))


def test_tied_scores_warn():
    issues = validate(good_submission().assign(risk_score=0.5), EVAL_PAIRS)
    assert [i.level for i in issues] == ["warning"]


@pytest.mark.data
@pytest.mark.skipif(not has_raw_data(), reason="competition data not available")
def test_sample_submission_only_misses_families():
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv", keep_default_na=False)
    eval_pairs = pd.read_csv(RAW_DIR / "evaluation_pairs.csv")
    issues = validate(sample, eval_pairs)
    errs = [i.message for i in issues if i.level == "error"]
    assert len(errs) == 1 and "never predicted" in errs[0]
    # sample_submission predicts 'none' with constant scores -> all ties
    assert any(i.level == "warning" for i in issues)
