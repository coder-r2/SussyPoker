import json

import numpy as np
import pandas as pd
import pytest

from suspoker.config import EXTERNAL_DIR
from suspoker.metric import (
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
    MetricError,
    average_precision,
    average_precision_at_5,
    build_solution,
    score,
)


def frame(rows):
    """rows: (pair_id, risk, behavior, [evidence...])"""
    out = []
    for pair_id, risk, behavior, ev in rows:
        ev = list(ev) + [NO_EVIDENCE] * (5 - len(ev))
        out.append({"pair_id": pair_id, "risk_score": risk, "predicted_behavior": behavior,
                    **dict(zip(EVIDENCE_COLUMNS, ev, strict=True))})
    return pd.DataFrame(out)


class TestAveragePrecision:
    def test_perfect_ranking(self):
        assert average_precision(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0

    def test_hand_computed(self):
        # ranking: 1,0,1,0 -> precisions at hits: 1/1, 2/3 -> AP = (1 + 2/3) / 2
        ap = average_precision(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]))
        assert ap == pytest.approx((1 + 2 / 3) / 2)

    def test_no_positives_is_zero(self):
        assert average_precision(np.array([0, 0]), np.array([0.5, 0.1])) == 0.0

    def test_ties_keep_input_order(self):
        # tie: positive listed second is ranked second -> AP = 1/2
        assert average_precision(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


class TestEvidenceAP5:
    def test_all_hits(self):
        assert average_precision_at_5({"a", "b"}, ["a", "b"]) == 1.0

    def test_hand_computed(self):
        # hits at ranks 2 and 3 of 3 relevant: (1/2 + 2/3) / 3
        assert average_precision_at_5({"a", "b", "c"}, ["x", "a", "b"]) == pytest.approx((1 / 2 + 2 / 3) / 3)

    def test_only_first_five_count(self):
        assert average_precision_at_5({"f"}, ["a", "b", "c", "d", "e", "f"]) == 0.0

    def test_denominator_capped_at_five(self):
        relevant = {f"h{i}" for i in range(7)}
        assert average_precision_at_5(relevant, [f"h{i}" for i in range(5)]) == 1.0


class TestScore:
    solution = frame([
        ("P1", 1, "soft_play", ["h1", "h2"]),
        ("P2", 1, "directed_transfer", ["h3"]),
        ("P3", 0, "none", []),
        ("P4", 1, "coordinated_isolation", ["h4"]),
        ("P5", 0, "none", []),
    ])

    def test_perfect_submission_scores_one(self):
        s = score(self.solution, self.solution.assign(risk_score=[0.9, 0.8, 0.1, 0.7, 0.2]))
        assert (s.pair_ap, s.evidence_map, s.behavior_map) == (1.0, 1.0, 1.0)
        assert s.final == pytest.approx(1.0)

    def test_unpredicted_family_falls_to_tie_order(self):
        # Nobody is predicted coordinated_isolation, so every class score is 0 and the
        # tie is broken by pair_id: P4 lands at rank 4 -> AP = 1/4 (not exactly 0).
        sub = self.solution.assign(risk_score=[0.9, 0.8, 0.1, 0.7, 0.2])
        sub.loc[sub.pair_id == "P4", "predicted_behavior"] = "other_coordination"
        s = score(self.solution, sub)
        assert s.behavior_ap["coordinated_isolation"] == pytest.approx(1 / 4)
        assert s.behavior_map == pytest.approx((1 + 1 + 1 / 4) / 3)

    def test_family_absent_from_truth_scores_zero(self):
        sol = self.solution.copy()
        sol.loc[sol.pair_id == "P4", ["risk_score", "predicted_behavior"]] = [0, "none"]
        s = score(sol, sol.assign(risk_score=[0.9, 0.8, 0.1, 0.7, 0.2]))
        assert s.behavior_ap["coordinated_isolation"] == 0.0

    def test_weights(self):
        sub = self.solution.assign(risk_score=[0.9, 0.8, 0.1, 0.7, 0.2])
        for col in EVIDENCE_COLUMNS:
            sub[col] = NO_EVIDENCE
        s = score(self.solution, sub)
        assert s.final == pytest.approx(0.7 * 1.0 + 0.2 * 0.0 + 0.1 * 1.0)

    def test_tie_broken_by_pair_id(self):
        # all scores tied -> ranked P1..P5 by id -> labels 1,1,0,1,0
        s = score(self.solution, self.solution.assign(risk_score=0.5))
        assert s.pair_ap == pytest.approx((1 + 1 + 3 / 4) / 3)

    def test_row_order_does_not_matter(self):
        sub = self.solution.assign(risk_score=[0.9, 0.8, 0.1, 0.7, 0.2])
        assert score(self.solution, sub).final == score(self.solution, sub.iloc[::-1]).final

    @pytest.mark.parametrize("mutate,msg", [
        (lambda d: d.assign(risk_score=1.5), "between 0 and 1"),
        (lambda d: d.assign(predicted_behavior="weird"), "invalid predicted_behavior"),
        (lambda d: d.assign(evidence_hand_2=d["evidence_hand_1"]), "must not repeat"),
        (lambda d: d.iloc[:-1], "coverage mismatch"),
        (lambda d: d.drop(columns="evidence_hand_5"), "missing columns"),
    ])
    def test_invalid_submissions_raise(self, mutate, msg):
        sub = self.solution.assign(risk_score=0.5, evidence_hand_1=["a", "b", "c", "d", "e"])
        with pytest.raises(MetricError, match=msg):
            score(self.solution, mutate(sub))


def test_build_solution():
    labels = pd.DataFrame({"pair_id": ["P1", "P2"], "player_1": ["a", "c"], "player_2": ["b", "d"],
                           "label": [1, 0], "label_status": ["confirmed_target", "confirmed_non_target"],
                           "behavior_family": ["soft_play", "none"]})
    evidence = pd.DataFrame({"pair_id": ["P1", "P1"], "evidence_rank": [2, 1], "hand_id": ["h2", "h1"],
                             "behavior_family": ["soft_play"] * 2})
    sol = build_solution(labels, evidence)
    assert sol.loc[0, ["evidence_hand_1", "evidence_hand_2", "evidence_hand_3"]].tolist() == ["h1", "h2", NO_EVIDENCE]
    assert sol.loc[1, "evidence_hand_1"] == NO_EVIDENCE
    assert sol["risk_score"].tolist() == [1, 0]


OFFICIAL_NOTEBOOK = EXTERNAL_DIR / "metric" / "slash-poker-competition-metric.ipynb"


@pytest.mark.skipif(not OFFICIAL_NOTEBOOK.exists(), reason="official metric notebook not downloaded")
def test_parity_with_official_metric():
    source = "\n".join("".join(c["source"]) for c in json.loads(OFFICIAL_NOTEBOOK.read_text("utf-8"))["cells"]
                       if c["cell_type"] == "code")
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - trusted, pinned local copy of the official metric
    official = namespace["score"]

    rng = np.random.default_rng(0)
    families = ["directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"]
    hands = [f"H{i}" for i in range(40)]
    for trial in range(25):
        n = 60
        ids = [f"P{i:03d}" for i in rng.permutation(n)]
        labels = rng.random(n) < 0.3
        sol = frame([
            (pid, int(lab), rng.choice(families) if lab else "none",
             list(rng.choice(hands, size=rng.integers(1, 6), replace=False)) if lab else [])
            for pid, lab in zip(ids, labels, strict=True)
        ])
        # coarse scores create ties on purpose
        sub = frame([
            (pid, float(np.round(rng.random(), 1)), rng.choice(["none", *families]),
             list(rng.choice(hands, size=rng.integers(0, 6), replace=False)))
            for pid in ids
        ]).sample(frac=1, random_state=trial)
        assert score(sol, sub).final == pytest.approx(official(sol.copy(), sub.copy(), "pair_id"), abs=1e-12)
