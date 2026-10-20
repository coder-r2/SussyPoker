"""Web export, part 1 (P5): the aggregate-only model card data, `web/public/data/model_card.json`.

Everything here is a count, rate, score or curve. No pair, player, table or hand appears, and the self-check
refuses to write the file if any string looks like a competition ID (SPEC C5, DEC-014).

    python -m suspoker.export_web --model-card
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from suspoker.config import ARTIFACTS_DIR, MODELS_DIR, ROOT
from suspoker.error_analysis import (
    behavior_confusion,
    decoy_profile,
    evidence_by_family,
    flag_top_k,
    load_oof,
    missed_profile,
    outcome_counts,
    pr_curve,
    recall_by_family,
    with_window_features,
)
from suspoker.explain import BENIGN, EXPLAIN_DIR, TAGS, family_importance, feature_info
from suspoker.modeling import FAMILIES

WEB_DATA = ROOT / "web" / "public" / "data"
# competition ID formats: hands, pairs, players, tables (also matches our synthetic pair ids)
ID_PATTERN = re.compile(r"\b(H[0-9A-F]{14}|P[0-9A-F]{12}|U[0-9A-F]{12}|T[0-9A-F]{8})\b")
TOP_FEATURES = 25

CITATION = ("@misc{detect-suspicious-value-transfers-in-poker, author = {Slash — Detect Suspicious Value Transfers "
            "in Poker}, title = {Detect Suspicious Value Transfers in Poker}, year = {2026}, howpublished = "
            "{\\url{https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker}}, note = {Kaggle}}")
# late submissions (scored, not ranked); the deployed model is #4, the best locally (decision rule: local decides)
KAGGLE = {
    "note": "Late submissions after the competition closed: scored by Kaggle, not ranked. Ranks are estimates "
            "against the final public leaderboard.",
    "leaderboard": {"teams": 368, "median": 0.805, "p90": 0.900, "top": 0.935},
    "submissions": [
        {"n": 1, "commit": "8422fef", "change": "P3 model: risk, behavior, novelty and evidence models",
         "local_window_final": 0.606, "public": 0.78585, "private": 0.79167},
        {"n": 2, "commit": "6ca4b28", "change": "Preflop edge signatures (folding a better hand, junk calls)",
         "local_window_final": 0.622, "public": 0.82451, "private": 0.82328},
        {"n": 3, "commit": "33e95f4", "change": "Hand-suspicion features, cleaned training negatives, repeated CV",
         "local_window_final": 0.6458, "public": 0.83251, "private": 0.83775},
        {"n": 4, "commit": "05f7db7", "change": "Exact-odds decision equity and expected-value cost features",
         "local_window_final": 0.6510, "public": 0.83107, "private": 0.83579, "deployed": True},
    ],
}
VIEWS = {
    "labeled": "1,860 labeled development pairs (20% colluders), full development period",
    "mirror": "122,580 development pairs that pass the evaluation filter; unlabeled pairs count as innocent "
              "(0.3% colluders), full development period",
    "window": "the same pairs seen through a 2,000-hand window, the length of the evaluation period: the closest "
              "local stand-in for the leaderboard",
}


class LeakError(ValueError):
    """The export contains something that looks like a competition ID."""


def leak_check(text: str) -> None:
    hits = ID_PATTERN.findall(text)
    if hits:
        raise LeakError(f"refusing to write: {len(hits)} competition-ID-like strings (first: {hits[0][:2]}...)")


def _round(x, nd: int = 4):
    if isinstance(x, float | np.floating):
        return None if np.isnan(x) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v, nd) for v in x]
    return x


def _records(df: pl.DataFrame) -> list[dict]:
    return _round(df.to_dicts())


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def results_table(results: dict) -> list[dict]:
    rows = []
    for name, r in results["models"].items():
        for view in VIEWS:
            s = r[view]
            rows.append({"model": name, "note": r["note"], "view": view, "final": s["final"], "pair_ap": s["pair_ap"],
                         "evidence_map": s["evidence_map"], "behavior_map": s["behavior_map"]})
    return _round(rows)


def repeated_cv(tag: str = "ev_equity") -> dict:
    ex = pl.read_csv(ARTIFACTS_DIR / "experiments.csv").filter(pl.col("tag") == tag)
    cols = ["window_final", "window_pair_ap", "window_evidence", "window_behavior", "mirror_pair_ap"]
    return _round({"tag": tag, "fold_seeds": ex["seed"].to_list(),
                   **{c: {"mean": ex[c].mean(), "sd": ex[c].std()} for c in cols}})


def importances() -> dict:
    imp = pl.read_parquet(EXPLAIN_DIR / "importance_m1.parquet")
    out = {"population": "evaluation pairs (final models, mean |SHAP| in log-odds)",
           "risk": _records(imp.head(TOP_FEATURES).select("feature", "family", "label", "mean_abs_shap")),
           "risk_by_family": _records(family_importance(imp)), "behavior": {}}
    for fam in FAMILIES:
        m2 = pl.read_parquet(EXPLAIN_DIR / f"importance_m2_{fam}.parquet")
        out["behavior"][fam] = _records(m2.head(10).select("feature", "family", "label", "mean_abs_shap"))
    return out


def error_analysis() -> dict:
    p = load_oof()
    f = flag_top_k(with_window_features(p))
    rates, tops = decoy_profile(f)
    labels, preds, matrix = behavior_confusion(p)
    lab = p.filter(pl.col("labeled"))
    return {
        "operating_point": "break-even cut on the window view: the K highest-risk pairs, K = number of colluders",
        "counts": _round(outcome_counts(f)),
        "recall_by_family": _records(recall_by_family(f)),
        "decoy_rates": _records(rates),
        "decoy_top_template": _records(tops),
        "missed_profile": _records(missed_profile(f)),
        "evidence_by_family": _records(evidence_by_family()),
        "confusion": {"view": "window", "true": labels, "predicted": preds, "matrix": matrix},
        "pr_curves": {"labeled": pr_curve(lab["label"].to_numpy(), lab["risk"].to_numpy()),
                      "window": pr_curve(p["label"].to_numpy(), p["risk_window"].to_numpy())},
    }


def model_card() -> dict:
    results = json.loads((ARTIFACTS_DIR / "results.json").read_text(encoding="utf-8"))
    meta = json.loads((MODELS_DIR / "meta.json").read_text(encoding="utf-8"))
    ablation = pl.read_csv(ARTIFACTS_DIR / "ablation.csv")
    families = sorted({feature_info(f)[0] for f in meta["pair_features"]})
    return {
        "generated": time.strftime("%Y-%m-%d"),
        "code_version": git_sha(),
        "model": {"pair_features": len(meta["pair_features"]), "hand_features": len(meta["hand_features"]),
                  "feature_families": families, "seeds": len(meta["seeds"]), "behaviors": list(FAMILIES),
                  "calibration": _round(meta["calibration"])},
        "data": {"source": "Kaggle competition 'Detect Suspicious Value Transfers in Poker' (Slash)",
                 "citation": CITATION,
                 "use": "Trained on the competition's development period. No competition hands, pairs, players or "
                        "labels are published; this card shows aggregates only."},
        "validation": {"folds": results["validation"]["folds"], "views": VIEWS},
        "results": results_table(results),
        "repeated_cv": repeated_cv(),
        "kaggle": KAGGLE,
        "ablation": {"note": "P3 model (73 features); drop one family at a time, 1 fold seed",
                     "rows": _records(ablation)},
        "importance": importances(),
        "errors": error_analysis(),
        "explanations": {"evidence_tags": {k: t.text for k, t in TAGS.items()},
                         "benign_templates": {k: {"title": b.title, "trigger": b.feature, "tail": b.tail}
                                              for k, b in BENIGN.items()}},
    }


def write_json(obj: dict, path: Path) -> None:
    text = json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False)
    leak_check(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-card", action="store_true", help="write web/public/data/model_card.json")
    args = ap.parse_args(argv)
    if not args.model_card:
        ap.error("nothing to do: pass --model-card")
    out = WEB_DATA / "model_card.json"
    write_json(model_card(), out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB); leak check passed")


if __name__ == "__main__":
    main()
