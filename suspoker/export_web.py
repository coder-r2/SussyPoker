"""Web export: the static JSON behind the dashboard and the model card (web/public/data/).

Part 1 (P5), `model_card.json`: aggregate-only. Every number is a count, rate, score or curve; no competition
pair, player, table or hand appears.
Part 2 (P6), the dashboard: the SimRoom (our own simulated hands) scored by the Kaggle-trained models.
    summary.json            headline numbers, SimRoom metrics, calibration
    alerts.json             every scored SimRoom pair: risk, behavior, ground truth
    pairs/<pair_id>.json    detail for the top pairs, every planted pair and every decoy: SHAP waterfall,
                            timeline, evidence hands with reason tags, innocent explanations, ground truth
    hands/<hand_id>.json    replays of the hands those pages reference

Every file goes through a self-check that refuses to write anything that looks like a competition ID
(SPEC C5, DEC-014).

    python -m suspoker.export_web --model-card
    python -m suspoker.export_web --simroom       # needs artifacts/simroom (python -m suspoker.game.simroom)
"""

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from suspoker.config import ARTIFACTS_DIR, MODELS_DIR, ROOT, load_config
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
from suspoker.explain import BENIGN, EXPLAIN_DIR, TAGS, family_importance, feature_info, fit_reference
from suspoker.modeling import FAMILIES

WEB_DATA = ROOT / "web" / "public" / "data"
# competition ID formats: hands, pairs, players, tables (also matches our synthetic pair ids)
ID_PATTERN = re.compile(r"\b(H[0-9A-F]{14}|P[0-9A-F]{12}|U[0-9A-F]{12}|T[0-9A-F]{8})\b")
TOP_FEATURES = 25
SIM_BENIGN_REFERENCE = MODELS_DIR / "benign_reference_simroom.json"

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


def simroom_card() -> dict | None:
    """SimRoom metrics and calibration for model card section 9 (written by --simroom)."""
    from suspoker.game.simroom import SIM_DIR

    path = SIM_DIR / "metrics.json"
    if not path.exists():
        return None
    out = json.loads(path.read_text(encoding="utf-8"))
    cal = SIM_DIR / "calibration.json"
    if cal.exists():
        out["calibration"] = json.loads(cal.read_text(encoding="utf-8"))["rows"]
    return _round(out)


def live_game_card() -> dict | None:
    """Detection rates of bot-only Beat the Detector sessions (scripts/simulate_sessions.py), aggregates only."""
    path = ARTIFACTS_DIR / "game_sessions.json"
    if not path.exists():
        return None
    g = json.loads(path.read_text(encoding="utf-8"))
    return _round({"sessions_per_config": g["sessions_per_config"], "max_hands": g["max_hands"],
                   "chance_top1": g["chance_top1"], "summary": g["summary"]})


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
        "simroom": simroom_card(),
        "live_game": live_game_card(),
        "explanations": {"evidence_tags": {k: t.text for k, t in TAGS.items()},
                         "benign_templates": {k: {"title": b.title, "trigger": b.feature, "tail": b.tail}
                                              for k, b in BENIGN.items()}},
    }


def write_json(obj: dict, path: Path, compact: bool = False) -> None:
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False,
                      **({"separators": (",", ":")} if compact else {"indent": 1}))
    leak_check(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")


# ----------------------------------------------------------------------------- part 2: SimRoom dashboard
DETAIL_TOP = 100          # pair pages for the highest-risk pairs (plus every planted pair and decoy)
TIMELINE_HANDS = 10       # biggest-flow shared hands replayed per pair, besides evidence and coordinated hands
COORDINATED_SHOWN = 5
ROLLING = 25              # shared hands in the rolling suspicion window
SHAP_TOP = 10
BUDGET_MB = 15
BANNER = ("Simulated hands: every hand on this dashboard was played by our bots in the SimRoom. The detector was "
          "trained on the Kaggle competition 'Detect Suspicious Value Transfers in Poker' (Slash); no competition "
          "data is shown.")


def pair_timeline(pid: str, a: str, b: str, t, flows: pl.DataFrame, hand_p: pl.DataFrame,
                  hand_no: pl.DataFrame) -> dict:
    """Every shared hand of (a, b) in time order: net chips a -> b, running total, rolling hand suspicion."""
    ids = t.player_ids
    ia = ids.filter(pl.col("player_id") == a)["player_idx"][0]
    ib = ids.filter(pl.col("player_id") == b)["player_idx"][0]
    shared = (t.seats.filter(pl.col("player_idx") == ia).select("hand_idx")
              .join(t.seats.filter(pl.col("player_idx") == ib).select("hand_idx"), on="hand_idx").sort("hand_idx"))
    f = flows.filter(pl.col("a").is_in([ia, ib]) & pl.col("b").is_in([ia, ib]))
    net = f.group_by("hand_idx").agg(pl.when(pl.col("a") == ia).then(pl.col("flow_bb")).otherwise(-pl.col("flow_bb"))
                                     .sum().alias("ab"))
    p = hand_p.filter(pl.col("pair_id") == pid).select("hand_idx", "p")
    tl = (shared.join(net, on="hand_idx", how="left").join(p, on="hand_idx", how="left")
          .join(hand_no, on="hand_idx", how="left").join(t.hand_ids, on="hand_idx", how="left")
          .sort("hand_idx").with_columns(pl.col("ab").fill_null(0.0), pl.col("p").fill_null(0.0))
          .with_columns(pl.col("ab").cum_sum().alias("cum"),
                        pl.col("p").rolling_mean(ROLLING, min_samples=1).alias("susp")))
    return {"a": a, "b": b, "hand_ids": tl["hand_id"].to_list(), "table_hand_no": tl["hand_no"].to_list(),
            "flow_bb": _round(tl["ab"].to_list(), 2), "cum_flow_bb": _round(tl["cum"].to_list(), 2),
            "suspicion": _round(tl["susp"].to_list(), 3)}


def waterfall(row: int, contrib: np.ndarray, values: dict, features: list[str], medians: dict) -> dict:
    """Top SHAP contributions of one pair's risk score, with the pair's value and the population median."""
    phi = contrib[row, :len(features)]
    order = sorted(range(len(features)), key=lambda i: -abs(phi[i]))
    return _round({
        "units": "log-odds", "base": float(contrib[row, -1]), "raw": float(contrib[row].sum()),
        "rest": float(sum(phi[i] for i in order[SHAP_TOP:])),
        "features": [{"feature": features[i], "family": feature_info(features[i])[0],
                      "label": feature_info(features[i])[1], "value": values[features[i]],
                      "median": medians[features[i]], "shap": float(phi[i])} for i in order[:SHAP_TOP]]})


def simroom_export(out_dir: Path = WEB_DATA) -> dict:
    from suspoker.game.calibrate import compare, load_reference, report_markdown, stats
    from suspoker.game.evaluate import simroom_metrics, with_truth
    from suspoker.game.replay import pair_highlights, replays
    from suspoker.game.simroom import SIM_DIR, load, pair_id
    from suspoker.interactions import flows, hand_context
    from suspoker.predict import Models
    from suspoker.scoring import attach_benign, score_tables

    t0 = time.perf_counter()
    hands, seats, actions, truth = load()
    w = score_tables(hands, seats, actions, Models.load(), pair_ids=pair_id)
    print(f"  scored {w.pairs.height:,} SimRoom pairs ({time.perf_counter() - t0:.0f}s)", flush=True)
    # innocent explanations compare a pair with the other pairs of the same card room (simulated aggregates);
    # the reference is saved for the live game (P8)
    sim_ref = fit_reference(w.pairs)
    SIM_BENIGN_REFERENCE.write_text(json.dumps({
        "population": f"{w.pairs.height} SimRoom pairs (percentiles only, simulated data)", "percentiles": sim_ref},
        indent=1) + "\n", encoding="utf-8", newline="\n")
    w.pairs = attach_benign(w.pairs.drop("benign"), sim_ref)

    metrics = _round(simroom_metrics(w.pairs, w.evidence, w.hands, truth))
    (SIM_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    ref = load_reference()
    sim_stats = stats(w.tables)
    rows, cal_ok = compare(sim_stats, ref)
    (SIM_DIR / "calibration.json").write_text(json.dumps({"simroom": sim_stats, "rows": rows, "ok": cal_ok},
                                                         indent=1), encoding="utf-8")
    (SIM_DIR / "calibration.md").write_text(report_markdown(rows, sim_stats, ref), encoding="utf-8")
    results_path = ARTIFACTS_DIR / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["simroom"] = metrics  # train.py rewrites results.json: re-run this export after retraining
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    bots = {b["player_id"]: b for tb in truth["tables"] for b in tb["bots"]}
    names = {pid: b["name"] for pid, b in bots.items()}
    planted = {x["pair_id"]: x for tb in truth["tables"] for x in tb["planted"]}
    decoys = {x["pair_id"]: x for tb in truth["tables"] for x in tb["decoys"]}
    p = with_truth(w.pairs, truth).with_row_index("row")
    p = p.sort(["risk", "pair_id"], descending=[True, False]).with_row_index("rank", offset=1)
    k = metrics["break_even"]["k"]
    threshold = float(p.filter(pl.col("rank") == k)["risk"][0]) if k else 0.5
    detail = set(p.filter(pl.col("rank") <= DETAIL_TOP)["pair_id"]) | ((set(planted) | set(decoys))
                                                                       & set(p["pair_id"]))
    probs = [f"p_{f}" for f in FAMILIES]

    alerts = [{
        "rank": r["rank"], "pair_id": r["pair_id"], "table_id": r["table_id"],
        "players": [{"player_id": x, "name": names[x]} for x in (r["player_lo"], r["player_hi"])],
        "shared_hands": r["n_shared"], "risk": r["risk"], "behavior": r["predicted_behavior"],
        "probs": {f: r[c] for f, c in zip(FAMILIES, probs, strict=True)},
        "truth": {"role": r["role"], "family": r["family"], "decoy": r["decoy"]},
        "detail": r["pair_id"] in detail} for r in p.iter_rows(named=True)]

    ctx = hand_context(w.tables, load_config()["features"]["block_hands"])
    fl = flows(w.tables, ctx)
    hand_no = w.tables.hands.select("hand_idx", (pl.col("hand_idx") - pl.col("hand_idx").min().over("table_idx"))
                                    .alias("hand_no"))
    hand_p = w.hands.select("pair_id", "hand_idx", "p")
    feats = w.features
    medians = {f: float(np.nanmedian(p[f].cast(pl.Float64).to_numpy())) for f in feats}
    ev_by = {kk[0]: g.sort("rank") for kk, g in w.evidence.group_by("pair_id")}
    pot = w.hands.select("pair_id", "hand_id", "h_pot_bb")

    pages, wanted = {}, set()
    for r in p.filter(pl.col("pair_id").is_in(list(detail))).iter_rows(named=True):
        pid = r["pair_id"]
        tl = pair_timeline(pid, r["player_lo"], r["player_hi"], w.tables, fl, hand_p, hand_no)
        ev = ev_by.get(pid)
        ev_rows = [] if ev is None else ev.join(pot, on=["pair_id", "hand_id"], how="left").to_dicts()
        coord = planted.get(pid, {}).get("coordinated_hands", [])
        biggest = sorted(range(len(tl["hand_ids"])), key=lambda i: -abs(tl["flow_bb"][i]))[:TIMELINE_HANDS]
        replay_ids = list(dict.fromkeys([e["hand_id"] for e in ev_rows] + coord[:COORDINATED_SHOWN]
                                        + [tl["hand_ids"][i] for i in biggest]))
        wanted |= set(replay_ids)
        truth_block = {"role": r["role"], "family": r["family"], "decoy": r["decoy"]}
        if pid in planted:
            x = planted[pid]
            truth_block |= {"intensity": x["intensity"], "windows": x["windows"], "giver": x["giver"],
                            "receiver": x["receiver"], "coordinated_hands": coord,
                            "behavior_correct": r["predicted_behavior"] == r["family"]}
        pages[pid] = {
            "pair_id": pid, "table_id": r["table_id"], "rank": r["rank"], "shared_hands": r["n_shared"],
            "players": [{"player_id": x, "name": names[x], "style": bots[x]["style"]}
                        for x in (r["player_lo"], r["player_hi"])],
            "risk": r["risk"], "behavior": r["predicted_behavior"],
            "probs": {f: r[c] for f, c in zip(FAMILIES, probs, strict=True)},
            "novelty_pct": r["novelty_pct"], "truth": truth_block,
            "shap": waterfall(r["row"], w.risk_contrib, {f: r[f] for f in feats}, feats, medians),
            "timeline": tl,
            "evidence": [{"rank": e["rank"], "hand_id": e["hand_id"], "score": e["score"], "pot_bb": e["h_pot_bb"],
                          "tags": e["tags"], "coordinated": e["hand_id"] in set(coord)} for e in ev_rows],
            "benign": r["benign"], "replays": replay_ids, "highlights": {}}

    reps = replays(sorted(wanted), hands, seats, actions, names)
    for page in pages.values():
        a, b = page["players"][0]["player_id"], page["players"][1]["player_id"]
        page["highlights"] = {h: pair_highlights(reps[h], a, b) for h in page["replays"]}

    for sub in ("pairs", "hands"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    summary = {
        "generated": time.strftime("%Y-%m-%d"), "code_version": git_sha(), "banner": BANNER, "citation": CITATION,
        "simroom": {"tables": truth["config"]["tables"], "hands_per_table": truth["config"]["hands"],
                    "bots_per_table": truth["config"]["pool"], "hands": hands.height,
                    "pairs_total": sum(len(tb["bots"]) * (len(tb["bots"]) - 1) // 2 for tb in truth["tables"]),
                    "stakes": sorted({f"{tb['small_blind']}/{tb['big_blind']}" for tb in truth["tables"]})},
        "metrics": metrics, "default_threshold": threshold, "calibration": {"ok": cal_ok, "rows": rows},
        "kaggle": {"best_private": max(x["private"] for x in KAGGLE["submissions"]),
                   "deployed": next(x for x in KAGGLE["submissions"] if x.get("deployed"))},
        "families": list(FAMILIES), "detail_pairs": len(pages), "replays": len(reps)}
    write_json(_round(summary), out_dir / "summary.json")
    write_json(_round({"threshold": threshold, "alerts": alerts}), out_dir / "alerts.json", compact=True)
    for pid, page in pages.items():
        write_json(_round(page), out_dir / "pairs" / f"{pid}.json", compact=True)
    for hid, rep in reps.items():
        write_json(rep, out_dir / "hands" / f"{hid}.json", compact=True)
    total = sum(f.stat().st_size for f in out_dir.rglob("*.json")) / 2**20
    if total > BUDGET_MB:
        raise ValueError(f"web export is {total:.1f} MB, over the {BUDGET_MB} MB budget")
    m = metrics["models"]["M_full"]
    print(f"SimRoom export: {len(alerts):,} alerts, {len(pages)} pair pages, {len(reps):,} replays, {total:.1f} MB "
          f"total; SimRoom final {m['final']:.3f} (Pair AP {m['pair_ap']:.3f}, evidence {m['evidence_map']:.3f}, "
          f"behavior {m['behavior_map']:.3f}); leak check passed ({time.perf_counter() - t0:.0f}s)")
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-card", action="store_true", help="write web/public/data/model_card.json")
    ap.add_argument("--simroom", action="store_true", help="score the SimRoom and write the dashboard data")
    args = ap.parse_args(argv)
    if not (args.model_card or args.simroom):
        ap.error("nothing to do: pass --simroom and/or --model-card")
    if args.simroom:
        simroom_export()
    if args.model_card:
        out = WEB_DATA / "model_card.json"
        write_json(model_card(), out)
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB); leak check passed")


if __name__ == "__main__":
    main()
