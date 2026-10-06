"""P3: train and validate baselines B0/B1 and models M1-M4 out of fold, then fit the final models.

    python -m suspoker.train            # OOF validation + final models (~2 min)

Outputs
  artifacts/results.json         every model/baseline x 3 metric components + final, on 3 validation sets
  artifacts/oof/pairs.parquet    OOF pair predictions (mirror set: labeled + eval-filtered unlabeled dev pairs)
  artifacts/oof/evidence.parquet OOF evidence scores for the hands of labeled positive pairs
  models/                        final LightGBM boosters (text), M3 Isolation Forest, meta.json

Models (SPEC 5.4): M1 risk = LightGBM binary on pair features, trained on labeled pairs plus a sample of
unlabeled development pairs as (noisy) negatives; M2 behavior = 3-class LightGBM on positives; M3 novelty
= Isolation Forest on lift features; M4 evidence = LightGBM binary at hand level on positive pairs' hands.
M1 and M2 train on every exposure view at once (full development period + 2,000-hand windows) so they also
learn what collusion looks like at the evaluation period's shorter exposure (P3 shift check).
"""

import json
import pickle
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from suspoker.config import ARTIFACTS_DIR, MODELS_DIR, load_config
from suspoker.metric import EVIDENCE_COLUMNS, NO_EVIDENCE, SUBMISSION_COLUMNS, Score, build_solution, score
from suspoker.modeling import (
    CROSS_FEATURES,
    FAMILIES,
    KEYS,
    LGB_BINARY,
    LGB_MULTI,
    PAIR_FEATURES,
    DevData,
    fit_predict,
    hand_context,
    hand_feature_names,
    load_dev,
    matrix,
    predict_proba,
)
from suspoker.pipeline import DEV_HANDS_DIR, FEATURES_DIR, WINDOW_HANDS
from suspoker.suspicion import HS_FEATURES, attach, fit_hand_models, labeled_hands, oof_suspicion

OOF_DIR = ARTIFACTS_DIR / "oof"
N_UNLABELED = 5000          # unlabeled development pairs added as negatives per view and M1 fit
SEEDS = (0, 1, 2)
WINDOW_VIEW = 1000          # validation window: development hands 1000-2999 (evaluation length, latest in time)
NONE_MIN_GAIN = 0.005       # Behavior MAP gain needed before predicting `none` for low-risk pairs
LGB_EVIDENCE = dict(LGB_BINARY, n_estimators=300)
NOVELTY_FEATURES = [c for c in PAIR_FEATURES if "_lift" in c or "_pres_" in c]
FEATURES = list(PAIR_FEATURES)  # M1/M2 inputs; main() appends CROSS_FEATURES when model.cross_period is on
FAMILY_PROBS = [f"p_{f}" for f in FAMILIES]


# ----------------------------------------------------------------------------- M1 risk
EXCLUDED_UNLABELED: pl.DataFrame | None = None  # likely hidden colluders kept out of the negatives (DEC-024)


def risk_training_set(d: DevData, train_mask: pl.Expr, sample_seed: int) -> pl.DataFrame:
    parts = []
    for i, view in enumerate(d.views()):
        lab = view.filter(pl.col("labeled") & train_mask)
        unl = view.filter(~pl.col("labeled") & train_mask)
        if EXCLUDED_UNLABELED is not None:
            unl = unl.join(EXCLUDED_UNLABELED, on="pair_id", how="anti")
        parts += [lab, unl.sample(min(N_UNLABELED, unl.height), seed=sample_seed + 100 * i)]
    return pl.concat([p.select("label", "behavior_family", *FEATURES) for p in parts])


def oof_views(d: DevData, params: dict, train_set, target) -> list[np.ndarray]:
    """Out-of-fold predictions for every view (rows in d.mirror order), from one model per fold."""
    out: list[np.ndarray] = [None] * len(d.views())  # type: ignore[list-item]
    for f in range(d.labeled["fold"].max() + 1):
        tr = train_set(f)
        test = (d.mirror["fold"] == f).to_numpy()
        tests = [matrix(v.filter(pl.Series(test)), FEATURES) for v in d.views()]
        _, models = fit_predict(params, matrix(tr, FEATURES), target(tr), seeds=SEEDS)
        for i, x in enumerate(tests):
            p = predict_proba(models, x)
            if out[i] is None:
                out[i] = np.zeros((d.mirror.height, *p.shape[1:]))
            out[i][test] = p
    return out


def oof_risk(d: DevData, clean_above: float | None = None) -> list[np.ndarray]:
    """OOF risk per view. With `clean_above`, a first pass scores every unlabeled pair out of fold, and pairs
    above the threshold (likely hidden colluders) are excluded from the negatives of the second pass and of
    the final fit. Their first-pass models saw other folds' labels: a small, second-order leak, accepted as in
    standard stacking."""
    global EXCLUDED_UNLABELED
    EXCLUDED_UNLABELED = None

    def run() -> list[np.ndarray]:
        return oof_views(d, LGB_BINARY, lambda f: risk_training_set(d, pl.col("fold") != f, sample_seed=f),
                         lambda tr: tr["label"].to_numpy())

    first = run()
    if clean_above is None:
        return first
    EXCLUDED_UNLABELED = d.mirror.filter(~pl.col("labeled") & pl.Series(first[0] > clean_above)).select("pair_id")
    print(f"  excluding {EXCLUDED_UNLABELED.height} unlabeled pairs with first-pass risk > {clean_above} "
          "from the negatives", flush=True)
    return run()


# ----------------------------------------------------------------------------- M2 behavior
def family_index(df: pl.DataFrame) -> np.ndarray:
    codes = {f: i for i, f in enumerate(FAMILIES)}
    return df["behavior_family"].replace_strict(codes, return_dtype=pl.Int32).to_numpy()


def behavior_training_set(d: DevData, train_mask: pl.Expr) -> pl.DataFrame:
    return pl.concat([v.filter(pl.col("labeled") & (pl.col("label") == 1) & train_mask)
                      .select("behavior_family", *FEATURES) for v in d.views()])


def oof_behavior(d: DevData) -> list[np.ndarray]:
    return oof_views(d, LGB_MULTI, lambda f: behavior_training_set(d, pl.col("fold") != f), family_index)


# ----------------------------------------------------------------------------- M3 novelty
def fit_novelty(train: pl.DataFrame, seed: int) -> IsolationForest:
    return IsolationForest(n_estimators=200, random_state=seed).fit(np.nan_to_num(matrix(train, NOVELTY_FEATURES)))


def novelty_percentile(model: IsolationForest, df: pl.DataFrame) -> np.ndarray:
    """0-100 percentile of novelty within the scored population (100 = most unusual)."""
    s = -model.score_samples(np.nan_to_num(matrix(df, NOVELTY_FEATURES)))
    return 100.0 * pd.Series(s).rank(pct=True).to_numpy()


def oof_novelty(view: pl.DataFrame, seed: int) -> np.ndarray:
    out = np.zeros(view.height)
    for f in range(view["fold"].max() + 1):  # unsupervised, but kept out of fold anyway
        test = (view["fold"] == f).to_numpy()
        out[test] = novelty_percentile(fit_novelty(view.filter(pl.Series(~test)), seed), view.filter(pl.Series(test)))
    return out


# ----------------------------------------------------------------------------- M4 evidence
def evidence_hands(hf: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Candidate hands of `pairs` (needs pair_id, player keys, family probs) with hand context added."""
    h = hf.join(pairs, on=KEYS, how="inner").sort("pair_id", "hand_idx")
    return hand_context(h)


def labeled_evidence_hands(d: DevData, hf: pl.DataFrame, probs: pl.DataFrame) -> pl.DataFrame:
    pos = d.labeled.filter(pl.col("label") == 1).select("pair_id", *KEYS, "fold").join(probs, on="pair_id")
    h = evidence_hands(hf.filter(pl.col("phase") == 0), pos)
    planted = d.evidence.select("pair_id", "hand_id", pl.lit(1).alias("y"))
    return h.join(planted, on=["pair_id", "hand_id"], how="left").with_columns(pl.col("y").fill_null(0))


def oof_evidence(d: DevData, hf: pl.DataFrame, probs: pl.DataFrame) -> pl.DataFrame:
    h = labeled_evidence_hands(d, hf, probs)
    feats = hand_feature_names(hf) + FAMILY_PROBS
    s = np.zeros(h.height)
    for f in range(d.labeled["fold"].max() + 1):
        tr = h.filter(pl.col("fold") != f)
        test = (h["fold"] == f).to_numpy()
        s[test], _ = fit_predict(LGB_EVIDENCE, matrix(tr, feats), tr["y"].to_numpy(),
                                 matrix(h.filter(pl.Series(test)), feats), seeds=SEEDS)
    return h.select("pair_id", "hand_id", "fold", "y", pl.Series("score", s))


def top5(scored: pl.DataFrame, col: str = "score") -> pl.DataFrame:
    """Top-5 hands per pair as evidence_hand_1..5 (NO_EVIDENCE fills pairs with fewer candidates)."""
    ranked = (scored.sort(["pair_id", col], descending=[False, True])
              .group_by("pair_id", maintain_order=True).head(5)
              .with_columns(pl.int_range(1, pl.len() + 1).over("pair_id").alias("rank")))
    wide = ranked.pivot(on="rank", index="pair_id", values="hand_id")
    wide = wide.rename({str(r): f"evidence_hand_{r}" for r in range(1, 6) if str(r) in wide.columns})
    return wide.with_columns(pl.col(c).fill_null(NO_EVIDENCE) if c in wide.columns else pl.lit(NO_EVIDENCE).alias(c)
                             for c in EVIDENCE_COLUMNS)


# ----------------------------------------------------------------------------- behavior policy
def assign_behavior(risk: np.ndarray, probs: np.ndarray, novelty: np.ndarray | None, cfg: dict,
                    none_below: float = 0.0) -> np.ndarray:
    """SPEC 5.6: argmax(M2) by default; `none` below a tuned risk threshold; other_coordination for novel
    pairs that fit no disclosed family (novelty percentile, low max family prob, above-median risk)."""
    out = np.asarray(FAMILIES, dtype=object)[probs.argmax(axis=1)]
    if novelty is not None:
        oc = cfg["other_coordination"]
        novel = ((novelty >= oc["novelty_percentile"]) & (probs.max(axis=1) < oc["max_family_prob"])
                 & (risk > np.quantile(risk, oc["min_risk_quantile"])))
        out[novel] = "other_coordination"
    out[risk < none_below] = "none"
    return out


# ----------------------------------------------------------------------------- scoring
def submission(pairs: pl.DataFrame, risk: np.ndarray, behavior: np.ndarray, evidence: pl.DataFrame | None
               ) -> pd.DataFrame:
    sub = pairs.select("pair_id").with_columns(pl.Series("risk_score", risk), pl.Series("predicted_behavior", behavior))
    if evidence is not None:
        sub = sub.join(evidence, on="pair_id", how="left")
    sub = sub.with_columns(pl.col(c).fill_null(NO_EVIDENCE) if c in sub.columns else pl.lit(NO_EVIDENCE).alias(c)
                           for c in EVIDENCE_COLUMNS)
    return sub.select(SUBMISSION_COLUMNS).to_pandas()


def solutions(d: DevData) -> dict[str, pd.DataFrame]:
    lab = build_solution(d.labels.to_pandas(), d.evidence.to_pandas())
    unl = d.mirror.filter(~pl.col("labeled")).select("pair_id").to_pandas()
    unl = unl.assign(risk_score=0, predicted_behavior="none", **{c: NO_EVIDENCE for c in EVIDENCE_COLUMNS})
    return {"labeled": lab, "mirror": pd.concat([lab, unl[list(SUBMISSION_COLUMNS)]], ignore_index=True)}


def evaluate(d: DevData, sols: dict, full: tuple[np.ndarray, np.ndarray], window: tuple[np.ndarray, np.ndarray],
             evidence: pl.DataFrame | None) -> dict[str, Score]:
    """Score (risk, behavior) predictions on the three validation sets.

    labeled = labeled pairs, full-period features; mirror = labeled + eval-filtered unlabeled pairs, full-period
    features; window = the mirror pairs with 2,000-hand window features (evaluation-like exposure and base rate).
    Evidence comes from the full development period in all three (only positive pairs are scored).
    """
    sub = submission(d.mirror, *full, evidence)
    is_lab = d.mirror["labeled"].to_numpy()
    return {"labeled": score(sols["labeled"], sub[is_lab].reset_index(drop=True)),
            "mirror": score(sols["mirror"], sub),
            "window": score(sols["mirror"], submission(d.mirror, *window, evidence))}


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True, method="average").to_numpy()


def baseline_evidence(hf: pl.DataFrame, d: DevData, col: str) -> pl.DataFrame:
    pos = d.labeled.filter(pl.col("label") == 1).select("pair_id", *KEYS)
    h = hf.filter(pl.col("phase") == 0).join(pos, on=KEYS)
    return top5(h.select("pair_id", "hand_id", pl.col(col).alias("score")))


# ----------------------------------------------------------------------------- calibration
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(raw: np.ndarray, y: np.ndarray) -> dict:
    """Monotone (ranking-preserving) calibration, fitted at a realistic base rate and exposure."""
    lr = LogisticRegression(C=1e6).fit(_logit(raw).reshape(-1, 1), y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def calibrate(raw: np.ndarray, c: dict) -> np.ndarray:
    return 1 / (1 + np.exp(-(c["a"] * _logit(raw) + c["b"])))


# ----------------------------------------------------------------------------- final fit
def fit_final(d: DevData, hf: pl.DataFrame, probs_oof: pl.DataFrame, calib: dict, none_below: float,
              cfg: dict) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    tr = risk_training_set(d, pl.lit(True), sample_seed=seed)
    _, m1 = fit_predict(LGB_BINARY, matrix(tr, FEATURES), tr["label"].to_numpy(), seeds=SEEDS,
                        feature_names=FEATURES)
    pos = behavior_training_set(d, pl.lit(True))
    _, m2 = fit_predict(LGB_MULTI, matrix(pos, FEATURES), family_index(pos), seeds=SEEDS,
                        feature_names=FEATURES)
    hpos = labeled_evidence_hands(d, hf, probs_oof)
    hfeats = hand_feature_names(hf) + FAMILY_PROBS
    _, m4 = fit_predict(LGB_EVIDENCE, matrix(hpos, hfeats), hpos["y"].to_numpy(), seeds=SEEDS, feature_names=hfeats)
    m3 = fit_novelty(pl.concat([v.select(NOVELTY_FEATURES) for v in d.views()]), seed)
    hs_cols: list[str] = []
    if option(cfg, "hand_suspicion"):
        htrain = labeled_hands(d, dev_hand_parts())
        hs_cols = [c for c in htrain.columns if c.startswith("h_")]
        for s, m in zip(SEEDS, fit_hand_models(htrain, hs_cols, SEEDS), strict=True):
            m.booster_.save_model(MODELS_DIR / f"h_suspicion_s{s}.txt")
    for name, models in (("m1_risk", m1), ("m2_behavior", m2), ("m4_evidence", m4)):
        for s, m in zip(SEEDS, models, strict=True):
            m.booster_.save_model(MODELS_DIR / f"{name}_s{s}.txt")
    with open(MODELS_DIR / "m3_novelty.pkl", "wb") as fh:
        pickle.dump(m3, fh)
    meta = {"pair_features": FEATURES, "cross_period": any(f.startswith("x_") for f in FEATURES),
            "hand_suspicion": bool(hs_cols), "suspicion_hand_features": hs_cols,
            "hand_features": hfeats, "novelty_features": NOVELTY_FEATURES,
            "families": list(FAMILIES), "seeds": list(SEEDS), "calibration": calib, "none_below_raw": none_below,
            "behavior_policy": cfg["behavior_policy"], "n_unlabeled_negatives_per_view": N_UNLABELED,
            "training_views": ["full development period",
                               *[f"development hands {w}-{w + WINDOW_HANDS - 1}" for w in d.windows]]}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------- repeated CV (experiments)
EXPERIMENTS = ARTIFACTS_DIR / "experiments.csv"
OVERRIDES: dict[str, bool | float] = {}  # command-line overrides of config.yaml `model:` options


def load_config_with_overrides() -> dict:
    cfg = load_config()
    cfg.setdefault("model", {}).update(OVERRIDES)
    return cfg


def option(cfg: dict, name: str):
    return cfg.get("model", {}).get(name, False)


def clean_threshold(cfg: dict) -> float | None:
    v = option(cfg, "clean_unlabeled")
    return None if v is False or v is None else float(v)


def configure(cfg: dict) -> bool:
    """Set the M1/M2 feature list from config; returns whether cross-period features are on."""
    global FEATURES
    cross = option(cfg, "cross_period")
    FEATURES = (PAIR_FEATURES + (CROSS_FEATURES if cross else [])
                + (HS_FEATURES if option(cfg, "hand_suspicion") else []))
    return cross


def dev_hand_parts() -> list:
    parts = sorted(DEV_HANDS_DIR.glob("part_*.parquet"))
    assert parts, "run `python -m suspoker.pipeline --stage S5H` first"
    return parts


def load_all(cfg: dict, seed: int | None = None) -> tuple[DevData, pl.DataFrame]:
    pf = pl.read_parquet(FEATURES_DIR / "pair_features.parquet")
    hf = pl.read_parquet(FEATURES_DIR / "hand_features.parquet")
    cross = configure(cfg)
    d = load_dev(pf, pl.read_parquet(FEATURES_DIR / "pair_features_windows.parquet"), seed=seed,
                 cross=pf.filter(pl.col("phase") == 1) if cross else None)
    if option(cfg, "hand_suspicion"):
        d = attach(d, oof_suspicion(d, dev_hand_parts(), SEEDS))
    return d, hf


def oof_scores(d: DevData, hf: pl.DataFrame, cfg: dict) -> dict[str, Score]:
    """M1 + M2 (argmax) + M4 out of fold, scored on the three validation sets (M3 omitted: no effect)."""
    policy = cfg["behavior_policy"]
    vi = 1 + list(d.windows).index(WINDOW_VIEW)
    risks, probs = oof_risk(d, clean_threshold(cfg)), oof_behavior(d)
    probs_df = d.mirror.select("pair_id").with_columns(
        pl.Series(n, probs[0][:, i]) for i, n in enumerate(FAMILY_PROBS))
    evidence = top5(oof_evidence(d, hf, probs_df))
    return evaluate(d, solutions(d), (risks[0], assign_behavior(risks[0], probs[0], None, policy)),
                    (risks[vi], assign_behavior(risks[vi], probs[vi], None, policy)), evidence)


def repeat_cv(tag: str, seeds: list[int]) -> pl.DataFrame:
    """Run the OOF evaluation under several fold seeds; append rows to artifacts/experiments.csv."""
    cfg = load_config_with_overrides()
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        d, hf = load_all(cfg, seed)
        sc = oof_scores(d, hf, cfg)
        row = {"tag": tag, "seed": seed, "time": time.strftime("%Y-%m-%d %H:%M")}
        for vset, x in sc.items():
            row |= {f"{vset}_final": x.final, f"{vset}_pair_ap": x.pair_ap, f"{vset}_evidence": x.evidence_map,
                    f"{vset}_behavior": x.behavior_map}
        rows.append(row)
        print(f"[{tag} seed {seed}] window final {row['window_final']:.4f} (AP {row['window_pair_ap']:.4f}, "
              f"ev {row['window_evidence']:.4f}) | mirror AP {row['mirror_pair_ap']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pl.DataFrame(rows)
    old = pl.read_csv(EXPERIMENTS) if EXPERIMENTS.exists() else None
    (pl.concat([old, df], how="diagonal_relaxed") if old is not None else df).write_csv(EXPERIMENTS)
    for c in ("window_final", "window_pair_ap", "window_evidence", "mirror_pair_ap"):
        print(f"  {c:<16} mean {df[c].mean():.4f}  sd {df[c].std():.4f}")
    return df


# ----------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config_with_overrides()
    policy = cfg["behavior_policy"]

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:5.0f}s] {msg}", flush=True)

    d, hf = load_all(cfg)
    cross = cfg.get("model", {}).get("cross_period", False)
    mir, win = d.mirror, d.windows[WINDOW_VIEW]
    sols = solutions(d)
    log(f"labeled {d.labeled.height:,} pairs, mirror {mir.height:,} pairs ({mir['label'].mean():.2%} positive), "
        f"views: full + windows {list(d.windows)}; {len(FEATURES)} pair features (cross-period: {cross})")

    results: dict[str, dict] = {}

    def record(name: str, scores: dict[str, Score], note: str) -> None:
        results[name] = {"note": note, **{k: v.as_dict() for k, v in scores.items()}}
        lab, m, w = scores["labeled"], scores["mirror"], scores["window"]
        log(f"{name:<15} labeled {lab.final:.4f} (AP {lab.pair_ap:.4f} ev {lab.evidence_map:.4f} beh "
            f"{lab.behavior_map:.4f}) | mirror {m.final:.4f} (AP {m.pair_ap:.4f}) | window {w.final:.4f} "
            f"(AP {w.pair_ap:.4f} beh {w.behavior_map:.4f})")

    # baselines
    all_dt = np.full(mir.height, "directed_transfer", dtype=object)

    def per_view(expr: pl.Expr) -> list[np.ndarray]:
        return [rank01(v.select(expr.fill_null(0)).to_series().to_numpy()) for v in (mir, win)]

    b0, b0w = per_view(pl.col("n_shared"))
    record("B0_shared_hands", evaluate(d, sols, (b0, all_dt), (b0w, all_dt), baseline_evidence(hf, d, "h_pot_bb")),
           "risk = shared hands; behavior = directed_transfer; evidence = biggest pots")
    b1, b1w = per_view(pl.col("netflow_bb100").abs())
    record("B1_net_flow", evaluate(d, sols, (b1, all_dt), (b1w, all_dt), baseline_evidence(hf, d, "h_netflow_bb")),
           "risk = |net chip flow| per 100 hands; behavior = directed_transfer; evidence = biggest flows")

    # models, out of fold
    vi = 1 + list(d.windows).index(WINDOW_VIEW)
    risks = oof_risk(d, clean_threshold(cfg))
    risk, risk_w = risks[0], risks[vi]
    log("M1 OOF done")
    probs = oof_behavior(d)
    prob, prob_w = probs[0], probs[vi]
    log("M2 OOF done")
    probs_df = mir.select("pair_id").with_columns(pl.Series(n, prob[:, i]) for i, n in enumerate(FAMILY_PROBS))
    ev = oof_evidence(d, hf, probs_df)
    evidence = top5(ev)
    log("M4 OOF done")
    nov, nov_w = oof_novelty(mir, cfg["seed"]), oof_novelty(win, cfg["seed"])
    log("M3 OOF done")

    record("M1_M2_no_evid", evaluate(d, sols, (risk, assign_behavior(risk, prob, None, policy)),
                                     (risk_w, assign_behavior(risk_w, prob_w, None, policy)), None),
           "M1 risk + M2 argmax, no evidence")

    # none-vs-argmax threshold, tuned on the window set (evaluation-like exposure and base rate). Adopted only
    # if it beats argmax-for-all by more than noise: keeping a pair in its family list (ordered by risk) is
    # rarely worse than dropping it into the tie pile at score 0.
    tuning = []
    for th in [0.0, *np.quantile(risk_w, [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995])]:
        s = evaluate(d, sols, (risk, assign_behavior(risk, prob, None, policy, th)),
                     (risk_w, assign_behavior(risk_w, prob_w, None, policy, th)), evidence)
        tuning.append({"none_below": float(th), **{f"{k}_behavior_map": v.behavior_map for k, v in s.items()}})
    best = max(tuning, key=lambda r: r["window_behavior_map"])
    if best["window_behavior_map"] - tuning[0]["window_behavior_map"] <= NONE_MIN_GAIN:
        best = tuning[0]
    none_below = best["none_below"]
    log(f"none threshold {none_below:.4f}: window Behavior MAP {best['window_behavior_map']:.4f} "
        f"(argmax for all: {tuning[0]['window_behavior_map']:.4f})")

    record("M_full_no_M3", evaluate(d, sols, (risk, assign_behavior(risk, prob, None, policy, none_below)),
                                    (risk_w, assign_behavior(risk_w, prob_w, None, policy, none_below)), evidence),
           "M1 + M2 (none policy) + M4")
    beh = assign_behavior(risk, prob, nov, policy, none_below)
    beh_w = assign_behavior(risk_w, prob_w, nov_w, policy, none_below)
    record("M_full", evaluate(d, sols, (risk, beh), (risk_w, beh_w), evidence),
           "M1 + M2 (none policy) + M3 other_coordination + M4")
    is_pos = mir["label"].to_numpy() == 1
    oc = {"assigned_window": int((beh_w == "other_coordination").sum()),
          "of_which_labeled_positive": int(((beh_w == "other_coordination") & is_pos).sum())}

    calib = platt(risk_w, mir["label"].to_numpy())
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    mir.select("pair_id", *KEYS, "table_id", "fold", "labeled", "label", "behavior_family", "n_shared").with_columns(
        pl.Series("risk_raw", risk), pl.Series("risk_window_raw", risk_w), pl.Series("risk", calibrate(risk, calib)),
        pl.Series("risk_window", calibrate(risk_w, calib)), pl.Series("novelty_pct", nov),
        pl.Series("predicted_behavior", beh), pl.Series("predicted_behavior_window", beh_w),
        *[pl.Series(n, prob[:, i]) for i, n in enumerate(FAMILY_PROBS)],
        *[pl.Series(f"{n}_window", prob_w[:, i]) for i, n in enumerate(FAMILY_PROBS)],
    ).write_parquet(OOF_DIR / "pairs.parquet")
    ev.write_parquet(OOF_DIR / "evidence.parquet")

    meta = fit_final(d, hf, probs_df, calib, none_below, cfg)
    log(f"final models saved to {MODELS_DIR}")

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "validation": {
            "folds": f"StratifiedGroupKFold({d.labeled['fold'].max() + 1}, group=table_id, "
                     f"stratify=behavior_family), seed {cfg['seed']}",
            "labeled": f"{d.labeled.height} labeled development pairs ({int(d.labeled['label'].sum())} positive), "
                       "full-period features",
            "mirror": f"{mir.height} development pairs passing the evaluation filter, unlabeled counted as negative "
                      f"({mir['label'].mean():.4f} positive), full-period features",
            "window": f"the mirror pairs with features from development hands {WINDOW_VIEW}-"
                      f"{WINDOW_VIEW + WINDOW_HANDS - 1} (evaluation-length window): the closest local proxy for the "
                      "leaderboard",
        },
        "models": results,
        "behavior_none_tuning": tuning,
        "none_below_raw": none_below,
        "other_coordination": oc,
        "calibration": calib,
        "n_pair_features": len(meta["pair_features"]),
        "n_hand_features": len(meta["hand_features"]),
    }
    (ARTIFACTS_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {ARTIFACTS_DIR / 'results.json'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train/validate (default) or compare a variant with repeated CV.")
    ap.add_argument("--repeats", help="comma-separated fold seeds, e.g. 42,43,44: OOF only, no final fit")
    ap.add_argument("--tag", default="current", help="experiment name for artifacts/experiments.csv")
    ap.add_argument("--set", action="append", default=[], metavar="model.KEY=true|false",
                    help="override a model option for this run, e.g. --set model.hand_suspicion=true")
    args = ap.parse_args()
    for kv in args.set:
        key, val = kv.split("=", 1)
        try:
            OVERRIDES[key.removeprefix("model.")] = float(val)
        except ValueError:
            OVERRIDES[key.removeprefix("model.")] = val.lower() in ("1", "true", "yes")
    if args.repeats:
        repeat_cv(args.tag, [int(x) for x in args.repeats.split(",")])
    else:
        main()
