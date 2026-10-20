import type { ReactNode } from "react";
import { HBars, Matrix, PrChart } from "../components/charts";
import { REPO } from "../components/Layout";
import { Status, Term } from "../components/ui";
import { loadModelCard, useData } from "../data";
import { FAMILY_LABEL, fmt } from "../format";
import type { Behavior, ModelCard as Card } from "../types";

// Text is authored here (SPEC C8: no Markdown is published); numbers come from model_card.json (aggregates only).

const SECTIONS = [
  ["task", "1. Task and data"], ["approach", "2. Approach"], ["results", "3. Results"], ["pr", "4. PR curves"],
  ["confusion", "5. Behavior confusion"], ["importance", "6. What drives the score"], ["ablation", "7. Ablations"],
  ["errors", "8. Error analysis"], ["simroom", "9. Generalization to our simulator"], ["limits", "10. Limitations and ethics"],
  ["repro", "11. Reproducibility"], ["attribution", "12. Data and attribution"],
] as const;

const MODEL_NAMES: Record<string, string> = {
  B0_shared_hands: "B0 · shared hands only", B1_net_flow: "B1 · net chip flow only", M1_M2_no_evid: "Risk + behavior, no evidence",
  M_full_no_M3: "Full model without novelty", M_full: "Full model",
};
const TEMPLATE_LABEL: Record<string, string> = {
  passive_player: "passive player", losing_player: "losing player", tilt: "tilt", aggressive_player: "aggressive player",
  similar_styles: "similar styles", schedule_overlap: "play together a lot", small_sample: "small sample", neighbours: "seated side by side",
  "none fits": "none fits",
};

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return <section id={id} style={{ scrollMarginTop: 70 }}><h2>{title}</h2>{children}</section>;
}

export function ModelCard() {
  const { data: mc, error } = useData(loadModelCard, "card");
  if (!mc) return <Status error={error} />;
  return (
    <div>
      <h1>Model card</h1>
      <p className="muted small">Generated {mc.generated} from code version <code>{mc.code_version}</code>. All competition
        numbers are aggregates; no competition hand, player, pair or label appears on this site.</p>
      <nav className="card small" aria-label="Sections" style={{ columns: "220px 3", marginBottom: 8 }}>
        {SECTIONS.map(([id, t]) => <div key={id}><a href={`#/model`} onClick={(e) => { e.preventDefault();
          document.getElementById(id)?.scrollIntoView({ behavior: "smooth" }); }}>{t}</a></div>)}
      </nav>
      <Task mc={mc} />
      <Approach mc={mc} />
      <Results mc={mc} />
      <Section id="pr" title="4. PR curves">
        <p className="small muted">Precision against recall for the risk ranking, out of fold. Per-family curves are not
          exported yet; section 8 gives recall per family.</p>
        <div className="card chart-card">
          <PrChart curves={[
            { name: "window view", curve: mc.errors.pr_curves.window, color: "var(--accent)" },
            { name: "labeled pairs", curve: mc.errors.pr_curves.labeled, color: "var(--muted)" },
          ]} />
          <p className="small muted" style={{ marginBottom: 0 }}>The labeled pairs (20% colluders) make ranking easy; the
            window view (0.3% colluders, evaluation-length exposure, unlabeled pairs counted as innocent) is the realistic one.</p>
        </div>
      </Section>
      <Section id="confusion" title="5. Behavior confusion">
        <p className="small muted">Window view, the colluders flagged at the break-even cut: true family (rows) vs
          predicted family (columns).</p>
        <Matrix rowsLabel="true" colsLabel="predicted" rows={mc.errors.confusion.true.map(lab)}
          cols={mc.errors.confusion.predicted.map(lab)} m={mc.errors.confusion.matrix} />
      </Section>
      <Importance mc={mc} />
      <Ablation mc={mc} />
      <Errors mc={mc} />
      <SimRoom mc={mc} />
      <Limits />
      <Repro mc={mc} />
      <Section id="attribution" title="12. Data and attribution">
        <p>{mc.data.source}. {mc.data.use}</p>
        <p className="small">The trained model files are published in the repository with this citation. Per the competition
          rules (competition use only), the data itself is never redistributed: no hands, IDs, labels or examples.</p>
        <pre className="small" style={{ whiteSpace: "pre-wrap", background: "var(--surface-2)", padding: 10, borderRadius: 8 }}>{mc.data.citation}</pre>
      </Section>
    </div>
  );
}

const lab = (f: string) => FAMILY_LABEL[f as Behavior] ?? f;

function Task({ mc }: { mc: Card }) {
  return (
    <Section id="task" title="1. Task and data">
      <p>
        The competition provides 2,000,000 synthetic six-max No-Limit Hold'em hands from 400 tables of 30 players, with
        every player's hole cards visible (an auditor's view). The task: for each of 112,540 pairs of players in the
        evaluation period, give a <Term k="risk">risk score</Term>, the most likely <Term k="behavior">behavior</Term>{" "}
        (directed transfer, soft play, coordinated isolation, or an undisclosed fourth kind), and up to five evidence hands.
      </p>
      <ul className="small">
        <li>Development period: the first 3,000 hands of each table, with 1,860 labeled pairs (372 colluders: 148 directed
          transfer, 132 soft play, 92 coordinated isolation).</li>
        <li>Evaluation period: the last 2,000 hands of each table; pairs share 38 to 419 hands (median 76).</li>
        <li>Scoring: <Term k="final">final = 0.70 × Pair AP + 0.20 × Evidence MAP@5 + 0.10 × Behavior MAP</Term>.</li>
        <li>Hard parts: collusion is episodic (mixed with honest play), decoys look suspicious (tilt, weak play, similar
          styles, players who sit together), and pairs share very different numbers of hands.</li>
      </ul>
      <p className="small muted">Source: {mc.data.source}. Full citation in section 12.</p>
    </Section>
  );
}

function Approach({ mc }: { mc: Card }) {
  const steps = ["Hand histories", "Decision events (who responded to whom, with exact odds)", "Pair features (lift vs each player's usual play)",
    "Risk model + behavior model + hand-suspicion model", "Evidence ranker + explanations"];
  return (
    <Section id="approach" title="2. Approach">
      <div className="row" style={{ gap: 6, alignItems: "stretch" }} role="img" aria-label="Pipeline diagram">
        {steps.map((s, i) => (
          <div key={s} className="row" style={{ gap: 6, flex: "1 1 160px" }}>
            <div className="card small" style={{ flex: 1, padding: 10 }}>{s}</div>
            {i < steps.length - 1 && <span className="muted" aria-hidden>→</span>}
          </div>
        ))}
      </div>
      <p style={{ marginTop: 12 }}>
        The core idea is <strong>lift</strong>: a player who folds a lot is not suspicious, but a player who folds far more
        to one particular opponent than to everyone else is. Every relationship feature compares a player's behavior
        against the partner with the same player's behavior against the rest of the table, shrunk toward zero when there
        are few hands. Because all cards are visible, the features can also ask whether a fold gave up the better hand or
        a call paid off a worse one.
      </p>
      <p className="small">
        {mc.model.pair_features} pair features in {mc.model.feature_families.length} families ({mc.model.feature_families.join(", ")});{" "}
        {mc.model.hand_features} hand features for the evidence ranker. Models are LightGBM, averaged over {mc.model.seeds} seeds:
        M1 risk (calibrated), M2 behavior family, M3 novelty (isolation forest, for the undisclosed kind), M4 evidence
        ranker, and H, a hand-level "does this hand look planted?" model whose top scores feed M1.
      </p>
    </Section>
  );
}

function Results({ mc }: { mc: Card }) {
  const deployed = mc.kaggle.submissions.find((s) => s.deployed)!;
  const models = ["B0_shared_hands", "B1_net_flow", "M_full"];
  const row = (m: string, view: string) => mc.results.find((r) => r.model === m && r.view === view)!;
  return (
    <Section id="results" title="3. Results">
      <p className="small muted">Out-of-fold (<Term k="oof">OOF</Term>) scores on the competition's development data, and
        the Kaggle late-submission score of the deployed model. Kaggle reports only the final score.</p>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr><th>Model</th><th><Term k="pair_ap">Pair AP</Term></th><th><Term k="evidence_map">Evidence MAP@5</Term></th>
              <th><Term k="behavior_map">Behavior MAP</Term></th><th><Term k="final">Final</Term> (OOF, window)</th>
              <th>Final (OOF, labeled)</th><th><Term k="kaggle">Kaggle public</Term></th><th>Kaggle private</th></tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const w = row(m, "window");
              return (
                <tr key={m}>
                  <td>{MODEL_NAMES[m]}</td>
                  <td className="num">{fmt.num(w.pair_ap)}</td><td className="num">{fmt.num(w.evidence_map)}</td>
                  <td className="num">{fmt.num(w.behavior_map)}</td><td className="num"><strong>{fmt.num(w.final)}</strong></td>
                  <td className="num">{fmt.num(row(m, "labeled").final)}</td>
                  <td className="num">{m === "M_full" ? fmt.num(deployed.public, 4) : "–"}</td>
                  <td className="num">{m === "M_full" ? fmt.num(deployed.private, 4) : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">Pair AP, Evidence and Behavior columns are the <Term k="window">window view</Term>. Over
        three fold seeds the full model's window final is {fmt.num((mc.repeated_cv.window_final as { mean: number }).mean)} ±{" "}
        {fmt.num((mc.repeated_cv.window_final as { sd: number }).sd)}.</p>

      <h3>Kaggle late submissions</h3>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>#</th><th>Change</th><th>Local window final</th><th>Public</th><th>Private</th></tr></thead>
          <tbody>
            {mc.kaggle.submissions.map((s) => (
              <tr key={s.n}><td>{s.n}{s.deployed ? " (deployed)" : ""}</td><td>{s.change}</td>
                <td className="num">{fmt.num(s.local_window_final)}</td><td className="num">{fmt.num(s.public, 4)}</td>
                <td className="num">{fmt.num(s.private, 4)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">{mc.kaggle.note} Final public leaderboard: {mc.kaggle.leaderboard.teams} teams, median{" "}
        {mc.kaggle.leaderboard.median}, top {mc.kaggle.leaderboard.top}. The deployed model is #4, the best on local
        validation; Kaggle scores are never used to choose. Kaggle scores sit above local ones because the local proxy
        counts unlabeled pairs as innocent, and some of them are hidden colluders.</p>
    </Section>
  );
}

function Importance({ mc }: { mc: Card }) {
  return (
    <Section id="importance" title="6. What drives the score">
      <p className="small muted">{mc.importance.population}. Higher = moves risk scores more on average.</p>
      <div className="grid grid-2">
        <div className="card chart-card">
          <h3>Top 15 features</h3>
          <HBars rows={mc.importance.risk.slice(0, 15).map((r) => ({ label: r.label, value: r.mean_abs_shap, sub: r.family }))} />
        </div>
        <div className="card chart-card">
          <h3>Share by feature family</h3>
          <HBars rows={mc.importance.risk_by_family.map((r) => ({ label: r.family, value: r.share, sub: `${r.n_features} features` }))}
            format={(v) => fmt.pct(v, 0)} />
          <p className="small muted" style={{ marginBottom: 0 }}>"Presence" asks whether a player opens differently when the
            partner is seated; "controls" are each player's overall style, which mostly push scores <em>down</em> for
            players who simply play that way with everyone.</p>
        </div>
      </div>
    </Section>
  );
}

function Ablation({ mc }: { mc: Card }) {
  return (
    <Section id="ablation" title="7. Ablations">
      <p className="small muted">{mc.ablation.note}. Δ = change in window Pair AP when the family is removed (negative =
        the family helps).</p>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Removed family</th><th>Features left</th><th>Window Pair AP</th><th>Δ window</th><th>Δ mirror</th></tr></thead>
          <tbody>
            {mc.ablation.rows.map((r) => (
              <tr key={r.dropped}><td>{r.dropped}</td><td className="num">{r.n_features}</td><td className="num">{fmt.num(r.window_ap)}</td>
                <td className="num">{fmt.signed(r.window_delta, 3)}</td><td className="num">{fmt.signed(r.mirror_delta, 3)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Run on the P3 model (before hand suspicion and exact-odds features); a refresh is planned.</p>
    </Section>
  );
}

function Errors({ mc }: { mc: Card }) {
  const e = mc.errors;
  const c = e.counts;
  const templates = Object.keys(TEMPLATE_LABEL).filter((t) => t !== "none fits");
  const rate = (o: string) => e.decoy_rates.find((r) => r.outcome === o)!;
  return (
    <Section id="errors" title="8. Error analysis">
      <p className="small muted">{e.operating_point}.</p>
      <div className="grid grid-4">
        <div className="card tile"><div className="label">Caught (TP)</div><div className="value num">{c.TP}</div></div>
        <div className="card tile"><div className="label">False alarms (FP)</div><div className="value num">{c.FP}</div>
          <div className="sub">{c.fp_confirmed_negative} of {fmt.int(c.confirmed_negatives)} confirmed-innocent pairs</div></div>
        <div className="card tile"><div className="label">Missed (FN)</div><div className="value num">{c.FN}</div></div>
        <div className="card tile"><div className="label"><Term k="precision">Precision</Term> = <Term k="recall">recall</Term></div>
          <div className="value num">{fmt.num(c.precision, 2)}</div></div>
      </div>
      <p style={{ marginTop: 12 }}>
        Every false alarm is an <em>unlabeled</em> pair: none of the {fmt.int(c.confirmed_negatives)} pairs confirmed innocent
        is flagged (the best ranks {c.best_confirmed_negative_rank}th). Some false alarms are probably colluders the labels
        don't list, so precision is a lower bound.
      </p>
      <h3>Which innocent explanation fits?</h3>
      <p className="small muted">Share of pairs in each outcome where a template fires (strongest 10%). Innocent pairs
        (TN) sit near 10% for every template by construction.</p>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Template</th>{["TP", "FP", "FN", "TN"].map((o) => <th key={o}>{o}</th>)}</tr></thead>
          <tbody>
            {templates.map((t) => (
              <tr key={t}><td>{TEMPLATE_LABEL[t]}</td>
                {["TP", "FP", "FN", "TN"].map((o) => <td key={o} className="num">{fmt.pct(Number(rate(o)[t]), 0)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small">
        A passive player who folds to everyone is never a false alarm (0%): the lift principle works. Pairs seated side
        by side or playing together a lot are slightly over-represented among false alarms, a known soft spot.
      </p>
      <h3>Recall by family</h3>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Family</th><th>Colluders</th><th>Caught</th><th>Recall</th><th>Median rank</th><th>Evidence AP@5</th></tr></thead>
          <tbody>
            {e.recall_by_family.map((r) => {
              const ev = e.evidence_by_family.find((x) => x.behavior_family === r.behavior_family);
              return <tr key={r.behavior_family}><td>{lab(r.behavior_family)}</td><td className="num">{r.positives}</td>
                <td className="num">{r.caught}</td><td className="num">{fmt.pct(r.recall, 0)}</td>
                <td className="num">{r.median_rank}</td><td className="num">{ev ? fmt.num(ev.evidence_ap5, 2) : "–"}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
      <h3>What missed colluders have in common</h3>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Family</th><th>Feature</th><th>Caught (median)</th><th>Missed (median)</th></tr></thead>
          <tbody>
            {e.missed_profile.map((r, i) => (
              <tr key={i}><td>{lab(r.family)}</td><td><code>{r.feature}</code></td><td className="num">{fmt.num(r.caught_median)}</td>
                <td className="num">{fmt.num(r.missed_median)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small">Missed colluders mostly had little to show: fewer shared hands and weaker partner-specific
        signals, rather than strong signals the model ignored. Concrete examples are on the{" "}
        <a href="#/alerts">SimRoom dashboard</a>.</p>
    </Section>
  );
}

function SimRoom({ mc }: { mc: Card }) {
  const s = mc.simroom;
  if (!s) return <Section id="simroom" title="9. Generalization to our simulator"><p className="muted">Pending.</p></Section>;
  const m = s.models;
  return (
    <Section id="simroom" title="9. Generalization to our simulator">
      <p>
        A domain-shift test: we wrote our own poker engine, bots and collusion (not the competition's generator), played
        12 tables × 2,500 hands, planted colluders and decoys, and scored every pair with the Kaggle-trained models
        unchanged. {s.population}: {fmt.int(s.pairs_scored)} pairs, {s.planted_scored} planted colluders
        (<Term k="base_rate">base rate</Term> {fmt.pct(s.base_rate)}).
      </p>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Model</th><th>Final</th><th>Pair AP</th><th>Evidence MAP@5</th><th>Behavior MAP</th></tr></thead>
          <tbody>
            {(["M_full", "B0_shared_hands", "B1_net_flow"] as const).map((k) => (
              <tr key={k}><td>{MODEL_NAMES[k]}</td><td className="num"><strong>{fmt.num(m[k].final)}</strong></td>
                <td className="num">{fmt.num(m[k].pair_ap)}</td><td className="num">{fmt.num(m[k].evidence_map)}</td>
                <td className="num">{fmt.num(m[k].behavior_map)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid grid-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Caught at the break-even cut (top {s.break_even.k})</h3>
          <HBars rows={s.recall_by_family.map((r) => ({ label: lab(r.family), value: r.caught / r.n,
            sub: `${r.caught} of ${r.n} · median rank ${r.median_rank} · family right ${fmt.pct(r.behavior_correct, 0)}` }))}
            format={(v) => fmt.pct(v, 0)} />
        </div>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Detection vs how often they colluded</h3>
          <HBars rows={[...s.detection_by_coordinated_hands].sort((a, b) => order(a.coordinated) - order(b.coordinated))
            .map((r) => ({ label: `${r.coordinated} coordinated hands`, value: r.n ? r.caught / r.n : 0,
              sub: `${r.caught} of ${r.n} caught · median rank ${r.median_rank}` }))} format={(v) => fmt.pct(v, 0)} />
        </div>
      </div>
      <p style={{ marginTop: 12 }}>
        <strong>Reading it.</strong> Directed transfer transfers almost perfectly (behavior AP{" "}
        {fmt.num(m.M_full.behavior_ap.directed_transfer, 2)}): chips moving one way look the same in any world. Soft
        play and coordinated isolation mostly do not: they are styles, and the model learned the competition generator's
        version of them, not ours. Detection rises steeply with how often a pair actually colluded. No decoy reaches the
        flagged list (best decoy rank {Math.min(...s.decoys.map((d) => d.best_rank))}). Live-game detection rates arrive
        with Beat the Detector.
      </p>
      {s.calibration && (
        <>
          <h3>Is the simulator realistic? (calibration)</h3>
          <p className="small muted">Bot parameters were tuned until the SimRoom's aggregate statistics fell inside the
            competition's range (player stats: median within the competition's 5–95% range).</p>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Statistic</th><th>Competition (p5 / p50 / p95)</th><th>SimRoom</th><th>OK</th></tr></thead>
              <tbody>
                {s.calibration.map((r) => (
                  <tr key={r.stat}><td>{r.stat}</td><td className="num">{r.competition}</td><td className="num">{r.simroom}</td>
                    <td>{r.ok ? <span className="badge-ok">✓</span> : <span className="badge-bad">✗</span>}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Section>
  );
}

const order = (b: string) => ({ "0-2": 0, "3-9": 1, "10-19": 2, "20+": 3 } as Record<string, number>)[b] ?? 9;

function Limits() {
  return (
    <Section id="limits" title="10. Limitations and ethics">
      <ul>
        <li><strong>Synthetic data on both sides.</strong> The competition hands are generated, and so is our SimRoom. Real
          poker has richer strategies, real money incentives and colluders who adapt. Section 9 shows how much the model
          depends on the world it learned from.</li>
        <li><strong>Labels are incomplete.</strong> Unlabeled development pairs are unknown, not innocent; local scores
          are therefore pessimistic.</li>
        <li><strong>False positives have a cost.</strong> Even at the break-even cut a quarter of flagged pairs are not
          known colluders. A flag is a reason to review hands, never proof; the innocent-explanation panel exists to push
          against confirmation bias.</li>
        <li><strong>Style combinations can look like collusion.</strong> In the SimRoom, 22 of the 23 false alarms at
          the break-even cut involve a calling station, and 11 are two calling stations together: two very passive
          players check strong hands down and pay each other off, which is what planted soft-play hands look like.
          The hand-suspicion features judge hands one at a time, so they cannot tell this apart; comparing with each
          player's other opponents did not fix it, because the pattern only appears for the combination.</li>
        <li><strong>The auditor sees all cards.</strong> Features use every player's hole cards, which a real site has but
          players do not. The model judges decisions with information the players could not see, by design.</li>
        <li><strong>Not a production fraud tool.</strong> A portfolio project: no appeal process, no monitoring for drift,
          no human review loop. Don't use it to sanction real players.</li>
        <li><strong>Families.</strong> The undisclosed fourth behavior is handled by a crude novelty rule; it is rarely
          predicted (37 of 112,540 evaluation pairs).</li>
      </ul>
    </Section>
  );
}

function Repro({ mc }: { mc: Card }) {
  const deployed = mc.kaggle.submissions.find((s) => s.deployed)!;
  return (
    <Section id="repro" title="11. Reproducibility">
      <pre className="small" style={{ whiteSpace: "pre-wrap", background: "var(--surface-2)", padding: 10, borderRadius: 8 }}>{
`pip install -e ".[dev,game]"                 # Python 3.13
python -m suspoker.pipeline --all             # raw competition data -> features -> models (needs the Kaggle data)
python -m suspoker.train --repeats 42,43,44   # local test over 3 fold seeds
python -m suspoker.predict --submission       # Kaggle submission
python -m suspoker.explain                    # SHAP, reason tags, innocent templates
python -m suspoker.game.simroom               # simulate the SimRoom (no competition data needed)
python -m suspoker.export_web --simroom --model-card   # this site's data`}</pre>
      <p className="small">Deployed model: commit <code>{deployed.commit}</code> (Kaggle submission #{deployed.n}). This page:
        code <code>{mc.code_version}</code>. Everything is seeded; the SimRoom is identical on every run.{" "}
        <a href={REPO} target="_blank" rel="noreferrer">Repository</a>.</p>
    </Section>
  );
}

