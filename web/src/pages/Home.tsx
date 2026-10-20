import { Link } from "react-router-dom";
import { REPO } from "../components/Layout";
import { Status, Term, Tile } from "../components/ui";
import { loadModelCard, loadSummary, useData } from "../data";
import { fmt } from "../format";

export function Home() {
  const s = useData(loadSummary, "summary");
  const mc = useData(loadModelCard, "card");
  if (!s.data || !mc.data) return <Status error={s.error ?? mc.error} />;
  const sum = s.data;
  const sim = sum.metrics.models.M_full;
  const window = mc.data.repeated_cv.window_final as { mean: number; sd: number };
  return (
    <div>
      <h1>Catching poker players who secretly work together</h1>
      <p style={{ maxWidth: 760 }}>
        Two players at the same table can quietly team up: one feeds chips to the other, they go easy on each other, or
        they squeeze everyone else out. SusPoker is a detector for that kind of collusion. It reads hand histories,
        ranks every pair of players by risk, names the likely scheme, and picks the five hands an investigator should
        look at first, with a plain-English reason for each. It was trained on the Kaggle competition{" "}
        <em>Detect Suspicious Value Transfers in Poker</em>, then tested on a card room we simulated ourselves.
      </p>

      <div className="grid grid-4" style={{ margin: "18px 0" }}>
        <Tile term="kaggle" label="Kaggle score (hidden test set)" value={fmt.num(sum.kaggle.deployed.private, 3)}
          sub={`deployed model, private leaderboard · best ${fmt.num(sum.kaggle.best_private, 3)}`} />
        <Tile term="window" label="Local validation" value={`${fmt.num(window.mean, 3)}`}
          sub={`final score, window view, ± ${fmt.num(window.sd, 3)} over 3 fold seeds`} />
        <Tile term="final" label="SimRoom (our simulator)" value={fmt.num(sim.final, 3)}
          sub={`Pair AP ${fmt.num(sim.pair_ap, 2)} vs ${fmt.num(sum.metrics.base_rate, 3)} for a random ranking`} />
        <Tile label="Simulated hands" value={fmt.int(sum.simroom.hands)}
          sub={`${sum.simroom.tables} tables · ${fmt.int(sum.metrics.pairs_scored)} pairs scored`} />
      </div>
      <p className="small muted" style={{ maxWidth: 760 }}>
        The three scores use the competition's <Term k="final">final score</Term> formula. Kaggle and local validation
        measure the competition's own world; the SimRoom measures how well the model transfers to collusion written by
        us, which it only partly does (see the model card, section 9).
      </p>

      <div className="grid grid-3" style={{ marginTop: 18 }}>
        <Link to="/alerts" className="section-link">
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Investigator dashboard →</h3>
            <p className="muted">Every pair in the SimRoom ranked by risk, with the simulator's answer key. Open a pair to
              see why it was flagged, its chip-flow timeline, its evidence hands and an innocent explanation to rule out.</p>
          </div>
        </Link>
        <Link to="/model" className="section-link">
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Model card →</h3>
            <p className="muted">How the detector was built and validated, Kaggle results, what drives its scores, where
              it fails, and how it transfers to a different poker world.</p>
          </div>
        </Link>
        <Link to="/play" className="section-link">
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Beat the Detector →</h3>
            <p className="muted">Sit at a table with five bots, one of them your secret partner, and try to collude
              without getting caught while the detector watches every hand.</p>
          </div>
        </Link>
      </div>
      <p style={{ marginTop: 18 }} className="small">
        Source code, data-use notes and how to reproduce everything: <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>.
      </p>
    </div>
  );
}
