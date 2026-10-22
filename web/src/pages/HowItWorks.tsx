import { Link } from "react-router-dom";
import { COMPETITION, REPO } from "../components/Layout";
import { Status, Term } from "../components/ui";
import { loadModelCard, useData } from "../data";
import { fmt } from "../format";

const STEPS = [
  ["👀", "It sees every card", "Like a casino's security team, the detector reads the full hand history, including the cards players kept hidden. So it can tell when someone folded the better hand or called with junk."],
  ["⚖️", "It compares players with themselves", "Folding a lot isn't suspicious; plenty of players are cautious. Folding far more to one particular opponent than to everyone else is. Every clue measures how a player treats one opponent versus the rest of the table."],
  ["🧠", "It scores every pair", "A machine-learning model turns those clues into a suspicion score for every pair of players and guesses the scheme: chip dumping, soft play or squeezing others out."],
  ["🔍", "It shows its evidence", "For each pair it picks the five hands an investigator should watch first, with a plain-English reason for each, plus an innocent explanation worth ruling out."],
];

export function HowItWorks() {
  const { data, error } = useData(loadModelCard, "card");
  const k = data?.kaggle;
  const deployed = k?.submissions.find((x) => x.deployed);
  const best = k ? Math.max(...k.submissions.map((x) => x.private)) : 0;
  return (
    <div style={{ maxWidth: 940, margin: "0 auto" }}>
      <span className="pill hot">💡 How it works</span>
      <h1 style={{ marginTop: 10 }}>A detector that thinks like a casino investigator</h1>
      <p style={{ fontSize: "1.08rem", maxWidth: 760 }} className="muted">
        Colluders rarely do anything that looks wrong in a single hand. The giveaway is a pattern: two players who treat each
        other differently from everyone else. Here's how the detector finds them.
      </p>
      <div className="grid grid-2" style={{ marginTop: 18 }}>
        {STEPS.map(([e, t, d], i) => (
          <div key={t} className="card" style={{ animation: `fadeUp .5s ${i * 0.1}s both` }}>
            <div className="row"><span className="avatar lg" style={{ background: "var(--surface-2)" }}>{e}</span>
              <h3 style={{ margin: 0 }}>{i + 1}. {t}</h3></div>
            <p style={{ margin: "10px 0 0" }}>{d}</p>
          </div>
        ))}
      </div>

      <h2>How it did on Kaggle</h2>
      {!k || !deployed ? <Status error={error} /> : (
        <div className="card" style={{ background: "var(--grad)", color: "#fff", border: 0 }}>
          <div className="row" style={{ gap: 28, alignItems: "flex-end" }}>
            <div><div className="stat-hero">{fmt.num(deployed.private, 3)}</div>
              <div className="small"><Term k="kaggle">Kaggle score</Term> of the detector you play against</div></div>
            <div><div className="display" style={{ fontSize: "2rem", fontWeight: 700 }}>{fmt.num(best, 3)}</div>
              <div className="small">best of our {k.submissions.length} submissions</div></div>
            <div><div className="display" style={{ fontSize: "2rem", fontWeight: 700 }}>{k.leaderboard.teams}</div>
              <div className="small">teams on the leaderboard</div></div>
          </div>
          <p style={{ marginTop: 16, marginBottom: 0 }}>
            Scored on {fmt.int(112540)} pairs of players it had never seen. The typical team scored {k.leaderboard.median}; the
            winner {k.leaderboard.top}. <Term k="final"><span style={{ color: "#fff", borderColor: "#fff" }}>What does the score measure?</span></Term>
          </p>
        </div>
      )}

      <h2>Good to know</h2>
      <ul>
        <li>The competition data is synthetic, and so is everything you see on this site: the case files come from our own
          simulated card room, and every player at the game table is a bot.</li>
        <li>A high score is a reason to look closer, never proof. That's why every case comes with an innocent explanation.</li>
        <li>It's a game and a portfolio project, not a real fraud tool.</li>
      </ul>
      <div className="row" style={{ marginTop: 18 }}>
        <Link to="/play" className="btn primary big">🎮 Try to beat it</Link>
        <a className="btn" href={REPO} target="_blank" rel="noreferrer">Source code</a>
        <a className="btn" href={COMPETITION} target="_blank" rel="noreferrer">The Kaggle competition</a>
      </div>
    </div>
  );
}
