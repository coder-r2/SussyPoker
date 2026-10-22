import { Link } from "react-router-dom";
import { Card } from "../components/Table";
import { loadModelCard, loadSummary, useData } from "../data";
import { FAMILY_BLURB, FAMILY_COLOR, FAMILY_EMOJI, fmt } from "../format";
import type { Family } from "../types";

const SCHEMES: { id: Family; name: string; how: string }[] = [
  { id: "directed_transfer", name: "Chip dump", how: "Your partner folds good hands to you and pays you off with bad ones." },
  { id: "soft_play", name: "Soft play", how: "You never bet into each other and check it down when it's just you two." },
  { id: "coordinated_isolation", name: "Squeeze play", how: "You open, your partner re-raises, and everyone else gets pushed out." },
];

export function Home() {
  const mc = useData(loadModelCard, "card");
  const s = useData(loadSummary, "summary");
  const k = mc.data?.kaggle;
  const deployed = k?.submissions.find((x) => x.deployed);
  return (
    <div>
      <section className="hero">
        <div>
          <span className="pill" style={{ background: "rgba(255,255,255,.2)", color: "#fff" }}>🎮 A poker game with a twist</span>
          <h1 style={{ marginTop: 12 }}>Can you cheat at poker without getting caught?</h1>
          <p className="lead">Team up with a secret bot partner, rig the game, and try to stay under the radar. An AI detector is
            watching every single hand.</p>
          <div className="row" style={{ marginTop: 18, gap: 12 }}>
            <Link to="/play" className="btn big">▶ Play Beat the Detector</Link>
            <Link to="/alerts" className="btn ghost">🕵️ Browse the case files</Link>
          </div>
        </div>
        <div className="hero-art" aria-hidden>
          <span style={{ position: "absolute", left: "8%", top: "12%", rotate: "-14deg" }}><Card card="Ah" /></span>
          <span style={{ position: "absolute", left: "34%", top: "2%", rotate: "4deg", animationDelay: "-1.5s" }}><Card card="Ks" /></span>
          <span style={{ position: "absolute", left: "22%", top: "42%", rotate: "-4deg" }}><Card card="Qd" /></span>
          <div className="eye">👁️</div>
        </div>
      </section>

      <h2 style={{ textAlign: "center", marginTop: 42 }}>How to play</h2>
      <div className="grid grid-3">
        {[
          ["🤝", "Pick your scheme", "Choose how you and your secret partner will cheat, and how often."],
          ["🃏", "Play the hands", "Fold, call or raise against five bots. Your partner plays along. Auto-play when you're in a hurry."],
          ["🕵️", "Dodge the detector", "After every hand the detector re-ranks all 15 pairs at the table. Stay off the top of its list!"],
        ].map(([e, t, d], i) => (
          <div key={t} className="card step" style={{ animation: `fadeUp .5s ${i * 0.12}s both` }}>
            <span className="big-emoji" style={{ animationDelay: `${i * 0.4}s` }}>{e}</span>
            <h3 style={{ marginTop: 0 }}>{i + 1}. {t}</h3>
            <p className="muted" style={{ marginBottom: 0 }}>{d}</p>
          </div>
        ))}
      </div>

      <h2 style={{ textAlign: "center", marginTop: 42 }}>Three ways to cheat</h2>
      <div className="grid grid-3">
        {SCHEMES.map((x) => (
          <div key={x.id} className="card scheme-card">
            <span className="ribbon" style={{ background: FAMILY_COLOR[x.id] }} />
            <div className="row"><span style={{ fontSize: "2rem" }}>{FAMILY_EMOJI[x.id]}</span>
              <div><h3 style={{ margin: 0 }}>{x.name}</h3><div className="small muted">{FAMILY_BLURB[x.id]}</div></div></div>
            <p style={{ margin: "10px 0 0" }}>{x.how}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-2" style={{ marginTop: 42 }}>
        <div className="card" style={{ background: "var(--grad-warm)", color: "#fff", border: 0 }}>
          <div className="pill" style={{ background: "rgba(255,255,255,.22)", color: "#fff" }}>🏆 Tested on Kaggle</div>
          <h3 style={{ color: "#fff" }}>Trained on a real data-science competition</h3>
          <p>The detector was built for Kaggle's <em>Detect Suspicious Value Transfers in Poker</em>, then scored on 112,540 player
            pairs it had never seen.</p>
          {deployed && k ? (
            <div className="row" style={{ alignItems: "flex-end", gap: 18 }}>
              <div><div className="stat-hero">{fmt.num(deployed.private, 3)}</div><div className="small">Kaggle score (hidden test set)</div></div>
              <div className="small" style={{ opacity: 0.95 }}>Better than the typical team ({k.leaderboard.median}) out of {k.leaderboard.teams};
                the winner scored {k.leaderboard.top}.</div>
            </div>
          ) : <p className="small">Loading score…</p>}
        </div>
        <div className="card">
          <div className="pill">🕵️ Case files</div>
          <h3>Play detective first</h3>
          <p>Our bots played {s.data ? fmt.int(s.data.simroom.hands) : "30,000"} hands in a simulated card room. Some of them were told to
            cheat. Browse every suspicious pair, replay the hands, and reveal who was really guilty.</p>
          <Link to="/alerts" className="btn cool">Open the case files →</Link>
        </div>
      </div>

      <div className="cta-band" style={{ marginTop: 42 }}>
        <span style={{ fontSize: "2.4rem" }}>😎</span>
        <div style={{ flex: 1, minWidth: 220 }}>
          <h2>Think you can fool it?</h2>
          <div style={{ opacity: 0.95 }}>One session takes a couple of minutes. No sign-up, nothing saved.</div>
        </div>
        <Link to="/play" className="btn big" style={{ background: "#fff", color: "#6d28d9" }}>Sit down at the table</Link>
      </div>
    </div>
  );
}
