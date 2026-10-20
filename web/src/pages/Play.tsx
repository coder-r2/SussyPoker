import { Link } from "react-router-dom";

export function Play() {
  return (
    <div style={{ maxWidth: 760 }}>
      <h1>Beat the Detector</h1>
      <p>
        Sit at a six-player No-Limit Hold'em table with five bots. One of them is your secret partner. Pick a scheme
        (feed chips one way, go easy on each other, or squeeze the others out), choose how often to collude, and try to
        stay off the top of the detector's list while it rescores every pair after each hand.
      </p>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Coming soon</h3>
        <p className="small" style={{ marginBottom: 0 }}>
          The game needs the live game server (poker engine, bots and real-time scoring), which is the next build
          step. Meanwhile, the <Link to="/alerts">dashboard</Link> shows the same bots and detector on 30,000
          simulated hands.
        </p>
      </div>
    </div>
  );
}
