import { useEffect, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";

const CITATION = `@misc{detect-suspicious-value-transfers-in-poker,
  author = {Slash — Detect Suspicious Value Transfers in Poker},
  title = {Detect Suspicious Value Transfers in Poker},
  year = {2026},
  howpublished = {\\url{https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker}},
  note = {Kaggle}
}`;
export const REPO = "https://github.com/coder-r2/SussyPoker";
const COMPETITION = "https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker";

type Theme = "system" | "light" | "dark";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("theme") as Theme) || "system";
    } catch {
      return "system";
    }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* storage may be unavailable */
    }
  }, [theme]);
  const next = () => setTheme((t) => (t === "system" ? "light" : t === "light" ? "dark" : "system"));
  return [theme, next];
}

export function Layout({ children }: { children: ReactNode }) {
  const [theme, nextTheme] = useTheme();
  return (
    <>
      <header className="site-header">
        <div className="wrap">
          <NavLink to="/" className="brand">SusPoker</NavLink>
          <nav className="nav" aria-label="Main">
            <NavLink to="/alerts">Alerts</NavLink>
            <NavLink to="/model">Model card</NavLink>
            <NavLink to="/play">Beat the Detector</NavLink>
            <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>
          </nav>
          <button className="icon-btn small" onClick={nextTheme} title="Theme: system → light → dark">
            Theme: {theme}
          </button>
        </div>
      </header>
      <main className="wrap">{children}</main>
      <footer className="site-footer">
        <div className="wrap">
          <p>
            <strong>Simulated data.</strong> Every hand, player and pair on this site comes from our own simulator (the
            SimRoom, IDs starting with <code>SIM-</code>). The detection models were trained on the Kaggle competition{" "}
            <a href={COMPETITION} target="_blank" rel="noreferrer">Detect Suspicious Value Transfers in Poker</a> by Slash.
            No competition data is shown or redistributed; the model card reports aggregates only.
          </p>
          <p>Please cite the competition data as:</p>
          <pre>{CITATION}</pre>
          <p>
            Code: <a href={REPO} target="_blank" rel="noreferrer">coder-r2/SussyPoker</a> (MIT). A portfolio project, not a
            production fraud tool.
          </p>
        </div>
      </footer>
    </>
  );
}

export function SimBanner({ text }: { text?: string }) {
  return (
    <div className="banner" role="note">
      {text ??
        "Simulated hands: every hand on this dashboard was played by our bots in the SimRoom. The detector was trained on the Kaggle competition 'Detect Suspicious Value Transfers in Poker' (Slash); no competition data is shown."}
    </div>
  );
}
