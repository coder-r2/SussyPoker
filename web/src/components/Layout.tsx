import { useEffect, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";

export const REPO = "https://github.com/coder-r2/SussyPoker";
export const COMPETITION = "https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker";

type Theme = "light" | "dark";

const systemDark = () => typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;

/** Follows the system until the visitor picks a theme with the sun/moon button (remembered in this browser). */
function useTheme(): [Theme, () => void] {
  const [chosen, setChosen] = useState<Theme | null>(() => {
    try {
      const t = localStorage.getItem("theme");
      return t === "light" || t === "dark" ? t : null;
    } catch {
      return null;
    }
  });
  const [sys, setSys] = useState<Theme>(systemDark() ? "dark" : "light");
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const on = () => setSys(mq.matches ? "dark" : "light");
    mq?.addEventListener("change", on);
    return () => mq?.removeEventListener("change", on);
  }, []);
  useEffect(() => {
    const root = document.documentElement;
    if (chosen) root.setAttribute("data-theme", chosen);
    else root.removeAttribute("data-theme");
    try {
      if (chosen) localStorage.setItem("theme", chosen);
    } catch {
      /* storage may be unavailable */
    }
  }, [chosen]);
  const theme = chosen ?? sys;
  return [theme, () => setChosen(theme === "dark" ? "light" : "dark")];
}

const Sun = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
    <circle cx="12" cy="12" r="4.5" />
    {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
      <line key={a} x1="12" y1="2.5" x2="12" y2="5" transform={`rotate(${a} 12 12)`} />
    ))}
  </svg>
);
const Moon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11Z" /></svg>
);

export function Layout({ children }: { children: ReactNode }) {
  const [theme, toggle] = useTheme();
  return (
    <>
      <header className="site-header">
        <div className="wrap">
          <NavLink to="/" className="brand"><span className="logo">♠</span>SusPoker</NavLink>
          <nav className="nav" aria-label="Main">
            <NavLink to="/play" className="play-link">🎮 Beat the Detector</NavLink>
            <NavLink to="/alerts">🕵️ Case files</NavLink>
            <NavLink to="/how">💡 How it works</NavLink>
          </nav>
          <button className="icon-btn" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
            {theme === "dark" ? <Sun /> : <Moon />}
          </button>
        </div>
      </header>
      <main className="wrap">{children}</main>
      <footer className="site-footer">
        <div className="wrap">
          <span><strong>♠ SusPoker</strong> · a portfolio project</span>
          <span>Detector trained on Kaggle's <a href={COMPETITION} target="_blank" rel="noreferrer">Detect Suspicious Value Transfers in Poker</a> (Slash)</span>
          <span className="spacer" />
          <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </footer>
    </>
  );
}
