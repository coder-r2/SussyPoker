import { lazy, Suspense, useEffect } from "react";
import { HashRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Alerts } from "./pages/Alerts";
import { Hand } from "./pages/Hand";
import { Home } from "./pages/Home";
import { Play } from "./pages/Play";

// the chart pages pull in Recharts: load them on demand
const Pair = lazy(() => import("./pages/Pair").then((m) => ({ default: m.Pair })));
const ModelCard = lazy(() => import("./pages/ModelCard").then((m) => ({ default: m.ModelCard })));

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0); // block body: newer browsers return a Promise, which React would treat as a cleanup
  }, [pathname]);
  return null;
}

const TITLES: Record<string, string> = { "": "SusPoker", alerts: "Alerts", pair: "Pair", hand: "Hand", model: "Model card",
  play: "Beat the Detector" };

function Title() {
  const { pathname } = useLocation();
  useEffect(() => {
    const t = TITLES[pathname.split("/")[1] ?? ""] ?? "SusPoker";
    document.title = t === "SusPoker" ? t : `${t} · SusPoker`;
  }, [pathname]);
  return null;
}

// Hash routing (/#/...) so GitHub Pages serves every route from one index.html.
export function App() {
  return (
    <HashRouter>
      <ScrollToTop />
      <Title />
      <Layout>
        <Suspense fallback={<p className="muted">Loading…</p>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/pair/:pairId" element={<Pair />} />
          <Route path="/hand/:handId" element={<Hand />} />
          <Route path="/model" element={<ModelCard />} />
          <Route path="/play" element={<Play />} />
          <Route path="*" element={<p>Page not found. <Link to="/">Home</Link></p>} />
        </Routes>
        </Suspense>
      </Layout>
    </HashRouter>
  );
}
