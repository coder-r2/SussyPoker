import { lazy, Suspense, useEffect } from "react";
import { HashRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Status } from "./components/ui";
import { Alerts } from "./pages/Alerts";
import { Hand } from "./pages/Hand";
import { Home } from "./pages/Home";
import { HowItWorks } from "./pages/HowItWorks";
import { Play } from "./pages/Play";

// the case-file page draws charts with Recharts: load it on demand
const Pair = lazy(() => import("./pages/Pair").then((m) => ({ default: m.Pair })));

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0); // block body: newer browsers return a Promise, which React would treat as a cleanup
  }, [pathname]);
  return null;
}

const TITLES: Record<string, string> = { "": "Beat the Detector", play: "Play", alerts: "Case files", pair: "Case file",
  hand: "Hand replay", how: "How it works", model: "How it works" };

function Title() {
  const { pathname } = useLocation();
  useEffect(() => {
    document.title = `SusPoker · ${TITLES[pathname.split("/")[1] ?? ""] ?? "Beat the Detector"}`;
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
        <Suspense fallback={<Status />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/play" element={<Play />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/pair/:pairId" element={<Pair />} />
            <Route path="/hand/:handId" element={<Hand />} />
            <Route path="/how" element={<HowItWorks />} />
            <Route path="/model" element={<HowItWorks />} />
            <Route path="*" element={<p>Page not found. <Link to="/">Home</Link></p>} />
          </Routes>
        </Suspense>
      </Layout>
    </HashRouter>
  );
}
