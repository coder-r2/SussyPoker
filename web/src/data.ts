import { useEffect, useState } from "react";
import type { AlertsFile, HandReplay, ModelCard, PairDetail, Summary } from "./types";

// Static JSON written by `python -m suspoker.export_web` into web/public/data/ (copied as-is into the build).
const BASE = `${import.meta.env.BASE_URL}data/`;
const cache = new Map<string, Promise<unknown>>();

function load<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    const p = fetch(BASE + path).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    });
    p.catch(() => cache.delete(path)); // allow a retry after a failure
    cache.set(path, p);
  }
  return cache.get(path) as Promise<T>;
}

export const loadSummary = () => load<Summary>("summary.json");
export const loadAlerts = () => load<AlertsFile>("alerts.json");
export const loadModelCard = () => load<ModelCard>("model_card.json");
export const loadPair = (id: string) => load<PairDetail>(`pairs/${encodeURIComponent(id)}.json`);
export const loadHand = (id: string) => load<HandReplay>(`hands/${encodeURIComponent(id)}.json`);

export type Loaded<T> = { data?: T; error?: string };

/** Load once per key; re-runs when the key changes. */
export function useData<T>(loader: () => Promise<T>, key: string): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({});
  useEffect(() => {
    let live = true;
    setState({});
    loader().then((data) => live && setState({ data }), (e: Error) => live && setState({ error: e.message }));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
