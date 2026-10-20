import { Link, useParams, useSearchParams } from "react-router-dom";
import { SimBanner } from "../components/Layout";
import { Replayer } from "../components/Replayer";
import { Status } from "../components/ui";
import { loadHand, loadPair, useData } from "../data";

export function Hand() {
  const { handId = "" } = useParams();
  const [params] = useSearchParams();
  const pairId = params.get("pair");
  const hand = useData(() => loadHand(handId), handId);
  const pair = useData(() => (pairId ? loadPair(pairId) : Promise.resolve(null)), pairId ?? "");
  if (!hand.data) return <Status error={hand.error} />;
  const p = pair.data;
  return (
    <div>
      <p className="small">{p ? <Link to={`/pair/${p.pair_id}`}>← Pair {p.pair_id}</Link> : <Link to="/alerts">← All alerts</Link>}</p>
      <h1>Hand {hand.data.hand_id}</h1>
      <SimBanner text="A simulated hand from the SimRoom, shown in investigator view: every player's cards are face-up." />
      <Replayer hand={hand.data} pair={p ? [p.players[0].player_id, p.players[1].player_id] : undefined}
        highlights={p?.highlights[handId] ?? []} />
    </div>
  );
}
