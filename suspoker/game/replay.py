"""Hand replays for the web replayer: one JSON object per hand, built from competition-schema tables.

The same format is written for the static dashboard (SimRoom hands) and served by the game API (P8), so the
replayer is one component. Only simulated hands ever go through here (SPEC C5).
"""

import polars as pl


def replays(hand_ids: list[str], hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame,
            names: dict[str, str] | None = None) -> dict[str, dict]:
    """hand_id -> replay dict for every requested hand."""
    names = names or {}
    want = pl.DataFrame({"hand_id": hand_ids}).unique()
    h = hands.join(want, on="hand_id", how="semi")
    s = seats.join(want, on="hand_id", how="semi").sort("hand_id", "seat_no")
    a = actions.join(want, on="hand_id", how="semi").sort("hand_id", "action_no")
    seats_by = {k[0]: g for k, g in s.group_by("hand_id")}
    acts_by = {k[0]: g for k, g in a.group_by("hand_id")}
    out = {}
    for r in h.iter_rows(named=True):
        hid = r["hand_id"]
        out[hid] = {
            "hand_id": hid, "table_id": r["table_id"], "started_at": r["started_at"].isoformat(),
            "small_blind": r["small_blind"], "big_blind": r["big_blind"], "button_seat": r["button_seat"],
            "board": r["board_cards"].split() if r["board_cards"] else [], "final_pot": r["final_pot"],
            "seats": [{"seat_no": x["seat_no"], "player_id": x["player_id"], "name": names.get(x["player_id"], ""),
                       "starting_stack": x["starting_stack"], "cards": [x["hole_card_1"], x["hole_card_2"]],
                       "contribution": x["total_contribution"], "net_chips": x["net_chips"], "folded": x["folded"],
                       "showdown": x["went_to_showdown"]} for x in seats_by[hid].iter_rows(named=True)],
            "actions": [{"action_no": x["action_no"], "street": x["street"], "player_id": x["player_id"],
                         "action": x["action"], "amount": x["amount"], "amount_to": x["amount_to"],
                         "pot_before": x["pot_before"], "stack_before": x["stack_before"], "to_call": x["to_call"]}
                        for x in (acts_by[hid].iter_rows(named=True) if hid in acts_by else [])],
        }
    return out


def pair_highlights(replay: dict, a: str, b: str) -> list[dict]:
    """Actions where one partner acts on the other: responding to the partner's bet or raise, or betting
    with the partner still in. These are the moments a reason tag is about."""
    out, aggressor, folded = [], {}, set()
    for x in replay["actions"]:
        street, pid = x["street"], x["player_id"]
        mate = b if pid == a else (a if pid == b else None)
        aggressive = x["action"] in ("bet", "raise") or (x["action"] == "all_in" and x["amount"] > x["to_call"])
        if mate is not None and mate not in folded:
            if x["to_call"] > 0 and aggressor.get(street) == mate:
                verb = {"fold": "folds to", "call": "calls", "raise": "re-raises", "all_in": "goes all-in against"}
                text = f"{verb.get(x['action'], x['action'])} the partner's bet"
                out.append({"action_no": x["action_no"], "text": text})
            elif aggressive:
                out.append({"action_no": x["action_no"], "text": "bets or raises with the partner in the pot"})
        if aggressive:
            aggressor[street] = pid
        if x["action"] == "fold":
            folded.add(pid)
    return out
