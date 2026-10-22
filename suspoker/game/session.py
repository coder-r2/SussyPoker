"""Beat the Detector: one live table (the user + 5 bots, one of them the user's secret partner), scored after
every hand by the Kaggle-trained models (SPEC §7.6).

Table rules (§7.6.3): blinds 1/2, 200-chip stacks topped back up to 200 below 80, the button rotates every hand,
six seats always filled (the user sits in seat 0). The user sees only their own cards and showdown cards until the
end-of-session report reveals everything.

Live scoring (§7.6.6) uses the same path as the SimRoom (`suspoker.scoring.score_tables`) with three adjustments:
    * equity is cached per hand, so a rescore only computes odds for the newest hand;
    * a virtual 30-player pool: at a full six-seat table every pair shares every hand, which the models never saw,
      so exposure shares are scaled to what six seats out of a 30-player pool would give (DEC-030);
    * F8 presence features are set to "no effect": they compare hands with and without the partner seated, and at
      a fixed table the partner is always seated;
    * no M3 novelty (a population percentile over 15 pairs is meaningless), so other_coordination is never shown.
"""

import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import polars as pl

from suspoker.cards import CARD_STRINGS
from suspoker.config import MODELS_DIR, load_config
from suspoker.features import CATALOG
from suspoker.game.bots import FAMILIES, STYLES, Bot, Collusion, TablePolicy, jitter, tilt_decoy
from suspoker.game.engine import N_SEATS, STREET_NAMES, Decision, Hand, SeatIn, View
from suspoker.game.logger import HandLogger
from suspoker.game.replay import replays
from suspoker.game.simroom import ADJECTIVES, NOUNS

SB, BB = 1, 2
START_STACK, TOP_UP_BELOW = 200, 80
MAX_HANDS = 500
MIN_SHARED = 38                 # the competition's evaluation filter: below this the detector gives no verdict
# six seats drawn from a 30-player pool: a pair shares 6*5/(30*29) = 1/29 of the hands, a player plays 6/30 = 1/5
VIRTUAL_POOL = {"n_shared": 30 * 29 / (6 * 5), "hands_max": 30 / 6, "hands_min": 30 / 6}
# presence ("plays differently when the partner is seated") needs hands without the partner: none at a fixed table
NEUTRAL = [f for f, (family, _) in CATALOG.items() if family == "F8 presence"]
STRATEGIES = ("soft_play", "directed_transfer", "coordinated_isolation", "honest")
REVIEW_THRESHOLD = 0.093        # SimRoom break-even risk: pairs above it would be reviewed by an investigator
# warning line: ~99th percentile of innocent pairs' risk at 50-100 hands in simulated live games. A top-3 rank alone
# is not suspicious (someone is always top 3 of 15), so "suspected" also needs the risk above this line.
WATCH_THRESHOLD = 0.005
YOU = "SIM-YOU"
TABLE = "SIM-GAME"
BOT_STYLES = ("TAG", "LAG", "nit", "calling_station", "maniac")


@dataclass
class Detector:
    status: str                  # "insufficient" | "scored"
    hands: int
    pairs: list[dict] = field(default_factory=list)
    you_rank: int | None = None


class Session:
    def __init__(self, strategy: str, intensity: float = 0.5, direction: str = "partner_to_you",
                 partner_seat: int | None = None, seed: int | None = None, models=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not 0 <= intensity <= 1:
            raise ValueError("intensity must be between 0 and 1")
        if partner_seat is not None and not 1 <= partner_seat < N_SEATS:
            raise ValueError("partner_seat must be 1-5 (you sit in seat 0)")
        self.id = str(uuid.uuid4())
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
        self.rng = random.Random(self.seed)
        self.strategy, self.intensity, self.direction = strategy, intensity, direction
        self.partner_seat = partner_seat if partner_seat is not None else self.rng.randrange(1, N_SEATS)
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        self.models = models
        self.revealed = False

        names = self.rng.sample([f"{a}_{b}" for a in ADJECTIVES for b in NOUNS], N_SEATS - 1)
        self.players = [YOU] + [f"SIM-B{k:02d}" for k in range(1, N_SEATS)]
        self.names = {YOU: "you", **dict(zip(self.players[1:], names, strict=True))}
        self.partner = self.players[self.partner_seat]
        styles = {pid: jitter(STYLES[self.rng.choice(BOT_STYLES)], self.rng) for pid in self.players[1:]}
        decoy = self.rng.choice([p for p in self.players[1:] if p != self.partner])
        styles[decoy] = tilt_decoy(styles[decoy])  # a non-partner bot tilts after big losses (a deliberate decoy)
        self.bots = {pid: Bot(pid, self.names[pid], st, random.Random(self.rng.random())) for pid, st in styles.items()}
        # the user's own seat is played by a TAG bot during auto-play
        self.bots[YOU] = Bot(YOU, "you", STYLES["TAG"], random.Random(self.rng.random()))
        self.pair: Collusion | None = None
        if strategy != "honest":
            giver, receiver = ((self.partner, YOU) if direction == "partner_to_you" else (YOU, self.partner))
            a, b = (giver, receiver) if strategy == "directed_transfer" else (self.partner, YOU)
            self.pair = Collusion(strategy, a, b, intensity)
        self.policy = TablePolicy(self.bots, [self.pair] if self.pair else [], random.Random(self.rng.random()))

        self.stacks = dict.fromkeys(self.players, START_STACK)
        self.button = self.rng.randrange(N_SEATS)
        self.log = HandLogger()
        self.equity: list[pl.DataFrame] = []
        self.coordinated: list[str] = []  # hands where the partner's overlay changed a decision (ground truth)
        self.start_time = datetime(2026, 7, 1, tzinfo=UTC)
        self.hand: Hand | None = None
        self.hand_id: str | None = None
        self.autoplay_mode: str | None = None
        self.detector = Detector("insufficient", 0)
        self.last_hand: dict | None = None
        self.events: list[dict] | None = []  # table snapshots after each step of the last request (for animation)
        self._deal()
        self._run_bots()

    # ----------------------------------------------------------------------------------------------- dealing
    @property
    def hands_played(self) -> int:
        return len(self.log)

    @property
    def finished(self) -> bool:
        return self.hands_played >= MAX_HANDS

    def _deal(self) -> None:
        if self.finished:
            self.hand = None
            return
        for p in self.players:
            if self.stacks[p] < TOP_UP_BELOW:
                self.stacks[p] = START_STACK
        n = self.hands_played
        self.hand_id = f"SIM-G{n + 1:04d}"
        self.policy.new_hand(n, self.players)
        self.hand = Hand([SeatIn(p, self.stacks[p]) for p in self.players], self.button, SB, BB,
                         self.rng.sample(range(52), 52))
        self._emit("new_hand", None)

    def _decide(self, v: View) -> Decision:
        return self.policy(v)

    def _run_bots(self) -> None:
        """Let bots act until it is the user's turn (manual play) or the hand is over."""
        while self.hand is not None:
            v = self.hand.view()
            if v is None:
                self._finish_hand()
                if self.autoplay_mode is None:
                    self._deal()
                    continue
                return
            if v.player_id == YOU and self.autoplay_mode is None:
                return
            if v.player_id == YOU and self.autoplay_mode == "honest":
                d, _ = _honest(self.policy, v)
            else:
                d = self._decide(v)
            self._emit("action", self.hand.act(d))

    def _finish_hand(self) -> None:
        rec = self.hand.record()
        self.policy.end_hand(rec.net, BB)
        if 0 in self.policy.memo.changed:
            self.coordinated.append(self.hand_id)
        for p, x in rec.net.items():
            self.stacks[p] += x
        when = self.start_time + timedelta(seconds=75 * self.hands_played)
        log = HandLogger()
        log.add(rec, self.hand_id, TABLE, when, "simulation")
        self.log.add(rec, self.hand_id, TABLE, when, "simulation")
        self.equity.append(_hand_equity(*log.frames()))
        showdown = rec.hand["players_at_showdown"] > 0
        if self.events is not None:
            self.events.append({"type": "hand_end", "hand_id": self.hand_id, "table": self._table(rec)})
        self.last_hand = {
            "hand_id": self.hand_id, "board": rec.hand["board_cards"].split() if rec.hand["board_cards"] else [],
            "pot": rec.hand["final_pot"],
            "results": [{"seat": s["seat_no"], "name": self.names[s["player_id"]], "net": s["net_chips"],
                         "cards": ([s["hole_card_1"], s["hole_card_2"]]
                                   if s["player_id"] == YOU or (showdown and s["went_to_showdown"]) else None)}
                        for s in rec.seats]}
        self.button = (self.button + 1) % N_SEATS
        self.hand = None

    # ----------------------------------------------------------------------------------------------- user API
    def act(self, action: str, amount: int | None = None, rescore: bool = True) -> bool:
        """Apply the user's action; returns whether at least one hand finished. With `rescore=False` the detector is
        brought up to date later (`ensure_scored`), so the moves can go back to the player first."""
        if self.hand is None:
            raise ValueError("the session is over")
        v = self.hand.view()
        if v is None or v.player_id != YOU:
            raise ValueError("it is not your turn")
        before = self.hands_played
        self.events = []
        self._emit("action", self.hand.act(_user_decision(v, action, amount)))
        self._run_bots()
        if self.hands_played > before and rescore:
            self.rescore()
        return self.hands_played > before

    def autoplay(self, hands: int, user_mode: str, animate: bool = False, rescore: bool = True) -> None:
        """Play `hands` hands with the user's seat run by bot logic ('colluding' or 'honest'). With `animate`, record
        every step so the front end can play the hands out at table pace."""
        if user_mode not in ("colluding", "honest"):
            raise ValueError("user_mode must be 'colluding' or 'honest'")
        self.autoplay_mode = user_mode
        self.events = [] if animate else None
        try:
            for _ in range(hands):
                if self.hand is None:
                    self._deal()
                if self.hand is None:
                    break
                self._run_bots()
        finally:
            self.autoplay_mode = None
        if self.hand is None:
            self._deal()
            self._run_bots()
        if rescore:
            self.rescore()

    # ----------------------------------------------------------------------------------------------- scoring
    def frames(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        return self.log.frames()

    def ensure_scored(self) -> Detector:
        """Rescore only if hands were played since the last scoring."""
        return self.detector if self.detector.hands == self.hands_played else self.rescore()

    def rescore(self) -> Detector:
        n = self.hands_played
        if n < MIN_SHARED:
            self.detector = Detector("insufficient", n)
            return self.detector
        from suspoker.scoring import score_tables

        hands, seats, actions = self.frames()
        w = score_tables(hands, seats, actions, self.models, pair_ids=self._pair_id, min_shared=MIN_SHARED,
                         equity=pl.concat(self.equity), exposure_scale=VIRTUAL_POOL, neutral=NEUTRAL,
                         novelty=False, workers=0, benign_ref=_benign_ref())
        self._world = w
        reasons = w.risk_reasons.filter(pl.col("direction") == "for").sort("id", "rank")
        drivers = {k[0]: g.head(3).select("label", "shap").to_dicts() for k, g in reasons.group_by("id")}
        rows = w.pairs.sort(["risk", "pair_id"], descending=[True, False]).with_row_index("rank", offset=1)
        seat = {p: i for i, p in enumerate(self.players)}
        pairs = []
        for r in rows.iter_rows(named=True):
            members = [r["player_lo"], r["player_hi"]]
            pairs.append({
                "pair_id": r["pair_id"], "rank": r["rank"], "shared_hands": r["n_shared"], "risk": r["risk"],
                "behavior": r["predicted_behavior"],
                "probs": {f: r[f"p_{f}"] for f in FAMILIES},
                "players": [{"seat": seat[m], "name": self.names[m], "is_you": m == YOU,
                             "is_partner": m == self.partner} for m in sorted(members, key=seat.get)],
                "you_and_partner": set(members) == {YOU, self.partner},
                "drivers": drivers.get(r["pair_id"], []), "flagged": r["risk"] >= REVIEW_THRESHOLD,
                "watched": r["risk"] >= WATCH_THRESHOLD})
        you_rank = next((p["rank"] for p in pairs if p["you_and_partner"]), None)
        self.detector = Detector("scored", n, pairs, you_rank)
        return self.detector

    def _pair_id(self, a: str, b: str) -> str:
        seat = {p: i for i, p in enumerate(self.players)}
        i, j = sorted((seat[a], seat[b]))
        return f"SIM-S{i}{j}"

    def report(self) -> dict:
        """End-of-session report: verdict, evidence hands of (you, partner), innocent explanation. Reveals all cards."""
        self.revealed = True
        if self.hands_played < MIN_SHARED:
            return {"verdict": "insufficient", "headline": f"Not enough hands: {self.hands_played}/{MIN_SHARED}",
                    "hands": self.hands_played, "evidence": [], "benign_alternative": None, "you": None,
                    "bots": self._bot_reveal(), "coordinated_hands": self.coordinated}
        d = self.rescore() if self.detector.hands != self.hands_played else self.detector
        mine = next(p for p in d.pairs if p["you_and_partner"])
        w = self._world
        ev = w.evidence.filter(pl.col("pair_id") == mine["pair_id"]).sort("rank")
        benign = w.pairs.filter(pl.col("pair_id") == mine["pair_id"])["benign"].to_list()[0]
        ranks = f"your pair ranks #{mine['rank']} of {len(d.pairs)}"
        if mine["rank"] == 1 and mine["flagged"]:
            verdict, headline = "caught", "Caught: you are the detector's #1 pair and above the review threshold."
        elif mine["flagged"] or (mine["watched"] and mine["rank"] <= 3):
            verdict, headline = "suspected", f"Suspected: {ranks} and is above the warning line."
        elif self.strategy == "honest":
            verdict, headline = "clear", f"Cleared: {ranks}, below the warning line."
        else:
            verdict, headline = "clear", f"You got away with it: {ranks}, below the warning line."
        coordinated = set(self.coordinated)
        return {"verdict": verdict, "headline": headline, "hands": self.hands_played, "you": mine,
                "evidence": [{"rank": r["rank"], "hand_id": r["hand_id"], "score": r["score"], "tags": r["tags"],
                              "coordinated": r["hand_id"] in coordinated} for r in ev.iter_rows(named=True)],
                "benign_alternative": benign[0] if benign else None, "bots": self._bot_reveal(),
                "coordinated_hands": self.coordinated}

    def _bot_reveal(self) -> list[dict]:
        return [{"seat": i, "name": self.names[p], "style": self.bots[p].style.name, "is_partner": p == self.partner}
                for i, p in enumerate(self.players) if p != YOU]

    def replay(self, hand_id: str) -> dict:
        hands, seats, actions = self.frames()
        rep = replays([hand_id], hands, seats, actions, self.names)
        if hand_id not in rep:
            raise KeyError(hand_id)
        r = rep[hand_id]
        if not self.revealed:  # during play: only your cards and cards shown at showdown
            for s in r["seats"]:
                if s["player_id"] != YOU and not s["showdown"]:
                    s["cards"] = ["??", "??"]
        return r

    # ----------------------------------------------------------------------------------------------- state
    def _emit(self, kind: str, row: dict | None) -> None:
        """Record a table snapshot for the front end to animate step by step (only during a user action)."""
        if self.events is None or self.hand is None:
            return
        ev = {"type": kind, "hand_id": self.hand_id, "table": self._table()}
        if row is not None:
            ev |= {"seat": self.players.index(row["player_id"]), "text": describe(row)}
        self.events.append(ev)

    def _table(self, rec=None) -> dict:
        """What everyone at the table can see: stacks, chips in front, folds, the board, the pot. Hole cards only for
        you, plus the cards shown at showdown once a hand (`rec`) is over."""
        h = self.hand
        seats = []
        showdown = rec is not None and rec.hand["players_at_showdown"] > 0
        by_player = {s["player_id"]: s for s in rec.seats} if rec is not None else {}
        for i, p in enumerate(self.players):
            k = h.idx[i]
            acts = [a for a in h.actions if a["player_id"] == p]
            seat = {"seat": i, "name": self.names[p], "is_you": p == YOU, "is_partner": p == self.partner,
                    "stack": h.stack[k], "committed": h.commit[k], "folded": h.folded[k],
                    "cards": [CARD_STRINGS[c] for c in h.hole[i]] if p == YOU else None,
                    "last_action": describe(acts[-1]) if acts else None, "net": None}
            if rec is not None:
                r = by_player[p]
                seat |= {"stack": r["starting_stack"] + r["net_chips"], "committed": 0, "net": r["net_chips"]}
                if showdown and r["went_to_showdown"]:
                    seat["cards"] = [r["hole_card_1"], r["hole_card_2"]]
            seats.append(seat)
        v = h.view() if rec is None else None
        return {"seats": seats, "board": [CARD_STRINGS[c] for c in h.board],
                "pot": rec.hand["final_pot"] if rec is not None else h.pot, "button_seat": h.button,
                "street": STREET_NAMES[h.street], "to_act": v.seat if v is not None else None}

    def take_events(self) -> list[dict]:
        ev, self.events = self.events or [], None
        return ev

    def state(self) -> dict:
        v = self.hand.view() if self.hand else None
        h = self.hand
        seats = []
        for i, p in enumerate(self.players):
            seat = {"seat": i, "name": self.names[p], "is_you": p == YOU, "is_partner": p == self.partner,
                    "stack": self.stacks[p], "cards": None, "folded": False, "committed": 0, "last_action": None}
            if h is not None:
                k = h.idx[i]
                seat |= {"stack": h.stack[k], "folded": h.folded[k], "committed": h.commit[k],
                         "cards": [CARD_STRINGS[c] for c in h.hole[i]] if p == YOU else None}
                acts = [a for a in h.actions if a["player_id"] == p]
                if acts:
                    seat["last_action"] = describe(acts[-1])
            seats.append(seat)
        legal = None
        if v is not None and v.player_id == YOU:
            legal = {"fold": v.to_call > 0, "check": v.to_call == 0, "call": v.to_call > 0,
                     "call_amount": v.to_call, "raise": ({"min_to": v.min_raise_to, "max_to": v.max_raise_to}
                                                         if v.min_raise_to is not None else None),
                     "pot": v.pot, "committed": v.committed}
        return {
            "session_id": self.id, "strategy": self.strategy, "intensity": self.intensity,
            "direction": self.direction if self.strategy == "directed_transfer" else None,
            "you_seat": 0, "partner_seat": self.partner_seat, "partner_name": self.names[self.partner],
            "hands_played": self.hands_played, "max_hands": MAX_HANDS, "finished": self.finished,
            "small_blind": SB, "big_blind": BB,
            "hand": None if h is None else {
                "hand_id": self.hand_id, "button_seat": h.button, "street": STREET_NAMES[h.street],
                "board": [CARD_STRINGS[c] for c in h.board], "pot": h.pot,
                "log": [{"seat": self.players.index(a["player_id"]), "name": self.names[a["player_id"]],
                         "street": a["street"], "action": a["action"], "amount": a["amount"],
                         "amount_to": a["amount_to"]} for a in h.actions]},
            "seats": seats, "to_act": 0 if legal else None, "legal": legal, "last_hand": self.last_hand}

    def detector_json(self) -> dict:
        d = self.ensure_scored()
        return {"status": d.status, "hands": d.hands, "needed": MIN_SHARED, "threshold": REVIEW_THRESHOLD,
                "watch": WATCH_THRESHOLD, "pairs": d.pairs, "you_rank": d.you_rank}


def describe(row: dict) -> str:
    """A logged action in words, as a player would say it."""
    a, amount = row["action"], row["amount"]
    if a == "all_in":
        return f"all-in {amount}"
    if a == "raise":
        return f"raises to {row['amount_to']}"
    return {"fold": "folds", "check": "checks", "call": f"calls {amount}", "bet": f"bets {amount}"}[a]


def _honest(policy: TablePolicy, v: View) -> tuple[Decision, float]:
    from suspoker.game.bots import base_decision

    return base_decision(policy.bots[v.player_id], v, policy.memo)


def _user_decision(v: View, action: str, amount: int | None) -> Decision:
    if action in ("fold",):
        return Decision("fold")
    if action in ("check", "call"):
        return Decision("call")
    if action == "all_in":
        return Decision("raise", v.max_raise_to) if v.max_raise_to is not None else Decision("call")
    if action in ("bet", "raise"):
        if amount is None:
            raise ValueError("bet/raise needs an amount (the total you commit on this street)")
        if v.min_raise_to is None:
            return Decision("call")
        return Decision("raise", int(amount))
    raise ValueError(f"unknown action {action!r}")


def _hand_equity(hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Decision equity of one hand, keyed by (hand_id, action_no), computed in-process."""
    from suspoker.ingest import encode
    from suspoker.scoring import equity_table

    t = encode(hands, seats, actions)
    eq = equity_table(t, load_config()["seed"], workers=0)
    return eq.join(t.hand_ids, on="hand_idx").select("hand_id", pl.col("action_no").cast(pl.Int16), "eq")


_REF: dict | None = None


def _benign_ref() -> dict:
    global _REF
    if _REF is None:
        import json

        path = MODELS_DIR / "benign_reference_simroom.json"
        _REF = json.loads(path.read_text(encoding="utf-8"))["percentiles"]
    return _REF


class SessionStore:
    """In-memory sessions: 30-minute idle expiry and a cap on concurrent sessions (SPEC §7.6.8)."""

    def __init__(self, models, cap: int = 25, ttl_s: float = 1800):
        self.models, self.cap, self.ttl = models, cap, ttl_s
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def sweep(self) -> None:
        now = time.monotonic()
        with self.lock:
            for k in [k for k, s in self.sessions.items() if now - s.touched > self.ttl]:
                del self.sessions[k]

    def create(self, **kw) -> Session:
        self.sweep()
        with self.lock:
            if len(self.sessions) >= self.cap:
                raise OverflowError("Table full, try again soon")
        s = Session(models=self.models, **kw)
        with self.lock:
            self.sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session:
        self.sweep()
        with self.lock:
            s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        s.touched = time.monotonic()
        return s
