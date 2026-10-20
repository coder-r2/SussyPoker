"""Rule-based bots: five style profiles, a tilt decoy and collusion overlays for the three disclosed families.

A bot sees only what a real player sees (its own cards, the board, the betting) through `engine.View`. Its base
decision comes from its style: preflop ranges by position (hand-class percentiles), postflop equity against random
hands of the remaining opponents compared with pot odds, and pot-fraction bet sizes.

A colluding pair adds an *overlay* on top of the base decision, but only in hands where the pair coordinates
(episodic: inside active windows, with probability = intensity). The overlay records every decision it changed;
those hands are the ground truth of "where they actually coordinated".

    soft_play               join the partner's pots, never bet or raise into the partner, check it down
                            heads-up, and fold medium hands to the partner's bets instead of contesting them
    directed_transfer       the giver folds hands that are ahead to the receiver's aggression and calls with
                            hands that are behind; the receiver opens light and bets into the giver relentlessly
    coordinated_isolation   open light when the partner can back you up; when the partner has opened, squeeze
                            third players out with a re-raise, otherwise fold behind and leave them alone; never
                            raise the partner

The collusion is written by us, not by the competition's generator, so how well the Kaggle-trained model catches
it is a domain-shift test (SPEC §7.6.6).
"""

import random
from dataclasses import dataclass, field, replace

from phevaluator import evaluate_cards

from suspoker.cards import hand_class, preflop_percentile
from suspoker.game.engine import Decision, View

FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")
EQUITY_SAMPLES = 120
MAX_OPPONENTS = 3  # equity is estimated against at most 3 random hands (multiway pots are rare and cheap to cap)
TILT_LOSS_BB = 30
RANGE_DISCOUNT = 3.0
# position relative to the button -> multiplier on how many hands a style enters
POSITION_WIDTH = {3: 0.70, 4: 0.85, 5: 1.10, 0: 1.40, 1: 0.90, 2: 1.00}  # UTG, MP, CO, BTN, SB, BB


@dataclass(frozen=True)
class Style:
    name: str
    vpip: float        # share of starting hands this style is willing to play
    pfr: float         # share it opens with a raise
    three_bet: float   # share it re-raises with when facing a raise
    limp: float        # chance to just call instead of folding a playable non-raising hand when unopened
    agg: float         # chance to bet / raise when it has a value hand postflop
    bluff: float       # chance to bet when checked to without a value hand
    cbet: float        # chance to continuation-bet the flop as preflop aggressor
    call_down: float   # multiplier on pot odds when deciding to call (lower = calls more)
    size: float        # bet size as a fraction of the pot
    tilt: float = 0.0  # how strongly a big loss loosens it up (0 = never tilts)


STYLES = {
    "TAG": Style("TAG", vpip=0.34, pfr=0.23, three_bet=0.16, limp=0.10, agg=0.75, bluff=0.20, cbet=0.65,
                 call_down=1.05, size=0.60),
    "LAG": Style("LAG", vpip=0.45, pfr=0.31, three_bet=0.10, limp=0.10, agg=0.85, bluff=0.35, cbet=0.75,
                 call_down=0.95, size=0.70),
    "nit": Style("nit", vpip=0.22, pfr=0.15, three_bet=0.06, limp=0.15, agg=0.60, bluff=0.08, cbet=0.50,
                 call_down=1.25, size=0.55),
    "calling_station": Style("calling_station", vpip=0.68, pfr=0.12, three_bet=0.04, limp=0.85, agg=0.35,
                             bluff=0.08, cbet=0.35, call_down=0.60, size=0.50),
    "maniac": Style("maniac", vpip=0.72, pfr=0.42, three_bet=0.26, limp=0.30, agg=0.95, bluff=0.55, cbet=0.90,
                    call_down=0.80, size=0.85),
}


def tilt_decoy(base: Style) -> Style:
    return replace(base, name=f"{base.name}+tilt", tilt=1.0)


def jitter(style: Style, rng: random.Random, sd: float = 0.12) -> Style:
    """A personal variant of a style: each numeric parameter scaled by a random factor around 1."""
    def f(x: float) -> float:
        return max(0.0, x * rng.lognormvariate(0.0, sd))
    return replace(style, vpip=min(f(style.vpip), 0.9), pfr=min(f(style.pfr), 0.7), three_bet=f(style.three_bet),
                   limp=min(f(style.limp), 1.0), agg=min(f(style.agg), 1.0), bluff=min(f(style.bluff), 0.9),
                   cbet=min(f(style.cbet), 1.0), call_down=f(style.call_down), size=f(style.size),
                   tilt=style.tilt or rng.choice([0.0, 0.0, 0.3]))


_PCT: dict[str, float] | None = None


def preflop_pct(hole: tuple[int, int]) -> float:
    global _PCT
    if _PCT is None:
        _PCT = preflop_percentile()
    return _PCT[hand_class(*hole)]


def equity(hole: tuple[int, int], board: list[int], n_opp: int, rng: random.Random,
           samples: int = EQUITY_SAMPLES) -> float:
    """Chance to win against `n_opp` random hands, by Monte Carlo over their cards and the rest of the board."""
    n_opp = max(1, min(n_opp, MAX_OPPONENTS))
    deck = [c for c in range(52) if c not in hole and c not in board]
    need = 5 - len(board)
    wins = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, need + 2 * n_opp)
        full = board + draw[:need]
        mine = evaluate_cards(*hole, *full)
        best = min(evaluate_cards(draw[need + 2 * k], draw[need + 2 * k + 1], *full) for k in range(n_opp))
        wins += 1.0 if mine < best else (0.5 if mine == best else 0.0)
    return wins / samples


@dataclass
class Bot:
    player_id: str
    name: str
    style: Style
    rng: random.Random
    tilt_left: int = 0

    def note_result(self, net_bb: float) -> None:
        if self.tilt_left:
            self.tilt_left -= 1
        if self.style.tilt and net_bb <= -TILT_LOSS_BB and self.rng.random() < self.style.tilt:
            self.tilt_left = self.rng.randint(8, 25)

    @property
    def effective(self) -> Style:
        s = self.style
        if not self.tilt_left:
            return s
        return replace(s, vpip=min(s.vpip * 1.8, 0.85), pfr=min(s.pfr * 1.6, 0.6), three_bet=s.three_bet * 1.8,
                       bluff=min(s.bluff * 2.0, 0.9), call_down=s.call_down * 0.7)


@dataclass
class HandMemo:
    """Per-hand scratch space: equity cache and which pairs coordinate in this hand."""
    equity: dict = field(default_factory=dict)
    coordinating: set = field(default_factory=set)       # pair indices active this hand
    changed: dict = field(default_factory=dict)          # pair index -> number of decisions the overlay changed


def position(v: View) -> int:
    return (v.seat - v.button) % 6


def raises_so_far(v: View) -> int:
    return sum(1 for a in v.history if a["street"] == ("preflop", "flop", "turn", "river")[v.street]
               and (a["action"] in ("bet", "raise") or (a["action"] == "all_in" and a["amount"] > a["to_call"])))


def entered(v: View, pid: str) -> bool:
    """Did `pid` put chips in voluntarily before now in this hand?"""
    return any(a["player_id"] == pid and a["amount"] > 0 for a in v.history)


def raise_to(v: View, target: float) -> Decision:
    """Raise to `target` chips; commit everything when the raise would leave little behind."""
    if v.min_raise_to is None:
        return Decision("call")
    t = int(round(target))
    if t >= v.committed + 0.6 * (v.stack + v.committed):
        t = v.max_raise_to
    return Decision("raise", max(v.min_raise_to, min(t, v.max_raise_to)))


def base_decision(bot: Bot, v: View, memo: HandMemo) -> tuple[Decision, float]:
    """The bot's honest decision and the strength it was based on (preflop percentile or equity)."""
    s, rng = bot.effective, bot.rng
    if v.street == 0:
        pct = preflop_pct(v.hole)
        width = POSITION_WIDTH[position(v)]
        n_raises = raises_so_far(v)
        price_bb = v.to_call / v.bb
        if n_raises == 0:
            if pct >= 1 - s.pfr * width:
                limpers = sum(1 for a in v.history if a["action"] == "call")
                return raise_to(v, v.bb * (2.5 + limpers + rng.choice((0, 0, 0.5)))), pct
            if v.to_call == 0:
                return Decision("call"), pct  # big blind checks
            if pct >= 1 - s.vpip * width and rng.random() < s.limp:
                return Decision("call"), pct
            return Decision("fold"), pct
        if n_raises == 1:
            if pct >= 1 - s.three_bet:
                return raise_to(v, v.max_commit * 3.2), pct
            call_width = s.vpip * width * 0.75 / (1 + max(price_bb - 2, 0) / 8)
            if pct >= 1 - call_width:
                return Decision("call"), pct
            return Decision("fold"), pct
        # facing a re-raise: continue with the top of the range only
        top = s.three_bet * 0.9 / (1 + max(price_bb - 6, 0) / 20)
        if pct >= 1 - top * 0.4:
            return raise_to(v, v.max_raise_to or 0), pct
        if pct >= 1 - top:
            return Decision("call"), pct
        return Decision("fold"), pct

    n_opp = len(v.active) - 1
    key = (v.player_id, v.street, n_opp)
    if key not in memo.equity:
        memo.equity[key] = equity(v.hole, v.board, n_opp, rng)
    eq = memo.equity[key]
    rel = eq * (n_opp + 1)  # equity relative to a fair share of the pot
    if v.to_call == 0:
        bet = v.pot * s.size * rng.uniform(0.8, 1.2)
        if rel >= 1.25 and rng.random() < s.agg:
            return raise_to(v, bet), eq
        cbet = v.street == 1 and v.preflop_aggressor == v.player_id and rng.random() < s.cbet
        if cbet or rng.random() < s.bluff * (0.6 if n_opp > 1 else 1.0) * (1.4 if v.street >= 2 else 1.0):
            return raise_to(v, bet * 0.85), eq
        return Decision("call"), eq
    # a bettor's range is stronger than random cards, and more so the bigger the bet and the more raises
    # there have been: discount equity accordingly before comparing it with the price
    bet_frac = v.to_call / max(v.pot - v.to_call, 1)
    eq_vs_range = eq ** (1 + RANGE_DISCOUNT * min(bet_frac, 2.0) + 0.5 * max(raises_so_far(v) - 1, 0))
    pot_odds = v.to_call / (v.pot + v.to_call)
    if eq_vs_range * (n_opp + 1) >= 1.15 and rng.random() < s.agg * 0.7:
        return raise_to(v, v.max_commit * 2.8 + v.pot * 0.3), eq
    if eq_vs_range >= pot_odds * s.call_down:
        return Decision("call"), eq
    if rng.random() < s.bluff * 0.15:
        return raise_to(v, v.max_commit * 2.6), eq
    return Decision("fold"), eq


@dataclass(frozen=True)
class Collusion:
    """A planted pair. For directed_transfer, `a` gives and `b` receives."""
    family: str
    a: str
    b: str
    intensity: float = 0.5
    windows: tuple[tuple[int, int], ...] = ((0, 10**9),)  # active hand-number ranges [start, end)

    def partner(self, pid: str) -> str | None:
        return self.b if pid == self.a else (self.a if pid == self.b else None)

    def active_at(self, hand_no: int) -> bool:
        return any(lo <= hand_no < hi for lo, hi in self.windows)


def overlay(c: Collusion, bot: Bot, v: View, base: Decision, strength: float) -> Decision:
    """The colluding decision for `bot` (a member of pair `c`), given its honest decision."""
    me, mate = v.player_id, c.partner(v.player_id)
    if mate not in v.active:
        return base
    rng = bot.rng
    hu = len(v.active) == 2
    facing_mate = v.to_call > 0 and v.street_aggressor == mate
    if c.family == "soft_play":
        if v.street == 0 and base.kind == "fold" and entered(v, mate) and strength >= 0.3 and v.to_call <= 4 * v.bb:
            return Decision("call")                         # join the partner's pot (a friendly pot to share)
        if v.to_call == 0 and hu and base.kind == "raise":
            return Decision("call")                         # check it down with the partner
        if facing_mate:
            if base.kind == "raise":
                return Decision("call")                     # never raise the partner
            if base.kind == "call" and v.street > 0 and strength < 0.85 and rng.random() < 0.7:
                return Decision("fold")                     # give the pot up rather than contest it
        return base
    if c.family == "directed_transfer":
        if me == c.a:  # giver
            if facing_mate:
                if base.kind in ("call", "raise") and (strength >= 0.5 if v.street else strength >= 0.6):
                    return Decision("fold")                 # fold a hand that is ahead
                if base.kind == "fold" and v.to_call <= 0.35 * v.stack:
                    return Decision("call")                 # pay off with a hand that is behind
            if v.to_call == 0 and hu and base.kind == "raise" and strength >= 0.6:
                return Decision("call")                     # do not bet a strong hand into the receiver
            return base
        # receiver
        if v.street == 0 and raises_so_far(v) == 0 and base.kind != "raise" and strength >= 0.2:
            return raise_to(v, v.bb * 3)                    # open: the giver will call and lose it later
        if hu and v.to_call == 0 and base.kind == "call" and rng.random() < 0.8:
            return raise_to(v, v.pot * 0.75)                # bet: the giver will fold or pay
        if facing_mate and base.kind == "fold":
            return Decision("call")
        return base
    # coordinated_isolation
    if v.street == 0:
        n_raises = raises_so_far(v)
        if n_raises == 0 and base.kind != "raise" and mate in v.active and strength >= 0.45:
            return raise_to(v, v.bb * 3)                    # light open with the partner behind
        if n_raises == 1 and v.street_aggressor == mate:
            callers = sum(1 for a in v.history if a["street"] == "preflop" and a["action"] == "call"
                          and a["player_id"] not in (me, mate))
            to_act = len(v.active) - 2 - callers
            if callers + to_act > 0 and callers > 0 and strength >= 0.35:
                return raise_to(v, v.max_commit * 3.5)      # squeeze the callers out
            if base.kind != "fold":
                return Decision("fold")                     # step aside for the partner
        return base
    if facing_mate and base.kind == "raise":
        return Decision("call")                             # never raise the partner
    if hu and v.to_call == 0 and base.kind == "raise":
        return Decision("call")
    if v.to_call > 0 and v.street_aggressor not in (me, mate) and mate in v.active and len(v.active) > 2 \
            and base.kind == "call" and strength >= 0.35:
        return raise_to(v, v.max_commit * 3)                # raise a third player out of the partner's pot
    return base


class TablePolicy:
    """Decides for every bot at a table; applies collusion overlays and records where they changed play."""

    def __init__(self, bots: dict[str, Bot], pairs: list[Collusion], rng: random.Random):
        self.bots, self.pairs, self.rng = bots, pairs, rng
        self.member = {}
        for i, c in enumerate(pairs):
            for pid in (c.a, c.b):
                if pid in self.member:
                    raise ValueError(f"{pid} is in two colluding pairs")
                self.member[pid] = i
        self.memo = HandMemo()

    def new_hand(self, hand_no: int, seated: list[str]) -> None:
        self.memo = HandMemo()
        for i, c in enumerate(self.pairs):
            if c.a in seated and c.b in seated and c.active_at(hand_no) and self.rng.random() < c.intensity:
                self.memo.coordinating.add(i)

    def __call__(self, v: View) -> Decision:
        bot = self.bots[v.player_id]
        d, strength = base_decision(bot, v, self.memo)
        i = self.member.get(v.player_id)
        if i is not None and i in self.memo.coordinating:
            od = overlay(self.pairs[i], bot, v, d, strength)
            if od != d:
                self.memo.changed[i] = self.memo.changed.get(i, 0) + 1
            d = od
        return d

    def end_hand(self, net: dict[str, int], bb: int) -> None:
        for pid, n in net.items():
            if pid in self.bots:
                self.bots[pid].note_result(n / bb)
