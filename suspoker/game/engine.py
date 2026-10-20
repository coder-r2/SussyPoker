"""One six-max No-Limit Hold'em hand on top of pokerkit, logged in the competition's semantics.

pokerkit owns the rules: whose turn it is, legal raise sizes, side pots, showdown and pot splitting. This module
keeps its own bookkeeping next to it (street commitments, pot, stacks) because the log needs values pokerkit does
not expose in the competition's form, and it cross-checks the two after every action.

    hand = Hand(seats, button=2, sb=1, bb=2, deck=shuffled_cards)
    while (view := hand.view()) is not None:
        hand.act(policy(view))
    record = hand.record()

Seats are numbered 0-5. SB = button + 1, BB = button + 2 and preflop action starts at button + 3, as in the data.
pokerkit numbers players from the small blind, so pokerkit index i is seat (button + 1 + i) % 6.
"""

from dataclasses import dataclass, field

from pokerkit import Automation, Mode, NoLimitTexasHoldem

from suspoker.cards import CARD_STRINGS

N_SEATS = 6
STREET_NAMES = ("preflop", "flop", "turn", "river")
AUTOMATIONS = (
    Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION, Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
)


@dataclass(frozen=True)
class SeatIn:
    player_id: str
    stack: int


@dataclass(frozen=True)
class Decision:
    """What a player chooses: "fold", "call" (check when nothing to call) or "raise" to `amount_to` (the total
    street commitment after the action; clamped to the legal range, and a call if raising is not allowed)."""
    kind: str
    amount_to: int = 0


@dataclass
class View:
    """Everything the player to act may know: own cards, the public board and the betting so far."""
    seat: int
    player_id: str
    hole: tuple[int, int]
    board: list[int]
    street: int
    pot: int
    to_call: int
    stack: int
    committed: int            # own commitment on this street
    max_commit: int           # highest commitment on this street
    min_raise_to: int | None  # None: raising is not allowed
    max_raise_to: int | None
    bb: int
    button: int
    seats: dict[int, str]     # seat -> player_id for every seat
    active: list[str]         # players not folded, in seat order
    all_in: list[str]         # players with no chips behind
    history: list[dict]       # action rows so far (competition schema without hand_id)
    street_aggressor: str | None  # last bet / raise / aggressive all-in on this street
    preflop_aggressor: str | None


@dataclass
class HandRecord:
    hand: dict
    seats: list[dict]
    actions: list[dict]
    net: dict[str, int] = field(default_factory=dict)  # player_id -> net chips


class Hand:
    def __init__(self, seats: list[SeatIn], button: int, sb: int, bb: int, deck: list[int]):
        if len(seats) != N_SEATS:
            raise ValueError("every hand is six-handed")
        self.seats, self.button, self.sb, self.bb = seats, button, sb, bb
        self.order = [(button + 1 + i) % N_SEATS for i in range(N_SEATS)]  # pokerkit index -> seat
        self.idx = {s: i for i, s in enumerate(self.order)}
        self.start = [seats[s].stack for s in self.order]
        self.state = NoLimitTexasHoldem.create_state(AUTOMATIONS, True, 0, (sb, bb), bb, tuple(self.start), N_SEATS,
                                                     mode=Mode.CASH_GAME)
        self.hole = {s: (deck[2 * k], deck[2 * k + 1]) for k, s in enumerate(self.order)}
        for s in self.order:
            self.state.deal_hole("".join(CARD_STRINGS[c] for c in self.hole[s]))
        self.deck = deck[2 * N_SEATS:]
        self.board: list[int] = []
        self.street = 0
        self.commit = [sb, bb] + [0] * (N_SEATS - 2)      # by pokerkit index
        self.stack = [st - c for st, c in zip(self.start, self.commit, strict=True)]
        self.contrib = list(self.commit)
        self.pot = sb + bb
        self.folded = [False] * N_SEATS
        self.actions: list[dict] = []
        self.street_aggressor: str | None = None
        self.preflop_aggressor: str | None = None
        self._advance()

    # -- dealing ---------------------------------------------------------------------------------------------
    def _advance(self) -> None:
        """Deal board cards until a player must act or the hand is over."""
        while self.state.status and self.state.actor_index is None:
            if self.state.can_burn_card():
                self.state.burn_card("??")
                n = 3 if not self.board else 1
                cards, self.deck = self.deck[:n], self.deck[n:]
                self.state.deal_board("".join(CARD_STRINGS[c] for c in cards))
                self.board += cards
                self.street = {3: 1, 4: 2, 5: 3}[len(self.board)]
                self.commit = [0] * N_SEATS
                self.street_aggressor = None
            elif self.state.can_no_operate():
                self.state.no_operate()
            else:  # pragma: no cover - pokerkit is waiting for an operation we do not automate
                raise RuntimeError("pokerkit stalled")

    # -- acting ----------------------------------------------------------------------------------------------
    def _pid(self, i: int) -> str:
        return self.seats[self.order[i]].player_id

    def view(self) -> View | None:
        i = self.state.actor_index
        if not self.state.status or i is None:
            return None
        seat = self.order[i]
        hi = max(self.commit)
        can_raise = self.state.can_complete_bet_or_raise_to()
        return View(
            seat=seat, player_id=self._pid(i), hole=self.hole[seat], board=list(self.board), street=self.street,
            pot=self.pot, to_call=min(hi - self.commit[i], self.stack[i]), stack=self.stack[i],
            committed=self.commit[i], max_commit=hi,
            min_raise_to=self.state.min_completion_betting_or_raising_to_amount if can_raise else None,
            max_raise_to=self.state.max_completion_betting_or_raising_to_amount if can_raise else None,
            bb=self.bb, button=self.button, seats={s: p.player_id for s, p in enumerate(self.seats)},
            active=[self.seats[s].player_id for s in range(N_SEATS) if not self.folded[self.idx[s]]],
            all_in=[self._pid(j) for j in range(N_SEATS) if self.stack[j] == 0 and not self.folded[j]],
            history=list(self.actions), street_aggressor=self.street_aggressor,
            preflop_aggressor=self.preflop_aggressor)

    def act(self, d: Decision) -> dict:
        """Apply a decision for the player to act and return the logged action row."""
        st, i = self.state, self.state.actor_index
        if i is None:
            raise RuntimeError("no player to act")
        hi = max(self.commit)
        to_call = min(hi - self.commit[i], self.stack[i])
        row = {"action_no": len(self.actions), "street": STREET_NAMES[self.street], "player_id": self._pid(i),
               "pot_before": self.pot, "stack_before": self.stack[i], "to_call": to_call,
               "players_active": N_SEATS - sum(self.folded)}
        kind = d.kind
        if kind == "raise" and not st.can_complete_bet_or_raise_to():
            kind = "call"
        if kind == "fold" and to_call == 0:
            kind = "call"  # never fold for free: check instead
        if kind == "fold":
            st.fold()
            self.folded[i] = True
            amount, name = 0, "fold"
        elif kind == "call":
            st.check_or_call()
            amount = to_call
            name = "check" if amount == 0 else ("all_in" if amount == self.stack[i] else "call")
        elif kind == "raise":
            target = min(max(d.amount_to, st.min_completion_betting_or_raising_to_amount),
                         st.max_completion_betting_or_raising_to_amount)
            st.complete_bet_or_raise_to(target)
            amount = target - self.commit[i]
            name = "all_in" if amount == self.stack[i] else ("bet" if hi == 0 else "raise")
            self.street_aggressor = self._pid(i)
            if self.street == 0:
                self.preflop_aggressor = self._pid(i)
        else:
            raise ValueError(f"unknown decision {d.kind!r}")
        if name == "all_in" and amount > to_call:
            self.street_aggressor = self._pid(i)
            if self.street == 0:
                self.preflop_aggressor = self._pid(i)
        self.commit[i] += amount
        self.stack[i] -= amount
        self.contrib[i] += amount
        self.pot += amount
        row.update(action=name, amount=amount, amount_to=self.commit[i])
        self.actions.append(row)
        self._advance()
        self._check_stacks()
        return row

    def _check_stacks(self) -> None:
        """Our stacks must match pokerkit's until the end, when pokerkit has paid out the pot."""
        if self.state.status and list(self.state.stacks) != self.stack:
            # pokerkit returns an uncalled bet as soon as betting closes; it is only allowed once nobody can act
            returned = [a - b for a, b in zip(self.state.stacks, self.stack, strict=True)]
            if any(r < 0 for r in returned) or self.state.actor_index is not None:
                raise AssertionError(f"stack mismatch: pokerkit {list(self.state.stacks)} vs log {self.stack}")

    @property
    def done(self) -> bool:
        return not self.state.status

    # -- result ----------------------------------------------------------------------------------------------
    def record(self) -> HandRecord:
        if not self.done:
            raise RuntimeError("hand not finished")
        final = list(self.state.stacks)
        pot = sum(self.contrib)
        live = [i for i in range(N_SEATS) if not self.folded[i]]
        showdown = len(live) >= 2
        seats, net = [], {}
        for s in range(N_SEATS):
            i = self.idx[s]
            n = final[i] - self.start[i]
            pid = self._pid(i)
            net[pid] = n
            seats.append({"player_id": pid, "seat_no": s, "starting_stack": self.start[i],
                          "hole_card_1": CARD_STRINGS[self.hole[s][0]], "hole_card_2": CARD_STRINGS[self.hole[s][1]],
                          "total_contribution": self.contrib[i], "net_chips": n, "folded": self.folded[i],
                          "went_to_showdown": showdown and not self.folded[i],
                          "won_share": (n + self.contrib[i]) / pot})
        hand = {"button_seat": self.button, "small_blind": self.sb, "big_blind": self.bb,
                "board_cards": " ".join(CARD_STRINGS[c] for c in self.board), "final_pot": pot,
                "players_dealt": N_SEATS, "players_at_showdown": len(live) if showdown else 0}
        return HandRecord(hand=hand, seats=seats, actions=list(self.actions), net=net)


def play_hand(seats: list[SeatIn], button: int, sb: int, bb: int, deck: list[int], policy) -> HandRecord:
    """Play a whole hand, asking `policy(view) -> Decision` for every action."""
    hand = Hand(seats, button, sb, bb, deck)
    while (v := hand.view()) is not None:
        hand.act(policy(v))
    return hand.record()
