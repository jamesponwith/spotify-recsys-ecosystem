"""Scenario parameters, and an honest account of which ones are guesses.

Every other application in this ecosystem measures a real corpus. Concerto does
not: no promoter publishes its on-sale logs, and the resale platforms publish
only what flatters them. So this is a *simulation*, and a simulation's headline
is worth exactly as much as the parameters underneath it.

Two things are done about that, and they are the reason the numbers here are
worth reading at all:

1. Every assumed quantity below carries a citation-grade comment saying where
   the value came from and how confident it is.
2. `concerto sensitivity` re-runs the whole comparison across a grid over the
   four parameters that could plausibly overturn it. A finding is only reported
   as a finding if its *ranking* survives the grid. The point estimates in the
   report are one cell of that grid; the claim rests on the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SEED = 20260821

CONCERTO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = CONCERTO_ROOT / "artifacts"


@dataclass(frozen=True)
class Scenario:
    """One on-sale: a house, a face price, and the people who want in."""

    # --- The house -------------------------------------------------------
    # A mid-size arena show. Capacity is the number of seats that actually
    # reach a public on-sale, which is the number that matters and is never the
    # number on the venue's website: holds for artist, promoter, label, venue,
    # fan club, credit-card presale and platinum inventory come off the top
    # first. 46% is the figure from the New York Attorney General's 2016
    # "Obstructed View" investigation, which found that on average only that
    # share of tickets reached the general public. It is the best-documented
    # number available and it is nine years old; the practice has not become
    # less common since, so treat it as an upper bound on public availability.
    capacity: int = 9_200
    public_share: float = 0.46
    face_price: float = 95.0

    # --- Demand ----------------------------------------------------------
    # Demand multiple: total fan ticket demand as a multiple of the tickets
    # actually on sale. This is the single most important assumption in the
    # model and it is a guess. A show that does not sell out has a multiple
    # below 1 and no scalping problem worth modelling; the shows that generate
    # the complaints run 5x-50x. 8x is a deliberately unspectacular choice --
    # a hot but not once-in-a-decade show. Swept in `sensitivity`.
    demand_multiple: float = 8.0

    # Party size distribution. People buy for groups, which is why a tight
    # per-identity cap is not free: it splits parties. P(1..4 tickets).
    party_probs: tuple[float, ...] = (0.18, 0.52, 0.14, 0.16)

    # Willingness to pay is built as `face * income_factor * (0.5 + affinity)`.
    # Splitting it this way is the whole argument of this project: WTP is the
    # product of how much money you have and how much you care, and a price
    # rations on the product while a fan wants it to ration on the second term.
    # Income factor is lognormal -- income always is -- with sigma 0.85, which
    # reproduces roughly the US household-income Gini of 0.48 within the
    # concert-going subpopulation.
    income_sigma: float = 0.85
    income_median_factor: float = 1.15

    # --- Brokers ---------------------------------------------------------
    # Identity acquisition is convex: the tenth account is easy, the ten
    # thousandth needs residential proxies, aged accounts, distinct payment
    # instruments and a person to answer a phone. cost(m) = c0 * m^gamma.
    # gamma > 1 -- rising marginal cost of a usable identity -- is the only
    # structural assumption here, and it is what makes the broker sector finite.
    #
    # c0 = 0.174 is not a measurement. It is fitted, by bisection, so that the
    # unrestricted `queue` arm produces a resale markup of exactly 3.0x face at
    # the reference demand multiple -- see `concerto calibrate`, which
    # regenerates the fit and the sweep around it. Markup is the right thing to
    # fit on because it is the only quantity here that is publicly observable:
    # resale listings are visible, broker inventory is not. Broker capture is
    # therefore an *output* of the model, not a target it was tuned to hit.
    identity_cost_c0: float = 0.174
    identity_cost_gamma: float = 1.55

    # Verified-fan / KYC raises the cost of an identity. Ticketmaster's own
    # Verified Fan claims a large reduction in bot purchasing; the multiplier
    # here is what that claim implies if it is taken at face value.
    verify_cost_multiplier: float = 12.0
    # ...and it is not free. Real fans fail verification: no purchase history,
    # a shared address, a prepaid card, a country mismatch. 4% is the low end
    # of what identity vendors quote as a false-reject rate.
    verify_false_reject: float = 0.04

    # Forging *affinity* -- a plausible multi-year listening history bound to a
    # payment identity -- is dearer again than forging an identity, because it
    # cannot be bought on the day. This multiplier is the load-bearing
    # assumption of the affinity arm and is swept in `sensitivity`.
    affinity_forge_multiplier: float = 60.0

    # --- Resale ----------------------------------------------------------
    # Marketplace take rate on a resale, buyer and seller side combined. The
    # major secondary platforms sit between 25% and 30% all-in.
    resale_fee: float = 0.27
    # Primary-market service fees, as a share of face. Multiple US class
    # actions have put all-in fees at 20-40% of face.
    primary_fee: float = 0.27

    # What a face-value-capped exchange allows above face, to cover fees.
    exchange_cap_uplift: float = 0.10

    # Leakage: the share of restricted inventory a broker can still move
    # off-platform (a StubHub listing, a Craigslist meet, a sold account).
    # This is the parameter the Cardano work exists to interrogate.
    off_platform_leak: float = 0.35
    # A buyer paying a stranger off-platform for a ticket they cannot verify
    # bears counterparty risk, and prices it in. The discount is what that risk
    # costs the seller.
    off_platform_trust_discount: float = 0.30

    # When the ticket is bound to an identity and checked at the gate, the only
    # thing left to sell is the account itself. Both numbers are far harsher
    # than the open-market ones, and deliberately so: the buyer is now trusting
    # a stranger not to walk in on their own credential first, and cannot
    # inspect anything before the doors open. `bound_leak_factor` scales the
    # off-platform channel down; `bound_trust_discount` prices the extra risk.
    # These two are exactly what the Cardano validator can and cannot move --
    # see `ledger.py`.
    bound_leak_factor: float = 0.25
    bound_trust_discount: float = 0.55

    # Binding a ticket to an identity means checking that identity at the door,
    # and a door check turns people away who should have got in: a changed
    # surname, a partner holding the pair, a phone that died, no accepted photo
    # ID at all. This is the cost of the strictest arm and the model would
    # flatter it badly by leaving the number at zero. Counted as a customer
    # harm, not as an unsold seat -- the promoter still banked the money.
    gate_id_failure: float = 0.02

    # --- Solver ----------------------------------------------------------
    # Brokers form expectations about the resale price, act on them, and their
    # actions move the price. The model solves for the fixed point rather than
    # assuming broker behaviour is exogenous -- see broker.py for why that
    # single choice decides most of the results.
    max_iterations: int = 200
    tolerance: float = 1e-4
    damping: float = 0.35

    n_trials: int = 24
    seed: int = SEED

    @property
    def on_sale(self) -> int:
        """Seats that reach the public on-sale."""
        return int(round(self.capacity * self.public_share))


@dataclass(frozen=True)
class Arm:
    """One allocation policy, as the combination of choices a promoter makes.

    Keeping these as four orthogonal knobs rather than seven named products is
    deliberate: it is the only way to attribute an effect to the *mechanism*
    rather than to the brand name a platform gave a bundle of mechanisms.
    """

    key: str
    label: str
    # How the primary allocation is rationed among those who ask.
    rule: str = "lottery"  # lottery | clearing | affinity
    # Tickets one identity may buy. None = no cap.
    cap: int | None = 8
    # Multiplier on the cost of acquiring one usable identity.
    identity_cost_multiplier: float = 1.0
    # Share of legitimate fans wrongly excluded by the identity check.
    false_reject: float = 0.0
    # What a holder may do with the ticket afterwards.
    transfer: str = "open"  # open | capped | bound
    # Share of the house sold at a market-clearing price rather than face.
    clearing_share: float = 0.0
    # Share of the *face-priced* seats drawn by open lottery among everyone who
    # registered, ignoring the arm's rule. Only meaningful under a merit rule:
    # an affinity cut is absolute, so a fan below it has no chance at all rather
    # than a small one, and this is the knob that gives it back.
    # `frontier.reserve_curve` is what sets it -- 20% was asserted in the first
    # draft of the policy document and measured afterwards.
    reserve_share: float = 0.0
    note: str = ""


ARMS: tuple[Arm, ...] = (
    Arm(
        key="queue",
        label="Queue (as it ships)",
        rule="lottery",
        cap=8,
        transfer="open",
        note="A virtual waiting room admits in random order. Every identity is a "
        "separate place in the line, which is the whole problem.",
    ),
    Arm(
        key="lottery",
        label="Registration lottery",
        rule="lottery",
        cap=4,
        transfer="open",
        note="Register first, draw later. Removes the speed advantage; leaves the "
        "identity advantage entirely intact.",
    ),
    Arm(
        key="verified",
        label="Verified fan",
        rule="lottery",
        cap=4,
        identity_cost_multiplier=12.0,
        false_reject=0.04,
        transfer="open",
        note="Identity verification before the draw. Makes each broker identity "
        "12x dearer -- and wrongly excludes some share of real fans.",
    ),
    Arm(
        key="capped",
        label="Verified + capped exchange",
        rule="lottery",
        cap=4,
        identity_cost_multiplier=12.0,
        false_reject=0.04,
        transfer="capped",
        note="The same draw, but resale is only legal through an official "
        "exchange at face + 10%. Attacks the margin instead of the bot.",
    ),
    Arm(
        key="bound",
        label="Identity-bound, refund-only",
        rule="lottery",
        cap=4,
        identity_cost_multiplier=12.0,
        false_reject=0.04,
        transfer="bound",
        note="The ticket is not transferable at all: unwanted seats go back to "
        "the exchange at face and are re-sold to the queue.",
    ),
    Arm(
        key="clearing",
        label="Market clearing (Dutch)",
        rule="clearing",
        cap=4,
        transfer="open",
        clearing_share=1.0,
        note="Price the house at what it will actually bear. Eliminates the "
        "arbitrage by taking it -- the question is who then cannot afford to go.",
    ),
    Arm(
        key="affinity",
        label="Affinity-rationed",
        rule="affinity",
        cap=4,
        identity_cost_multiplier=60.0,
        false_reject=0.04,
        transfer="capped",
        note="Seats go to the deepest verified listening histories, at face. "
        "Rations on caring rather than on paying.",
    ),
    Arm(
        key="affinity_bound",
        label="Affinity + identity-bound",
        rule="affinity",
        cap=4,
        identity_cost_multiplier=60.0,
        false_reject=0.04,
        transfer="bound",
        note="The two mechanisms that each worked, together: seats rationed on "
        "listening depth, and no resale market for a broker to sell into.",
    ),
    Arm(
        key="hybrid",
        label="Affinity + bound + 15% cleared",
        rule="affinity",
        cap=4,
        identity_cost_multiplier=60.0,
        false_reject=0.04,
        transfer="bound",
        clearing_share=0.15,
        note="The best mechanism in the set, and not the one to ship first. "
        "Affinity-rationed at face and unresellable for 85% of the house; the "
        "remaining 15% priced at what it will bear, so the artist's upside goes "
        "to the artist rather than a broker. 15% is a judgement, not an optimum "
        "-- `concerto frontier` shows the clearing curve has no knee, and that "
        "this share buys $63 a seat while leaving 89% of low-income access and "
        "all superfan access intact. Bound tickets are also restricted by law in "
        "several US states, so the shippable default is `capped`.",
    ),
)

ARM_BY_KEY = {a.key: a for a in ARMS}


@dataclass(frozen=True)
class SensitivityGrid:
    """The parameters that could overturn the result, and their plausible range.

    Chosen by asking, for each assumption: if a sceptic said "you only got that
    because you assumed X", is there a value of X that flips the ordering? These
    four are the ones where the answer was not obviously no.
    """

    demand_multiple: tuple[float, ...] = (3.0, 5.0, 8.0, 15.0, 30.0)
    identity_cost_gamma: tuple[float, ...] = (1.25, 1.55, 2.0)
    off_platform_leak: tuple[float, ...] = (0.10, 0.35, 0.65, 0.90)
    affinity_forge_multiplier: tuple[float, ...] = (10.0, 60.0, 200.0)
    trials: int = 8
    arms: tuple[str, ...] = field(
        default_factory=lambda: (
            "queue",
            "verified",
            "capped",
            "bound",
            "clearing",
            "affinity",
            "affinity_bound",
        )
    )
