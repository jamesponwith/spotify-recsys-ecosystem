"""What a Cardano validator can enforce, and the one thing it structurally cannot.

The pitch for NFT ticketing is that a smart contract can make a ticket obey
rules: no resale above face, no transfer at all, royalties to the artist on every
hop. All three are true and all three are enforceable on Cardano, cheaply, in a
Plutus validator over a CIP-68 asset pair. The design is in `docs/CARDANO.md`.

The problem is the sentence underneath: *the chain enforces what it can see.*

A broker who cannot transfer a ticket can sell the wallet that holds it. That is
not a transfer, it is a disclosure -- a seed phrase read out over a messaging
app -- and no validator observes it, because nothing happens on chain at all.
The ticket never moves. Its owner does.

The one real defence the ledger provides here is not enforcement, it is
*adverse selection*. A key sale cannot be escrowed: a smart contract can hold an
asset until both sides perform, but it cannot hold a secret, because a secret
that has been shown has been given away. So the seller keeps a copy of the key
and can walk in first, and the buyer knows it, and prices it in. On-chain
restriction turns the resale market into a lemons market. It does not close it.

This module measures how much of the spread survives each level of enforcement,
and then measures something entirely different that a real deployment hits
first: eUTxO contention, which is the reason a naive on-chain drop cannot sell
nine thousand seats.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import ARM_BY_KEY, Scenario
from .market import solve
from .metrics import summarise
from .population import draw_for


@dataclass(frozen=True)
class Enforcement:
    """One rung of the on-chain enforcement ladder."""

    key: str
    label: str
    # Share of broker inventory that can still be monetised at all.
    leak: float
    # Price cut the buyer demands for the risk they are now carrying.
    trust_discount: float
    # Whether an on-chain price ceiling binds the sales that do happen.
    price_capped: bool
    mechanism: str
    # A cut the validator takes on every transfer, on top of marketplace fees.
    royalty: float = 0.0


# The rug probability the buyer is pricing when they buy a stranger's wallet:
# the seller kept the seed phrase and can enter first, or move the ticket out,
# or sell the same wallet again. There is no on-chain remedy -- see the module
# docstring on why a secret cannot be escrowed. 0.25 is a guess and is swept.
RUG_PROBABILITY = 0.25

# Share of the ticket-buying public that will hold a self-custodied wallet at
# all, let alone buy someone else's. Everything web3 does here is gated on this
# number and it is small. Custodial wallets raise it -- and a custodial wallet
# is an account with extra steps, which forfeits the property that made the
# chain interesting.
SELF_CUSTODY_SHARE = 0.35


LADDER: tuple[Enforcement, ...] = (
    Enforcement(
        key="plain_nft",
        label="Plain NFT ticket",
        leak=1.0,
        trust_discount=0.0,
        price_capped=False,
        mechanism="CIP-25/CIP-68 asset, freely transferable. Provenance is public "
        "and counterfeiting is impossible -- neither of which is the problem.",
    ),
    Enforcement(
        key="royalty",
        label="On-chain royalty",
        leak=1.0,
        trust_discount=0.0,
        price_capped=False,
        royalty=0.10,
        mechanism="A validator takes 10% of every transfer. Taxes the spread; "
        "does not cap it. The broker pays it out of a margin they still keep.",
    ),
    Enforcement(
        key="capped_script",
        label="Validator-capped resale",
        leak=SELF_CUSTODY_SHARE,
        trust_discount=RUG_PROBABILITY,
        price_capped=True,
        mechanism="The user token may only be spent into the exchange script, "
        "which refuses any output above face + uplift. On-chain resale is now "
        "capped -- and a key sale routes around it entirely.",
    ),
    Enforcement(
        key="soulbound",
        label="Soulbound (non-transferable)",
        leak=SELF_CUSTODY_SHARE,
        trust_discount=RUG_PROBABILITY,
        price_capped=True,
        mechanism="The validator requires the token to return to the same key. "
        "Strictly stronger than the cap on chain, and *identical* off it: the "
        "only remaining route was already the only route.",
    ),
    Enforcement(
        key="soulbound_gate",
        label="Soulbound + identity at the gate",
        leak=SELF_CUSTODY_SHARE * 0.12,
        trust_discount=RUG_PROBABILITY * 2.2,
        price_capped=True,
        mechanism="Redemption is a signature over a fresh challenge *plus* an ID "
        "whose name matches the datum. The buyer of a wallet still cannot get in. "
        "This is the rung that works, and it is the one that is not on the chain.",
    ),
)


def leak_ladder(scn: Scenario, *, arm_key: str = "capped", trials: int = 8) -> dict:
    """Run the market once per enforcement rung and report what the broker keeps.

    The comparison that matters is `capped_script` against `soulbound`: those two
    are wildly different on chain and nearly identical here, because the channel
    they both leave open is the one neither of them can see.
    """
    arm = ARM_BY_KEY[arm_key]
    rows: list[dict] = []
    for rung in LADDER:
        cell = replace(
            scn,
            off_platform_leak=rung.leak,
            off_platform_trust_discount=rung.trust_discount,
            bound_leak_factor=1.0,
            bound_trust_discount=rung.trust_discount,
            resale_fee=scn.resale_fee + rung.royalty,
        )
        cell_arm = replace(arm, transfer="capped" if rung.price_capped else "open")
        per_trial = []
        for t in range(trials):
            pop = draw_for(cell, t)
            per_trial.append(summarise(solve(cell_arm, cell, pop), cell_arm, cell, pop))
        rows.append(
            {
                "key": rung.key,
                "label": rung.label,
                "mechanism": rung.mechanism,
                "leak": rung.leak,
                "trust_discount": rung.trust_discount,
                "price_capped": rung.price_capped,
                "royalty": rung.royalty,
                "broker_capture": float(np.mean([r["broker_capture"] for r in per_trial])),
                "broker_profit": float(np.mean([r["broker_profit"] for r in per_trial])),
                "price_multiple": float(np.mean([r["price_multiple"] for r in per_trial])),
                "face_access": float(np.mean([r["face_access"] for r in per_trial])),
            }
        )

    base = float(rows[0]["broker_profit"])
    for r in rows:
        r["spread_retained"] = float(r["broker_profit"]) / base if base > 0 else 0.0
    return {
        "arm": arm_key,
        "rug_probability": RUG_PROBABILITY,
        "self_custody_share": SELF_CUSTODY_SHARE,
        "rungs": rows,
    }


def contention(
    seats: int,
    concurrent_buyers: int,
    shards: int,
    *,
    block_seconds: float = 20.0,
    seats_per_purchase: float = 2.28,
) -> dict:
    """How long an on-chain drop takes, given that a UTxO can be spent once.

    This is the Cardano-specific engineering constraint and it is not a
    performance detail -- it is the difference between a design that works and
    one that cannot exist. In the eUTxO model a transaction consumes specific
    outputs. Two buyers who build transactions against the *same* inventory UTxO
    produce two valid-looking transactions of which exactly one can be accepted;
    the other fails at validation, having already been signed and submitted.

    So a single "9,200 seats" UTxO sells one seat per block. The fix is to shard
    the inventory across many UTxOs and let buyers pick one at random, which
    turns the question into a balls-in-bins problem: with `n` buyers choosing
    among `k` shards, the expected number of shards receiving at least one bid
    is k(1 - (1 - 1/k)^n), and each of those settles exactly one.
    """
    if shards <= 0 or concurrent_buyers <= 0:
        raise ValueError("shards and concurrent_buyers must be positive")
    k, n = float(shards), float(concurrent_buyers)
    settled_per_block = k * (1.0 - (1.0 - 1.0 / k) ** n)
    success_rate = settled_per_block / n
    purchases_needed = seats / seats_per_purchase
    blocks = purchases_needed / max(settled_per_block, 1e-9)
    return {
        "seats": seats,
        "concurrent_buyers": concurrent_buyers,
        "shards": shards,
        "settled_per_block": settled_per_block,
        "success_rate": success_rate,
        "failed_per_block": n - settled_per_block,
        "blocks_to_clear": blocks,
        "minutes_to_clear": blocks * block_seconds / 60.0,
        # The number that decides whether this is shippable. `minutes_to_clear`
        # looks respectable at almost any shard count; what it hides is that
        # every buyer is submitting a signed transaction every block and having
        # it rejected. At 100 shards against 40,000 buyers that is ~400 failed
        # submissions per person before one lands. A waiting room is a queue. A
        # contention storm is 400 error dialogs.
        "attempts_per_purchase": 1.0 / max(success_rate, 1e-12),
    }


def contention_curve(
    seats: int,
    concurrent_buyers: int,
    shard_options: tuple[int, ...] = (1, 10, 100, 500, 2000, 10000),
) -> dict:
    return {
        "rows": [contention(seats, concurrent_buyers, k) for k in shard_options],
        "note": "One transaction per shard per block. Every other bid in that "
        "block is a signed, submitted, rejected transaction -- a user-visible "
        "failure, not a queue position.",
    }
