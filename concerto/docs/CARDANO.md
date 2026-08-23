# Cardano

The design, what it buys, and the two things it cannot do that decide where it
belongs in the stack.

## Why Cardano rather than a general-purpose chain

Three properties are a genuinely good fit for ticketing, and one is a serious
problem. All four are consequences of the same extended-UTxO model.

**Fees are knowable before submission.** A Cardano transaction's fee is
`155381 + 44 x size_bytes` lovelace plus the cost of the execution units the
script consumes, and all of those are computed locally before the transaction is
signed. There is no gas auction and no failed-and-still-charged transaction from
a price spike. For a consumer product where the buyer is told a total price
before they agree to it, this matters more than throughput.

**Validation is local and deterministic.** A script's result depends only on the
transaction and the outputs it consumes, not on global chain state at inclusion
time. You can prove a purchase transaction will validate before you show the
buyer a spinner.

**Tokens are first class.** Minting a ticket does not require deploying a
contract — the ledger tracks multi-asset values natively, so a minting policy
governs issuance and ordinary transfer needs no script at all. A ticket, a
stablecoin and ADA can move in a single atomic transaction with no bridge and no
wrapped anything.

**And the problem: a UTxO can be spent exactly once.** This is the thing that
decides the architecture, and it is covered under [contention](#contention-is-the-real-constraint).

## The ticket

CIP-68, which splits an asset into two tokens under the same policy:

| Token | Label | Lives | Holds |
|---|---|---|---|
| reference | `(100)` | at a validator address | the datum: event, tier, seat, face price, transfer policy, binding |
| user | `(222)` | in the holder's wallet | nothing — it *is* the entitlement |

The split is what makes the metadata updatable without touching the holder's
wallet. A seat re-assignment, a tier change, a policy update after a legal
ruling — all of them are a spend of the reference UTxO, and the fan's token
never moves. CIP-25, where metadata is written into the minting
transaction, is effectively frozen — changing it means re-minting under a policy
you deliberately left open, which is the wrong shape for anything with a
lifecycle and a bad thing to leave open on a ticket.

```
TicketDatum {
  event_id:      ByteArray,
  tier:          ByteArray,
  seat:          Option<ByteArray>,     -- None until assigned
  face_lovelace: Int,
  policy:        TransferPolicy,        -- Open | Capped { uplift_bps } | Bound
  bound_to:      Option<VerificationKeyHash>,
  transferable_after: Option<PosixTime>,
  redeemed:      Bool,
}
```

## The validator

Three rules, in Aiken shape. The interesting one is the second.

```aiken
validator ticket {
  spend(datum: TicketDatum, redeemer: Action, own_ref: OutputReference, tx: Transaction) {
    when redeemer is {
      // 1. Ordinary transfer -- permitted only if the policy allows it at all.
      Transfer { to } ->
        when datum.policy is {
          Open -> after(tx, datum.transferable_after)
          Bound -> False
          Capped { .. } -> False   // must go through Exchange, below
        }

      // 2. Resale through the official exchange. The validator does not
      //    "cap the price" in any abstract sense -- it refuses to validate a
      //    transaction whose value flows are wrong. The ceiling is a property
      //    of the outputs, which is the only thing a script can actually see.
      Exchange { buyer } ->
        when datum.policy is {
          Capped { uplift_bps } -> {
            let ceiling = datum.face_lovelace * (10000 + uplift_bps) / 10000
            let paid = value_paid_by(tx, buyer)
            and {
              paid <= ceiling,
              value_paid_to(tx, seller_of(datum)) >= datum.face_lovelace,
              value_paid_to(tx, promoter) >= fee_of(paid),
              continuing_output_preserves(tx, own_ref, datum),
            }
          }
          _ -> False
        }

      // 3. Redemption at the gate. A signature over a transaction the gate
      //    built seconds ago -- so a replayed or screenshotted proof is dead.
      Redeem ->
        and {
          !datum.redeemed,
          list.has(tx.extra_signatories, expect_some(datum.bound_to)),
          marks_redeemed(tx, own_ref, datum),
        }
    }
  }
}
```

Every one of those rules is enforceable, cheap, and correct. That is not the
problem.

## The hole

**A broker who cannot transfer a ticket can sell the wallet.**

That is not a transfer. It is a disclosure — a seed phrase read out over a
messaging app — and it produces no transaction, so no validator observes it.
The ticket never moves. Its owner does.

The measured consequence is in [RESULTS.md](RESULTS.md) and it is stark: a
**validator-capped resale** and a **fully soulbound token** are entirely
different contracts on chain and produce *the same* broker economics, because
the channel neither of them can see is the one carrying the volume. Adding
strictness on chain, past the first rung, buys nothing.

What the chain does buy is **adverse selection**, and this is worth stating
precisely because it is the one real benefit and it is usually claimed as
something stronger. A key sale cannot be escrowed. A smart contract can hold an
asset until both sides perform; it cannot hold a *secret*, because a secret that
has been shown has been given away. So the seller keeps a copy of the key and
can walk in first, or move the ticket out, or sell the same wallet twice — and
the buyer knows it and pays less. On-chain restriction turns the resale market
into a lemons market. It does not close it.

The rung that closes it is not on the chain: an identity check at the door,
where the buyer of a wallet still cannot get in. Which means **the strongest
anti-scalping component in a web3 ticketing system is a person with a scanner.**

## Contention is the real constraint

In the eUTxO model a transaction names the outputs it consumes. Two buyers who
build transactions against the same inventory UTxO produce two individually
valid transactions of which exactly one can be accepted — and the loser has
already signed and submitted. It does not queue. It fails.

So a single "9,200 seats" UTxO sells **one seat per block**, about one every
twenty seconds. Sharding the inventory across `k` UTxOs and letting buyers pick
at random turns it into a balls-in-bins problem: `k(1 - (1 - 1/k)^n)` shards
receive at least one bid and each settles exactly one. The numbers are in
[RESULTS.md](RESULTS.md) and they are not encouraging — at 100 shards against
40,000 concurrent buyers, about 0.25% of submitted transactions succeed, which
is roughly 400 signed, submitted, rejected transactions per person who ends up
with a ticket. A waiting room is a queue. That is 400 error dialogs.

There is a second cost that is easy to miss: **every UTxO holding a token must
carry a minimum ADA balance**, around 1.2–1.5 ADA depending on datum size. Ten
thousand inventory shards is ~15,000 ADA of capital locked up for the duration
of the on-sale, per event, before a single ticket sells.

The standard answers are all off-chain in the end:

- **A batcher** collects intents and submits them in ordered batches. It works,
  and it is a trusted sequencer — the thing the chain was supposed to remove.
- **Hydra** moves the on-sale into an isomorphic state channel and settles the
  result on L1. This is the right answer technically and it needs every
  participant in the head, which a public on-sale does not have.
- **Do not put the on-sale on chain.** Which is the recommendation.

## What to actually ship

**The chain is a settlement and provenance layer, not the on-sale path.**

| Runs off chain | Runs on chain |
|---|---|
| registration, verification, the draw | minting the ticket after settlement |
| checkout, payment authorisation | the transfer policy, as an enforceable rule |
| the gate scan itself | redemption receipt, written after the fact |
| customer support, refunds, appeals | the published draw seed |

That split keeps the properties worth having — a ticket whose provenance is
publicly verifiable, a resale rule that is a matter of arithmetic rather than of
a platform's goodwill, and a draw whose fairness a third party can check — and
drops the ones that do not survive contact with forty thousand people pressing a
button.

## Currencies

Native multi-asset value means a single transaction can move a ticket against
whatever the buyer holds, atomically, with no bridge:

| Rail | Note |
|---|---|
| ADA | Volatile between purchase and event. Price in fiat, quote ADA at settlement, and say so at checkout. |
| Cardano-native stablecoins (DJED, iUSD, USDM) | The honest caveat: aggregate liquidity is thin next to the volumes a major on-sale moves, and depeg risk on an algorithmic design is a real operational exposure. Treat as a settlement asset, not a treasury. |
| Fiat | Stripe. This is what almost everyone will use. |

The uncomfortable part of "supports all currencies": most ticket buyers will not
self-custody. Give them a custodial wallet and you have rebuilt an account
system with extra steps and forfeited the property that made the chain
interesting — the holder no longer holds anything. The model puts self-custody
at 35%, and every benefit above is gated on that number.

## Honest summary

Web3 gives this product three things: **verifiable provenance**, a **transfer
rule that is arithmetic rather than policy**, and a **publicly auditable draw
seed**. Those are real and none of them is nothing.

It does not give you an anti-scalping mechanism. The measurement says the
broker's margin is closed by the exchange design and the door, and the chain is
where you record that it happened.
