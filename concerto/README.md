# Concerto

**What actually stops a scalper — measured against an adversary who adapts.**

A *concerto* is written for one soloist and one crowd, and the whole form is
about how the two are balanced.

The five applications alongside this one ask whether a *listener* was served
well. Concerto asks whether a **market** serves anyone, using the same
discipline: paired arms against a shared control, claims written down before the
grid runs, and the hypotheses that died kept on the record.

It simulates a ticket on-sale — 4,232 seats reaching a public drop, $95 face,
eight times oversubscribed — under eight allocation policies, each solved to a
rational-expectations equilibrium against a broker sector that decides for
itself how many purchasing identities to buy.

---

## Start here

**Replacing Ticketmaster is mostly not a technical problem.** Their position
rests on exclusive multi-year venue contracts and on Live Nation owning the
promoters and the buildings, which is the substance of the DOJ's 2024 complaint.
You can ship a strictly better stack and have zero inventory.

What *is* a real technical problem is the part underneath the complaints: ticket
drops, bots and instant sellouts are an **allocation and arbitrage** problem.
That is simulable, falsifiable, and it turned out to contain two results that
contradict how the industry talks about it — and one that contradicted the
hypothesis this project started from.

## Results

### No policy makes a hot ticket cheap

Across all 180 parameter cells, **every** arm leaves the average fan paying more
than 1.2x face. There is no mechanism in the set that produces a cheap ticket
for a show eight times oversubscribed. What a policy chooses is not the price.
It is who gets in, and who keeps the difference.

### Dynamic pricing removes the scalper by becoming one

| | fan pays | broker takes | artist gets | bottom income quartile served |
|---|---:|---:|---:|---:|
| Queue, as it ships | **$221** (2.33x) | 61.1% | $95/seat | 4.9% |
| Market clearing | **$291** (3.06x) | 0.0% | $229/seat | **0.0%** |

Pricing the house at what it will bear does eliminate the broker. It also
charges every fan the market price, where the queue at least let a lucky third
in at face — so the average fan pays *more*, not less. The surplus moves from a
broker to the artist, which is a real improvement and an entirely different
claim from the one usually made for it.

### Most of what scalping costs is not scalper income

Under the queue, $19 per ticket reaches broker profit and **$35** per ticket is
burned on the identity infrastructure used to win the draw — proxies, aged
accounts, payment instruments. At equilibrium the sector grosses `gamma` times
what it spends on identities, so brokers set about 64% of their take on fire.

That money leaves the fan and reaches nobody who made the music. Reporting
broker *profit* alone understates the waste by roughly four times, which is what
the first version of this report did.

### The lever is the margin, not the bot

Capping resale beat hardening identity on broker capture in **180 of 180**
cells. Verified-fan schemes do work — but the first hypothesis this project
tested was that they were theatre, and that was wrong in an instructive way:
the identity-cost elasticity is 1.8, so a 12x cost rise cuts broker entry ~92x
before competition among the survivors competes about four fifths of it back.
Large, real, and still beaten outright by attacking the margin.

### Affinity is the only thing that moves who gets in

Every price-rationed arm serves 13–22% of superfan demand. Rationing on verified
listening depth instead serves **96%**, at face, with the income skew of who
gets in falling to 1.01x — an essentially income-blind allocation.

**And it has a named failure region.** With resale left open, the claim held in
177 of 180 cells. All three failures sit in one corner — hot demand paired with
a wide-open off-platform channel — and they are near-ties that move with trial
count (10 of 180 at a quarter of the trials, which is how the boundary was found
in the first place).

The mechanism is clean. The model assumes a broker can forge whatever listening
history clears the cut, so under affinity rationing brokers are served *first*:
with resale open and a show hot enough, affinity rationing is a **priority lane
for brokers**. Pair it with a closed resale channel and the same claim holds in
**180 of 180** — which is why the recommendation is the pair, and why the pair
is tested as its own claim rather than assumed.

### The chain enforces what it can see

A Cardano validator can refuse any transfer, cap any resale price, and take a
royalty on every hop. All three are cheap and correct. None of them is the
problem.

> A **validator-capped resale** and a **fully soulbound token** are entirely
> different contracts and produce the **same** broker economics — 7% of the
> original spread each — because the channel neither can see is the one carrying
> the volume. A broker who cannot transfer a ticket sells the *wallet*, which
> produces no transaction at all.

A smart contract can escrow an asset. It cannot escrow a secret, because a
secret that has been shown has been given away — so the seller keeps a copy of
the key, and the buyer prices that in. On-chain restriction turns resale into a
lemons market; it does not close it. The only rung that closed the spread is not
on the chain: an identity check at the door.

Which makes the strongest anti-scalping component in a web3 ticketing system **a
person with a scanner**.

### And the drop does not fit on chain

In the eUTxO model a transaction consumes specific outputs, so a single
inventory UTxO sells one seat per block. Shard the house and it becomes a retry
storm: at 100 shards against 40,000 concurrent buyers, **0.25% of submitted
transactions succeed** — about 400 signed, submitted, rejected transactions per
person who gets a ticket. Plus ~1.2–1.5 ADA of min-UTxO locked per shard.

The chain belongs where it is good: minting after settlement, the transfer rule
as enforceable arithmetic, and a **publicly committed draw seed** so the lottery
can be audited by someone who does not trust us. Not on the on-sale path.

## What it recommends

Ship the **capped face-value exchange** as the default — legal essentially
everywhere, and it beat identity-hardening in every cell. Add affinity
rationing where a listening signal exists, but only with resale closed. Reserve
a lottery block so the casual fan is not sealed out, because affinity rationing
shuts them out completely and that is a choice, not a side effect.

And note what the strictest arm costs: the identity-bound design that shows zero
broker entry is restricted by law in New York and several other US states, and
turns away ~2% of holders at the gate — about 180 people per house who paid,
travelled and did not get in. [POLICY.md](docs/POLICY.md) carries both.

## What this cannot do

Unlike the other five projects here, Concerto has **no corpus**. No promoter
publishes on-sale logs. The demand side is invented, and no amount of care about
the solver changes that.

Two things are done about it. The single free parameter — what a usable
purchasing identity costs a broker — is fitted in the open by bisection against
resale markup, the one publicly observable quantity, with the sweep published.
And every claim is tested across 180 cells spanning the four assumptions that
could overturn it, and reported with the fraction that held.

The point estimates are one cell of that grid. **The orderings are what this
project is asking to be believed.**

## Reading it

| Document | |
|---|---|
| [RESULTS.md](docs/RESULTS.md) | every arm, every claim, generated from the JSON |
| [FINDINGS.md](docs/FINDINGS.md) | the hypothesis that was backwards, the metric that reads backwards, three solver bugs |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | the full stack, and why a waiting room is the wrong primitive |
| [CARDANO.md](docs/CARDANO.md) | CIP-68 ticket, the validator, the wallet-sale hole, eUTxO contention |
| [POLICY.md](docs/POLICY.md) | the customer-facing rules, their measured costs, and where they are illegal |

## Running it

```bash
make install     # uv venv + editable install
make all         # lint, test, simulate, sensitivity, ledger, report
```

```bash
concerto simulate                     # every arm at equilibrium
concerto demo queue affinity_bound    # twelve people, followed across two policies
concerto ledger                       # the enforcement ladder and eUTxO contention
concerto sensitivity                  # 180 cells, ~8 minutes
concerto calibrate                    # the one fitted parameter, in the open
```

Everything is CPU-only and regenerates from a fixed seed. The full grid is the
slow part at about eight minutes; everything else is seconds.

## Stack

Python 3.12, NumPy, Typer, `uv`, `ruff`, `mypy`, `pytest`. No corpus, no GPU, no
network. The Cardano work is a design and a measured model — the validator
sketch in [CARDANO.md](docs/CARDANO.md) is Aiken-shaped pseudocode, not deployed
code, and is labelled as such.
