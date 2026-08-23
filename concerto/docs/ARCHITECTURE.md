# Architecture

What you would actually build, and the order the problems come at you in.

## The part that is not a technical problem

Ticketmaster's position does not rest on its software. It rests on exclusive
multi-year contracts with the buildings, signed against upfront advances and
rebates the venue keeps, and on Live Nation owning promoters, venues and the
artists' touring business at the same time. A venue on an exclusivity deal
cannot sell you inventory even if it prefers your product. That is the substance
of the US Department of Justice's May 2024 complaint, and it is why a better
on-sale system is necessary and nowhere near sufficient.

Nothing below changes that. What it does is make the thing worth switching to
once a door opens — a single-venue independent promoter, a festival, an artist
with contract leverage, a market where exclusivity is being unwound.

So the sequencing is deliberate: this repository builds and measures **the
allocation mechanism** first, because it is the only part where the answer was
not already known, and the measurement changed what the product should be.

## What the measurement changed

Three results from [RESULTS.md](RESULTS.md) that the design has to respect:

1. **A waiting room is the wrong primitive.** A queue rewards the party with the
   most simultaneous identities. Broker entry is an *investment* decision, so
   hardening identity raises the cost of capture without touching the reward,
   and roughly four fifths of the gain is competed back by the survivors winning
   more often. Do not build a virtual waiting room. Build a registration window
   and a deterministic draw.
2. **Attack the margin, not the bot.** Capping resale beat hardening identity on
   broker capture in every one of the 180 sensitivity cells. Anti-bot work is
   perimeter defence against an adversary who is buying capacity at market rate.
3. **Dynamic pricing does not help the fan.** Clearing the house at market price
   removes the broker completely and leaves the average fan paying *more* than
   the queue it replaced, with the bottom income quartile shut out entirely. It
   is a revenue policy for the artist, and it should be sold as one.

## Domain model

```
Event ──< PriceTier ──< InventoryBlock ──< Seat
  │                                        │
  └──< Drop ──< Registration ──< Allocation ┘
                     │              │
                  Identity       Ticket ──< Credential (rotating)
                     │              │
                  Person        Transfer / Return
```

The unusual pieces:

- **`InventoryBlock`** sits between tier and seat so holds are taken against a
  block, not a seat. A drop that assigns specific seats at request time
  serialises on the hottest rows.
- **`Registration`** is separated from **`Allocation`** in time. This is the
  whole anti-scalping design: nothing is first-come, so speed buys nothing.
- **`Credential`** is separate from `Ticket`. The ticket is an entitlement; the
  credential is a short-lived rotating proof of possession. Cloning a screenshot
  gets you a credential that expired.
- **`Transfer`** and **`Return`** are distinct. Return-at-face is the mechanism
  that makes a non-transferable ticket humane; without it the policy is just
  "you lose your money if you get sick."

## The on-sale path

```
 register        verify           draw            offer          settle
   │               │               │                │              │
   ▼               ▼               ▼                ▼              ▼
 window opens   identity +      seeded, auditable  time-boxed    payment +
 for 48h        affinity        allocation over    purchase      credential
                scoring         all registrants    window        issued
```

**Register (48h window).** No advantage to being early. This alone removes the
entire bot-speed industry, at the cost of an on-sale that is no longer an event
— which some promoters actively do not want, because the stampede is marketing.

**Verify.** Identity checks are graded rather than binary, because the
simulation prices the false rejects: at a 4% false-reject rate, a 40,000-person
registration turns away 1,600 real fans. Tiering means a soft signal costs a
review, not a rejection, and every rejection is appealable inside the window.

**Draw.** A single batch job over the full registration set, seeded from a
value published *before* registration closed (a block hash works well and is
the one genuinely good use of a public chain here). Deterministic, replayable,
and auditable by a third party. This is the step that must never be a race.

**Offer.** Winners get a time-boxed purchase window, not a countdown. Offers
that lapse cascade to the next-ranked registrants.

**Settle.** Payment, then credential issuance.

## Where the load actually is

| Phase | Shape | Hard part |
|---|---|---|
| Registration | 40k writes over 48h | trivial |
| Draw | one batch over ~40k rows | must be reproducible, not fast |
| Offer/settle | 4.2k checkouts over hours | payment idempotency |
| Gate | 9.2k scans in 90 minutes, patchy signal | offline validation |

Reading that table is the point. **The registration model moves the load off the
critical millisecond entirely.** A first-come on-sale has to survive 40,000
people arriving in the same second; this one never has more than a few hundred
concurrent checkouts. Most of the engineering folklore about ticketing —
waiting-room services, queue tokens, aggressive edge caching of a countdown
page — exists to solve a problem this design does not create.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Inventory + ledger | PostgreSQL | Holds and allocations are financial records. `SELECT ... FOR UPDATE SKIP LOCKED` over an inventory block is the whole concurrency story once the race is gone. |
| Allocation service | Go | One batch job, heavy on determinism, light on cleverness. Boring is the requirement. |
| Modelling / policy | Python | This repository. Policy parameters are simulated before they ship. |
| Affinity scoring | Python + offline batch | Listening history is a slow-moving feature. Nothing about it belongs on the request path. |
| Event log | NATS JetStream | Every state change is an event: registrations, draws, offers, returns, scans. Audit is a replay. |
| API | FastAPI or Go, REST + webhooks | Promoters integrate with webhooks and CSV, not GraphQL. |
| Front end | Next.js, server-rendered | The pages that matter are readable under bad signal in a stadium car park. |
| Payments | Stripe primary, Cardano optional | See [CARDANO.md](CARDANO.md) on why the chain is a settlement layer here and not an on-sale path. |
| Gate | Native app, offline-first | See below. |

## The gate is the load-bearing component

The single result from the Cardano work is that a validator cannot stop a broker
selling the wallet, and the only rung of enforcement that closed the spread was
an identity check at the door. That makes the gate — the least glamorous part of
a ticketing platform — the thing the whole anti-scalping design rests on.

It has to work with no connectivity, against a device someone is holding in
freezing rain, in under two seconds per person, at 100+ scans a minute per lane.

- **Rotating credential.** A short-lived code derived from a per-ticket secret
  and the current time interval, verified offline from a pre-synced key set.
  A screenshot is worthless because the code has already rotated.
- **Offline-first.** Scanners hold the event's key set and an allow list, sync
  opportunistically, and reconcile double-scans afterwards rather than blocking
  on the network. A gate that fails closed on a dead uplink is a riot.
- **Identity binding is graded.** Name-matched ID on the lead ticket only, not
  every seat in the party, or a family of four becomes four failure points.

The 2% gate-denial rate the simulation charges the bound arms is not a
rounding error. On a 9,200-seat house that is ~180 people who paid, travelled
and did not get in. Every one of them is a support case and some of them are a
news story. Budget staff for it, publish the appeal path, and refund
immediately and without argument.

## Beyond concerts

Concerts are the right first niche because they are the worst case: fixed
capacity, a single night, and demand that cannot be met at any price. Everything
adjacent is easier, and in instructive ways.

**The strongest intervention is not in the arm set.** Demand multiple is the
single most powerful lever in the whole model — resale markup runs 1.9x at 3x
oversubscription and 6.0x at 30x — and the way to move it is to *add dates*.
No allocation policy in this repository comes close to what a second night does.
It is left out of the arm set because it is a promoter's routing decision rather
than a platform's mechanism, but any honest account has to say it first: a tour
that plays one night in a city has chosen the scalping.

| Vertical | What changes | Which findings transfer |
|---|---|---|
| **Theatre, residencies** | Long runs mean supply is elastic. The oversubscription multiple that drives everything here is structurally low. | Margin-over-bot holds; the scalping problem is much smaller and mostly does not need solving. |
| **Sport** | Season tickets are the base and the marginal single-game seat is a thin slice. Demand is known months ahead and moves with form. | Dynamic pricing's equity cost is muted — the season-ticket base is already served at a fixed price, so clearing the marginal seat is not clearing the house. This is why it is defensible in sport and not in touring. |
| **Festivals** | Multi-day, camping, tiered, and identity-bound tickets are already the cultural norm — Glastonbury has run photo-matched tickets for years. | The bound arm is both lawful and *expected* here, which makes festivals the right place to prove it before touring. |
| **Sport / festivals, affinity** | Attendance history is a stronger and harder-to-forge affinity signal than listening history. | The affinity arm gets better, and its forgery-cost assumption gets more defensible. |

The order to enter, on this reading, is **festivals → independent single-venue
promoters → touring**, which is also the order of decreasing venue-exclusivity
lock-in. That is not a coincidence.

## What this repository does not build

No payment integration, no seat map, no promoter console, no gate app. This is
the policy engine and its evidence — the thing that decides *what the product's
rules should be* — and it is deliberately the part that was built first, because
it is the part where the intuitive answer turned out to be wrong twice.

See [POLICY.md](POLICY.md) for the customer-facing rules this implies, including
the ones that are illegal in some US states.
