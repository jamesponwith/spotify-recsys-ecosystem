# Customer policy

The rules a fan actually experiences, what each one costs, and the places where
the design the simulation recommends is restricted by law.

> Not legal advice. Statutes below are named so the constraints are concrete and
> checkable; jurisdictions differ and this area is moving quickly. Verify with
> counsel before shipping any of it.

## The principle

Every anti-scalping measure has a customer cost, and the industry practice is to
publish the benefit and not the cost. This project counts both. The `Harm`
column in [RESULTS.md](RESULTS.md) sums three real harms that policies here
create:

| Harm | Cause | Scale on a 9,200-seat house |
|---|---|---|
| Parties split | per-identity purchase cap below party size | a cap of 2 turns away a third of a family's demand |
| Real fans rejected | identity verification false positives | 4% of 40,000 registrants is 1,600 people |
| Turned away at the gate | ID mismatch on a bound ticket | 2% of holders is ~180 people who paid and travelled |

None of those numbers is published by any platform that ships the corresponding
measure. That is the gap this document exists to close.

## Pricing

**All-in from the first screen.** The price shown in search, on the seat map and
at checkout is the price charged. No fee reveal at the last step.

This is now largely required rather than optional in the US — the FTC's rule on
unfair or deceptive fees, finalised in December 2024 and effective in 2025,
covers live-event ticketing specifically and requires total price up front. Do
it because it is right; note that it is also no longer a differentiator.

**Dynamic pricing is disclosed as what it is.** The simulation is unambiguous:
clearing the house at market price removes the broker entirely, raises what the
average fan pays *above* what the queue charged, and shuts out the bottom income
quartile. It is a revenue policy for the artist and it should be labelled one.
If a tour uses it, the on-sale page should say which portion of the house is
priced dynamically, before registration, not after.

**Face-value returns, always.** Any ticket can be handed back for a full refund
of face and fees at any time before the event, and is re-offered to the queue at
face. This is not generosity — it is the mechanism that makes a restrictive
transfer policy survivable, and it is a large part of why the bound arms show no
broker entry at all: it bounds the broker's upside without stranding a fan who
gets sick.

## The drop

**Registration window, not a race.** A 48-hour registration window, then a
seeded draw. Nothing is first-come. Being fast, being awake, having a better
connection and running fifty browser tabs are all worth exactly nothing.

**The seed is published in advance.** Committed before registration closes and
verifiable afterwards, so the draw can be checked by someone who does not trust
us. This is the one part of the system where a public chain earns its place.

**Purchase caps are set at party size, not lower.** Caps below 4 do more damage
to families than to brokers, whose constraint is identity cost rather than
per-identity limit. The cap is a blunt instrument and the simulation prices its
bluntness.

**Every rejection is appealable inside the window.** A 4% false-reject rate on a
verification step is thousands of real people per major on-sale. If there is no
staffed appeal path that resolves before the draw, the verification step should
not ship.

## Transfer, and where it becomes illegal

The strongest arm in the simulation binds the ticket to an identity and offers
no transfer at all — only return at face. It is also the arm most likely to be
unlawful where you want to sell.

- **New York** (Arts and Cultural Affairs Law, Article 25) constrains an
  issuer's ability to make tickets non-transferable and requires a transferable
  purchase option. A pure bound-ticket product is not straightforwardly
  shippable there.
- **Several other US states** — Colorado, Illinois, Utah and Virginia among
  them — have transferability protections of varying strength.
- **Federally**, the BOTS Act of 2016 makes it unlawful to circumvent access
  controls to acquire tickets in excess of posted limits. It criminalises the
  bot, which is precisely the lever the measurement says is the weaker one.
- **Ireland's** Sale of Tickets Act 2021 bans above-face resale for designated
  events. **The UK** requires face value and restrictions to be disclosed on
  resale listings under the Consumer Rights Act 2015, with resale price caps
  under active consultation.

The practical consequence: **ship the capped face-value exchange as the default,
not the bound ticket.** It is legal essentially everywhere, it attacks the
margin rather than the bot, and it beat identity-hardening on broker capture in
all 180 sensitivity cells. Reserve bound tickets for jurisdictions and events
where they are lawful, and never as the only option on sale.

## Affinity, and its limits

The arm that protects deep fans rations on verified listening history rather
than on money. It is the only thing in the set that moves that axis, and it
carries three costs that have to be stated.

**It shuts out the casual fan completely.** Under a lottery, someone who
discovered the artist last month has a small chance. Under affinity rationing
they have none. That is not a bug in the implementation; it is what rationing on
merit means, and whether it is right depends on what a concert is for. Reserve a
lottery block — 20% of the house drawn among everyone — so the door is not
sealed.

**It is only safe with resale closed.** This is the sharpest boundary the
sensitivity grid found. With resale left open, affinity rationing beat every
price-rationed arm on superfan access in 177 of 180 cells — and **all three
failures sit in one corner: hot demand paired with a wide-open off-platform
channel.** With resale closed as well, it holds in 180 of 180.

The mechanism is why this matters operationally. The model assumes brokers can
present whatever listening history clears the cut, so under affinity rationing
they are served *first*. A forged history is a **priority lane**. Affinity
rationing shipped without a closed resale channel is worse than no affinity at
all, and it will look fine right up until a show is hot enough.

**It makes a fandom score into a gate.** Once listening history decides who gets
into a room, it stops being telemetry and becomes a record with consequences.
That implies, at minimum: explicit opt-in separate from any listening product's
terms; a visible score with the inputs that produced it; a human review path,
since an automated decision determining access is close enough to the line drawn
by GDPR Article 22 that it should not be tested; retention limits; and a
standing refusal to sell or share it. The failure mode is a fandom credit score,
and it arrives by drift, not by decision.

## Accessibility

**Accessible seating goes on sale at the same time, in the same drop, through
the same path.** In the US this is a requirement — the ADA regulations at 28 CFR
36.302(f) — and it is routinely met in the letter and not the spirit, with
accessible inventory behind a phone line that opens later.

**ID requirements exclude people.** Survey estimates of the share of US adults
without current government-issued photo ID range from the low-to-high single
digits up to around one in ten depending on method and vintage, and every
estimate finds it concentrated by income, age and race — exactly the population
the low-income metrics in this simulation track. A bound-ticket policy that accepts only a
driver's licence is an income filter with extra steps. Accept a wide document
set, allow a named companion to present for the party, and staff the exception
path.

**Gifting is a legitimate use.** Most transfers are not resales; they are a
parent buying for a child and a friend covering a mate. A transfer policy that
cannot express "this is a gift" will generate more angry customers than blocked
brokers. Free transfers to a verified contact, rate-limited, cost the model
nothing and cost a broker everything — a broker needs liquidity, not three
friends.

## What we publish

Committing to publish is what keeps the rest honest, since every number below is
one this project already computes:

- tickets sold at face, per on-sale
- verification rejections, appeals, and appeal outcomes
- gate denials and their resolutions
- returns handled and re-offered at face
- what share of the house was held back — artist, promoter, sponsor, platinum —
  before the public on-sale, which is the single number the industry most
  reliably does not disclose and the one that decides everything downstream

