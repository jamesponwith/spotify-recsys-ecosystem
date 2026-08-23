# Things that were wrong, and stayed on the record

Every project in this ecosystem keeps one of these, because the errors were the
most useful output. Concerto's are unusual in that most of them are errors of
*measurement design* rather than of code — this is a simulation, so a badly
chosen metric does not crash, it publishes.

## The first hypothesis was backwards

The model was built to demonstrate that hardening identity is theatre: make
accounts twelve times dearer, brokers buy fewer, but the survivors face less
competition and win more often, so capture barely moves.

That is not what happened. The identity-cost elasticity is `1/(gamma - 1)`, and
at the calibrated `gamma = 1.55` that is **1.8, not 1**. A 12x cost rise cuts
the sector's unconstrained best response by about 92x. The win-rate offset then
competes roughly four fifths of that back — which is a large, real effect, and
still leaves a 20x reduction.

So verified-fan schemes work. They are simply much weaker than their own
arithmetic implies, and they lose outright to attacking the margin: capping
resale beat hardening identity on broker capture in **180 of 180** sensitivity
cells. The corrected claim is in `broker.py`, and
`tests/test_broker.py::test_identity_hardening_is_mostly_absorbed_but_not_futile`
pins both halves so it cannot quietly revert.

## The obvious equity metric reads backwards

`access_gini` — the income inequality of the people who got in — is the metric
anyone would reach for, and it scores the **most exclusive** policy best.

Market clearing shuts out the bottom income quartile completely and posts an
access Gini of 0.24, against 0.45 for the arm that admits everyone at face. An
allocation that admits only rich people admits a set of people who are all
equally rich, so dispersion within the admitted set is low. Dispersion among the
admitted is not the same question as who was admitted.

This is the same shape of error as [Gamut's](../../gamut) flat exposure frontier
— a statistic computed over the wrong set, invariant to the thing being
measured. It is kept in the report, deliberately not featured, with
`income_ratio` next to it as the metric it is usually mistaken for.

## Fan fill is 100% under every policy

The headline number the industry quotes — "tickets that reached real fans" —
is above 99% for every arm here, including the one where a broker resold 61% of
the house at three times face. The fan is a fan; the ticket reached them.

It is useless on its own and it is the reason this report leads with what fans
*paid* and *which* fans got in.
`tests/test_market.py::test_fan_fill_is_useless_on_its_own` asserts it.

## Restricting brokers raises the resale price

Counterintuitive and correct: fewer broker-held tickets means the ones that
exist are sold into the very top of the residual demand curve. Going from the
open queue to the capped exchange takes broker capture from 61% to 1.8% and
takes the resale markup from 3.0x to 9.9x.

Both are true and only one is the story. Total fan outlay falls sharply, because
almost everybody is now buying at face instead of a few people buying very high.
An honest report has to carry the markup *and* the volume, or a restrictive
policy looks like it made resale worse.

The degenerate case is worse still: as broker inventory approaches zero, the
clearing price of that inventory runs off to the top of the demand curve, and
the bound arm posts a **22x markup on about six seats**. Markup is now
suppressed below 0.5% resale volume, which is a reporting decision made because
the first draft of the report credited the strictest arm with the most
spectacular scalping in the table.

## Three solver bugs that produced plausible output

- **`price_for(0)` returned infinity.** Economically the price at which zero
  tickets clear is the top of the demand curve, not infinity. Returning infinity
  handed a broker sector holding no inventory an unbounded expected margin, and
  every restrictive arm diverged on the first iteration.
- **`price_for(total_demand)` returned zero.** The "demand ran out" guard fired
  at *exactly* full clearing, reporting a sold-out house as worthless. Off by
  one boundary.
- **The weighted Gini dropped the first Lorenz segment.** A perfectly equal
  distribution scored 0.111 rather than 0. Small enough to look like a result.
  Caught only because a test asserted the trivial property, which is what those
  tests are for.

## Damped iteration hid its own failure

The equilibrium was originally found by damped fixed-point iteration on the
resale price. It failed to converge in roughly one sensitivity cell in eight —
worst in the stiff region where identities are expensive and a small price move
swings broker entry hard — and it failed *silently*, returning whatever the last
iterate happened to be alongside a `converged: false` nobody was reading.

The fixed point is the root of a monotone scalar function, so it is now found by
bisection, which cannot fail and is faster. The two methods agree everywhere the
old one converged, which is the only reason to trust either.

## The recommendation has a named failure region

Affinity rationing beat every price-rationed arm on superfan access in **177 of
180** cells. It is reported as 177 of 180, not as a result with a footnote.

All three failures sit in the same corner: demand at 15x, the least convex
identity cost curve, and off-platform leakage at 0.65 or above. They are
near-ties — at a quarter of the trials the same region produced 10 failures
rather than 3, which is what a boundary looks like when the two arms are
close, and is the reason the region is reported as a box rather than a count.

The mechanism is clean. Forged listening histories are assumed to clear whatever
cut is set, so under affinity rationing brokers are served *first*. With resale
open and a show hot enough, affinity rationing is a priority lane for brokers.

Pair it with a closed resale channel and the claim holds in **180 of 180**.
Which is why the recommendation is the pair, and why the pair is tested as a
separate claim rather than assumed.

## A zero that is exactly zero

Market clearing serves **0.0%** of bottom-income-quartile demand, and a round
zero in a simulation is usually a bug. This one is arithmetic.

The clearing price is $229. Willingness to pay is built as
`face x income_factor x (0.5 + affinity)`, so the ceiling for anyone in the
bottom income quartile — the richest of them, at maximum affinity — is $78. Not
one person in that quartile can reach the price, at any level of fandom. The
number is not rounded down from something small; there is nobody there.

It is worth stating because it also names the limit of the finding. That gap is
a property of the assumed income spread (lognormal, sigma 0.85) and the assumed
affinity range. Narrow either and the zero becomes a small number. What does not
depend on the parameterisation is the direction, which held in 180 of 180 cells.

## Most of what scalping costs is not scalper income

The queue transfers $19 per ticket to broker profit, and burns **$35** per
ticket on identity infrastructure — proxies, aged accounts, payment
instruments. At equilibrium the sector's revenue is exactly `gamma` times its
identity spend, so at the calibrated gamma = 1.55 brokers set about 64% of what
they gross on fire.

That money leaves the fan and reaches nobody who made the music. The first
version of the report folded it into an "other" bucket alongside platform fees,
which understated the waste by roughly four times and made the queue look far
less destructive than it is. It is now its own segment in the money chart.

## What this project cannot do

Unlike the other five applications here, Concerto has no corpus. No promoter
publishes on-sale logs; the resale platforms publish what flatters them. The
demand side is invented.

Two things are done about that and neither is a substitute for data: the single
free parameter is fitted in the open, by bisection, against the one publicly
observable quantity (resale markup), with the sweep published — and every claim
is tested across 180 cells spanning the four assumptions that could overturn it,
and reported with the fraction that held.

A reader who disagrees with the demand model should disagree with the numbers.
The orderings are what this project is asking to be believed.
