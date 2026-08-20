# Gamut

**Catalog exposure and popularity-bias audit for the
[Cadence](../cadence) ecosystem.**

Cadence measures whether a playlist matches the request. [Segue](../segue)
measures whether the next track is the right one. [Timbre](../timbre) measured
whether audio can stand in for missing history. All three ask the same question
from the listener's side: *were you served well?*

None of them asks **which artists were served at all.**

Gamut is that measurement. It is not a fourth recommender — it is the axis the
other three are missing, and it runs over their output.

The name is the range: in medieval music theory the gamut was the complete span
of recognised notes. This asks how much of the catalog's range is ever actually
heard.

---

## What it found

Over 400 held-out title-only queries, counting the top 20 as what a listener sees:

| | |
|---|---|
| Exposure held by the top 1% of artists | **43.6%** (274 of 27,442) |
| Catalog ever shown | **2.99%** |
| Catalog reaching the candidate pool at all | 10.71% |
| Artist exposure Gini | **0.951** |
| Long-tail share of recommendations | 58.5% (**1.09× lift**) |

**Cadence is not popularity-biased in the usual sense.** 58.5% of what it shows
comes from the bottom half of the catalog by playlist count — the tail is
slightly *over*-served. The concentration problem is about **artists**, not hits.

**The most accurate channel is the most concentrated.** `tag_exact` has the best
R-precision (0.0598) and a long-tail share of 4.7% against 58.5% overall.
Accuracy in this system is bought with concentration — and now that trade has a
number on it.

**Two of seven channels do nothing here.** `collaborative` and `cooccurrence`
need seed tracks, and these are cold natural-language queries. For the case the
system exists for, five channels are carrying it.

### The finding that matters

Across **every** intervention tested — nine popularity-penalty strengths and four
artist caps — artist Gini moves only from **0.951 to 0.936**.

> You cannot re-rank your way out of a retrieval problem.

Concentration is already present in the candidate pool. Re-ordering a hundred
candidates cannot introduce the 89% of the catalog that never became a candidate.
That lines up with Timbre's result from the other direction: 76.6% of distinct
tracks are filtered out before Cadence's index is built. **The exposure ceiling is
set upstream of everything ranked.**

### What intervening costs

A popularity penalty of 0.3 moves long-tail share 58.5% → 64.4% for a 15%
R-precision cost. The full frontier for both knobs is in
[docs/RESULTS.md](docs/RESULTS.md); whether any point on it is worth paying for
is a product decision, which is the point of measuring instead of guessing.

## Running it

```bash
make install
make collect   # run Cadence over the battery, cache what it surfaced
make audit     # per-channel attribution + both intervention frontiers
make report    # render artifacts/results.html
```

```
src/gamut/
  collect.py    one retrieval pass, cached with per-channel ranks
  exposure.py   coverage, Gini, long-tail share, artist concentration
  rerank.py     popularity penalty and artist cap
  audit.py      baseline, per-channel attribution, both frontiers
```

## Honest limits

- Audits the **retrieval and ranking** stage. Cadence's final assembly applies
  MMR and its own two-per-artist cap, so the shipped playlist is more diverse
  than the top-20 measured here. Candidate reach is the binding constraint either way.
- Coverage percentages grow with query volume; the **concentration ratios** are
  the stable quantities.
- Popularity is proxied by playlist count in the MPD sample — a stand-in for
  exposure, not a measurement of streams or revenue.
- The interventions are deliberately simple. The contribution is the frontier and
  a harness to place any future method on it.
