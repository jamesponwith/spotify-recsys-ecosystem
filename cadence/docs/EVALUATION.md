# Evaluation

## Protocol

The task is reconstructed from the **RecSys Challenge 2018** setup on the
Million Playlist Dataset, so the numbers are comparable to published work
rather than being a bespoke score only this repo understands.

2 000 playlists are drawn once with a fixed seed and frozen in
`data/processed/splits.json`. They are excluded from **every** training
matrix — the collaborative factorisation, the rebuilt folksonomy, the served
co-occurrence matrix, and the reranker's sampling pool. Without that, the model
is scored on playlists it has already read.

For each evaluation playlist the system sees the **title** plus the first `k`
tracks and must predict the rest:

| `k` | What it tests |
|---|---|
| **0** | **Pure natural-language cold start.** Title only. Collaborative filtering has nothing to work with, so the folksonomy has to carry the query. This is the case the project exists for. |
| 1, 5 | Sparse seeds — the realistic "I've added a couple of songs" state |
| 10, 25 | Conventional playlist continuation |

Retrieval depth is 500. Every cell is 400 playlists.

## Metrics

**R-precision** — overlap of the top-|G| predictions with the withheld tracks,
where |G| is the number withheld. The official MPD variant also awards 0.25 for
matching the *artist* of a withheld track, on the reasoning that a different
song by the right artist is a near miss rather than a total miss. Artist credit
is consumed once per withheld artist, so ten songs by one withheld artist
cannot farm the score. Both variants are reported.

**NDCG@100** — position-discounted gain; rewards putting hits near the top.

**Clicks** (lower is better) — how many "show 10 more" presses a listener would
need before hitting a withheld track: `floor(rank_first_hit / 10)`, capped at
51. The most product-shaped of the three: it approximates *time to first good
song*.

**Recall@500** — how much of the withheld set is recoverable at all, which
bounds what any reranker could achieve.

### Beyond accuracy

Accuracy alone rewards recommending the head of the popularity distribution, so
these are reported alongside and never separately:

- **Catalog coverage@100** — fraction of the catalog that appears in any result
- **Gini@100** — inequality of exposure (0 = uniform, 1 = winner-takes-all)
- **Long-tail share** — fraction of recommendations below the 80th popularity percentile
- **Intra-list distance** — mean pairwise cosine distance inside one playlist

## Baselines

| Baseline | What it establishes |
|---|---|
| **popularity** | The floor. MPD is heavily popularity-skewed, so this is a genuinely strong baseline and the one that catches a system doing nothing useful. |
| **item-kNN** | Classic neighbourhood CF over the raw interaction matrix. The learned collaborative channel is a compressed version of this, so it is the comparison that says whether the embedding earned its keep. Falls back to popularity at k=0, where it has no seeds — the two are identical there by construction. |
| **lexical-title** | Matches the playlist title against track/artist/album/genre strings only. Isolates how much of the task is plain string matching, which is the fair floor for any natural-language claim. |

## Ablations

Each retrieval channel is removed (`− channel`) and isolated (`only channel`),
holding everything else fixed. The `only_tag` row at k=0 is the direct test of
the folksonomy thesis: with no seeds and no metadata match, it is the only
channel with anything to say.

## Reproducing

```bash
make all                                  # build → splits → train → reranker → eval
python scripts/render_results.py          # regenerate the tables below
```

Results are written to `artifacts/eval_report.json`. Every cell carries a
standard error (`*_se`) so a difference between two runs can be read as
meaningful or not rather than eyeballed.


## Constraint satisfaction (the assembly stage)

The ranking harness above stops at retrieval — it never calls selection or
sequencing. That leaves the stages enforcing the listener's actual
*requirements* unmeasured, and "the playlist respects what you asked for" is at
least as much of a promise as "the playlist is relevant". A separate battery of
20 constrained requests runs end to end (`cadence eval-constraints`):

| Requirement | Satisfaction | Queries testing it |
|---|---|---|
| Exact track count | **100 %** | 12 |
| Per-artist cap | **100 %** | 20 |
| Duration target (±10 %) | **100 %** | 4 |
| BPM window | **100 %** | 7 |
| No known-explicit track | **100 %** | 5 |

Zero failures. This is what [ADR-004](DECISIONS.md) buys: constraints are
filters and caps, so satisfaction is a property of the code path rather than
something to hope for.

**Mood adherence** — absolute gap between the delivered playlist's mean audio
feature and the value the request implied:

| Dimension | Mean error |
|---|---|
| speechiness | 0.012 |
| valence | 0.021 |
| danceability | 0.036 |
| acousticness | 0.042 |
| energy | 0.043 |
| **instrumentalness** | **0.130** |

Five of six dimensions land within 0.05 on a 0–1 scale. Instrumentalness is the
outlier and the reason is in the data, not the model: MPD playlists are
overwhelmingly vocal music, so a request for "instrumental focus music" is asking
for a region of the catalog that barely exists here.

Getting adherence to this level required three fixes that only surfaced by
looking at real output rather than at metrics:

1. **A curve conflict.** "chill rainy morning" matched both `chill` (energy
   0.32) and `morning` (curve `build`), and the implied curve won — producing a
   playlist that ramped *up* into high-energy tracks. An implied curve now has
   to agree with the energy target or it is dropped.
2. **Min-max normalisation hiding the audio term.** The reranker emits
   probabilities with median ~0.002 and max ~0.7. Min-max normalising that
   leaves nearly every candidate at ~0 and a handful near 1, so blending audio
   affinity against it did nothing at any weight. Rank-normalising relevance
   before the MMR trade fixed it: mean playlist energy for a "chill acoustic"
   request moved from 0.63 to 0.43 against a 0.35 target, and acousticness from
   0.23 to 0.62 against 0.67.

3. **Constraint words leaking into the query.** "nothing explicit" correctly set
   the clean filter — and then `nothing` and `explicit`, both present in the tag
   vocabulary, *also* became query themes and pulled the tag centroid toward
   unrelated music. Words consumed by constraint parsing are now excluded from
   theme extraction. This alone cut mean instrumentalness error from 0.221 to
   0.130 and energy error from 0.056 to 0.043.

None was visible in R-precision. All three were obvious in printed output.

**Diversity is the weak spot.** Mean long-tail share across the battery is
**0.057** — the system draws heavily from the popular head. Intra-list distance
is high (0.909), so playlists are varied *within* themselves, but they are not
digging into the catalog. MMR and artist caps are working on the wrong axis for
this: they diversify the selection, they do not push it deeper. Fixing it would
mean an explicit popularity-discount term, which would cost accuracy — a trade
worth measuring rather than assuming.

## Threats to validity

**Offline metrics are a proxy.** A track a listener would have loved but never
added counts as a miss. Every number here understates true quality by an
unknown amount, and the ranking between systems is more trustworthy than any
absolute value.

**One corpus, one period.** MPD is US-centric, English-language and ends in
2017. Nothing here demonstrates generalisation past that.

**Title-conditioned evaluation favours the folksonomy channel.** The k=0 task
uses playlist titles as queries, and the tag space is built from playlist
titles. The construction is not circular — evaluation playlists are excluded
from the tag matrix, so the tags come from *other people's* playlists — but the
query distribution is closer to the training vocabulary than free-text user
queries would be. Real queries are longer, more compositional and messier. The
honest reading is that k=0 measures *title-like* natural language, and the LLM
planner exists to close the gap between that and how people actually type.

**No multi-constraint query set.** Playlist titles carry one or two themes;
real requests stack an activity, a mood, a genre and an era. The tag channel
embeds a query as the centroid of its theme vectors, so a five-theme request
dilutes each one — visible in output, invisible to this evaluation. Building a
multi-constraint query set is the prerequisite for fixing it, not the other way
round.

**Single split.** One frozen 2 000-playlist sample, not k-fold. Standard errors
quantify sampling noise within the split, not variance across splits.

**The two harnesses measure different things and neither covers everything.**
Ranking metrics say nothing about whether the delivered playlist honours the
request; the constraint battery says nothing about whether the tracks are any
good. Sequencing quality — whether the energy arc and harmonic transitions
actually sound intentional — is not measured at all here. That needs listening
tests, which is the honest gap in this evaluation.

---


## What the numbers say

**Cold start works.** At k=0 — title only, no seed tracks — the full system
reaches R-precision **0.1422** against **0.0404** for popularity: a **3.5×**
improvement, and 7× over lexical title matching. Time-to-first-good-song
(clicks) drops from 13.0 pages to 3.4.

**The folksonomy is what makes it work, and the ablation proves it.** At k=0:

- removing the tag channels collapses R-precision from 0.0698 to **0.0191** (−73 %)
- the tag channels *alone* score 0.0659 — still 63 % above popularity
- both collaborative channels alone score exactly **0.0000**

That last row is the cleanest result in the study. With no seed tracks,
collaborative filtering has literally nothing to condition on and returns
nothing useful. Every bit of cold-start performance comes from the bridge
between playlist titles and tracks. This is also what the reranker independently
concluded: `tag_hits` is its highest-importance feature by a factor of two.

**The reranker is the single biggest win.** It lifts R-precision by 40–104 %
across every seed count (k=0: 0.0698 → 0.1422; k=25: 0.1116 → 0.1916) without
changing the candidate set — recall@500 is identical by construction, so the
entire gain is better ordering of the same 500 candidates. Validation AUC 0.849
versus 0.639 for the fusion score alone.

**Both collaborative channels earn their place.** Removing exact co-occurrence
costs 25 % of R-precision at k=5; removing the learned embedding costs 11 %.
Neither dominates: the embedding is stronger at k=25 (0.1065 alone vs 0.0807),
the exact counts are stronger at k=1 (0.1215 vs 0.0686). They fail differently,
which is exactly why fusing them beats either — and why the first version of
this system, which shipped only the embedding, lost to a plain item-kNN
baseline (see [ADR-003](DECISIONS.md)).

**The audio channel contributes nothing to ranking accuracy.** `− audio` is
within noise of the full system at every k. This is worth stating plainly
rather than burying: audio features are not in this system to improve
relevance. They are there to satisfy hard constraints ("120–140 BPM"), to steer
mood targeting, and to drive sequencing — none of which these metrics measure.
Judging that channel by R-precision would be judging it by the wrong test.

**R-precision peaks at k=5 and declines by k=25.** This is a property of the
metric, not a regression: as more of a playlist is revealed, fewer tracks remain
withheld, so |G| shrinks and the top-|G| cutoff gets stricter. Clicks and
recall@500 tell the more stable story.

**The reranker buys accuracy with concentration.** At k=0 it nearly doubles
R-precision but cuts catalog coverage@100 from 0.105 to 0.060 and raises Gini
from 0.50 to 0.68. That is a real trade, not a free win: trained on "what did
humans actually add", it learns that popular tracks are usually the safe answer.
Whether that trade is correct is a product decision, not a modelling one — and
it is visible here rather than hidden, which is the point of reporting
beyond-accuracy metrics in the same table.

---

### Main results by seed count

`k` = number of seed tracks revealed. **k=0 is the pure natural-language cold-start case**: title only, nothing for collaborative filtering to use.


**k = 0 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.1422 | 0.1647 | 0.1952 | 3.37 | 0.2975 |
| Cadence (fusion only) | 0.0698 | 0.0873 | 0.0913 | 8.05 | 0.2975 |
| item-kNN baseline | 0.0404 | 0.0558 | 0.0545 | 13.02 | 0.1552 |
| popularity baseline | 0.0404 | 0.0558 | 0.0545 | 13.02 | 0.1552 |
| lexical-title baseline | 0.0197 | 0.0278 | 0.0266 | 26.19 | 0.0672 |

**k = 1 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.1792 | 0.2037 | 0.2407 | 1.70 | 0.3575 |
| Cadence (fusion only) | 0.1273 | 0.1543 | 0.1761 | 2.76 | 0.3575 |
| item-kNN baseline | 0.1226 | 0.1501 | 0.1702 | 4.04 | 0.3408 |
| popularity baseline | 0.0389 | 0.0543 | 0.0528 | 14.03 | 0.1508 |
| lexical-title baseline | 0.0183 | 0.0257 | 0.0245 | 33.79 | 0.0497 |

**k = 5 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.2483 | 0.2804 | 0.3352 | 0.57 | 0.4492 |
| Cadence (fusion only) | 0.1782 | 0.2137 | 0.2531 | 0.77 | 0.4492 |
| item-kNN baseline | 0.1621 | 0.1970 | 0.2254 | 1.33 | 0.4514 |
| popularity baseline | 0.0339 | 0.0483 | 0.0470 | 16.98 | 0.1368 |
| lexical-title baseline | 0.0174 | 0.0246 | 0.0234 | 34.56 | 0.0477 |

**k = 10 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.2470 | 0.2790 | 0.3382 | 0.58 | 0.4571 |
| Cadence (fusion only) | 0.1752 | 0.2105 | 0.2502 | 0.91 | 0.4571 |
| item-kNN baseline | 0.1448 | 0.1795 | 0.2054 | 2.40 | 0.4419 |
| popularity baseline | 0.0277 | 0.0412 | 0.0389 | 20.33 | 0.1203 |
| lexical-title baseline | 0.0163 | 0.0233 | 0.0225 | 35.13 | 0.0467 |

**k = 25 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.1916 | 0.2190 | 0.2736 | 3.18 | 0.4130 |
| Cadence (fusion only) | 0.1116 | 0.1414 | 0.1731 | 4.97 | 0.4130 |
| item-kNN baseline | 0.0812 | 0.1099 | 0.1172 | 10.38 | 0.3373 |
| popularity baseline | 0.0142 | 0.0245 | 0.0197 | 30.91 | 0.0703 |
| lexical-title baseline | 0.0103 | 0.0164 | 0.0171 | 37.77 | 0.0405 |


### Channel ablations

R-precision; each row removes or isolates one retrieval channel.

| Configuration | k=0 | k=1 | k=5 | k=10 | k=25 |
|---|---|---|---|---|---|
| Cadence (fusion only) | 0.0698 | 0.1273 | 0.1782 | 0.1752 | 0.1116 |
| − exact co-occurrence | 0.0698 | 0.0956 | 0.1327 | 0.1362 | 0.1011 |
| − collaborative embedding | 0.0698 | 0.1356 | 0.1586 | 0.1421 | 0.0815 |
| − folksonomy tags | 0.0191 | 0.1094 | 0.1700 | 0.1693 | 0.1122 |
| − lexical | 0.0715 | 0.1267 | 0.1775 | 0.1741 | 0.1108 |
| − audio | 0.0705 | 0.1273 | 0.1783 | 0.1753 | 0.1116 |
| only co-occurrence | 0.0000 | 0.1215 | 0.1607 | 0.1430 | 0.0807 |
| only collaborative | 0.0000 | 0.0686 | 0.1235 | 0.1337 | 0.1065 |
| only folksonomy tags | 0.0659 | 0.0641 | 0.0578 | 0.0512 | 0.0320 |


### Beyond-accuracy and latency

| System | k | Coverage@100 | Gini@100 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0 | 0.0603 | 0.684 | 144 | 183 |
| Cadence (fusion only) | 0 | 0.1045 | 0.502 | 63 | 155 |
| popularity baseline | 0 | 0.0006 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 1 | 0.0941 | 0.529 | 156 | 205 |
| Cadence (fusion only) | 1 | 0.0874 | 0.551 | 99 | 161 |
| popularity baseline | 1 | 0.0006 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 5 | 0.0801 | 0.580 | 165 | 193 |
| Cadence (fusion only) | 5 | 0.0718 | 0.611 | 89 | 135 |
| popularity baseline | 5 | 0.0007 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 10 | 0.0762 | 0.599 | 166 | 214 |
| Cadence (fusion only) | 10 | 0.0691 | 0.628 | 93 | 145 |
| popularity baseline | 10 | 0.0007 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 25 | 0.0756 | 0.607 | 183 | 240 |
| Cadence (fusion only) | 25 | 0.0697 | 0.635 | 104 | 166 |
| popularity baseline | 25 | 0.0007 | 0.000 | 0 | 0 |


Evaluated on 400 held-out playlists per cell, retrieval depth 500, catalog 159,338 tracks.

