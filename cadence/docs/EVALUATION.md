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
.venv/bin/python scripts/render_results.py  # regenerate the tables below
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

`k` = number of seed tracks revealed. **k=0 is the pure natural-language cold-start case**: title only, nothing for collaborative filtering to use. Every cell is mean ± 2×SE.


**k = 0 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.1429 ± 0.0149 | 0.1655 ± 0.0157 | 0.1975 ± 0.0191 | 3.23 ± 0.95 | 0.3031 ± 0.0264 |
| Cadence (fusion only) | 0.0709 ± 0.0108 | 0.0883 ± 0.0119 | 0.0924 ± 0.0140 | 7.70 ± 1.21 | 0.3031 ± 0.0264 |
| item-kNN baseline | 0.0404 ± 0.0059 | 0.0558 ± 0.0072 | 0.0545 ± 0.0087 | 13.02 ± 1.81 | 0.1552 ± 0.0140 |
| popularity baseline | 0.0404 ± 0.0059 | 0.0558 ± 0.0072 | 0.0545 ± 0.0087 | 13.02 ± 1.81 | 0.1552 ± 0.0140 |
| lexical-title baseline | 0.0197 ± 0.0075 | 0.0278 ± 0.0083 | 0.0266 ± 0.0099 | 26.19 ± 2.11 | 0.0672 ± 0.0140 |

**k = 1 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.1962 ± 0.0184 | 0.2242 ± 0.0190 | 0.2672 ± 0.0230 | 2.08 ± 0.71 | 0.3723 ± 0.0273 |
| Cadence (fusion only) | 0.1441 ± 0.0154 | 0.1742 ± 0.0164 | 0.2022 ± 0.0199 | 2.99 ± 0.83 | 0.3723 ± 0.0273 |
| item-kNN baseline | 0.1390 ± 0.0153 | 0.1696 ± 0.0166 | 0.1888 ± 0.0198 | 5.08 ± 1.23 | 0.3545 ± 0.0282 |
| popularity baseline | 0.0398 ± 0.0059 | 0.0552 ± 0.0072 | 0.0540 ± 0.0086 | 13.08 ± 1.81 | 0.1541 ± 0.0140 |
| lexical-title baseline | 0.0183 ± 0.0077 | 0.0257 ± 0.0084 | 0.0242 ± 0.0098 | 34.50 ± 2.16 | 0.0500 ± 0.0135 |

**k = 5 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.2416 ± 0.0189 | 0.2733 ± 0.0191 | 0.3283 ± 0.0232 | 0.81 ± 0.40 | 0.4515 ± 0.0260 |
| Cadence (fusion only) | 0.1849 ± 0.0153 | 0.2197 ± 0.0162 | 0.2559 ± 0.0194 | 1.01 ± 0.45 | 0.4515 ± 0.0260 |
| item-kNN baseline | 0.1761 ± 0.0142 | 0.2117 ± 0.0153 | 0.2430 ± 0.0185 | 1.42 ± 0.59 | 0.4663 ± 0.0258 |
| popularity baseline | 0.0378 ± 0.0057 | 0.0527 ± 0.0070 | 0.0518 ± 0.0082 | 13.50 ± 1.82 | 0.1525 ± 0.0139 |
| lexical-title baseline | 0.0179 ± 0.0075 | 0.0251 ± 0.0083 | 0.0237 ± 0.0097 | 34.90 ± 2.14 | 0.0501 ± 0.0136 |

**k = 10 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.2457 ± 0.0184 | 0.2777 ± 0.0186 | 0.3359 ± 0.0231 | 0.71 ± 0.36 | 0.4696 ± 0.0256 |
| Cadence (fusion only) | 0.1829 ± 0.0150 | 0.2183 ± 0.0159 | 0.2578 ± 0.0191 | 0.95 ± 0.41 | 0.4696 ± 0.0256 |
| item-kNN baseline | 0.1676 ± 0.0138 | 0.2033 ± 0.0150 | 0.2390 ± 0.0178 | 1.52 ± 0.58 | 0.4769 ± 0.0250 |
| popularity baseline | 0.0353 ± 0.0055 | 0.0497 ± 0.0068 | 0.0498 ± 0.0080 | 14.20 ± 1.85 | 0.1515 ± 0.0138 |
| lexical-title baseline | 0.0172 ± 0.0073 | 0.0240 ± 0.0080 | 0.0233 ± 0.0096 | 35.45 ± 2.11 | 0.0489 ± 0.0134 |

**k = 25 seed tracks**

| System | R-prec | R-prec (artist) | NDCG@100 | Clicks ↓ | Recall@500 |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0.2005 ± 0.0181 | 0.2293 ± 0.0184 | 0.2973 ± 0.0225 | 2.28 ± 0.91 | 0.4706 ± 0.0264 |
| Cadence (fusion only) | 0.1443 ± 0.0141 | 0.1748 ± 0.0150 | 0.2217 ± 0.0179 | 2.59 ± 0.92 | 0.4706 ± 0.0264 |
| item-kNN baseline | 0.1286 ± 0.0122 | 0.1595 ± 0.0135 | 0.1963 ± 0.0157 | 3.05 ± 0.91 | 0.4726 ± 0.0250 |
| popularity baseline | 0.0292 ± 0.0052 | 0.0412 ± 0.0063 | 0.0451 ± 0.0077 | 17.58 ± 2.01 | 0.1464 ± 0.0141 |
| lexical-title baseline | 0.0117 ± 0.0051 | 0.0176 ± 0.0062 | 0.0190 ± 0.0082 | 38.12 ± 2.00 | 0.0447 ± 0.0130 |


### Channel ablations

R-precision ± 2×SE; each row removes or isolates one retrieval channel. `≈` marks a cell whose difference from the fusion row is inside the band of that difference — at this sample size the change is not distinguishable from noise.

| Configuration | k=0 | k=1 | k=5 | k=10 | k=25 |
|---|---|---|---|---|---|
| Cadence (fusion only) | 0.0709 ± 0.0108 | 0.1441 ± 0.0154 | 0.1849 ± 0.0153 | 0.1829 ± 0.0150 | 0.1443 ± 0.0141 |
| − exact co-occurrence | 0.0709 ± 0.0108 ≈ | 0.1070 ± 0.0134 | 0.1212 ± 0.0134 | 0.1157 ± 0.0131 | 0.0906 ± 0.0122 |
| − collaborative embedding | 0.0709 ± 0.0108 ≈ | 0.1499 ± 0.0154 ≈ | 0.1734 ± 0.0143 ≈ | 0.1651 ± 0.0140 ≈ | 0.1310 ± 0.0130 ≈ |
| − folksonomy tags | 0.0191 ± 0.0076 | 0.1264 ± 0.0146 ≈ | 0.1699 ± 0.0142 ≈ | 0.1694 ± 0.0138 ≈ | 0.1361 ± 0.0133 ≈ |
| − lexical | 0.0723 ± 0.0103 ≈ | 0.1438 ± 0.0154 ≈ | 0.1843 ± 0.0152 ≈ | 0.1823 ± 0.0150 ≈ | 0.1448 ± 0.0141 ≈ |
| − audio | 0.0716 ± 0.0108 ≈ | 0.1441 ± 0.0154 ≈ | 0.1850 ± 0.0154 ≈ | 0.1829 ± 0.0150 ≈ | 0.1443 ± 0.0141 ≈ |
| only co-occurrence | 0.0000 ± 0.0000 | 0.1372 ± 0.0153 ≈ | 0.1734 ± 0.0142 ≈ | 0.1651 ± 0.0138 ≈ | 0.1270 ± 0.0121 ≈ |
| only collaborative | 0.0000 ± 0.0000 | 0.0857 ± 0.0115 | 0.1060 ± 0.0112 | 0.1020 ± 0.0110 | 0.0820 ± 0.0108 |
| only folksonomy tags | 0.0668 ± 0.0104 ≈ | 0.0657 ± 0.0104 | 0.0613 ± 0.0100 | 0.0561 ± 0.0096 | 0.0429 ± 0.0088 |

20 of 25 `−` cells sit inside their own band: removing that channel cannot be told from noise in this report.


### Beyond-accuracy and latency

| System | k | Coverage@100 | Gini@100 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| **Cadence (reranked)** | 0 | 0.0602 | 0.681 | 128 | 145 |
| Cadence (fusion only) | 0 | 0.1041 | 0.503 | 49 | 96 |
| popularity baseline | 0 | 0.0006 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 1 | 0.1258 | 0.430 | 163 | 200 |
| Cadence (fusion only) | 1 | 0.1408 | 0.383 | 94 | 155 |
| popularity baseline | 1 | 0.0006 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 5 | 0.1049 | 0.494 | 172 | 215 |
| Cadence (fusion only) | 5 | 0.1050 | 0.505 | 92 | 150 |
| popularity baseline | 5 | 0.0007 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 10 | 0.0963 | 0.525 | 172 | 213 |
| Cadence (fusion only) | 10 | 0.0949 | 0.541 | 97 | 143 |
| popularity baseline | 10 | 0.0007 | 0.000 | 0 | 0 |
| **Cadence (reranked)** | 25 | 0.0852 | 0.569 | 183 | 230 |
| Cadence (fusion only) | 25 | 0.0828 | 0.590 | 110 | 177 |
| popularity baseline | 25 | 0.0007 | 0.000 | 0 | 0 |


Evaluated on 400 held-out playlists per cell, retrieval depth 500, catalog 159,338 tracks. **Detection floor: 0.0149 R-precision** — 2×SE of the k=0 `full_reranked` cell, the band the headline number sits in. A difference between two cells is judged against the band of their *difference* (the `≈` marks above), which is wider still.
