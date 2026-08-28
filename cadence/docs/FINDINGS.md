# Two defects found by reading the demo output, and what measuring them showed

Both were found the same way: generating real playlists for the report page and
looking at them, rather than reading a metric. Neither was visible in any
aggregate number, and the second one still isn't.

---

## 1. The exact-tag channel scored multi-concept queries as ANY-match

`sparse_tag_channel` summed raw folksonomy counts across the requested concepts
and *then* took `log1p`:

```python
scores = np.log1p(np.asarray(sub.sum(axis=1)).ravel())
```

For *"90s alternative rock for a road trip"* the planner resolved three tags
correctly — `road trip` (14,212 tracks), `alternative rock` (2,246), `1990s`
(8,790). But a track on 500 playlists tagged `road trip` and none tagged `1990s`
scored `log1p(500) = 6.2`, while one matching **both** at 50 each scored
`log1p(100) = 4.6`. Riding one popular tag beat satisfying the whole request.

**Fix:** take the log per concept, then sum — diminishing returns within a
concept, additive across them.

```python
scores = np.asarray(sub.log1p().sum(axis=1)).ravel()
```

The same three-concept track now scores 10.30 against 4.51 for the single-concept
one. Covered by `tests/test_channels_tags.py`.

**What measuring it showed: nothing.** Across all five seed counts, R-precision
moved by at most 0.0011 against a 2×SE band of 0.021–0.027 — **0 of 5 cells
changed by more than the noise**. The fix is kept because it is correct and
tested, not because it improved a number. It doesn't: the exact-tag channel
rarely determines the fused top-k on these queries.

## 2. Selection blended a rank-normalised score with a min-maxed one

`assemble/select.py` carries a long docstring explaining why min-max is the wrong
normalisation for a skewed scorer — and the line immediately below it min-maxed
the other term:

```python
relevance = _rank_normalize(raw)          # uniform across the pool
aff = _minmax(...)                        # full 0-1 swing
relevance = (1 - w) * relevance + w * aff
```

Rank-normalised relevance is uniform, so the gap between the best candidate and
the fiftieth in a 500-pool is 0.1. A min-maxed affinity swings the full range. At
`audio_affinity_weight = 0.35` the audio term could therefore overturn a hundred
places of retrieval ranking.

It did. Retrieval for the 90s query returned Pixies, Third Eye Blind, Savage
Garden, Dave Matthews Band and No Doubt — genuinely correct. Selection replaced
them with tracks matching a valence target inferred from the words *"road trip"*.
**The retrieval stage was never the problem.**

**Fix:** rank-normalise the affinity too, so both terms are on one scale.

**What measuring it showed: a real trade, priced.** Constraint satisfaction stays
at 100% across all five constraints with zero failures, and Red Hot Chili
Peppers, 311 and Savage Garden return to the 90s playlist. The cost is audio
precision:

| mood target | before | after |
|---|---:|---:|
| valence | 0.0206 | **0.0514** |
| acousticness | 0.0416 | 0.0487 |
| energy | 0.0429 | 0.0482 |
| danceability | 0.0360 | 0.0289 |
| instrumentalness | 0.1302 | 0.1153 |

That is the trade working as designed — less audio dominance means looser audio
targeting — but it is a cost, not a free win.

**Unfinished.** `audio_affinity_weight = 0.35` was chosen when the term was
min-maxed, so changing the normalisation without retuning the weight is half a
change. The principled fix is a weight that depends on the query: a request
naming a genre or an era should lean on tags, a request naming only a mood should
lean on audio. That needs an evaluation to tune against, and the one here is
`n = 20` queries — far too small.

---

## The finding underneath both

**Cadence's evaluation harness cannot see changes of this size.** At its default
`limit=400`, the k=0 cell has SE = 0.0074, so the detection floor is
**2×SE = 0.0149 R-precision — about 10% of the metric's own value.** Reaching
half a point of resolution would take 16× the challenges.

Most of the channel weights in `config.py` differ by less than that. They were
not validated against this harness, because they could not have been. The same
lesson [Ostinato](../../ostinato) learned about its own dose applies here:
before believing a null result, check whether the instrument could have detected
a positive one.

---

## 3. The evaluation harness handed out the wrong seeds

Found by [Segue](../../segue), which needed playlist order and discovered Cadence
could not supply it.

`eval/splits.py` documented itself as exposing *"the title plus the first `k`
tracks"* and implemented:

```python
trs = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
seed_tracks = trs[:k].tolist()
```

`trs` comes back in ascending **track-id** order, so `trs[:k]` is *the k
lowest-numbered tracks in the playlist*, not its first k. Because ids are assigned
in corpus-wide first-seen order during the build, low id correlates with
popular-and-early — the seeds were quietly biased toward the catalog's head rather
than being a neutral prefix.

**The cause was a layer deeper than it looked.** The obvious suspect is SciPy,
which keeps CSR column indices sorted within a row. SciPy was innocent. The order
had already been destroyed in `data/build.py`, pass 1:

```python
# A track counts once per playlist even if the playlist repeats it.
rows.append(np.unique(row))
```

`np.unique` deduplicates — the stated intent — and **sorts as a side effect**,
which was not. By the time the matrix was constructed there was no order left in
the pipeline to preserve. Dedup by first occurrence instead:

```python
_, first = np.unique(row, return_index=True)
rows.append(row[np.sort(first)])
```

The first attempt at this fix captured the per-playlist sequence from pass 2 and
verified additive — identical counts, identical `track_uri` at every index. It
would have shipped an `order.npz` that looked entirely correct and contained
ascending ids, because what it faithfully captured was already sorted. A
verifier written *before* trusting the rebuild caught it: 99 of 99 sampled rows
came back ascending.

**Why it survived a review.** Nothing is wrong with any line in isolation. The
bug lives in the gap between a call chosen for one property and used for another,
and a docstring three modules away that promised something the data no longer
supported.

**Fix.** Order now comes from the build rather than being re-derived:

* `data/build.py` writes `order.npz`, the ragged per-playlist track sequence,
  captured in the same pass that fills the matrix — so the matrix and the order
  cannot disagree about which tracks a playlist contains.
* `eval/splits.py` reads it and takes a genuinely first-k prefix. If the file is
  absent it **raises**, rather than falling back to track-id order. A silent
  fallback is how this lasted as long as it did.

The rebuild was verified purely additive: identical track count, identical
`track_uri` at every index, identical interaction count. That mattered because
Timbre, Segue, Gamut and Ostinato all store track *indices* — a shifted index
would have silently invalidated four downstream projects.

**How biased were the old seeds?** Measured against the corrected ones, on the
same 2,000 held-out playlists:

| k | old seed popularity | true first-k | ratio | seed overlap |
|---:|---:|---:|---:|---:|
| 1 | 1103 | 640 | **1.72x** | 2.1% |
| 5 | 1015 | 595 | 1.71x | 9.3% |
| 10 | 962 | 573 | 1.68x | 18.1% |
| 25 | 838 | 543 | 1.54x | 42.8% |

(Catalog mean is 36.9 playlists per track, median 9. Any playlist track is
popular by selection; the point is the *ratio* between the two seed sets.)

The old seeds were **1.5–1.7x more popular** than the real ones, and at k = 1 the
two sets shared **2.1%** of their tracks — they were very nearly different
experiments. Popular seeds carry richer co-occurrence, so the prediction was that
the published k >= 1 numbers were optimistic.

**That prediction was wrong.** Re-running with correct seeds:

| k | biased seeds | true first-k | delta | 2xSE |
|---:|---:|---:|---:|---:|
| 0 | 0.1429 | 0.1429 | +0.0000 | 0.0210 |
| 1 | 0.1785 | 0.1962 | +0.0177 | 0.0252 |
| 5 | 0.2472 | 0.2416 | -0.0056 | 0.0265 |
| 10 | 0.2472 | 0.2457 | -0.0015 | 0.0262 |
| 25 | 0.1925 | 0.2005 | +0.0081 | 0.0262 |

No cell moves by more than two standard errors, and the direction is mixed rather
than uniformly down. The k = 0 row is bit-identical, which is the control working:
a title-only challenge has no seeds to select, so nothing there could change.

So the harness was **wrong in method and, as far as this instrument can tell,
right in value**. The seeds were drawn from a systematically more popular slice of
each playlist and the headline metric did not notice. Clicks is the only measure
with a consistent pattern — worse at k = 1, 5, 10 (+0.29, +0.19, +0.14) and better
at k = 25 (-0.85) — which is suggestive and under-powered for the same reason
everything else here is.

The fix is kept regardless. A harness whose seeds do not mean what its docstring
says is broken whether or not the breakage happens to move a number, and the next
person to build a sequence-aware model on it — which is exactly what Segue is —
would have been misled.

**The ablation claims survive unchanged.** Re-run against corrected seeds, removing
the folksonomy channels still collapses k = 0 R-precision by **73%**, and both
collaborative channels alone still score exactly **0.0000**. Worth stating because
the first pass at checking this compared the ablations — which are fusion-level —
against `full_reranked`, got 87%, and nearly "corrected" a number that was right.
Ablations belong against `full_fusion`.

**What it changes.** Every seeded cell (k >= 1) of the published evaluation. The
k = 0 cell is untouched: a title-only challenge has no seeds to select, which is
also why the two fixes above could be measured against it while this one was
still outstanding.

**Independently cross-checked.** Segue rebuilt playlist order from the raw MPD
slices by a completely different path. On a 235-playlist sample the two agree on
**100%** of sequences once duplicate policy is reconciled — Cadence dedups by
first occurrence, Segue keeps repeats, and every one of the 55 initial
disagreements was exactly that.

---

## 4. The weight could not be retuned, because the harness could only see one side

`audio_affinity_weight` trades audio adherence against tag relevance. Fixing the
normalisation mismatch in §2 changed what the weight *means* without retuning the
weight itself, which left the job half done — so this went looking for the right
value and found that the question was not yet answerable.

**Nothing measured the cost.** `constraints_eval` reports mood error, which falls
monotonically as the weight rises. No metric anywhere in the repo measured whether
the genre or era the listener asked for actually arrived, so nothing ever pushed
back and the term looked free at any strength. It was not: at 0.35 it was
overturning a hundred places of correct retrieval. **The problem was never that the
battery had 20 queries. A one-sided metric cannot find an optimum at any n.**

### The missing counterweight

**Tag adherence**: of the tracks delivered, do they carry the tags the request
named? Measured against the folksonomy — the same human behaviour retrieval is
built on — rather than against a genre string nobody curated. It is reported two
ways because the obvious one is useless:

| weight | tag share | tag strength | mood error |
|---:|---:|---:|---:|
| 0.00 | 1.0000 | 3.426 | 0.1926 |
| 0.20 | 0.9969 | 3.101 | 0.1096 |
| **0.35** | 0.9910 | 2.820 | 0.0735 |
| 0.45 | 0.9770 | 2.605 | 0.0561 |
| 0.80 | 0.5468 | 1.405 | 0.0070 |

*Share* — the fraction of tracks carrying at least one requested tag — is flat to
three decimal places until 0.45 and then collapses. A track filed under `rock`
once out of five hundred playlists clears that bar, so it cannot separate a
plausible playlist from a good one. *Strength*, the mean `log1p` of the requested
tag count, is what `sparse_tag_channel` itself ranks on and moves smoothly across
the whole range.

### Two findings, one of them a surprise

**Query-dependence is already implemented, and I did not know it.** The hypothesis
was that genre requests should lean on tags and mood requests on audio. But a pure
genre or era query states no audio target, so `_audio_affinity` returns `None` and
the weight is forced to zero. Across the whole sweep the tag-led family's strength
is *identical* at 3.889 for every weight from 0.0 to 0.8. The contested case is
narrower than it looked: only queries naming **both** a mood and a genre — which
is exactly what "90s alternative rock for a road trip" is.

**The weight cannot be chosen from this data, and the knee criterion was fooling
me.** Distance to the ideal corner on min-max-normalised axes puts the knee at
0.35 for both families — reassuringly, the shipped value. But that is an artifact:

| swept range | knee (mood) | knee (mixed) |
|---|---:|---:|
| 0.0–0.8 | 0.35 | 0.35 |
| ≤ 0.6 | 0.30 | 0.30 |
| ≤ 0.45 | 0.20 | 0.20 |
| excluding 0.0 | 0.45 | 0.45 |

The knee moves wherever the sweep's endpoints move, because the trade is smoothly
convex with no kink in it. Reporting 0.35 as "confirmed optimal" would have been
reporting my own choice of sweep range back as a finding.

**So the weight is unchanged at 0.35, and this is not a retune.** What is now
true that was not before: the cost of the weight is measurable, the exchange rate
at the shipped value is **≈0.12 tag strength per 0.01 of mood error**, and both
query families agree on that rate at every swept range — which is the robust part,
and the reason a query-dependent weight is *not* justified.

Choosing a different value needs a stated preference between mood fidelity and
genre fidelity. That is a product judgement, and no amount of held-out data
supplies it. The contribution here is that the trade now has numbers on both
sides; `cadence eval-affinity` regenerates them.
