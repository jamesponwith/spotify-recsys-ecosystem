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
