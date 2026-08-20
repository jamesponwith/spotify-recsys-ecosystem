# Architecture

## The problem

Natural-language playlist requests fail in two characteristic ways.

A pure LLM invents tracks. Ask for "90s R&B for a rainy drive" and you get a
plausible tracklist containing songs that do not exist, are not in the catalog,
or are attributed to the wrong artist. The output is fluent and unusable.

A pure embedding search cannot honour requirements. "Nothing explicit, 45
minutes, 120–140 BPM, no more than two songs per artist" are not soft
preferences to be approximated by cosine similarity — they are constraints that
are either met or not.

Cadence separates the two concerns. Language understanding produces a
*structured intent*; retrieval and assembly produce a *playlist* from the real
catalog under hard constraints. The model never touches the tracklist.

---

## Pipeline

```
  "chill 90s R&B for a rainy drive, nothing explicit, about 45 minutes"
        │
   ┌────▼─────────────────────────────────────────────────────────────┐
   │ 1. PLAN            rules (default) or Claude (structured output) │
   │    → PlaylistIntent{themes, eras, audio targets, tempo,          │
   │                     constraints, energy_curve, seeds}            │
   └────┬─────────────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────────────┐
   │ 2. MASK            hard constraints as a boolean filter          │
   │    explicit · duration · tempo window · avoided artists          │
   └────┬─────────────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────────────┐
   │ 3. RETRIEVE        seven channels, each answering differently    │
   │                                                                  │
   │   collaborative   SPPMI+SVD embedding of playlist co-occurrence  │
   │   cooccurrence    exact neighbourhood counts (sharp, sparse)     │
   │   tag             folksonomy embedding: text ↔ music bridge      │
   │   tag_exact       raw title-token counts (precise, no smoothing) │
   │   lexical         TF-IDF over track/artist/album/genre strings   │
   │   audio           k-NN in standardised audio-feature space       │
   │   popularity      prior; backstop only when others come up thin  │
   └────┬─────────────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────────────┐
   │ 4. FUSE            reciprocal rank fusion → ~1500 candidates     │
   │ 5. RERANK          gradient-boosted trees, 30 features           │
   └────┬─────────────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────────────┐
   │ 6. SELECT          MMR diversity + artist caps + duration target │
   │ 7. SEQUENCE        beam search: energy arc · key · tempo         │
   │ 8. EXPLAIN         grounded reasons + validated copy             │
   └────┬─────────────────────────────────────────────────────────────┘
        │
        ▼   GeneratedPlaylist{tracks, stats, constraint_report, timings}
```

---

## Why the folksonomy is the interesting part

The bridge from language to music is the hardest link in the chain, and track
metadata does not provide it. The string "Holocene — Bon Iver — Bon Iver" does
not contain the words *calm*, *winter*, *introspective* or *driving*. No
encoder can extract meaning that is not in the text.

But the corpus contains ~98 000 playlists that humans named. Those names are
free-text descriptions of *how the music is used*: `rainy day`, `gym`,
`studying`, `90s throwbacks`, `late night drive`. Building a track × title-token
matrix and factorising its Shifted PPMI yields a space where a *phrase* and a
*track* are directly comparable, grounded entirely in human behaviour.

This is what makes cold-start natural-language retrieval work. On title-only
queries (no seed tracks — nothing for collaborative filtering to use), the tag
channels alone beat the popularity baseline substantially; see
[EVALUATION.md](EVALUATION.md).

---

## Representation learning

Both learned spaces come from the same primitive: **Shifted Positive PMI +
truncated SVD**.

```
PMI(i,j) = log( c_ij · N / (c_i · c_j^α) )      α = popularity damping
SPPMI    = max(0, PMI − log k)                   k = negative-sample shift
M ≈ U Σ Vᵀ                                       randomized SVD
rows = U√Σ,  cols = V√Σ                          both scaled so rows·colsᵀ ≈ M
```

Levy & Goldberg (2014) showed that skip-gram with negative sampling is
implicitly factorising exactly this matrix. Doing it explicitly removes epochs,
learning rates and sampling noise, fits the full corpus in ~40 s, and is exactly
reproducible from a seed — see [ADR-002](DECISIONS.md).

Scaling both sides by `√Σ` is what lets a **tag vector and a track vector live
in one space**, so `tag · track` approximates their SPPMI association directly.

| Space | Matrix | Purpose |
|---|---|---|
| collaborative | playlist × track | "these songs go together" |
| tag | track × title-token | "this phrase means this music" |
| lexical | TF-IDF over metadata strings | exact entity matches |
| audio | standardised Spotify features | mood/tempo targeting, sequencing |

---

## Fusion and reranking

Channel scores are mutually incomparable (cosine, TF-IDF dot products, damped
counts, negative-exponential distances). Reciprocal rank fusion uses only the
ordering:

```
score(d) = Σ_c  w_c / (k + rank_c(d))          k = 60
```

The learned reranker then fixes what RRF's fixed weights cannot: per-query
channel trust. Supervision is free — hide part of a real playlist, retrieve
against its title and remaining seeds, label each candidate by whether a human
actually put it there. Gradient-boosted trees over 30 features (per-channel
ranks/scores/presence, popularity, audio distance, tag mass, tempo fit).

---

## Assembly

**Selection** is not top-k. Twenty near-identical songs by three artists is a
perfect ranking and a bad playlist. Maximal Marginal Relevance trades relevance
against novelty in the collaborative space, while artist caps and the duration
target are enforced inside the loop.

**Sequencing** treats the playlist as an arc. Beam search minimises

```
Σ  w_e·|energy(t) − curve(position)|          energy arc: build/wind_down/wave/…
 + w_k·keyDistance(t, t+1)                    Camelot wheel compatibility
 + w_t·tempoDistance(t, t+1)                  half/double-time treated as close
 + w_a·[same artist adjacent]
```

Key compatibility uses the Camelot wheel: position `((root·7) mod 12 + 7) mod 12 + 1`
walking the circle of fifths, with minor keys placed on their relative major.
Same key, a fifth away, or relative major/minor are cheap; a tritone is not.

---

## Serving

`Catalog.load()` preloads every array the request path touches; pandas is used
only to build lookup tables. Warm latency is ~100 ms end to end, dominated by
retrieval. Measured per-stage timings ship on every response in `timings_ms`,
so regressions are visible rather than inferred.

The `/explain` endpoint returns the retrieval trace — parsed intent, resolved
tags, per-channel candidate counts, what each mask removed — which answers
"why did the engine consider these at all", a different question from the
per-track reasons.

---

## Failure behaviour

| Failure | Behaviour |
|---|---|
| No LLM key / API error | Falls back to the rule-based planner, warns, continues |
| LLM copy names an absent artist | Detected, replaced with template copy, warned |
| Reranker missing or raising | Falls back to fusion order, warns |
| Over-constrained request | Returns fewer tracks; `constraint_report` shows which filter bit |
| Unparseable query | Popularity backstop keeps the system responsive |
| Missing audio features | Track keeps neutral cost; never imputed to the mean |
