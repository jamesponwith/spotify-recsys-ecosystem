# Timbre — evaluation plan

The project's credibility rests on this document rather than on the model. The
data constraint in [SPEC.md §2](SPEC.md#2-why-the-obvious-approach-is-unavailable)
means no single experiment proves the whole claim, so the evaluation is built as
a **ladder**: each rung states precisely what it establishes and what it does
not, and there is a **decision gate** before the expensive work begins.

---

## Phase 0: the falsification test

**Run this first. It uses only data already on disk, needs no audio, and takes
under an hour. It exists to kill the project cheaply if the premise is wrong.**

Everything downstream assumes one thing: *acoustic content predicts how humans
tag music in playlist titles.* If that is false, no encoder architecture rescues
it, and three weeks of audio work would produce a well-engineered null result.

### Design

Cadence already holds both halves for 159,338 tracks — Spotify's 11 engineered
audio descriptors, and a 128-d folksonomy tag embedding learned from ~98 k
playlist titles. So the premise is directly testable *without any audio at all*,
using the engineered descriptors as a stand-in for what an encoder would extract.

```
input   11 Spotify audio features   (energy, valence, danceability,
                                     acousticness, instrumentalness,
                                     speechiness, liveness, loudness,
                                     tempo, key, mode)
        -> 22-d design matrix       9 continuous + one-hot key (12) + mode
target  128-d Cadence tag embedding, L2-normalised
        (retrieval is cosine, so only direction carries meaning; fitting the
         raw vector would spend capacity on a norm the metric discards)
model   ridge regression, plus a 2-layer MLP as a stronger reference
split   80/20 by track, seeded
```

`key` is one-hot rather than ordinal: pitch class 11 is a semitone *below* 0,
not eleven units above it, and Cadence's own `AUDIO_FEATURE_COLS` drops key and
mode entirely for exactly this reason.

### The metric that decides

Embedding cosine is too forgiving — a prediction can look close and still be
useless for retrieval. The gate is a **retrieval** measurement:

1. Hold out 20% of tracks; overwrite each one's tag embedding with the
   *predicted* one.
2. For each held-out track, take the titles of the playlists that actually
   contain it and use them as queries.
3. Measure **recall@100** — does Cadence surface the track for a query drawn
   from a real playlist it genuinely belongs to?

Five systems, all on identical queries:

| System | Embedding used | Role |
|---|---|---|
| `random` | random unit vector | chance floor |
| `mean` | catalog-mean embedding | no-information floor |
| `content_ridge` | predicted, linear | **the thing being tested** |
| `content_mlp` | predicted, nonlinear | **the thing being tested** |
| `oracle` | the track's true embedding | upper bound |

Both regressors are reported. Fitting two and showing only the winner would make
the headline number unreproducible; the gate is ruled on the better of the two,
and which one that was is recorded in the artifact.

Only the held-out 20% of tracks have their embedding replaced — the rest of the
catalog keeps its true vectors under every system. That is what makes this a
cold-start simulation rather than a wholesale catalog swap.

**The scoring index must be rebuilt per system.** Cadence's `DenseIndex` marks
all-zero rows dead and scores them `-inf`. A predicted embedding is never exactly
zero, so a harness that scored raw cosine would let `content` compete for
candidate slots `oracle` is structurally barred from — inflating the very ratio
Gate 0 divides by.

### Gate 0 — go / no-go

> Proceed only if `content` reaches **≥ 3× the `random` floor** *and*
> **≥ 25% of `oracle`** on recall@100.

If it fails, the honest conclusion is that playlist-title vocabulary is driven
by context and social convention more than by sound, and the pivots are: predict
the **collaborative** embedding instead of the tag embedding; restrict the
target to the genre/texture tags the bridge actually covers; or move to the
sequential-transformer project instead. Record the null result either way — it
is a finding about the data, and it is worth writing up.

> **Interpretation caveat.** Passing Gate 0 shows the *target* is learnable from
> acoustic descriptors. It does not show a CNN on raw audio will reach the same
> place from a different, non-overlapping corpus. It removes one way to be
> wrong, not all of them.

---

## The ladder

### Rung 1 — encoder quality (fully validatable)

Standard MagnaTagATune top-50 auto-tagging benchmark, conventional folder split,
chunk-averaged inference.

| Metric | Why |
|---|---|
| **ROC-AUC** (macro) | the conventional headline; comparable to the literature |
| **PR-AUC** (macro) | far more informative here — tags are heavily imbalanced, and ROC-AUC flatters models on sparse positives |
| Per-tag PR-AUC | shows *which* concepts the encoder learns; the bridged tags are the ones that matter downstream |

Compared against published CNN baselines on this benchmark — FCN, musicnn,
sample-level CNN, short-chunk CNN — as reported in Won et al. (2020),
*Evaluation of CNN-based Automatic Music Tagging Models*. Those figures cluster
around **~0.90–0.92 ROC-AUC** and **~0.35–0.39 PR-AUC**; the exact numbers must
be read off the paper and cited precisely rather than quoted from memory.

Timbre's encoder is ~0.73 M parameters against baselines several times larger,
so the target is *landing near the published band at a fraction of the
parameters*, and reporting the gap honestly if it does not.

**Establishes:** the encoder extracts real semantic structure from audio.
**Does not establish:** anything about Cadence, MPD repertoire, or cold start.

### Rung 2 — the bridge (measurable)

| Measurement | Detail |
|---|---|
| Vocabulary coverage | 31/50 (62%) — already measured, reported per tag with Cadence track counts |
| Mass coverage | share of Cadence's *tag mass* reachable through the bridged tags, which matters more than the raw count |
| Calibration | reliability curves per bridged tag on the MTAT validation split; ECE before and after temperature scaling |
| Weighting scheme | uniform vs IDF vs learned, selected on validation, reported for all three |

**Establishes:** how much of Cadence's semantic space is addressable from audio.
**Does not establish:** that the projection preserves *neighbourhood structure* —
Rung 3 tests that.

### Rung 3 — cold-start retrieval (the headline)

The experiment the project exists for. Runs entirely inside Cadence, so it is a
genuine held-out retrieval measurement rather than a demonstration.

```
1. Sample 2,000 catalog tracks with ≥ 20 playlists  → "simulated new releases"
2. Remove their interactions and tag rows from every training matrix
3. Retrain the collaborative and tag spaces without them   (~100 s)
4. Give each one a content-predicted tag embedding
5. Query with the titles of the held-out playlists that really contained them
6. Measure whether they come back
```

Sampling tracks with ≥ 20 playlists is deliberate: they have enough real history
to compute a trustworthy oracle, which is what makes the upper bound meaningful.

| System | Embedding | Role |
|---|---|---|
| `random` | random unit vector | chance floor |
| `mean` | catalog-mean | no-information floor |
| `audio-features` | Phase 0's feature model | how far engineered descriptors get |
| `timbre` | the CNN encoder's projection | **the contribution** |
| `oracle` | true embedding, computed *with* history | upper bound |

Popularity — Cadence's strongest baseline elsewhere — is deliberately **not**
available: a genuinely new item has no popularity. That absence is the point.

**Metrics:** recall@100, recall@500, MRR, and the headline —

> **Oracle recovery ratio** = `recall@100(timbre) / recall@100(oracle)`
>
> *"Content recovers N% of what full listening history would have given you."*

One number, honest about its ceiling, and it makes the contribution legible to
someone who does not know the metric conventions.

**Establishes:** content-derived placement recovers a measurable fraction of
history-derived placement, on Cadence's own repertoire and queries.
**Does not establish:** that the *audio encoder* achieves this on MPD tracks —
the `timbre` row depends on the encoder transferring from Creative Commons
repertoire to mainstream US repertoire, which is Risk R1.

### Rung 4 — end-to-end injection (demonstration, not measurement)

Take real MTAT clips — genuinely absent from Cadence — encode, project, inject
as catalog rows flagged `has_history = False`, and query.

Reported as **qualitative**, with:

- constraint satisfaction, which *is* checkable (do injected tracks respect
  BPM windows and duration targets?);
- explanations that never claim playlist evidence for a track that has none;
- a fixed set of queries with output printed in full, so a reader can judge.

**Establishes:** the pipeline runs end to end and behaves sanely.
**Does not establish:** relevance. There is no ground truth for whether a
Magnatune post-rock clip belongs in a "rainy day study" playlist, and no amount
of presentation should imply otherwise.

---

## Reporting rules

Carried over from Cadence, because they are what made its evaluation worth
reading:

1. Every accuracy number ships with **catalog coverage and Gini** beside it.
   Cold-start injection that only surfaces the same twenty tracks is a
   regression wearing a good number.
2. Every cell carries a **standard error**, so run-to-run differences can be
   read as meaningful or not.
3. All splits are **frozen and seeded** before any modelling, and the same
   `splits.json` discipline applies.
4. Ablations are mandatory: no-calibration, uniform-vs-IDF weighting,
   bridged-tags-only-vs-all-50, and encoder-vs-engineered-features.
5. **Negative results are published in the same document as positive ones.**

---

## Threats to validity

**Domain shift is the dominant threat.** MagnaTagATune is Magnatune's Creative
Commons catalogue — heavy on ambient, classical, electronic and world, light on
the mainstream US pop, hip-hop and rock that dominate MPD. An encoder trained
there and applied to MPD repertoire is doing unmeasured transfer. Rung 3's
`audio-features` row exists partly to bound this: it shows what content
*of any kind* achieves on native Cadence data, so the gap between it and
`timbre` isolates the transfer cost.

**The bridge is lossy in a biased direction.** Instrument and voice-type
concepts do not survive; genre and texture do. Cold-start placement will
therefore be systematically better for tracks whose identity is carried by genre
than for tracks whose identity is carried by instrumentation.

**Tag co-occurrence is not relevance.** Recovering a track's tag neighbourhood
is a proxy for recovering its *audience*, and the two come apart — two songs can
share every descriptor and appeal to different listeners.

**The oracle is not perfect knowledge.** It is the embedding Cadence's own
factorisation produces, inheriting all of that model's error. The recovery ratio
is measured against an achievable ceiling, not a true one, which makes it
generous to Timbre. Say so wherever it is quoted.

**Rung 4 has no ground truth,** and the temptation to present a nice-looking
injected playlist as evidence should be resisted in the write-up.
