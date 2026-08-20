# Design decisions

Short records of the choices that shaped this system, including the ones that
went the other way. Each is written so a reviewer can disagree with it on the
merits.

---

## ADR-001 — Ground natural language in playlist titles, not in a text encoder

**Context.** The core problem is mapping "rainy day study music" onto tracks.
The obvious move is a pretrained sentence encoder over track metadata.

**Decision.** Build a *folksonomy* instead: tokenise the titles of ~98 k real
playlists, and represent every track by the bag of title tokens of the
playlists it appears on.

**Why.** Track metadata (title, artist, album) contains almost no mood or
context information. "Holocene — Bon Iver" does not contain the word "calm";
no encoder can extract meaning that is not in the string. But 400 humans put
that track on a playlist called something with "chill" in it, and *that* is
direct evidence about how the song is used. The folksonomy turns free text and
music into the same space using human behaviour as the bridge.

**Consequences.** Vocabulary is limited to what people actually write in
playlist titles — strong on mood, activity, genre and era, weak on anything
niche. Tracks on few playlists get thin tag profiles (the cold-start-item
problem, unsolved here — see LIMITATIONS in the README). The upside is that
the signal is real: the `only_tag` ablation beats the popularity baseline on
title-only queries by a wide margin (see `docs/EVALUATION.md`).

---

## ADR-002 — Factorise Shifted PPMI instead of training word2vec

**Context.** Item embeddings from playlist co-occurrence are usually produced
with SGNS ("item2vec"), which means an epoch loop, a learning rate, negative
sampling and run-to-run variance.

**Decision.** Explicitly factorise the Shifted PPMI matrix with a truncated
randomized SVD.

**Why.** Levy & Goldberg (2014) showed SGNS is *implicitly* factorising a
shifted PMI matrix. Doing it explicitly gives the same class of embedding with
no hyperparameter search, no epochs, exact reproducibility from a seed, and a
40-second fit on the full corpus. For a system whose evaluation must be
trustworthy, removing a source of run-to-run variance is worth more than the
last fraction of a point.

**Consequences.** No incremental updates: adding playlists means refitting
(cheap here, would matter at Spotify's scale). Rank is capped by the SVD
dimension rather than learned adaptively.

---

## ADR-003 — Keep exact co-occurrence as its own channel

**Context.** The first working version used only the learned collaborative
embedding. On seeded challenges it *lost* to a plain item-kNN baseline:
R-precision 0.129 vs 0.142 at k=5.

**Decision.** Add exact neighbourhood co-occurrence as a separate retrieval
channel and let fusion arbitrate, rather than tuning the embedding until it
wins.

**Why.** The measurement was unambiguous, and the honest reading is that 160
dimensions genuinely lose information that exact counts retain. The embedding
still earns its place — it generalises where counts are sparse, and it is what
makes the diversity and selection stages work, since MMR needs a metric space.
Running both and fusing lifted k=5 R-precision to 0.169, beating the baseline
the embedding alone had lost to.

**Consequences.** The serving footprint now includes the interaction matrix
(~70 MB) alongside the embeddings. Worth it.

**Note.** This is the decision most likely to be wrong in the long run: at
larger scale, exact co-occurrence gets expensive and the embedding's
generalisation matters more. The channel design means that trade can be
re-measured rather than re-argued.

---

## ADR-004 — Hard constraints are filters, never prompts

**Context.** "Nothing explicit", "12 songs", "130-150 BPM" are non-negotiable.
An LLM can be asked to respect them.

**Decision.** Enforce every hard constraint structurally — a boolean mask
before scoring, a cap inside the selection loop, a duration check during
assembly — and never rely on instruction-following for correctness.

**Why.** A filter cannot be talked out of its job. Asking a model to honour a
constraint converts a guarantee into a probability, and the failure is silent:
a playlist with one explicit track looks exactly like a correct one until a
parent complains. The `constraint_report` on every response records
satisfaction per constraint, so compliance is observable rather than assumed.

**Consequences.** Some requests over-constrain and return few tracks. That is
the correct behaviour — returning a wrong playlist is worse than returning a
short one — and the response says which filter did the damage.

---

## ADR-005 — The LLM expresses intent; it never picks tracks

**Context.** The straightforward LLM playlist generator asks a model for songs.

**Decision.** The model's only outputs are (a) a structured `PlaylistIntent`
and (b) the title/description copy, written *after* selection. Retrieval picks
every track from the catalog.

**Why.** Track hallucination becomes structurally impossible rather than
merely unlikely: the model has no channel through which to emit a track. It
also keeps the system honest about catalog reality — a model asked for "90s
R&B" will happily name songs that are not licensed, not in the catalog, or not
real. The remaining exposure is the copy naming an artist who is not on the
playlist, which `explain.validate_copy` detects by scanning the generated text
against the catalog's artist index and falling back to template copy.

**Consequences.** The system cannot surface a track the catalog lacks, even
when the model "knows" it. That is a property, not a bug.

---

## ADR-006 — Deterministic planner as the default, LLM as an upgrade

**Context.** The natural design puts an LLM on the request path.

**Decision.** Ship a 49-entry mood lexicon and a regex/rule parser as the
default planner. The Anthropic planner is opt-in and degrades to the rules on
any failure.

**Why.** Three reasons, in order of importance. (1) Evaluation: a
nondeterministic planner injects variance into every retrieval metric, and
ablations stop being comparable. (2) Availability: an LLM outage becomes a
quality regression instead of an outage. (3) Latency and cost: the rules run in
~4 ms with no per-request spend, which matters when most queries are
"workout" or "chill".

**Consequences.** The rules cannot handle open-ended compositional phrasing
("something like the first half of a Sofia Coppola soundtrack"). That gap is
exactly what the LLM planner is for, and the two are directly comparable
because they satisfy one interface.

---

## ADR-007 — Reciprocal-rank fusion before a learned reranker

**Context.** Five channels return scores on incomparable scales — cosine
similarity, TF-IDF dot products, negative exponential distances, raw counts.

**Decision.** Fuse by rank (RRF), then learn a reranker on top of the fused
candidate set.

**Why.** Calibrating heterogeneous scores against each other is its own
research project, and RRF sidesteps it by using only the ordering each channel
agrees on. It also gives a strong, dependency-free system on day one, which
means the reranker can be evaluated against a real baseline instead of against
nothing. The reranker then learns what RRF's fixed weights cannot: that
"songs like Radiohead" is a collaborative-filtering question while "rainy day
study" is a tag question.

**Consequences.** RRF discards score magnitude, so a channel that is
overwhelmingly confident about its top hit gets no extra credit for it. The
reranker sees the raw scores as features and can recover that.

---

## ADR-008 — Exact search over an ANN index

**Context.** Nearest-neighbour search over ~159 k × 160-dimensional vectors.

**Decision.** Blocked exact float32 matmul, behind a `DenseIndex` interface.

**Why.** At this scale exact search takes single-digit milliseconds, well
inside budget. More importantly it removes approximation error from the
evaluation: every number the harness reports is a property of the model, not
of an index's recall/latency trade. Swapping in HNSW at larger scale is a
change behind one interface.

**Consequences.** Linear in catalog size. Past a few million tracks this must
become approximate, and the eval must then report ANN recall separately.

---

## ADR-009 — Sequencing as a first-class stage

**Context.** Most recommenders stop at a ranked list.

**Decision.** Order the selected tracks with beam search over a transition cost
combining an energy-curve target, Camelot-wheel key compatibility and tempo
continuity.

**Why.** A playlist is an experience with a shape, not a set. The same twenty
tracks in a different order are a different product, and the signals that make
an order good — harmonic compatibility, tempo continuity, an energy arc — are
already in the audio features. This is where domain knowledge shows up as
engineering rather than as vocabulary.

**Consequences.** Depends on audio-feature coverage; tracks without key/tempo
get neutral costs, so sequencing quality degrades gracefully with coverage
rather than failing. Beam search is exact enough at playlist length (≤ 100)
and runs in ~10 ms.

---

## ADR-010 — Impute nothing; record coverage instead

**Context.** Audio features cover part of the catalog. The convenient move is
to fill missing values with the column mean.

**Decision.** Carry a `has_audio` flag, exclude unmeasured tracks from
audio-space scoring, and give them neutral (not average) costs in sequencing.

**Why.** An imputed 0.5 energy is indistinguishable downstream from a measured
0.5 energy, which quietly converts "we don't know" into "we know it's average".
Every track without features would then look like a perfect match for any
mid-range mood request. Keeping the distinction means the audio channel is
smaller but truthful, and coverage is a reported number rather than a hidden
assumption.

**Consequences.** The audio channel ranks over a subset of the catalog. The
tempo filter deliberately *keeps* tracks with unknown tempo rather than
dropping them, since excluding them would silently restrict results to the
measured subset.
