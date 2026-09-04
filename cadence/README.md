# Cadence

**Natural-language playlist generation with grounded retrieval and learned sequencing.**

Ask for *"chill 90s R&B for a rainy drive, nothing explicit, about 45 minutes"*
and get back a real, ordered, constraint-satisfying playlist — every track from
a 159 000-track catalog built from the Spotify Million Playlist Dataset, every
recommendation explained by evidence, and not one track invented.

```
$ cadence play "chill acoustic songs for a rainy sunday morning, nothing explicit, 8 songs"

╭──── "chill acoustic songs for a rainy sunday morning, nothing explicit, 8 songs" ────╮
│ Acoustic Chill                                                                      │
│ 8 tracks, for acoustic, chill, rainy day.                                           │
╰─────────────────────────────────────────────────────────────────────────────────────╯
  #  Track                  Artist            BPM   Nrg  Why
  1  Breakdown              Jack Johnson       75  0.39  on 49 playlists tagged "chill"
  2  Ho Hey                 The Lumineers      80  0.47  on 105 playlists tagged "chill"
  3  I'm Not The Only One   Sam Smith          81  0.50  on 74 playlists tagged "chill"
  4  Location               Khalid             80  0.45  on 287 playlists tagged "chill"
  5  Say It                 Flume              75  0.53  on 77 playlists tagged "chill"
  6  She Is Love            Parachute         134  0.17  on 63 playlists tagged "chill"
  7  Planez                 Jeremih           129  0.56  on 88 playlists tagged "chill"
  8  Bloom - Bonus Track    The Paper Kites    96  0.42  on 90 playlists tagged "chill"

8 tracks · 28 min · 8 artists · long-tail 0% · energy holds steady (0.43 → 0.49)
constraints: {'no_known_explicit': True, 'track_count': True, 'artist_cap': True}
note: explicit flags are known for only 3% of the catalog; known-explicit tracks
      were removed, unlabelled ones could not be checked
```

---

## The result that matters

On **title-only queries** — no seed tracks, nothing for collaborative filtering
to work with, which is the actual natural-language cold-start case:

| System | R-precision | NDCG@100 | Clicks ↓ |
|---|---|---|---|
| **Cadence** | **0.1429 ± 0.0149** | **0.1975 ± 0.0191** | **3.2 ± 1.0** |
| item-kNN baseline | 0.0404 ± 0.0059 | 0.0545 ± 0.0087 | 13.0 ± 1.8 |
| popularity baseline | 0.0404 ± 0.0059 | 0.0545 ± 0.0087 | 13.0 ± 1.8 |
| lexical title matching | 0.0197 ± 0.0075 | 0.0266 ± 0.0099 | 26.2 ± 2.1 |

**3.5x better than popularity**, using the official RecSys Challenge 2018
metrics on 400 held-out playlists. With seeds revealed the margin holds:
at k=5, R-precision **0.2416 ± 0.0189** vs **0.1761 ± 0.0142** for item-kNN
(+37 %).

> Bands are ±2×SE. The headline band, **0.0149**, is also the harness's
> detection floor: a difference smaller than that — which is most of the
> channel weights in `config.py` — cannot be told from sampling noise at this
> sample size. It is printed beside every number so it cannot be missed, not
> because the harness got any sharper.

> Seeds are the playlist's genuine first k. They used not to be — the harness
> handed out the k lowest track ids, which are 1.5-1.7x more popular. Fixing
> that cut the margin over item-kNN from +53 % to +37 %, because the old
> seeds flattered the comparison. See [docs/FINDINGS.md](docs/FINDINGS.md).

Relevance is only half the promise. Across a 20-query battery of constrained
requests, the assembly stage satisfies **100 %** of stated requirements — exact
track counts, per-artist caps, duration targets within 10 %, BPM windows and the
explicit filter — with zero failures, because constraints are enforced as
filters rather than requested in a prompt.

Full protocol, ablations and threats to validity: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

---

## The idea

Track metadata contains no mood. The string *"Holocene — Bon Iver — Bon Iver"*
does not contain the words *calm*, *winter* or *driving*, and no encoder can
extract meaning that is not in the text. So how do you match free text to music?

**Use what humans already wrote.** The corpus contains 98 334 playlists that
people named — `rainy day`, `gym`, `studying`, `90s throwbacks`, `late night
drive`. Those titles describe *how music is used*. Build a track × title-token
matrix, factorise its Shifted PPMI, and a phrase and a song end up in the same
space, grounded entirely in human behaviour.

The ablation is unambiguous. On title-only queries, removing the folksonomy
channels collapses R-precision by **73 %**, while both collaborative channels
*alone* score exactly **0.0000** — with no seeds, CF has nothing to condition
on. Every bit of cold-start performance comes from that bridge. The learned
reranker reached the same conclusion independently: `tag_hits` is its
highest-importance feature by a factor of two.

## The architecture

```
query → PLAN → MASK → RETRIEVE (7 channels) → FUSE → RERANK → SELECT → SEQUENCE → EXPLAIN
```

Two principles do most of the work:

**The LLM expresses intent; it never picks tracks.** Its outputs are a
structured `PlaylistIntent` and the title/description copy. Retrieval selects
every track from the real catalog, which makes track hallucination
*structurally impossible* rather than merely unlikely.

**Hard constraints are filters, not prompts.** "Nothing explicit", "12 songs",
"130–150 BPM" are enforced by a boolean mask and by caps inside the selection
loop. A filter cannot be talked out of its job; an instruction can. Every
response ships a `constraint_report` recording satisfaction per constraint.

| Stage | What it does |
|---|---|
| **Plan** | Free text → `PlaylistIntent`. Deterministic rules by default (49-entry mood lexicon, ~4 ms, no API key); Claude with structured output optionally. |
| **Retrieve** | 7 channels: SPPMI+SVD collaborative embedding, exact co-occurrence, folksonomy embedding, exact tag counts, TF-IDF lexical, audio-feature k-NN, popularity prior. |
| **Fuse + rerank** | Reciprocal rank fusion, then gradient-boosted trees over 30 features (validation AUC **0.849** vs 0.639 for fusion alone). |
| **Select** | MMR diversity, per-artist caps, duration targeting. A playlist is not a top-k list. |
| **Sequence** | Beam search over an energy arc, Camelot-wheel key compatibility and tempo continuity. |
| **Explain** | Per-track reasons grounded in real counts; generated copy validated against the tracklist. |

Details in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**; the reasoning
behind each choice, including the one that was wrong first, in
**[docs/DECISIONS.md](docs/DECISIONS.md)**.

---

## Quickstart

```bash
make venv install          # Python 3.12 + deps
make data                  # download MPD + audio features (~15 GB, filtered down)
make all                   # build → splits → train → reranker → eval  (~10 min)

cadence play "upbeat 90s hip hop for a workout, 15 tracks, nothing explicit"
cadence eval-constraints   # does the assembly stage honour what was asked?
cadence serve              # HTTP API on :8000
```

No API key is needed: the deterministic planner is the default. To use Claude
for open-ended phrasing:

```bash
export ANTHROPIC_API_KEY=...
cadence play "something like the quiet half of a Sofia Coppola soundtrack" --provider anthropic
```

If the LLM is unreachable or returns malformed output, the system falls back to
the rule-based planner and says so in `warnings`. An outage is a quality
regression, not an outage.

### Without the 12 GB feature dump

```bash
python scripts/download_data.py --slices 20 --skip-audio
cadence build --min-tag-playlists 3 && cadence splits --n-eval 200 && cadence train
```

Everything runs; you lose mood targeting and sequencing quality.

### HTTP

```bash
curl -s localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"query":"late night jazz for studying","n_tracks":10}' | jq '.tracks[].track.name'

curl -s localhost:8000/explain -H 'content-type: application/json' \
  -d '{"query":"90s workout bangers"}' | jq '{intent:.intent.themes, channels:.channel_sizes}'
```

---

## What's in here

```
src/cadence/
  text.py            normalisation shared by the offline build and the query path
  catalog.py         preloaded serving view over catalog + learned spaces
  engine.py          orchestrator: plan → retrieve → fuse → rerank → assemble
  explain.py         grounded reasons + the copy hallucination guard
  data/build.py      MPD → catalog, interactions, folksonomy
  models/            factorize.py (SPPMI+SVD) · train.py (spaces) · reranker.py
  retrieval/         channels.py (7 sources) · fusion.py (RRF) · ann.py
  assemble/          select.py (MMR + constraints) · sequencer.py (energy/key/tempo)
  planner/           offline.py (rules) · anthropic_planner.py · lexicon.py
  eval/              metrics.py · splits.py · baselines.py · run_eval.py
  service/api.py     FastAPI
docs/                ARCHITECTURE · EVALUATION · DECISIONS · DATA_CARD · MODEL_CARD
tests/               100 tests; integration tests skip cleanly without artifacts
```

Everything is seeded and reproducible: `make all` from a clean checkout
reproduces every number in the docs.

---

## Honest limitations

- **The explicit filter is best-effort.** The flag is known for only **2.76 %**
  of the catalog. Known-explicit tracks are removed; unlabelled ones cannot be
  checked, and the system says so on every affected response. Do not deploy this
  as a child-safety control.
- **Era targeting is inferred, not factual.** Neither data source carries a
  release date, so `1990s` means "music people file under the 90s", derived from
  playlist titles — not "released 1990–1999".
- **The corpus ends in 2017** and is US-centric and English-language. Non-English
  queries fall back to the lexical and collaborative channels.
- **k=0 evaluation uses playlist titles as queries**, which are shorter and
  tidier than real user input. Evaluation playlists are excluded from the tag
  matrix so the setup is not circular, but the query distribution is friendlier
  than production. Closing that gap is what the LLM planner is for.
- **Many themes dilute each other.** The tag channel embeds a query as the
  *centroid* of its theme vectors. A request combining an activity, a mood, a
  genre and an era ("upbeat 90s hip hop for a workout") averages five tag
  vectors, and the result drifts toward generic workout music rather than
  honouring the genre and era. Weighting identity tags (genre, era) above soft
  ones (mood, activity), or scoring by max-similarity instead of centroid, is
  the obvious fix — but the k=0 evaluation uses playlist titles, which usually
  carry one or two themes, so it would barely register the change either way.
  Shipping it unmeasured would be guessing; it needs an evaluation set of
  multi-constraint queries first.
- **Cold-start items are unsolved.** A track on fewer than four playlists is
  filtered out rather than handled; that needs content-based audio embeddings.
- **The reranker trades diversity for accuracy** — measured, reported, not hidden.

---

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, representation learning, fusion, assembly, failure behaviour |
| [EVALUATION.md](docs/EVALUATION.md) | Protocol, metrics, baselines, ablations, results, threats to validity |
| [DECISIONS.md](docs/DECISIONS.md) | 10 ADRs — why each choice was made, and what it costs |
| [DATA_CARD.md](docs/DATA_CARD.md) | Sources, processing, skews, privacy |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Components, training protocol, risks, maintenance |

## Licence

Apache-2.0. Datasets retain their own terms — see [DATA_CARD.md](docs/DATA_CARD.md).
