# Timbre

**Content-based cold start for [Cadence](../cadence). Phase 0 built and run — [Gate 0 failed](#status), and the project stops there.**

Cadence retrieves music by learning from how humans *use* it: which playlists a
track appears on, and what those playlists are called. That works, and its own
evaluation says where it stops working:

> **Cold-start items are unsolved.** A track on fewer than four playlists is
> filtered out rather than handled; that needs content-based audio embeddings.
> — `cadence/README.md`

Timbre is the project that closes that gap. Cadence knows what a track *means to
listeners*; Timbre learns what a track *sounds like*, so a song with no listening
history can still be found.

The name is the thesis. Timbre is the quality of a sound that is left once you
remove pitch and loudness — the part that is genuinely in the audio and nowhere
in the metadata.

---

## Why this gap is worth closing

Not a rounding error. From Cadence's own build:

| | |
|---|---|
| Distinct tracks seen in the corpus | 679,889 |
| Tracks that survive `min_playlists ≥ 4` | 159,338 |
| **Tracks discarded as too sparse** | **520,551 (76.6%)** |
| Tracks *inside* the catalog with < 10 playlists | 85,430 (**53.6%**) |
| Median playlist count per catalog track | **9** |

The filter is not hiding a small tail. It hides three quarters of the music, and
over half of what remains is thinly observed. A brand-new release has zero
history and is invisible to every channel Cadence has except lexical string
matching.

This is the shape of the problem at any streaming service: catalog grows
continuously, listening evidence accrues slowly, and the newest music is the
music you most need to route to someone.

---

## What Timbre proposes

Train an audio encoder that maps a waveform to the **same semantic space Cadence
already retrieves in**, so a track with no listening history can be placed in the
catalog and served.

```
   raw audio ──▶ log-mel ──▶ CNN ──▶ tag posterior ──▶ Cadence tag space
   (no history)                                              │
                                                             ▼
                                              retrievable for "chill acoustic",
                                              "90s rock", "workout" …
```

Two things make this more than a diagram:

1. **A measured vocabulary bridge.** 31 of the 50 tags in the standard
   MagnaTagATune benchmark already exist in Cadence's 4,569-tag folksonomy
   (62%), and they are the heavy ones — `rock` (22,234 tracks), `dance`
   (14,451), `pop` (14,318), `country` (12,288), `slow` (11,601). The 19 that
   fail to map are almost entirely instrument and voice-type tags
   (`harpsichord`, `cello`, `choral`, `male voice`) — the category people simply
   do not put in playlist titles. The mismatch is systematic and explainable,
   not noise.

2. **A falsification test before any of it gets built.** Phase 0 asks whether
   the folksonomy is predictable from acoustic content *at all*, using data
   already on disk, in about an hour. If it is not, the project stops there.
   See [docs/EVALUATION.md](docs/EVALUATION.md#phase-0-the-falsification-test).

---

## The honest constraint, stated up front

**There is no corpus with both raw audio and Spotify playlist co-occurrence.**
Spotify preview URLs are no longer dependable for new applications, and the
audio-bearing corpora (MagnaTagATune, FMA, MTG-Jamendo) are Creative Commons
catalogues that do not overlap MPD's mainstream US repertoire.

So Timbre cannot regress Cadence's collaborative embedding directly from audio
for the same tracks, which is what the canonical approach
([van den Oord et al., 2013](https://papers.nips.cc/paper/2013/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html))
did. Every downstream claim is scoped to that limit, and the evaluation is built
as a ladder where each rung says exactly what it does and does not establish
([docs/EVALUATION.md](docs/EVALUATION.md)).

Scraping YouTube to manufacture the missing pairs is **explicitly rejected** —
see [docs/RISKS.md](docs/RISKS.md#rejected-alternatives).

---

## Running it

Timbre depends on Cadence's built artifacts. Build those first (`make all` in
`../cadence`), then:

```bash
make install    # venv + Cadence and Timbre installed editable
make phase0     # the falsification test; rules on Gate 0
make demo       # the joint Cadence + Timbre cold-start demo
make report     # renders artifacts/results.html and docs/RESULTS.md
```

Every number in `docs/RESULTS.md` and `artifacts/results.html` is read out of the
JSON the code writes, so the prose and the measurements cannot drift apart.

```
src/timbre/
  config.py        paths, seed, all hyperparameters and the Gate 0 thresholds
  predictor.py     persisted audio -> folksonomy model; Phase 1's CNN slots in here
  demo.py          freeze-out / graft surgery on a live Cadence catalog
  report.py        the joint demo across a query battery
  phase0/
    features.py    11 Spotify descriptors -> 22-d design matrix
    data.py        Cadence artifacts -> split, targets, held-out queries
    fit.py         ridge + MLP onto the L2-normalised tag embedding
    retrieval.py   batched cosine scorer mirroring Cadence's DenseIndex
    run.py         the experiment, and the Gate 0 verdict
scripts/
  build_report.py  JSON -> results.html + docs/RESULTS.md
  _theme.py        design tokens; palette is validator-checked in both modes
```

---

## Documents

| Document | Contents |
|---|---|
| [SPEC.md](docs/SPEC.md) | Architecture, data, model, interfaces, proposed layout, compute budget |
| [EVALUATION.md](docs/EVALUATION.md) | The validation ladder, metrics, baselines, oracle bounds, decision gates |
| [RISKS.md](docs/RISKS.md) | Ranked risks with kill criteria; rejected alternatives and why |
| [MILESTONES.md](docs/MILESTONES.md) | Five phases with go/no-go gates and effort estimates |
| [PHASE0_NOTES.md](docs/PHASE0_NOTES.md) | Implementation notes for Phase 0: exclusions, contamination guards, settled design choices |

## Status

**Phase 0 is built and run. Gate 0 failed. The project stops here.**

The falsification test says the premise does not hold at the strength the project
needs. `content_mlp` recovers **13.2%** of oracle recall@100; the bar, set before
the experiment ran, was 25%. Phase 1 does not start. Full numbers in
[docs/RESULTS.md](docs/RESULTS.md), interpretation in
[docs/PHASE0_NOTES.md](docs/PHASE0_NOTES.md).

| System | recall@100 | share of oracle |
|---|---:|---:|
| random unit vector | 0.0000 | 0% |
| catalog mean | 0.0000 | 0% |
| audio → ridge | 0.0000 | 0% |
| audio → MLP | 0.0037 | **13%** |
| true embedding (oracle) | 0.0281 | 100% |

### The result worth keeping

**Ridge reached a mean cosine of 0.198 with the true embeddings and retrieved
nothing at all** — zero hits across 1,718 queries. It shrinks every cold track
into a tight cone near the conditional mean: close to the truth on average, never
the nearest thing to any query, which is the only property a top-100 cut rewards.

That is the entire argument for scoring this with retrieval instead of cosine,
demonstrated rather than asserted. A cosine-based gate would have passed this
model and sent the project into three weeks of audio work.

The corollary: going from ridge to an MLP improves cosine by 23% and improves
retrieval from *nothing* to *something*. Tuning a content encoder on embedding
distance would optimise a quantity nearly uninformative about the goal.

### The joint demo

`timbre demo-report` runs the integration end to end: Cadence answers a query,
its own top 20 picks are stripped of every history-derived signal, and Timbre
hands back an embedding predicted from audio alone.

The mechanism works — 9 of 120 frozen tracks go from unranked to ranked. The best
lands at rank 1,058 against a cut of 200. **Timbre recovers zero tracks that
Cadence's content-based channels did not already recover on their own.** Gate 0's
verdict reappearing exactly where it should.

### What would come next, if it were continued

The pivots are named in [docs/EVALUATION.md](docs/EVALUATION.md): predict the
**collaborative** embedding instead of the folksonomy one; restrict the target to
the genre and texture tags the vocabulary bridge actually covers; or accept that
playlist-title language encodes social context more than sound, and move to the
sequential-transformer project instead.

The result is recorded rather than retried until it passes. A gate that only ever
returns the answer you wanted is not a gate.

### Verified along the way

- MagnaTagATune audio (25,863 clips, 2.97 GB) is publicly downloadable and
  decodes via `soundfile` 1.2.2, which bundles MP3 support — no `ffmpeg`.
- Phase 0 runs end to end in **16.8 minutes** on CPU, inside the hour the spec
  promised. Getting there required a real fix: sklearn's default `batch_size=200`
  put the MLP at 157 minutes, and widening it to 512 cut that to 14.2 while
  *improving* mean cosine. The MLP early-stopped at 53 of 60 epochs, so the cap
  did not bind and the number is not a truncated fit.
