# Timbre — technical specification

Status: **proposal**. Nothing here is implemented.

---

## 1. Problem statement

Given a track for which the system has **no listening history** — no playlist
co-occurrence, no folksonomy tags, possibly not even reliable genre metadata —
produce a representation that lets Cadence retrieve it for an appropriate
natural-language request.

Formally: learn `f: audio → R^d` such that `f(x)` is usable in place of the
track's folksonomy tag embedding `t ∈ R^128`, which Cadence's `tag` channel
currently obtains only by observing the track on many named playlists.

### Success is not "good tag accuracy"

Tag accuracy is an instrument, not the goal. The goal is **retrieval under zero
history**, so the headline number must be a retrieval number measured against an
oracle that has full history. A model with excellent AUC that cannot recover
cold-start tracks has not solved the stated problem.

---

## 2. Why the obvious approach is unavailable

The canonical method (van den Oord, Dieleman & Schrauwen, NIPS 2013) regresses
the *collaborative* latent factors from audio, trained on tracks that have both.
That requires a corpus with audio **and** interaction data for the same items.

No such corpus is available here:

| Corpus | Audio | Playlist / interaction data | Overlaps MPD |
|---|---|---|---|
| Spotify MPD (Cadence's corpus) | ✗ | ✓ 98,334 playlists | — |
| MagnaTagATune | ✓ 25,863 clips | ✗ (tags only) | ✗ |
| MTG-Jamendo | ✓ 55k tracks | ✗ (tags only) | ✗ |
| FMA | ✓ up to 106k | ✗ | ✗ |
| Spotify preview URLs | restricted for new applications | — | — |

Consequence: **the bridge between audio and Cadence must be a shared
*vocabulary*, not shared *items*.** That is the central design commitment, and
it is the thing most likely to be wrong. Section 5 and
[EVALUATION.md](EVALUATION.md) are built around testing it rather than assuming
it.

---

## 3. Data

### 3.1 MagnaTagATune — encoder training

| Property | Value |
|---|---|
| Clips | 25,863 (29 s, MP3) |
| Archive | 2.97 GB, `confit/magnatagatune` on HuggingFace |
| Annotations | 188 binary tags; the standard benchmark uses the **top 50** |
| Split | The conventional folder split: `0–b` train, `c` validation, `d–f` test |
| Usable after filtering | ≈ 21 k (clips with no top-50 tag are conventionally dropped) |

Chosen over the larger MTG-Jamendo because it is the **standard auto-tagging
benchmark**, which means published baselines exist to compare against — turning
"my model gets 0.89" into "my model gets 0.89 against a published 0.91 at a
fraction of the parameters", which is a claim rather than a number.

Known data-quality issues to handle explicitly: a small number of MP3s in the
archive are zero-length or truncated, and tag frequencies are extremely
long-tailed. Both are standard and both must be logged, not silently dropped.

### 3.2 Cadence artifacts — the target space

Consumed read-only from `../cadence`:

| Artifact | Use |
|---|---|
| `data/processed/tag_vocab.json` | 4,569-tag folksonomy vocabulary |
| `data/processed/tags.npz` | track × tag counts, for bridge weighting |
| `artifacts/spaces.npz` → `tag_vectors` | 128-d tag embeddings — the target space |
| `artifacts/spaces.npz` → `tag_track_vectors` | track embeddings, for the oracle |
| `data/processed/tracks.parquet` | Spotify audio features, for Phase 0 |

**Timbre never writes into Cadence.** It reads artifacts and, at the
demonstration stage, produces additional rows that Cadence can index. Cadence
stays a dependency, not a fork.

### 3.3 The measured bridge

31 of the MTAT top-50 tags appear in Cadence's folksonomy (62%). By tracks
carrying the tag in Cadence:

```
rock 22,234 · dance 14,451 · pop 14,318 · country 12,288 · slow 11,601
classic 9,519 · beats 7,751 · soft 5,376 · metal 4,415 · electronic 3,894
quiet 2,732 · classical 2,199 · beat 2,015 · weird 1,478 · loud 1,438 …
```

Absent (19): `drums, opera, male, no vocals, harpsichord, flute, male vocal,
no vocal, sitar, solo, choir, voice, male voice, female vocal, harp, cello,
no voice, female voice, choral`.

The pattern is clean: **genre, tempo and texture concepts survive the bridge;
instrument and voice-type concepts do not**, because people name playlists after
moods and activities, not after instrumentation. This asymmetry is a finding in
its own right and should be reported, not smoothed over.

---

## 4. Model

### 4.1 Front end

Follows the reference configuration used by the published baselines, so results
are comparable:

| Parameter | Value |
|---|---|
| Sample rate | 16 kHz mono |
| Window / hop | `n_fft` 512 / hop 256 |
| Mel bands | 128 |
| Chunk | 3.69 s → **128 × 231** log-mel |
| Normalisation | per-bin mean/var from the training split only |

Spectrograms are precomputed once and cached. Cached as `float16`, the full
corpus is ≈ 6 GB; an `int8` quantised cache with per-clip scale halves it again
and is worth doing if disk pressure appears.

### 4.2 Encoder — short-chunk CNN

Seven 3×3 conv blocks (conv → batch-norm → ReLU → 2×2 max-pool), then global
max- and average-pool concatenated, a 256-unit dense layer with dropout 0.5, and
a 50-way sigmoid head.

```
channels:  1 → 32 → 32 → 64 → 64 → 128 → 128 → 256
head:      concat(GMP, GAP) = 512 → 256 → 50   (sigmoid)
params:    ≈ 0.73 M
loss:      binary cross-entropy, multi-label
optimiser: Adam, lr 1e-3, cosine decay, early stop on validation PR-AUC
```

**Deliberately ~7× smaller than the published baselines.** A small deficit
against them is expected and is itself the interesting measurement: it
quantifies what the parameter budget buys. Reporting a smaller model that lands
close is a stronger result than reporting a big model with no reference point.

Inference over a full 29 s clip averages the sigmoid outputs of overlapping
chunks — the standard protocol, and it must match between validation and test or
the benchmark comparison is void.

### 4.3 Projection into Cadence's space

The encoder emits `p ∈ [0,1]^50`, a posterior over MTAT tags. Mapping that to a
128-d Cadence tag embedding:

```
t̂ = Σ_{j ∈ bridge}  w_j · p_j · v_j        v_j = Cadence tag vector for tag j
```

Three candidate weightings for `w_j`, to be chosen empirically in Phase 2:

1. **Uniform** over the 31 bridged tags — the baseline.
2. **IDF-style**, down-weighting tags carried by very many Cadence tracks, so
   `rock` (22,234 tracks) does not swamp `ambient` (822).
3. **Learned** — a ridge regression from `p` to `t`, fitted on any tracks where
   both are known. Strongest in principle, but see the Phase 0 caveat: fitting
   it needs paired data that only Phase 0's proxy provides.

Whichever wins, the calibration of `p` matters more than the architecture here,
so per-tag threshold/temperature calibration on the validation split is part of
the deliverable, not an afterthought.

### 4.4 A second head worth having

Alongside the 50 tag logits, a small regression head predicting **Spotify-
comparable descriptors** — energy, acousticness, instrumentalness, tempo. Cadence
already indexes these in its `audio` channel, so predicting them gives a *second,
independent* injection route that needs no vocabulary bridge at all.

Caveat that must be stated wherever this is reported: without a corpus carrying
both raw audio and Spotify's proprietary features, agreement between predicted
and true Spotify features **cannot be measured**. Tempo is the exception —
it is verifiable against DSP beat tracking. The other three are trained against
proxy targets derived from the audio (RMS loudness, spectral flatness, the
encoder's own vocal/non-vocal tags) and are therefore *plausible but unvalidated*.
Ship them behind a flag, off by default, until that changes.

---

## 5. Integration with Cadence

Cadence's engine already accepts candidate contributions per channel, so
integration is additive rather than invasive:

```python
# proposed — a new Cadence channel, populated by Timbre
def content_channel(catalog, intent, k, mask):
    """Retrieval over content-predicted tag embeddings for history-less tracks."""
```

Cold-start tracks enter as ordinary catalog rows whose `tag_track_vectors` come
from `f(audio)` rather than from observation, and carry a `has_history = False`
flag so that:

- explanations say *"placed by audio similarity"*, never *"on 1,204 playlists
  tagged chill"* — a grounded-explanation system must not fabricate evidence for
  a track that has none;
- evaluation can always separate observed from inferred placements;
- ranking can apply a deliberate confidence discount to inferred rows.

---

## 6. Proposed layout

```
timbre/
  README.md
  docs/            SPEC · EVALUATION · RISKS · MILESTONES
  src/timbre/
    config.py         seeds, paths, all hyperparameters in one place
    data/
      download.py     fetch + verify MTAT
      annotations.py  top-50 selection, official split, corrupt-clip handling
      melspec.py      decode → log-mel → cache (soundfile, no ffmpeg)
      dataset.py      chunk sampling, augmentation, batching
    models/
      cnn.py          short-chunk CNN
      train.py        loop, early stopping, checkpoints
      calibrate.py    per-tag temperature/threshold fitting
    bridge/
      vocabulary.py   MTAT ↔ Cadence tag alignment + weighting schemes
      project.py      posterior → 128-d Cadence tag embedding
    eval/
      tagging.py      ROC-AUC / PR-AUC vs published baselines
      coldstart.py    the Phase 3 held-out retrieval experiment
      report.py       render results to markdown
    cli.py            typer entry points
  tests/
```

Mirrors Cadence's conventions deliberately: `uv` + Python 3.12, `ruff` + `mypy`
clean, `pytest` with unit tests hermetic and integration tests skipping when
artifacts are absent, `make all` reproducing every number from a fixed seed.

New dependency beyond Cadence's set: **PyTorch (CPU wheel)**, plus `soundfile`
for decoding. No `librosa` — the mel front end is ~40 lines of `torch.stft` and
avoiding the dependency keeps the audio path inspectable and fast.

---

## 7. Compute budget

No GPU. 8 CPU cores, ~16 GB usable RAM, ~83 GB free disk.

| Stage | Estimate |
|---|---|
| Download MTAT | ~6 min at observed throughput |
| Decode + mel cache (25.8 k clips, 8 workers) | ~15–20 min |
| Mel cache on disk | ~6 GB `float16` (~3 GB `int8`) |
| Training, one random chunk per clip per epoch | **~5.5 min/epoch** |
| 40 epochs | **~3.7 h**, backgroundable |
| Phase 0 falsification test | < 1 h total, no audio needed |

If training proves slower than estimated, the levers in order of preference are:
subsample the training set, halve the mel resolution to 64 bands, drop hop to
512 — the last one costs benchmark comparability and should be the final resort.
The parameter count is *not* a lever; shrinking further undermines the
comparison the benchmark exists to support.
