# Model card

## Overview

**Cadence** converts a free-text playlist request into an ordered playlist of
real catalog tracks. It is a retrieval-and-assembly system with an optional LLM
front-end, not a generative music model.

**Intended use.** Music discovery and playlist generation research;
demonstration of grounded natural-language retrieval.

**Out of scope.** Not a licensed catalog. Not a content-safety classifier — see
the explicit-flag limitation. Not tuned for any individual user; there is no
per-user personalisation.

## Components

| Component | Type | Training data | Output |
|---|---|---|---|
| Collaborative space | SPPMI + truncated SVD (d=160) | 96 334 training playlists | Track embeddings |
| Tag space | SPPMI + truncated SVD (d=128) | Track × 4 569 title tokens | Joint track/tag embeddings |
| Lexical index | TF-IDF (1–2 grams, 146 k features) | Track/artist/album/genre strings | Sparse vectors |
| Audio space | Standardised Spotify features (7 dims) | 99.98 % of catalog | Z-scored features |
| Reranker | `HistGradientBoostingClassifier`, 30 features | 1 932 held-in playlists → 418 716 labelled pairs | P(relevant) |
| Planner (default) | Rules + 49-entry mood lexicon | — | `PlaylistIntent` |
| Planner (optional) | Claude (`claude-opus-5`), structured output | — | `PlaylistIntent`, copy |

All learned components are fit with a fixed seed (`20260815`) and are exactly
reproducible.

## Training protocol

Evaluation playlists (2 000, drawn once and frozen in `splits.json`) are
excluded from **every** training matrix: the collaborative factorisation, the
rebuilt tag matrix, the served co-occurrence matrix and the reranker's sampling
pool. Without this the model would be scored on playlists it had already read.

The reranker's train/validation split is **by query, not by row**. Candidate
rows from one playlist are highly correlated, so a random row split leaks the
answer across the boundary and inflates AUC.

## Reranker performance

| Metric | Value |
|---|---|
| Validation AUC | 0.849 |
| Validation AUC, fusion score alone | 0.639 |
| Uplift | **+0.210** |
| Labelled pairs | 418 716 (50 683 positive) |

Top features by permutation importance:

| Feature | Importance |
|---|---|
| `tag_hits` | 0.188 |
| `tag_exact_score` | 0.104 |
| `tag_exact_rank` | 0.037 |
| `n_tag_cols` | 0.029 |
| `tag_rank` | 0.023 |
| `collaborative_rank` | 0.016 |

The folksonomy features dominate. That is the central claim of this project
arriving as a measurement rather than an assertion: how humans *title* the
playlists a track appears on is the most informative signal available for
matching free text to music.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for the full protocol, metrics, ablations
and baselines.

## Ethical considerations and risks

**Popularity amplification.** Recommenders trained on playlist co-occurrence
reinforce existing popularity. Mitigated by popularity damping in the PPMI
weighting, MMR diversification and artist caps; measured by long-tail share,
catalog coverage and Gini, all reported alongside accuracy. Not eliminated.

**Explicit content.** The filter is best-effort at 2.76 % flag coverage and is
reported as such on every affected response. Do not deploy this as a
child-safety control.

**Cultural skew.** English-language, US-centric, 2010–2017. Non-English and
post-2017 requests degrade in quality, and the folksonomy vocabulary will not
contain their concepts.

**LLM-specific risks.** The model cannot hallucinate tracks — it has no channel
to emit one. It can hallucinate *prose*, so generated copy is scanned against
the catalog artist index and replaced with template copy when it names an
artist who is not on the playlist.

## Maintenance

Rebuilding is a full refit (~4 minutes end to end). There is no incremental
update path: adding playlists means re-running `make all`. At production scale
this would need to change — see [ADR-002](DECISIONS.md).
