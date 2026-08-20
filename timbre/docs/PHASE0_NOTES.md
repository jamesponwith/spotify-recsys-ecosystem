# Phase 0 — implementation notes (pre-flight)

Findings from reading Cadence before writing any Phase 0 code. Everything Phase 0
needs is already on disk; nothing has to be rebuilt.

## Inputs, confirmed present

| What | Where | Shape / note |
|---|---|---|
| 11 audio descriptors | `cadence/data/processed/tracks.parquet` | per-track columns; `has_audio` flag; **but** see caveat below |
| 128-d tag embedding (target) | `cadence/artifacts/spaces.npz` → `tag_track_vectors` | 159,338 × 128 |
| Tag vectors (query side) | same file → `tag_vectors` | 4,569 × 128 |
| Playlist titles | `cadence/data/processed/playlists.parquet` | `name` column |
| Playlist × track | `cadence/artifacts/train_interactions.npz` | holdout-excluded copy |
| Tag vocabulary | `cadence/artifacts/tag_vocab.json` | 4,569 tokens |

Seed is `20260815` (`cadence.config.SEED`); reuse it so splits line up.

## Two things the spec did not anticipate

1. **`AUDIO_FEATURE_COLS` is only 7 columns, not 11.** `models/train.py` deliberately
   excludes `key`/`mode` (categorical — Euclidean distance over pitch class is
   meaningless) and `tempo`/`loudness` are absent from the standardised block too.
   Phase 0 should read the raw 11 from `tracks.parquet` directly rather than reuse
   `audio_z`, and one-hot `key` / handle `mode` as binary. Report the 7-col variant
   as an ablation — it isolates whether the extra four carry the signal.
2. **Retrieval must go through `DenseIndex`, not raw cosine.** `tag_tracks` is an
   L2-normalised index that sets all-zero rows to `-inf` (`ann.py:28`). A predicted
   embedding is never exactly zero, so a naive harness would let `content` compete
   against rows that the real system excludes — inflating it against `oracle`.
   Rebuild the index over the substituted matrix each time.

## Query construction for the gate

`tag_channel` (`retrieval/channels.py:75`) embeds a query as the **mean of its tag
vectors**, not as free text. So a Phase 0 query is: playlist title →
`text.title_tokens()` → tag columns → mean of `tag_vectors[cols]` → cosine against
the (substituted) track matrix. Titles whose tokens are all out-of-vocabulary are
unanswerable and must be dropped *before* scoring, not scored as zero.

## Status

Nothing implemented. Next session starts at the data-loading script.

## Phase 0 needs no new dependencies

Verified in Cadence's venv: `scikit-learn` 1.9.0 with both `Ridge` and
`MLPRegressor`, `numpy` 2.5.2. **PyTorch is absent and not required** — SPEC §6
lists torch as the new dependency, but that is Phase 1's encoder. Phase 0 runs
entirely on Cadence's existing environment, which keeps the falsification test
genuinely cheap: no install, no download, no GPU question.

Implication for layout: do **not** stand up the full SPEC §6 tree yet. Phase 0
needs `src/timbre/config.py` + `src/timbre/phase0/` (features, fit, retrieval,
report) and nothing else. Building the audio scaffolding before Gate 0 rules
would be exactly the scope creep RISKS.md calls R6.

## Query construction, confirmed

`text.title_tokens()` caps at **6 tokens**, emits unigrams + bigrams + canonical
decade tags, and drops a 90-word stopword list. Bigrams are in the tag vocabulary
("rainy day" is its own tag), so token→column lookup must try bigrams too, not
just whitespace splits.

## Data probe — exclusions and design decisions settled

Measured, not assumed:

| Fact | Value | Consequence |
|---|---|---|
| Zero-norm tag embeddings | **125** / 159,338 (0.08%) | `DenseIndex` hides them at `-inf`. Unanswerable under `oracle`, so drop from relevant sets **and** from regression targets — otherwise `content` can score hits `oracle` structurally cannot. |
| Tracks lacking audio | **30** (`has_audio` = 159,308) | Drop; all 11 descriptor columns are NaN on exactly these rows. |
| Degenerate descriptors | `tempo` min 0.0, `loudness` min −60 dB | Real Spotify sentinels for undetectable values, not corruption. Leave as-is; standardisation absorbs them. |
| Holdout playlists | `splits.json` → `holdout_rows`, 2,000 | Never entered the tag factorisation. |

**Two design choices this settles.**

*Queries come from the 2,000 holdout playlists, not from training playlists.*
Their titles never entered the factorisation, so no system gets query-side
contamination. A held-out track's `oracle` embedding still comes from its *other*
appearances — which is precisely the contrast being measured: full history versus
audio alone.

*Regress onto L2-normalised targets.* Retrieval is cosine, so only direction
carries meaning; fitting raw vectors would spend model capacity on norm, which
the metric discards.

*Query definition.* One query per holdout playlist: relevant set = the test-split
tracks it contains, `recall@100 = |top100 ∩ relevant| / |relevant|`. Proper recall
rather than a single-target hit rate, and 2,000 queries keeps the four-system
sweep to a few seconds of BLAS.

## The MLP blew the budget — measured, then capped

The spec promises Phase 0 in "under an hour". The first full run did not come
close, and the cause is worth recording because it is a property of the setup,
not of the machine.

`MLPRegressor`'s default `batch_size=200` over 127,346 training rows is **637
minibatches per epoch**, and with `early_stopping` judging a 128-dimensional
target by R², the improvement per epoch stays above `tol` for a long time. The
uncapped run passed **95 minutes of wall clock / 123 CPU-minutes** without
triggering early stopping, and was killed rather than allowed to finish.

Two changes, both defensible on their own terms:

* `batch_size` 200 → **512**, cutting per-epoch Python/BLAS dispatch overhead by
  ~2.5x on a problem whose gradient is not batch-noise-limited.
* `max_iter` 300 → **60**, exposed as `--mlp-max-iter`.

The cap is reported in the artifact as `hit_iter_cap`, so a truncated fit can
never be mistaken for a converged one. **Ridge remains the unbounded,
fully-converged reference** — it fits in ~60 s with leave-one-out alpha
selection, so the gate is never resting solely on a truncated model.

If the MLP is capped and still beats ridge, the nonlinearity is real and the
reported number is a *lower* bound on it. That is the right direction for a
gate to be biased.

---

## Outcome: Gate 0 failed, and the failure is legible

`content_mlp` recovers **13.2%** of oracle recall@100 against a **25%** bar set
before the experiment ran. The project does not proceed to Phase 1. Full numbers
in [RESULTS.md](RESULTS.md); three things in them are worth more than the verdict.

### 1. Ridge scored exactly 0.0000 at a mean cosine of 0.198

This is the argument for using retrieval as the metric, made concrete. Ridge
lands measurably close to the true embedding on average — a mean cosine of 0.198
against ~0 for random vectors — and retrieves **nothing**, on any of 1,718
queries.

Ridge shrinks toward the conditional mean, so all 31,837 cold tracks are
predicted into a tight cone around one direction. Every one of them is
*near* the truth and none is the *nearest* thing to any query, which is the only
property that survives a top-100 cut against 127,346 tracks holding real,
well-spread embeddings. A cosine-based gate would have passed this model.

### 2. The nonlinearity matters far more than cosine suggests

| | mean cosine | recall@100 |
|---|---:|---:|
| ridge | 0.198 | 0.0000 |
| MLP | 0.243 | 0.0037 |

A **23%** improvement in cosine is the difference between nothing and something.
Anyone tuning this on embedding distance would be optimising a quantity that is
nearly uninformative about the thing they want.

### 3. The random floor is 0.0000, which makes half the gate vacuous

Random unit vectors in 128-d never enter a top-100 against 127k real embeddings,
so the "≥ 3x random" criterion cannot be failed by anything that scores above
zero, and cannot be passed by anything that does not. It carried no information.
The binding criterion was always the oracle ratio.

Recorded as `random_criterion_vacuous: true` rather than quietly dropped. The
design lesson is that a floor worth gating against has to be reachable — the
`mean` baseline was intended to be that and also scored 0.0000.

### What it does not show

Oracle recall@100 is itself only **0.028**: title-only cold-start retrieval is
hard even with perfect embeddings, and 83% of queries get no hit at all from the
tag channel alone. A low ceiling makes the *ratio* the honest statistic, but it
also means the whole experiment sits in a regime where small absolute differences
decide the outcome. A larger `recall_k`, or queries restricted to titles with
richer tag coverage, would move the absolute numbers without changing the
ordering.

### Runtime, after the batch fix

**16.8 minutes end to end**, inside the one-hour budget the spec promised — and
the MLP early-stopped at 53 of 60 epochs, so the cap did not bind
(`hit_iter_cap: false`) and the number is not a truncated fit. Widening the batch
from 200 to 512 cut that stage from **157 minutes to 14.2** *and* improved mean
cosine from 0.2392 to 0.2430.
