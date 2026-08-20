# Timbre — phased plan

Five phases. Phases 0-3 each end in a **gate** that can stop the project;
Phase 4 is the write-up and is ungated. Effort is in
focused working sessions, not calendar time, and assumes the CPU-only machine
described in [SPEC.md §7](SPEC.md#7-compute-budget).

The ordering is deliberate: **the cheapest thing that can falsify the premise
runs first**, and the expensive audio work is not started until it survives.

---

## Phase 0: falsify the premise · ~1 session

No audio. No PyTorch. Only data already on disk.

- [x] Load Cadence's audio features and tag embeddings for 159,338 tracks
- [x] Fit ridge + small MLP: 11 features → 22-d design matrix → 128-d tag
      embedding, seeded 80/20 split by track
- [x] Build the held-out retrieval harness (`random` / `mean` / `content_ridge` /
      `content_mlp` / `oracle`), rebuilding the index per system
- [x] Report recall@100 and the oracle recovery ratio → [RESULTS.md](RESULTS.md)

Implemented in `src/timbre/phase0/`; notes on the exclusions, the contamination
guards and the runtime budget are in [PHASE0_NOTES.md](PHASE0_NOTES.md).

**Gate 0 —** proceed only if `content` ≥ 3× `random` **and** ≥ 25% of `oracle`.

> ### ✗ Gate 0 failed — 2026-08-16
>
> `content_mlp` reached **13.2%** of oracle recall@100. The `random` floor came
> out at exactly 0.0000, which made the first criterion vacuous (recorded as
> `random_criterion_vacuous`), so the verdict rested on the oracle ratio alone.
>
> **Phases 1–4 below were not started.** They are left in place as the plan that
> the gate was protecting, which is the point of having had one.
>
> Results: [RESULTS.md](RESULTS.md) · reading: [PHASE0_NOTES.md](PHASE0_NOTES.md)

> Failing here is a *good outcome for the cost*: one session spent to avoid three
> weeks, plus a genuine finding about how much of playlist-title vocabulary is
> acoustic rather than social. Write it up either way.

---

## Phase 1 — the encoder · ~3–4 sessions · **not started (Gate 0 failed)**

- [ ] Download and verify MagnaTagATune (2.97 GB); log corrupt clips explicitly
- [ ] Top-50 tag selection and the conventional folder split, matching the
      published protocol exactly
- [ ] Mel front end on `torch.stft` (no `librosa`); cache as `float16`
- [ ] Short-chunk CNN, ~0.73 M params; training loop with early stopping
- [ ] Chunk-averaged inference; ROC-AUC and PR-AUC, macro and per-tag
- [ ] Compare against published baselines, citing figures from the paper

**Gate 1 —** macro ROC-AUC ≥ **0.86**. Below that the encoder is not learning
enough for anything downstream to be interesting, and the failure is in the
training setup rather than in the thesis — fix it before continuing.

*Deliverable even if everything later fails: a from-scratch music tagger
benchmarked against the literature. That stands on its own.*

---

## Phase 2 — the bridge · ~2 sessions

- [ ] Vocabulary alignment; report coverage by count **and** by tag mass
- [ ] Per-tag calibration (temperature scaling); reliability curves, ECE
      before and after
- [ ] Implement all three projection weightings — uniform, IDF, learned
- [ ] Select on validation; report all three rather than only the winner
- [ ] Neighbourhood-preservation check: do projected embeddings land near the
      right Cadence tracks?

**Gate 2 —** the projection beats the `mean` floor on neighbourhood overlap. If
the posterior carries signal but the projection destroys it, the fault is the
bridge and it should be fixed here, not compensated for downstream.

---

## Phase 3 — cold-start retrieval · ~2–3 sessions

The headline experiment.

- [ ] Sample 2,000 catalog tracks with ≥ 20 playlists; freeze the selection
- [ ] Strip their interactions and tag rows; retrain both spaces (~100 s)
- [ ] Score all five systems: `random`, `mean`, `audio-features`, `timbre`, `oracle`
- [ ] recall@100 / @500, MRR, coverage, Gini, surfacing rate
- [ ] Report the **oracle recovery ratio** as the headline
- [ ] Ablations: calibration on/off, weighting scheme, bridged-only vs all-50

**Gate 3 —** `timbre` beats the `mean` floor. If it does not while
`audio-features` does, that is Risk R1 (domain shift), and the honest response is
to report negative transfer rather than to tune toward a number.

---

## Phase 4 — integration and write-up · ~2 sessions

- [ ] `content_channel` in Cadence, behind a flag, reading Timbre's artifacts
- [ ] `has_history = False` propagated through explanations so no injected track
      is ever credited with playlist evidence it does not have
- [ ] Rung 4 injection demo: real MTAT clips, full printed output, labelled
      qualitative
- [ ] Docs matching Cadence's standard: README, SPEC, EVALUATION, DECISIONS,
      DATA_CARD, MODEL_CARD
- [ ] `make all` reproduces every number from a clean checkout

---

## Total

**~10–12 focused sessions**, with the falsification gate at ~1 session and the
first standalone-publishable artifact (the benchmarked tagger) at ~5.

## What "done" looks like

A reader should be able to answer three questions from the repository alone:

1. **Does Cadence's cold-start gap actually close, and by how much?**
   One number: the oracle recovery ratio, with its ceiling stated.
2. **Is the encoder any good on its own terms?**
   ROC-AUC and PR-AUC against published baselines, at a stated parameter count.
3. **Where does it still fail?**
   Domain shift, the instrument/voice half of the vocabulary that does not
   bridge, and the fact that Rung 4 has no ground truth.

## Interaction with Cadence

Cadence is a **read-only dependency** throughout. Timbre consumes its artifacts
and, in Phase 4, adds one optional channel behind a flag. If Timbre is abandoned
at any gate, Cadence is untouched and still works.
