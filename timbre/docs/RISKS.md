# Timbre — risks and rejected alternatives

Ranked by expected damage. Each carries a **kill criterion** — the observation
that would mean stopping rather than pushing on — because a plan without one
tends to absorb any result as encouragement.

---

## R1 — Domain shift between MagnaTagATune and MPD repertoire

**Severity: high. Likelihood: high. This is the one that decides the project.**

MagnaTagATune is Magnatune's Creative Commons catalogue: ambient, classical,
electronic, world, and independent rock. MPD is mainstream US streaming,
2010–2017: pop, hip-hop, R&B, country, rock. An encoder trained on the first and
applied to the second is performing unmeasured transfer, and the genres are not
merely different — production conventions, loudness, and vocal presence all
differ systematically.

**Mitigations**
- Rung 3 reports `audio-features` alongside `timbre`. Because the former uses
  native Cadence descriptors, the gap between them *is* the transfer cost, made
  visible rather than assumed away.
- Restrict the bridge to concepts with support in both corpora; report per-tag
  results so failures are attributable.
- If transfer proves poor, MTG-Jamendo (55 k tracks, broader and more
  contemporary) is the fallback training corpus, at the cost of losing the
  published benchmark comparison.

**Kill criterion**
> `timbre` fails to beat the `mean` no-information floor in Rung 3 while
> `audio-features` clears it comfortably. That isolates the failure to transfer
> rather than to the premise, and the correct response is to report it as a
> negative transfer result — a genuinely useful finding — not to keep tuning.

---

## R2 — The premise is wrong: playlist-title vocabulary is not acoustic

**Severity: fatal. Likelihood: moderate. Detected in Phase 0, before any cost.**

People name playlists after *context* — `gym`, `study`, `road trip`, `sunday
morning` — and context is not a property of the waveform. A calm song and a loud
song can both be filed under `driving`. If the folksonomy is mostly social
convention, audio cannot predict it.

**Mitigation:** this is exactly what
[Phase 0](EVALUATION.md#phase-0-the-falsification-test) is for. It costs under an
hour and uses only data already on disk.

**Kill criterion**
> Gate 0 fails: content-predicted embeddings reach under 3× the random floor or
> under 25% of oracle on recall@100. Stop, write up the null result, and pivot to
> the collaborative target or to the sequential-transformer project.

---

## R3 — CPU training is slower than estimated

**Severity: moderate. Likelihood: moderate.**

The ~5.5 min/epoch estimate is derived from parameter count and input size, not
measured. Real PyTorch CPU throughput on this machine is unknown until tried, and
could plausibly be 2–3× worse.

**Mitigations, in order of preference**
1. Subsample the training set — costs statistical power, keeps comparability.
2. Halve mel resolution to 64 bands — costs some accuracy, keeps the protocol.
3. Increase hop to 512 — **costs benchmark comparability**; last resort.

Shrinking the model further is *not* on the list: the parameter count is load-
bearing for the "small model, near-published results" claim.

**Kill criterion**
> A measured epoch exceeds 25 minutes even after mitigations 1 and 2, making a
> 40-epoch run over 16 hours. At that point rent a GPU for an afternoon or
> descope to a frozen pretrained encoder used as a feature extractor.

---

## R4 — Corrupt and mislabelled MagnaTagATune data

**Severity: low. Likelihood: certain.**

The archive contains a small number of zero-length or truncated MP3s, and the
annotations contain near-duplicate tags with inconsistent usage (`vocal` /
`vocals` / `voice`, `no vocal` / `no vocals` / `no voice`).

**Mitigation:** validate every decode, log and exclude failures explicitly with
counts in the build metadata, and follow the conventional top-50 synonym merge
so the split matches the published protocol. Silent dropping is the failure mode
to avoid — a benchmark comparison against a quietly different subset is not a
comparison.

**Kill criterion:** none. This is hygiene, not a risk to the thesis.

---

## R5 — Cold-start injection amplifies popularity bias, or hides behind it

**Severity: moderate. Likelihood: moderate.**

Cadence already draws heavily from the popular head (long-tail share 0.057).
Content-placed tracks land wherever their predicted embedding puts them, which
may be a dense region full of popular tracks — where they will never outrank
established items. A cold-start system that technically indexes new tracks but
never surfaces them has not solved anything.

**Mitigation:** report coverage and Gini for injected tracks separately, and
measure *surfacing rate* — how often an injected track appears in a top-100 at
all — not just whether it is indexed.

**Kill criterion**
> Injected tracks appear in fewer than 5% of top-100 result sets across the
> query battery. That means the placement works and the ranking buries it, which
> is a ranking problem and should be attacked as one rather than dressed up as a
> cold-start success.

---

## R6 — Scope creep into a second Cadence

**Severity: moderate. Likelihood: high.**

Timbre touches retrieval, so there is constant temptation to re-litigate
Cadence's ranking, sequencing and serving decisions inside it.

**Mitigation:** Timbre owns exactly one thing — *placing history-less tracks into
Cadence's semantic space*. Retrieval, ranking, assembly and serving stay in
Cadence. Timbre reads Cadence's artifacts and never writes to them.

---

## Rejected alternatives

### Scraping YouTube for MPD audio — rejected

Matching MPD tracks to YouTube and downloading audio with `yt-dlp` would create
the paired corpus that would make this project straightforward, and it is
rejected on two independent grounds:

1. **Terms and copyright.** It is a bulk download of commercial music against the
   host's terms of service, for a portfolio project. Neither the scale nor the
   purpose makes that acceptable, and building an interview piece on a licence
   violation is a poor way to demonstrate judgment.
2. **Reproducibility.** Video availability, audio codecs and rate limits change
   constantly. Any number produced this way would be unreproducible within
   months, which defeats the point of a seeded, re-runnable pipeline.

The paired-corpus gap is a real constraint. The response is to design an
evaluation honest about it, which is what [EVALUATION.md](EVALUATION.md) does.

### Spotify preview URLs — rejected as unreliable

30-second previews would be the licensed route, but `preview_url` availability
has been restricted for newly registered applications, so a pipeline depending
on it cannot be reproduced by a reader. If access is available, it is the
**best** path and would collapse the entire bridge problem — the spec should be
revisited immediately if that changes.

### A large pretrained audio encoder (MusicFM, CLAP, MERT) — deferred, not rejected

Using frozen embeddings from a pretrained music model would very likely beat a
0.73 M-parameter CNN trained from scratch, and is the right engineering choice
for a product.

Deferred here because the point of this project is to demonstrate *training* an
audio model end to end, which is the capability Cadence does not evidence. It
belongs in the evaluation as a **strong reference row** — "frozen pretrained
encoder + the same projection head" — to show the from-scratch model's cost
honestly. Adding that row is a stretch goal, not a substitute.

### Training on MTG-Jamendo instead of MagnaTagATune — held as fallback

Larger (55 k), more contemporary, richer mood/theme vocabulary that would bridge
better to Cadence's folksonomy. Not the default because it lacks the standard
benchmark with published baselines, which is what turns Rung 1 into a claim
rather than an unanchored number. Promote it if R1 materialises.
