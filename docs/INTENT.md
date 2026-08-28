# Intent

What is open in this ecosystem, what each piece of work would establish, and —
where it matters more — what it would **not**.

Five research agents were fanned out across the open threads. Every claim below
that came back from one was re-verified here before it was written down; where my
verification disagreed with the agent, both numbers are given. Two of the findings
are **errors in already-published work** and are corrections, not proposals.

Tracked as beads under the `spot-` prefix. `bd list` for status.

---

## 1. Corrections to work already published

These are not proposals. The pages are live and wrong.

### 1.1 Gamut's funnel measures the wrong window

`gamut/src/gamut/config.py` sets `depth = 100`. `cadence/src/cadence/config.py`
sets `fused_candidates = 1500`. Gamut measured the top 100 of a 1500-deep pool and
labelled that bar **"Reached by retrieval"**.

Union catalog reach, replayed over the same battery:

| stage | reach |
|---|---:|
| what `retrieve()` actually returns (1500) | **70.6%** |
| what `select()` actually reads (500) | 38.4% |
| what Gamut measured and labelled "retrieval" (100) | 10.7% |
| shown to the listener (20) | 3.0% |

*Verified independently at n=60 (a lower bound, since reach grows with query
count): 26.9% / 11.4% / 2.7% at depths 1500 / 500 / 100 — the same
order-of-magnitude error.*

So the published sentence **"89% of the catalog never becomes a candidate"** is
arithmetic on the wrong denominator. The real figure is ~29%. The headline
finding — that concentration is upstream of ranking — survives directionally, but
it was measured on a window 15× shallower than the pool, and a pilot at the true
depth pushes artist Gini to **0.9321**, past the 0.936 floor the page calls the
limit of all thirteen interventions. It costs 62% of R-precision instead of 15%,
so the conclusion holds; the number does not.

### 1.2 Gamut's per-channel attribution is contaminated

`gamut/src/gamut/audit.py:38` treats `ranks[i] >= 0` as "this channel returned this
candidate". But `cadence/src/cadence/retrieval/fusion.py:78` fills absent
candidates with `depth + 1`, not `-1` — so the sentinel passes the filter and each
"channel" block is the whole pool re-sorted.

Measured contamination of stored channel ranks: tag 33.7%, cooccurrence 52.0%,
lexical 57.1%, tag_exact 64.3%, collaborative 66.0%, popularity 71.5%, audio 88.3%.

This is why lexical's published row (R-prec 0.0227, Gini 0.953, top-1% 43.7%) is
indistinguishable from the baseline row (0.0229, 0.951, 43.6%) — **it is the
baseline row**. The `tag_exact` headline is unaffected, so "the most accurate
channel is also the most concentrated" stands.

### 1.3 Cadence's evaluation tag matrix contains the answer key

`Catalog.load` reads `data/processed/tags.npz`, built over all 98,334 playlists
**including the 2,000 held out**. So when a held-out challenge asks for `rock`, a
track can be credited for carrying that tag by the very playlist being scored
against.

For 400 sampled held-out playlists: the median share of a track's *requested-tag*
credit supplied by its own scored playlist is **33.3%**, and for **35.4%** of
tracks the entire credit is self-supplied.

This touches `sparse_tag_channel`, the reranker's top feature `tag_hits`
(importance 0.188), and `tag_adherence` itself. `models/train.py::_rebuild_tags`
already builds a holdout-free matrix in ~12s and is simply never persisted.

> My first measurement of this got 1.9% and nearly dismissed it. I had computed
> share of *total* tag mass rather than of the requested tags — the wrong
> denominator, which is the same error as §1.1, made twenty minutes after
> catching it there.

---

## 2. The live decision: mood fidelity vs genre fidelity

The open question was whether to trust Spotify's audio descriptors or the
folksonomy when they disagree about what "chill" means. The research changed the
question.

### The proposed decisive experiment is not decisive

Held-out playlist membership cannot arbitrate. The ground truth is co-filing, and
the folksonomy *is* co-filing aggregated — asking it to choose between co-filing
and audio adherence is asking a ruler which of two things is longer when one of
them is the ruler. It can price the audio term's cost; it can never credit its
benefit.

### But its control is decisive, and it is damning

Running the same weights with the affinity vector **randomly permuted**
(n=60, paired, precision@20 against withheld tracks):

| weight | real affinity | shuffled | delta (t) |
|---:|---:|---:|---|
| 0.20 | .1242 | .1117 | +0.0125 (1.37) |
| **0.35** | .0967 | .1000 | **−0.0033 (−0.37)** |
| 0.60 | .0442 | .0742 | −0.0300 (−3.18) |
| 0.80 | .0092 | .0408 | −0.0317 (−3.97) |

**At the shipped 0.35 the audio term is indistinguishable from a random
permutation of itself.** Above 0.45 it is significantly worse than noise.

### And the target may be the real problem, not the weight

Of 106 `MOOD_LEXICON` (word, dimension) pairs where the word is also a real tag,
**72 assert a target further from where humans actually file that word than the
plain catalog mean is** — worse than doing nothing:

| word | dimension | asserted | humans | catalog | tracks |
|---|---|---:|---:|---:|---:|
| sleep | energy | 0.12 | **0.509** | 0.632 | 9,115 |
| study | instrumentalness | 0.62 | **0.279** | 0.099 | 11,088 |
| acoustic | acousticness | 0.85 | **0.481** | 0.270 | 5,283 |
| happy | valence | 0.85 | **0.536** | 0.472 | 12,558 |

The audio term may be performing no better than noise because it is aimed
somewhere humans do not put this music.

### Blast radius

Only **323 of 2,000** held-out titles (16.2%) activate an audio target at all. On
the other 83.8% the weight is a literal no-op — which is also why any naive
full-set evaluation of it would be diluted ~6× and underpowered, exactly as
Ostinato's first run was.

**Decision: do not retune the weight yet.** Fix the target first (§3.1), then
re-ask. The FINDINGS §4 conclusion — that choosing a weight needs a stated product
preference no held-out data supplies — survives intact.

### 1.4 The seed fix was incomplete — the reranker still trains on the old bug

On 2026-08-21 `eval/splits.py` was corrected to take a playlist's genuine first
*k* tracks instead of its *k* lowest track ids. `models/reranker.py:205` still
does the old thing:

```python
trs = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
```

So the **evaluation** seeds were fixed and the **training** seeds were not. The
reranker — which roughly doubles k=0 R-precision, 0.0709 → 0.1429, the single
biggest win in the system — is supervised on head-biased, unordered prefixes.
`artifacts/reranker.pkl` (2026-08-15) predates `order.npz` (2026-08-21) by six
days.

I grepped every call site rather than assuming this was the only one. Five exist;
`build.py` and `train.py` build the tag matrix with `np.tile` and Timbre's loader
builds a filtered *set* — all three are order-independent and correct. Exactly one
was a real miss, and it was mine: I fixed the call site I was looking at and never
grepped for its siblings.

---

## 3. Segue: the result survives fusion, and the harness cannot see it

Prototyped rather than argued — `collaborative_channel` was swapped for Segue's
operator and run through Cadence's real fusion and reranker on the frozen splits.
Baselines reproduce `eval_report.json` exactly, so the harness is faithful.

| k=25, n=400 | base | +Segue | paired Δ |
|---|---:|---:|---|
| fusion R-precision | 0.1443 | 0.1508 | **+0.0064 ± 0.0023** |
| reranked R-precision | 0.2005 | 0.2095 | **+0.0090 ± 0.0027** |
| recall@500 | 0.4706 | 0.4775 | +0.0069 ± 0.0023 |

Null at k≤10, and inert at k=0 by construction — which is the case
`docs/EVALUATION.md` calls "the case the project exists for".

**The instrument was the limit.** `eval/metrics.py` reports only *unpaired* SE:
2×SE is 0.0181 reranked at k=25, larger than the real +0.0090 effect. **The
shipped harness would have called this null.** For scale, the collaborative
channel's entire marginal contribution at k=25 is 0.0133 — also below its own
floor. Everything Cadence has ever tuned sits under this.

**Order is what survives.** Decomposing via `segue_shuffled`, the order-attributable
share of the clicks win is 0% at k=1, 13% at k=5, 57% at k=10, **101% at k=25**.
The one seed count where fusion shows a gain is the one where the win is entirely
an order win — and Segue's headline "13.7% average" is majority order-*free* at
low k.

**Two claims are now stale.** `order.npz` is Segue's own sequences deduplicated by
first occurrence — rows align 1:1, prefixes byte-identical for 91.3% of k=25
challenges — so Segue's "order is not recoverable from Cadence" is no longer true.
And the dependency problem dissolves: export 0.62 MB of `beta` as an artifact and
copy ~35 lines of pure-numpy encoder, rather than importing the package and
inverting the star topology.

**But there is no product surface.** `seed_indices_from_intent` emits tracks sorted
by popularity, then per-artist expansions — resolution order, not listening order.
Feeding that to an operator whose slot 0 means "most recently played" is
meaningless. Ordered prefixes exist only in `eval/splits.py`. **Scope this to the
eval path; the write-up is worth more than the +0.009.**

---

## 4. The measurement floor is the binding constraint on everything

Cadence's harness reports **unpaired** standard errors. At n=400 that is 2×SE =
0.0149 at k=0 — **10.4% of the metric's own value**. Against that band:

| perturbation | Δ R-prec | published band | paired band | ρ |
|---|---:|---:|---:|---:|
| `rrf_k` 60→30 | −0.00109 | 0.0209 | **0.00185** | 0.9922 |
| `tag_exact` 0.7→0.9 | +0.00186 | 0.0211 | **0.00165** | 0.9939 |
| `cooccurrence` 1.3→1.0 (k=5) | −0.00167 | 0.0267 | 0.00183 | 0.9953 |
| `popularity` 0.25→0.40 | +0.00003 | 0.0210 | 0.00045 | 0.9995 |

**Every plausible change is 6–190× under the published floor.** Halving `rrf_k` —
an enormous change to the fusion prior — moves R-precision by 0.0011 against a
0.0209 band. Only 5 of 25 ablation cells clear it, and all five are *total channel
removals*. Complete removal of the collaborative channel fails to clear it at
every k.

### The pairing already exists and is thrown away

`run_eval.py` gives every arm the identical challenge list in identical order, and
`MetricAccumulator.values` already holds the per-challenge vector. `summary()`
collapses it to mean + SE and **discards the vector**.

The SE it keeps is documented as being *"so a difference between two runs can be
read as meaningful or not"* — but an unpaired SE is the wrong statistic for that
job, and it is the only one the harness has ever reported.

Measured per-challenge correlation between arms: **ρ = 0.9922–0.9995**. Pairing
buys **128–2,171×** variance reduction for **zero extra compute** — an order of
magnitude beyond the 17× Ostinato got from fixing its query set, because two
configs differ in a handful of candidate positions rather than a whole sample.

| | unpaired (today) | paired (same n) |
|---|---|---|
| floor at k=0 | 0.0149 — 10.4% of the metric | 0.00045–0.00185 — **0.32–1.29%** |
| n needed to resolve 0.005 | 7,014–11,540 | **28–82** |

It resolves things immediately: `rrf_k` 60→30 costs **−0.00579 NDCG@100 against a
paired band of 0.00222** — a 2.6σ regression the shipped harness reports as
"no difference".

### Two structural limits pairing does not fix

**The harness never runs assembly.** `run_eval.py` stops after rerank, so
`mmr_lambda`, `beam_width` and the four sequencer weights are constrained by *no
number in the report at all*, at any n. `affinity_sweep.py` shows the fix —
retrieve once, replay many, 0.83s per extra arm — and applies it to one knob.

**`audio: 0.6` is a query-set problem, not a sample-size one.** The audio channel
is inert unless the query states a mood: 71 of 400 k=0 challenges (17.8%). No
statistic rescues a knob that fires on a fifth of the set.

### A config value that never runs

`config.py:28` declares `min_tag_playlists: int = 12`. `cli.py:48` defaults to 5,
`cli.py:209` hardcodes 5, and `build_meta.json` records 5. The shipped default is
a third value that nothing was ever built with, and 12-vs-5 has never been
measured.

---

## 5. What we are deliberately not doing

- **Brute-force more challenges.** `--limit 0` costs 2.7h and only reaches
  2×SE = 0.0066. A resplit to reach 0.005 costs 4.8h, invalidates every published
  number, and *still* leaves the audio weight, the lexical weight and all eight
  assembly knobs unmeasurable. Pairing buys 10× more resolution in 45 minutes on
  data already on disk.
- **Retune `audio_affinity_weight`.** Not until the target is fixed (§2). The
  term is currently indistinguishable from a random permutation of itself.
- **Build a serving surface for ordered seeds.** There is no product path that
  supplies one, and the eval-only result is worth more than the feature.
- **Raise catalog reach for its own sake.** Humans put Gini 0.965 / top-1% 53.7%
  on these same playlists; Cadence delivers 0.951 / 43.6% and serves the tail
  8.5× more than the humans did. Concentration here is demand-shaped. *(Pending
  independent verification — see spot beads.)*

---

## 6. The pattern, sixth and seventh instances

This repo keeps finding tools whose stated scope and actual scope differ, where
passing reads as a stronger guarantee than it is. Previously: an unanchored
`data/`; a lint target that had never passed; ruff flagging dead imports but not
dead functions; a claim-checker blind to its own first sentence.

Now:

6. **A standard error that cannot answer the question it documents.** The harness
   reports unpaired SE "so a difference between two runs can be read as meaningful
   or not". Unpaired SE is the wrong statistic for that, by 128–2,171×.
7. **A fix applied to one call site and not its siblings.** `splits.py` was
   corrected; `reranker.py` was not.

And one committed here: I measured tag leakage against total tag mass, got 1.9%,
and nearly dismissed a real 35.4% finding — the same wrong-denominator error I had
caught in Gamut's funnel twenty minutes earlier.


---

## 7. The beads

14 tracked under the `spot-` prefix. `bd ready` shows what is unblocked;
`bd show <id>` carries the full establishes / does-not-establish / acceptance text.

| bead | title |
|---|---|
| `spot-2ig` | Audit MOOD_LEXICON's audio targets against where humans actually file each word |
| `spot-36u` | Extend retrieve-once/replay-many to mmr_lambda and give the sweep error bars |
| `spot-4ec` | Add cadence eval-ab and price rrf_k and the seven channel weights |
| `spot-4gd` | Persist a holdout-free tag matrix and re-read every tag-derived number |
| `spot-4q0` | Fix Gamut's per-channel sentinel - lexical's row IS the baseline row |
| `spot-7lj` | Persist per-challenge metric vectors and report paired deltas |
| `spot-bv0` | Move the CI workflow to the repo root, where GitHub will read it |
| `spot-kmc` | Price the audio term against a shuffled-affinity control |
| `spot-ktp` | Print the detection floor beside every published number |
| `spot-p7s` | Make the coverage tools fail instead of silently checking nothing |
| `spot-ph0` | Correct Gamut's funnel - it measured depth 100 and labelled it 'retrieval' |
| `spot-q15` | Wire Segue as an eval-only channel and publish the fused A/B |
| `spot-qyk` | Verify independently that concentration here is demand-shaped |
| `spot-z2q` | Fix the reranker's training seeds - the 2026-08-21 fix was incomplete |

**Start here.** `spot` P0s split into two groups. The three *corrections* —
Gamut's funnel, Gamut's sentinel, the tag leak — are wrong things currently
published, and should land before anything builds on them. The two *unlocks* —
paired deltas, and the lexicon audit — each make a whole class of later work
possible: without pairing, every A/B in this list reports null; without the
lexicon audit, the mood-vs-genre question is being asked about the wrong
parameter.
