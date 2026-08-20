# A music recommendation ecosystem, and what measuring it honestly turned up

Five applications built on the [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).
They are not four demos of the same idea. Each one was built to answer a question
the previous one raised, and two of them answered it *no*.

| | | |
|---|---|---|
| **[cadence](cadence/)** | Natural-language playlist generation | 7 retrieval channels, RRF fusion, learned reranker, MMR selection, beam-search sequencing |
| **[timbre](timbre/)** | Content-based cold start | **Killed at its own gate.** Audio descriptors recover 13.2% of what playlist history gives; the bar was 25% |
| **[segue](segue/)** | Sequence-aware continuation | Playlist order carries real signal — 13.7% fewer Clicks — that set metrics cannot see |
| **[gamut](gamut/)** | Catalog exposure audit | 43.6% of exposure goes to 1% of artists, and no amount of re-ranking fixes it |
| **[ostinato](ostinato/)** | Feedback-loop simulation | No detectable runaway in 5 rounds — but the exposure penalty holds its gain in 6 of 6 |

---

## The thread

**Cadence** turns *"chill rainy morning, nothing explicit, about an hour"* into a
sequenced playlist. Its own evaluation named its biggest limitation: tracks with
little playlist history are filtered out rather than handled — 76.6% of the
distinct tracks in the corpus.

**Timbre** was built to close that gap by predicting a track's position in
Cadence's folksonomy space from audio. Phase 0 was a falsification test designed
to kill the project in one session before any audio work began. It did.

> Ridge regression reached a mean cosine of 0.198 with the true embeddings and
> retrieved **nothing** — zero hits across 1,718 queries. It lands close to the
> target on average and is never the nearest thing to any query, which is the
> only property a top-100 cut rewards.

A cosine-based gate would have passed that model and authorised three weeks of
work. The project stopped instead, and the null result is written up.

**Segue** took the pivot Timbre's own documents named. Cadence's collaborative
channel sums the seed tracks — shuffle them and nothing changes. Segue asks
whether that discards information, and proves it does with a shuffle control:
destroying prefix order costs 10.0% of R-precision at 25 seeds, and the effect
grows monotonically with prefix length.

The subtler result is *which* metric noticed. R-precision and NDCG are **set**
metrics; reading order better moves good tracks up the list rather than adding
new ones, so they see nothing. Clicks — how many "give me 10 more" presses before
a hit — improves at every seed count.

**Gamut** measures the axis none of the other three do. Cadence, Timbre and Segue
all ask whether the *listener* was served. None asks which artists were served at
all.

> Across every intervention tested — nine popularity-penalty strengths and four
> artist caps — artist Gini moves only from 0.951 to 0.936.
> **You cannot re-rank your way out of a retrieval problem.**

Only 10.7% of the catalog reaches the candidate pool and 2.99% is ever shown.
Re-ordering a hundred candidates cannot introduce what was never a candidate —
which is Timbre's finding arriving from the opposite direction. **The exposure
ceiling is set upstream of everything that gets ranked.**

---

**Ostinato** closes the loop the first four leave open. A recommender is trained
on interaction data its own output helped create, so Gamut's snapshot may be a
frame from a film. Five rounds of recommend → accept → refit, against an organic
control.

> The runaway did not happen. Artist Gini drifts +0.0014 under the closed loop
> against +0.0010 under the control — an excess of +0.0004 against a ±0.0023
> noise band. That was not the hypothesis, and it is the headline anyway.

What is real is the third arm: Gamut's popularity penalty holds **+4.8 points**
more long-tail share in **6 of 6 rounds**. That comparison is *paired* — both arms
see the same queries and the listener accepts the same positions, so only the
ranking differs. An intervention that looked marginal in a snapshot does not decay
under compounding.

---

## Things that were wrong, and stayed on the record

Every project here has a findings document, because the errors were the most
useful output.

- **Cadence** shipped a ranking bug that only reading real output caught:
  min-max normalisation over a reranker whose probabilities are heavily skewed
  (median ~0.002, max ~0.7) silently nullified audio affinity at any weight.
- **Segue's** first objective was wrong before it was tuned. Training on
  "predict the next track" lost to the order-free baseline everywhere above one
  seed, because the task scores against *all* withheld tracks. The losing run is
  kept, not deleted.
- **Segue** also found that playlist order is not recoverable from Cadence's
  processed data at all — a CSR matrix returns rows in ascending track-id order —
  which means Cadence's evaluation harness hands out "the k lowest track ids"
  where its docstring promises "the first k tracks". Documented rather than
  patched, because fixing it would move every number in Cadence's published
  report. See [segue/docs/FINDINGS.md](segue/docs/FINDINGS.md).
- **Ostinato's** first run was underpowered and the control proved it: a 0.011%
  corpus perturbation per round against a noise floor of σ = 0.0012 in artist
  Gini. It was killed before finishing rather than reported, because it would
  have produced a confident-looking null that was really a power failure. Its
  query sample is also redrawn each round, which confounds within-arm
  trajectories — named on the page rather than quietly smoothed over.
- **Gamut's** first exposure frontier came out perfectly flat. Exposure metrics
  computed over the candidate pool are invariant to re-ranking — reordering a set
  does not change the set. Exposure has to be counted at the cut the listener
  actually sees.

## Demos

Each app has a working demo command and a published report showing real output.

| App | Demo command | Shows |
|---|---|---|
| cadence | `cadence play "<request>"` | a generated playlist, with a grounded reason and a Camelot segue per track |
| timbre | `timbre demo "<request>"` | Cadence's own picks frozen out, then handed back an audio-only embedding |
| segue | `segue demo` | a held-out playlist continued, against what the person really played next |
| gamut | `gamut demo` | one query before and after the exposure-aware re-rank |
| ostinato | `ostinato simulate` | exposure drift across rounds, three arms |

## Running any of it

Cadence is the base; the other three install it editable as a sibling.

```bash
cd cadence && make install && make data && make all   # builds the catalog
cd ../segue && make install && make all               # or timbre / gamut
```

The corpora are not committed. The Million Playlist Dataset is licensed for
research use and is not redistributable; `cadence/scripts/download_data.py`
re-fetches everything. Every number in every report is regenerated from a fixed
seed by `make all`, and is read out of JSON the code writes rather than typed by
hand.

## Stack

Python 3.12, NumPy/SciPy/scikit-learn, FastAPI, Typer, `uv`, `ruff`, `mypy`,
`pytest`. No GPU anywhere — the most expensive thing in the repo trains in under
three minutes on CPU. The Anthropic SDK is an optional planner backend; Cadence
falls back to a deterministic rules planner with no API key.
