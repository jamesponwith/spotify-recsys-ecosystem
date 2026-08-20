# Segue

**Sequence-aware playlist continuation for the [Cadence](../cadence) catalog.**

Cadence answers *"make me a playlist about X"*. Segue answers the other half of
the product: *"here is a playlist someone is already building — what comes
next?"*

The thesis is one sentence. **Cadence's collaborative channel treats a playlist
prefix as a bag** — it sums the seed embeddings and searches that
neighbourhood, so shuffling the seeds changes nothing. A playlist is not a bag.
It has an arc: an opener, a build, a comedown. Segue asks whether that order
carries information a recommender can actually use, and measures the answer
against the system that ignores it.

---

## The experiment

Same catalog, same held-out playlists, same metrics as Cadence — four systems
ranked on the RecSys Challenge 2018 task:

| System | What it does | Order-aware |
|---|---|---|
| `popularity` | global top tracks | no |
| `centroid` | sum of seed embeddings, nearest neighbours — **exactly Cadence's collaborative channel** | no |
| `last` | nearest neighbours of the most recent seed only | yes, crudely |
| `segue` | learned transition operator over the ordered prefix | yes |

Plus the check that makes the claim falsifiable:

| `segue_shuffled` | the same model, the same tracks, **prefix order destroyed at inference** |
|---|---|

If `segue` and `segue_shuffled` score the same, the model never used order, and
the project's premise is decoration. That comparison is run at every seed count
and reported whatever it says.

## The model

A **position-weighted linear transition operator**. Not a transformer — the
honest description is a ridge regression that maps an ordered prefix to the
direction of the next track:

```
features   last 5 tracks, each in its own slot (5 x 160-d)
         + per-slot presence flags               (5)
         + mean of the whole prefix (160-d)      <- what centroid also sees
         + intercept
target     the next track's collaborative embedding, L2-normalised
ranking    cosine against the catalog, seeds excluded
```

The order-free mean is included deliberately: the model is handed everything the
centroid baseline has, plus ordered slots. If order were uninformative it could
collapse onto the mean and tie. It is not told which to prefer.

Fitted by normal equations — the 966x966 Gram accumulates in chunks, so peak
memory is megabytes where `RidgeCV`'s SVD path would materialise ~3 GB, and each
candidate alpha costs one Cholesky solve instead of a refit.

## Getting the order back

Playlist order **is not recoverable from Cadence's processed data**. Its
`interactions.npz` is a CSR matrix with sorted indices, so every row reads back
in ascending track-id order. Segue rebuilds true order from the raw MPD slices
(98,334 playlists, 5,962,343 positions, 36 s).

That dig turned up a defect in Cadence's evaluation harness, written up in
[docs/FINDINGS.md](docs/FINDINGS.md).

## Running it

```bash
make install
make build      # rebuild ordered sequences from raw MPD
make train      # fit the transition operator
make evaluate   # head-to-head at k = 1, 5, 10, 25
make demo       # continue a real held-out playlist
make report     # render artifacts/results.html
```

```
src/segue/
  sequences.py   raw MPD -> ordered catalog indices
  features.py    ordered prefix -> design vector
  model.py       transition operator, fitted by normal equations
  baselines.py   popularity / last / centroid
  evaluate.py    RecSys 2018 metrics, batched scoring
  demo.py        continue a held-out playlist, with ground truth
```

## Results

**Order carries real signal. The standard offline metric is mostly blind to it.**

Against the order-free centroid — Cadence's collaborative channel — on 2,000
held-out playlists:

| seeds | R-precision | vs centroid | Clicks | vs centroid | order gain* |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0883 | +5.7% | 6.23 | **−11.0%** | +0.0% |
| 5 | 0.1043 | −0.4% | 2.87 | **−17.5%** | +0.2% |
| 10 | 0.1025 | −0.5% | 2.79 | **−14.7%** | +1.8% |
| 25 | 0.0857 | +3.4% | 4.85 | **−11.8%** | **+10.0%** |

\* R-precision lost when the same model is run on the same prefix with its order
shuffled — the causal check.

Three things, in order of how much they matter.

**Clicks improves everywhere, by 13.7% on average.** Clicks is the official
metric that counts how many "give me 10 more" presses a listener needs before
hitting something they wanted. It is the most product-shaped of the three, and
Segue wins it at every seed count.

**R-precision is a wash.** +5.7% and +3.4% at the extremes, −0.5% in the middle.
Reporting the Clicks win without this would be marketing.

**The disagreement between those two is the actual finding.** R-precision and
NDCG are *set* metrics — they ask which tracks land in the top |G|, not where.
Reading order better mostly moves good tracks *up* the list rather than adding
new ones, so a set metric cannot see the improvement and Clicks can.

**The shuffle control proves the mechanism.** Destroying prefix order costs
10.0% of R-precision at 25 seeds, and the effect grows monotonically with prefix
length (0.0% → 0.2% → 1.8% → 10.0%). At one seed there is no order to read and
the two are byte-identical, which is the harness confirming its own wiring.

### The objective was wrong before it was tuned

The first model trained on the obvious target — predict the very next track — and
**lost to the centroid at every seed count above one**. R-precision scores against
*all* withheld tracks, so that objective was strictly narrower than the metric.
Widening the target to the mean direction of the next 10 tracks moved validation
cosine from 0.5625 to 0.7311 and closed the gap. Both runs are kept in
[docs/FINDINGS.md](docs/FINDINGS.md); full tables in
[docs/RESULTS.md](docs/RESULTS.md).

### Cost

Ordered sequences rebuilt from raw MPD in 36 s; the transition operator fits in
167 s; the full four-way evaluation over 8,000 challenges runs in 181 s. No GPU.
