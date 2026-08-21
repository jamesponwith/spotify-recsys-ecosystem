# Findings about Cadence, surfaced by building Segue

Segue needed one thing Cadence had never needed: the order a human put tracks in.
Going after it turned up a defect in Cadence's evaluation harness.

---

## Playlist order is not recoverable from `interactions.npz`

```python
m = sparse.load_npz("data/processed/interactions.npz").tocsr()
m.has_sorted_indices          # True
m.indices[m.indptr[500]:...]  # [1231, 1265, 1268, 1543, 2425, ...]  ascending
```

SciPy keeps CSR column indices sorted within a row. So reading a playlist's
tracks out of that matrix returns them in **ascending track-id order**, never in
the order they were arranged. Cadence's track ids are assigned in first-seen
order during the build, which makes the first few playlists look correctly
ordered and hides the problem on casual inspection — 34 of the first 200
playlists come out ascending by coincidence of id assignment.

For Cadence itself this costs nothing. Its collaborative channel sums seed
embeddings and its co-occurrence channel counts neighbours; both are order-free
by construction, and a bag of tracks is all either one ever consumed.

## Where it does leak: the challenge splits

`cadence/src/cadence/eval/splits.py` says:

> For each evaluation playlist we expose the title plus the first `k` tracks and
> withhold the rest.

and implements:

```python
trs = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
seed_tracks = trs[:k].tolist()
```

`trs` is ascending, so `seed_tracks` is **the k lowest-numbered track ids in the
playlist**, not its first k tracks. Because ids are assigned in corpus-wide
first-seen order, low id correlates with *popular and early-appearing*, so the
seeds are biased toward the catalog's head rather than being a neutral sample.

**How much this matters depends on what you are measuring.** The official RecSys
Challenge 2018 has both a *first-k* and a *random-k* seed variant, and Cadence's
reported numbers are internally consistent — every system it compares is handed
the identical seed set, so its ablations and channel comparisons stand. What is
not safe is the docstring's claim, or any future work that reads order out of
that matrix.

**Fixed in Cadence, 2026-08-21 — and the cause was not the one named above.**
SciPy's sorted CSR indices were a symptom, not the source. `data/build.py`
destroyed the order in pass 1 with `np.unique(row)`, which deduplicates (the
intent) and sorts (not the intent). Dedup is now by first occurrence, and
`order.npz` — the per-playlist sequence — is written from the same pass that
fills the matrix, so the two cannot disagree about membership. `eval/splits.py` reads it and takes a
genuinely first-k prefix; if the file is missing it raises rather than falling
back to track-id order, because a silent fallback is how this survived in the
first place. Cadence's `eval_report.json` was regenerated from the corrected
splits.

This was originally left as a finding rather than a patch, on the grounds that
changing seed selection moves every published number. That was the wrong call:
the numbers were wrong, and reissuing them with the reason recorded beats leaving
a known-bad harness in place. The before/after is in
[cadence/docs/FINDINGS.md](../../cadence/docs/FINDINGS.md).

Segue keeps its own `src/segue/sequences.py`, which rebuilds order from the raw
slices in 36 s. It now duplicates what Cadence emits and should eventually read
`order.npz` instead — noted, not yet done.

---

## Consequence for Segue

Segue rebuilds order from the raw MPD slices and evaluates on challenges it
constructs itself — same 2,000 held-out playlists as Cadence, genuinely ordered
prefixes. Its `centroid` baseline reproduces Cadence's collaborative channel
exactly, so the head-to-head remains apples to apples; only the seed *selection*
differs, and it differs in the direction of being correct.

---

## The objective was wrong before it was tuned

The first version of Segue trained on the obvious target: **predict the next
track**. It lost to the order-free centroid at every seed count above one.

| seeds | centroid | segue (next-track) | vs centroid | order gain vs shuffled |
|---:|---:|---:|---:|---:|
| 1 | 0.0836 | 0.0879 | **+5.2%** | +0.0% |
| 5 | 0.1047 | 0.1003 | −4.2% | +1.6% |
| 10 | 0.1030 | 0.0961 | −6.7% | +2.0% |
| 25 | 0.0829 | 0.0805 | −3.0% | **+14.7%** |

Two things are true at once in that table, and separating them is the whole
diagnosis.

**Order is genuinely informative, and increasingly so.** The gap between `segue`
and `segue_shuffled` is the same model on the same tracks with sequence
destroyed, and it grows monotonically with prefix length — 0%, 1.6%, 2.0%,
14.7%. With one seed there is no order to read and the two are identical, which
is the control behaving exactly as it must.

**But the objective was narrower than the task.** RecSys R-precision scores a
system against *every* withheld track — often fifty of them. A model trained to
hit the single immediate successor is optimising a strictly smaller target than
it is judged on, and the centroid, which implicitly summarises the playlist's
whole neighbourhood, is better aligned with the metric by accident.

The fix is one line of target construction: regress onto the mean direction of
the next `horizon` tracks instead of the next one. `horizon=1` recovers the naive
objective and is retained as the ablation above. Validation cosine moved from
**0.5625 to 0.7311** — the wider target is not just better aligned, it is a
smoother and more learnable quantity.

This is the sort of thing an offline metric catches and an eyeball does not. The
next-track model produced *perfectly plausible* continuations; it was simply
answering a different question than the one being scored.
