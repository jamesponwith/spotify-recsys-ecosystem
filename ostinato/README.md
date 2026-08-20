# Ostinato

**What happens to the catalog when a recommender trains on its own output.**

An *ostinato* is a figure that repeats until it becomes the whole piece.

[Gamut](../gamut) measured exposure at one instant and found it badly
concentrated: 43.6% of exposure to 1% of artists, and no re-ranking that moved
artist Gini more than 0.015. But a recommender is not a static object. It is
trained on interaction data that its own past output helped create — today's
recommendations become tomorrow's playlists become next month's training set.

Ostinato closes that loop and runs it.

---

## The mechanism

```
   embeddings ──▶ recommendations ──▶ what the listener keeps
        ▲                                       │
        └──────── refit on the grown corpus ◀───┘
```

The listener is a **position-based acceptance model**: a track near the top of
the list is kept far more often than one near the bottom. That is the crude part
of the simulation and also the load-bearing part — position bias is the mechanism
that converts a *ranking* bias into a *data* bias, and then into a training bias
on the next round.

## Three arms, same starting state

| Arm | Ranking comes from | Role |
|---|---|---|
| `organic` | the catalog's own popularity distribution | control — adds the same volume of data, so drift cannot be blamed on having more of it |
| `closed_loop` | Cadence as it ships | the system under test |
| `exposure_aware` | Cadence + [Gamut's](../gamut) popularity penalty at 0.3 | does an intervention that looked useless statically matter dynamically? |

Every round is a **full refit** of the collaborative and folksonomy spaces on the
grown corpus — not an incremental update — so the drift being measured is the
real thing and not an artifact of a cheaper approximation.

## Results

**The runaway did not happen. The intervention's effect did survive.**

Artist Gini drifts **+0.0014** under the closed loop against **+0.0010** under the
organic control — an excess of **+0.0004** against a noise band of **±0.0023**,
taken as twice the control's own round-to-round standard deviation. At this dose
and horizon there is no detectable homogenisation. That was not the hypothesis
going in, and it is the headline anyway.

**What is real** is the third arm. Gamut's popularity penalty holds **+4.8
points** more long-tail share than the unmodified loop, in **6 of 6 rounds**:

| round | closed loop | + penalty | gap |
|---:|---:|---:|---:|
| 0 | 62.5% | 66.8% | +4.3pp |
| 1 | 59.4% | 64.9% | +5.4pp |
| 2 | 58.0% | 62.5% | +4.5pp |
| 3 | 56.5% | 60.8% | +4.2pp |
| 4 | 55.4% | 60.8% | +5.4pp |
| 5 | 65.4% | 70.2% | +4.8pp |

This one is trustworthy in a way the trajectories are not, because it is
**paired**: both loop arms are driven by the same random stream, so at every
round they see the identical query sample *and* the simulated listener accepts
the identical *positions*. Only the track at each position differs. The gap is
attributable to the ranking and nothing else.

Full tables in [docs/RESULTS.md](docs/RESULTS.md).

### The control earned its keep twice

The first run used dose 1 — 150 accepted playlists a round against a corpus of
5.88M interactions, a **0.011%** perturbation. The organic arm's noise floor
(σ = 0.0012 in artist Gini, net five-round drift +0.0005) showed that no effect
below ~0.003 could be distinguished from the noise of refitting a randomised SVD.

That run was killed before it finished rather than reported. It would have
produced a confident-looking null that was really a power failure, and nothing in
its output would have distinguished the two. The `dose` parameter exists because
of it, and the reasoning is a comment in `config.py`.

### A design flaw worth naming

The query sample is **redrawn every round**, so within-arm trajectories confound
system drift with a change of question. Cross-arm comparisons at a fixed round
are clean, because the arms share a sample — which is precisely why the paired
result above is reported and the trend lines are not. Holding one query set fixed
across rounds is the first thing to change.

## Running it

```bash
make install
make simulate   # 3 arms x 5 rounds, full refit each round
make report     # render artifacts/results.html
```

```
src/ostinato/
  config.py      arms, rounds, the acceptance model's parameters
  simulate.py    one turn of the wheel: recommend, accept, append, refit
```

## What this is not

- **Not a claim about real listeners.** The acceptance model is position bias and
  nothing else — no taste, no satiation, no repeat plays, no discovery outside
  the recommender. It is a claim about what a ranking bias does to a corpus once
  the corpus sits downstream of the ranking.
- **Not calibrated to a real time-scale.** A round is 150 queries against a corpus
  of ~98k playlists, so the per-round effect is small by construction. The
  direction and the ordering of the arms are the findings; the magnitudes are not
  transferable.
- **The organic control is popularity-proportional**, which is itself a
  rich-get-richer process. Deliberately so: the question is whether the
  recommender concentrates *faster* than the world already does, not whether
  concentration exists at all.
- Only the collaborative and folksonomy spaces are refit. The lexical and audio
  channels are content-based and do not move.
