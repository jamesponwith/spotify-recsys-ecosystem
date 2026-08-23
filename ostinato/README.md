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

**The loop does not run away. The control does.**

Over 5 rounds, with a fixed query set and dose 25:

| arm | artist Gini drift | long-tail share | catalog reach |
|---|---:|---:|---:|
| organic control | **+0.00153** | −0.73pp | +0.021pp |
| closed loop | −0.00003 | +0.67pp | +0.006pp |
| closed loop + penalty | −0.00078 | +1.00pp | +0.014pp |

Noise band is ±0.00020, taken as twice the standard deviation of the *steadiest*
arm — not the control, whose own spread mixes noise with the drift being judged.

The ordering is the finding: **the recommender concentrates less than
popularity-shaped listening does.** That is the opposite of the hypothesis this
project was built to test, and it agrees with what [Gamut](../gamut) found in a
single frame — Cadence already over-serves the long tail at a lift of 1.09×.
Feeding its own output back does not reverse that; the organic arm, sampling in
proportion to popularity, is the one that compounds.

The caution that matters: Cadence's lexical and audio channels are content-based
and do not move when the corpus does, which plausibly anchors it. A purely
collaborative system has no such anchor, and that is where the runaway story came
from in the first place.

### Why this run is readable and the first one was not

The first version redrew the query sample every round, so a change between rounds
mixed system drift with a change of question. Fixing the set — one draw, reused by
every round of every arm — cut round-to-round variance in artist Gini by **17×**
on the closed-loop arm:

| arm | sd, resampled | sd, fixed | noise cut |
|---|---:|---:|---:|
| organic | 0.00117 | 0.00059 | 2.0× |
| closed loop | 0.00168 | **0.00010** | **17.2×** |
| exposure-aware | 0.00160 | 0.00042 | 3.8× |

The drift being measured is *smaller than the noise the old design generated*, so
the earlier run could not have found it at any dose. That run reported the paired
comparison and explicitly refused to report its trend lines. Those lines are now
reportable, and they say something the paired result could not.

### The result that held

The exposure penalty holds **+4.76 points** more long-tail share than the
unmodified loop, in **6 of 6** rounds — the same magnitude the noisier run found.
It is trustworthy because it is *paired*: both loop arms run off one random
stream, so at every round they see identical queries and the listener accepts
identical *positions*. Only the track at each position differs.

Surviving a redesign that changed everything else is the strongest thing that can
be said for a number here.

Full tables in [docs/RESULTS.md](docs/RESULTS.md).

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
  concentration exists at all. It turned out not to — the control is the arm
  that compounds.
- Only the collaborative and folksonomy spaces are refit. The lexical and audio
  channels are content-based and do not move.
