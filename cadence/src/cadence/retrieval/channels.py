"""Candidate-generation channels.

Each channel answers the same question from a different angle and returns
``(indices, scores)`` sorted best-first. Keeping them separate — rather than
fusing signals inside one scorer — is what makes the ablation study in
``docs/EVALUATION.md`` possible, and what lets the reranker learn how much to
trust each source per query type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from ..catalog import Catalog
from ..models.train import AUDIO_FEATURE_COLS
from ..types import AudioTargets, PlaylistIntent


@dataclass
class ChannelResult:
    name: str
    indices: np.ndarray
    scores: np.ndarray
    detail: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)

    @classmethod
    def empty(cls, name: str) -> ChannelResult:
        return cls(name, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))


def _topk_from_scores(
    scores: np.ndarray, k: int, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    s = scores.astype(np.float32, copy=True)
    if mask is not None:
        s[~mask] = -np.inf
    finite = int(np.isfinite(s).sum())
    k = int(min(k, finite))
    if k <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    return top.astype(np.int64), s[top]


def collaborative_channel(
    catalog: Catalog,
    seed_indices: np.ndarray,
    k: int,
    mask: np.ndarray | None = None,
    seed_weights: np.ndarray | None = None,
) -> ChannelResult:
    """Item-item CF: what co-occurs with the seeds on real playlists."""
    seeds = np.asarray(seed_indices, dtype=np.int64)
    if seeds.size == 0:
        return ChannelResult.empty("collaborative")
    vecs = catalog.collab.vectors[seeds]
    if seed_weights is not None:
        w = np.asarray(seed_weights, dtype=np.float32).reshape(-1, 1)
        centroid = (vecs * w).sum(axis=0)
    else:
        centroid = vecs.sum(axis=0)
    if not np.any(centroid):
        return ChannelResult.empty("collaborative")
    scores = catalog.collab.scores(centroid, mask=mask)
    # Never recommend a seed back to the listener.
    scores[seeds] = -np.inf
    idx, sc = _topk_from_scores(scores, k, None)
    return ChannelResult("collaborative", idx, sc, {"n_seeds": int(seeds.size)})


def tag_channel(
    catalog: Catalog,
    tag_cols: list[int],
    k: int,
    mask: np.ndarray | None = None,
    negative_cols: list[int] | None = None,
) -> ChannelResult:
    """Folksonomy space: free-text phrases -> tag vectors -> tracks.

    This is the channel that makes cold-start natural language work. A query
    with no seed tracks has nothing for CF to chew on, but "rainy day study" maps
    straight onto tags that tens of thousands of humans actually used.
    """
    if not tag_cols:
        return ChannelResult.empty("tag")
    centroid = catalog.tag_vectors[np.asarray(tag_cols, dtype=np.int64)].mean(axis=0)
    if negative_cols:
        neg = catalog.tag_vectors[np.asarray(negative_cols, dtype=np.int64)].mean(axis=0)
        centroid = centroid - 0.5 * neg
    if not np.any(centroid):
        return ChannelResult.empty("tag")
    scores = catalog.tag_tracks.scores(centroid, mask=mask)
    idx, sc = _topk_from_scores(scores, k, None)
    return ChannelResult("tag", idx, sc, {"tags": [catalog.tag_vocab[c] for c in tag_cols[:12]]})


def lexical_channel(
    catalog: Catalog, query_text: str, k: int, mask: np.ndarray | None = None
) -> ChannelResult:
    """Sparse TF-IDF over track/artist/album/genre strings.

    Latent spaces blur exact entities; this channel is how "some Radiohead"
    reliably surfaces Radiohead rather than merely Radiohead-adjacent music.
    """
    text = (query_text or "").strip()
    if not text:
        return ChannelResult.empty("lexical")
    q = catalog.lexical_vectorizer.transform([text]).astype(np.float32)
    if q.nnz == 0:
        return ChannelResult.empty("lexical")
    scores = np.asarray((catalog.lexical @ q.T).todense()).ravel().astype(np.float32)
    idx, sc = _topk_from_scores(scores, k, mask)
    idx = idx[sc > 0]
    sc = sc[sc > 0]
    return ChannelResult("lexical", idx, sc)


def audio_channel(
    catalog: Catalog,
    targets: AudioTargets,
    k: int,
    mask: np.ndarray | None = None,
) -> ChannelResult:
    """Distance to a point in standardised audio-feature space.

    Only the dimensions the listener actually specified participate. Scoring an
    unspecified dimension against a defaulted 0.5 would silently pull every
    result toward the catalog mean.
    """
    active = targets.active()
    if not active:
        return ChannelResult.empty("audio")
    dims = [AUDIO_FEATURE_COLS.index(name) for name in active if name in AUDIO_FEATURE_COLS]
    if not dims:
        return ChannelResult.empty("audio")
    values = np.array([active[AUDIO_FEATURE_COLS[d]] for d in dims], dtype=np.float32)
    z_target = (values - catalog.audio_mu[dims]) / catalog.audio_sigma[dims]

    sub = catalog.audio_z[:, dims]
    dist = np.linalg.norm(sub - z_target[None, :], axis=1) / np.sqrt(len(dims))
    scores = np.exp(-dist).astype(np.float32)
    scores[~catalog.audio_valid] = -np.inf
    idx, sc = _topk_from_scores(scores, k, mask)
    return ChannelResult("audio", idx, sc, {"dims": [AUDIO_FEATURE_COLS[d] for d in dims]})


def popularity_channel(catalog: Catalog, k: int, mask: np.ndarray | None = None) -> ChannelResult:
    """Popularity prior. Weak on its own, but it is the honest baseline and a
    useful backstop when every other channel comes back thin."""
    idx, sc = _topk_from_scores(catalog.popularity, k, mask)
    return ChannelResult("popularity", idx, sc)


def build_mask(catalog: Catalog, intent: PlaylistIntent) -> tuple[np.ndarray, dict]:
    """Hard structural filter applied before scoring.

    Hard constraints are enforced here, by construction, rather than by asking a
    model to respect them. A filter cannot be talked out of its job.
    """
    n = len(catalog)
    mask = np.ones(n, dtype=bool)
    applied: dict[str, int] = {}

    c = intent.constraints
    if c.exclude_explicit:
        before = int(mask.sum())
        mask &= ~catalog.col("explicit")
        applied["exclude_explicit"] = before - int(mask.sum())
        # The explicit flag is only partially observed. Removing every track
        # without a flag would shrink the catalog to a few percent, so we filter
        # what is known and report the coverage instead of implying a guarantee.
        known = catalog.col("explicit_known")
        applied["explicit_flag_coverage_pct"] = int(round(100 * float(known.mean())))

    if c.min_duration_s is not None:
        before = int(mask.sum())
        mask &= catalog.col("duration_ms") >= c.min_duration_s * 1000
        applied["min_duration"] = before - int(mask.sum())
    if c.max_duration_s is not None:
        before = int(mask.sum())
        mask &= catalog.col("duration_ms") <= c.max_duration_s * 1000
        applied["max_duration"] = before - int(mask.sum())

    if intent.tempo.is_set():
        tempo = catalog.col("tempo")
        known = np.isfinite(tempo)
        ok = np.ones(n, dtype=bool)
        if intent.tempo.min_bpm is not None:
            ok &= tempo >= intent.tempo.min_bpm
        if intent.tempo.max_bpm is not None:
            ok &= tempo <= intent.tempo.max_bpm
        # Tracks with no tempo reading are kept: excluding them would silently
        # restrict the catalog to the ~30% with audio features.
        before = int(mask.sum())
        mask &= ok | ~known
        applied["tempo_range"] = before - int(mask.sum())

    for artist in intent.avoid_artists:
        hits = catalog.resolve_artist(artist, limit=10_000)
        if hits:
            before = int(mask.sum())
            mask[np.asarray(hits, dtype=np.int64)] = False
            applied[f"avoid:{artist}"] = before - int(mask.sum())

    return mask, applied


def seed_indices_from_intent(
    catalog: Catalog, intent: PlaylistIntent, per_artist: int = 25
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolve named artists/tracks into weighted catalog rows.

    Named tracks are weighted above named artists: "songs like Holocene" is a
    sharper signal than "something like Bon Iver".
    """
    idx: list[int] = []
    weights: list[float] = []
    detail: dict[str, list[str]] = {"resolved": [], "unresolved": []}

    for t in intent.seed_tracks:
        hits = catalog.resolve_track(t)
        if hits:
            idx.append(hits[0])
            weights.append(1.0)
            detail["resolved"].append(f"track:{t}")
        else:
            detail["unresolved"].append(f"track:{t}")

    for a in intent.seed_artists:
        hits = catalog.resolve_artist(a, limit=per_artist)
        if hits:
            # Weight per-track so a prolific artist does not outvote a sparse one.
            w = 1.0 / np.sqrt(len(hits))
            for h in hits:
                idx.append(h)
                weights.append(float(w))
            detail["resolved"].append(f"artist:{a}")
        else:
            detail["unresolved"].append(f"artist:{a}")

    if not idx:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), detail
    return (
        np.asarray(idx, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        detail,
    )


def sparse_tag_channel(
    catalog: Catalog, tag_cols: list[int], k: int, mask: np.ndarray | None = None
) -> ChannelResult:
    """Exact tag match against raw folksonomy counts (no dimensionality
    reduction). Complements the dense tag channel: precise where the dense one
    generalises."""
    if not tag_cols:
        return ChannelResult.empty("tag_exact")
    cols = np.asarray(tag_cols, dtype=np.int64)
    sub = catalog.tag_matrix_csc[:, cols]
    # log1p *per concept*, then sum across concepts -- not log1p of the total.
    #
    # Summing raw counts first makes a multi-concept query an ANY-match: for
    # "90s alternative rock for a road trip", a track on 500 playlists tagged
    # `road trip` and none tagged `1990s` outscored one matching both at 50
    # each, because 500 > 100. The era was parsed correctly and then had no
    # force. Taking the log inside gives each concept diminishing returns of its
    # own, so breadth of match beats depth in whichever concept happens to be
    # the most popular tag in the corpus.
    scores = np.asarray(sub.log1p().sum(axis=1)).ravel().astype(np.float32)
    idx, sc = _topk_from_scores(scores, k, mask)
    idx = idx[sc > 0]
    sc = sc[sc > 0]
    return ChannelResult("tag_exact", idx, sc)


def as_sparse_vector(indices: np.ndarray, scores: np.ndarray, n: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (scores, (np.zeros_like(indices), indices)), shape=(1, n), dtype=np.float32
    )


def cooccurrence_channel(
    catalog: Catalog,
    seed_indices: np.ndarray,
    k: int,
    mask: np.ndarray | None = None,
    *,
    damping: float = 0.5,
    max_rows: int = 40_000,
    rng_seed: int = 0,
) -> ChannelResult:
    """Exact neighbourhood co-occurrence over the raw interaction matrix.

    The learned collaborative channel compresses co-occurrence into 160
    dimensions, which generalises well but measurably loses precision against
    exact counts on this corpus (see docs/EVALUATION.md). Rather than pretend
    the embedding subsumes the count, both run and fusion arbitrates: the
    embedding generalises to sparse seeds, the counts are sharp on dense ones.

    Popularity damping divides by ``count ** damping`` so that globally huge
    tracks do not win every neighbourhood vote regardless of the seeds.
    """
    seeds = np.asarray(seed_indices, dtype=np.int64)
    if seeds.size == 0:
        return ChannelResult.empty("cooccurrence")

    xt = catalog.interactions_t
    rows_list = [xt[int(s)].indices for s in seeds[:200]]
    if not rows_list:
        return ChannelResult.empty("cooccurrence")
    rows = np.unique(np.concatenate(rows_list))
    if rows.size == 0:
        return ChannelResult.empty("cooccurrence")
    if rows.size > max_rows:
        # Bound the work on ultra-popular seeds; a 40 k-playlist neighbourhood
        # already estimates the co-occurrence distribution well.
        rng = np.random.default_rng(rng_seed)
        rows = rng.choice(rows, size=max_rows, replace=False)

    scores = np.asarray(catalog.interactions[rows].sum(axis=0)).ravel().astype(np.float32)
    norm = np.power(np.maximum(catalog.track_playlist_counts, 1.0), damping)
    scores /= norm
    scores[seeds] = -np.inf
    idx, sc = _topk_from_scores(scores, k, mask)
    idx = idx[sc > 0]
    sc = sc[sc > 0]
    return ChannelResult("cooccurrence", idx, sc, {"n_rows": int(rows.size)})
