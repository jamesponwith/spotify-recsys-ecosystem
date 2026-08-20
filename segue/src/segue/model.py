"""The transition operator: prefix features -> the next track's direction.

Fitted by normal equations rather than an sklearn estimator for two reasons.
Accumulating the 966x966 Gram in chunks keeps peak memory in megabytes where
`RidgeCV`'s SVD path would materialise a 400k x 966 float64 copy (~3 GB). And
because the Gram is built once, every candidate alpha costs a single Cholesky
solve, so the regularisation sweep is effectively free.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ARTIFACTS, SegueConfig
from .features import encode, feature_dim, l2_normalize


@dataclass
class SegueModel:
    beta: np.ndarray  # (dim, emb_dim)
    window: int
    horizon: int
    alpha: float
    emb_dim: int
    valid_cosine: float
    seconds: float

    def predict(self, prefixes: list[np.ndarray], embeddings: np.ndarray) -> np.ndarray:
        x = encode(prefixes, embeddings, self.window)
        return l2_normalize(x @ self.beta)

    def save(self, path: Path | None = None) -> Path:
        path = path or ARTIFACTS / "segue_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> SegueModel:
        path = path or ARTIFACTS / "segue_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `segue train` first.")
        obj = pickle.loads(path.read_bytes())
        assert isinstance(obj, cls)
        return obj


def _targets(
    positions: np.ndarray,
    owners: np.ndarray,
    seq_tracks: np.ndarray,
    ends: np.ndarray,
    embeddings: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Mean direction of the next `horizon` tracks, clipped at the playlist end.

    Averaging unit vectors rather than raw ones keeps a single loud embedding
    from dominating the target the way a raw mean would.
    """
    out = np.empty((positions.size, embeddings.shape[1]), dtype=np.float32)
    for i, (p, o) in enumerate(zip(positions, owners, strict=True)):
        stop = min(p + horizon, ends[o])
        out[i] = l2_normalize(embeddings[seq_tracks[p:stop]]).mean(axis=0)
    return l2_normalize(out)


def _gram(
    positions: np.ndarray,
    owners: np.ndarray,
    seq_tracks: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    embeddings: np.ndarray,
    window: int,
    horizon: int,
    chunk: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate X'X and X'Y over sampled positions, in float64."""
    dim = feature_dim(window, embeddings.shape[1])
    xtx = np.zeros((dim, dim), dtype=np.float64)
    xty = np.zeros((dim, embeddings.shape[1]), dtype=np.float64)
    buf = np.zeros((chunk, dim), dtype=np.float32)

    for start in range(0, positions.size, chunk):
        sl = slice(start, start + chunk)
        idx, own = positions[sl], owners[sl]
        # `owners` is aligned with `positions`, not indexed by position: each
        # entry names the playlist a sampled position belongs to, which is what
        # bounds the prefix on the left.
        prefixes = [seq_tracks[starts[o] : p] for p, o in zip(idx, own, strict=True)]
        x = encode(prefixes, embeddings, window, out=buf[: idx.size])
        y = _targets(idx, own, seq_tracks, ends, embeddings, horizon)
        xtx += x.T.astype(np.float64) @ x.astype(np.float64)
        xty += x.T.astype(np.float64) @ y.astype(np.float64)
    return xtx, xty


def train(
    sequences,
    embeddings: np.ndarray,
    holdout_rows: set[int],
    cfg: SegueConfig,
    verbose: bool = True,
) -> SegueModel:
    """Fit on training playlists only; pick alpha on a held-out slice of them."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)

    # Every position with at least one predecessor, excluding evaluation playlists.
    starts, ends = sequences.offsets[:-1], sequences.offsets[1:]
    eligible_rows = np.array(
        [i for i in range(len(sequences)) if int(sequences.rows[i]) not in holdout_rows],
        dtype=np.int64,
    )
    # Split *by playlist*, so a prefix and its own continuation cannot land on
    # opposite sides of the alpha-selection split.
    shuffled = rng.permutation(eligible_rows)
    n_valid = max(1, int(0.05 * shuffled.size))
    valid_rows, fit_rows = shuffled[:n_valid], shuffled[n_valid:]

    def sample(rows: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
        pos = np.concatenate([np.arange(starts[r] + 1, ends[r]) for r in rows])
        owner = np.repeat(rows, (ends[rows] - starts[rows] - 1).astype(np.int64))
        if pos.size > cap:
            keep = rng.choice(pos.size, size=cap, replace=False)
            pos, owner = pos[keep], owner[keep]
        return pos, owner

    fit_pos, fit_owner = sample(fit_rows, cfg.max_train_positions)
    valid_pos, valid_owner = sample(valid_rows, 20_000)
    if verbose:
        print(
            f"fitting on {fit_pos.size:,} positions, validating on {valid_pos.size:,}", flush=True
        )

    xtx, xty = _gram(
        fit_pos, fit_owner, sequences.tracks, starts, ends, embeddings, cfg.window, cfg.horizon
    )

    valid_x = encode(
        [sequences.tracks[starts[o] : p] for p, o in zip(valid_pos, valid_owner, strict=True)],
        embeddings,
        cfg.window,
    )
    valid_y = _targets(valid_pos, valid_owner, sequences.tracks, ends, embeddings, cfg.horizon)

    best: tuple[float, float, np.ndarray] | None = None
    eye = np.eye(xtx.shape[0])
    eye[-1, -1] = 0.0  # never penalise the intercept
    for alpha in cfg.ridge_alphas:
        beta = np.linalg.solve(xtx + alpha * eye, xty)
        cos = float((l2_normalize(valid_x @ beta) * valid_y).sum(axis=1).mean())
        if verbose:
            print(f"  alpha={alpha:>9,.0f}  valid cosine={cos:.4f}", flush=True)
        if best is None or cos > best[0]:
            best = (cos, alpha, beta)
    assert best is not None

    return SegueModel(
        beta=best[2].astype(np.float32),
        window=cfg.window,
        horizon=cfg.horizon,
        alpha=best[1],
        emb_dim=embeddings.shape[1],
        valid_cosine=best[0],
        seconds=round(time.perf_counter() - t0, 1),
    )
