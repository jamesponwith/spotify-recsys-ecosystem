"""Shifted-PPMI matrix factorisation.

Why this and not SGNS/word2vec: Levy & Goldberg (2014) showed that skip-gram
with negative sampling is implicitly factorising a shifted PMI matrix. Doing the
factorisation explicitly gives the same class of embedding with a deterministic,
dependency-light, single-pass implementation — no epochs, no learning rate, no
sampling noise, and exactly reproducible from a seed.

The same primitive serves two different jobs:

* ``playlist x track``  -> collaborative item embeddings (co-occurrence)
* ``track x title-token`` -> a joint track/tag space (free text <-> music)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd


def sppmi(
    matrix: sparse.spmatrix,
    *,
    shift_k: float = 5.0,
    popularity_damping: float = 1.0,
) -> sparse.csr_matrix:
    """Shifted positive pointwise mutual information.

    ``PMI(i, j) = log( c_ij * N / (c_i * c_j) )`` then ``max(0, PMI - log k)``.

    ``popularity_damping`` raises the column marginal to a power < 1, which is
    the standard word2vec unigram-smoothing trick: it stops globally popular
    items from being penalised into irrelevance while still discounting them.
    """
    coo = matrix.tocoo(copy=False)
    data = coo.data.astype(np.float64, copy=False)
    total = data.sum()
    if total <= 0:
        raise ValueError("cannot compute PPMI on an all-zero matrix")

    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    col_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    if popularity_damping != 1.0:
        col_sums = np.power(col_sums, popularity_damping)
        col_sums *= total / max(col_sums.sum(), 1e-12)

    denom = row_sums[coo.row] * col_sums[coo.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.log(np.maximum(data * total, 1e-12) / np.maximum(denom, 1e-12))
    vals -= np.log(max(shift_k, 1e-12))
    np.maximum(vals, 0.0, out=vals)

    mask = vals > 0
    out = sparse.coo_matrix(
        (vals[mask].astype(np.float32), (coo.row[mask], coo.col[mask])),
        shape=matrix.shape,
    ).tocsr()
    out.eliminate_zeros()
    return out


@dataclass
class Factorization:
    """Row and column embeddings living in one shared space.

    Both are scaled by ``sqrt(S)`` so that ``row_vectors @ col_vectors.T``
    approximates the original SPPMI values, which is what makes a *tag* vector
    directly comparable to a *track* vector.
    """

    rows: np.ndarray  # (n_rows, dim) float32
    cols: np.ndarray  # (n_cols, dim) float32
    singular_values: np.ndarray
    explained_energy: float


def factorize(
    matrix: sparse.spmatrix,
    dim: int,
    *,
    shift_k: float = 5.0,
    popularity_damping: float = 1.0,
    seed: int = 0,
    n_oversamples: int = 12,
    n_iter: int = 5,
) -> Factorization:
    m = sppmi(matrix, shift_k=shift_k, popularity_damping=popularity_damping)
    dim = int(min(dim, min(m.shape) - 1))
    u, s, vt = randomized_svd(
        m, n_components=dim, n_oversamples=n_oversamples, n_iter=n_iter, random_state=seed
    )
    scale = np.sqrt(np.maximum(s, 0.0)).astype(np.float32)
    rows = (u * scale).astype(np.float32)
    cols = (vt.T * scale).astype(np.float32)
    # Fraction of the SPPMI Frobenius energy the truncation retains — a cheap
    # sanity check that `dim` is not absurdly small for the matrix.
    total_energy = float((m.data.astype(np.float64) ** 2).sum())
    kept = float((s.astype(np.float64) ** 2).sum())
    return Factorization(
        rows=rows,
        cols=cols,
        singular_values=s.astype(np.float32),
        explained_energy=kept / total_energy if total_energy > 0 else 0.0,
    )


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows stay zero rather than becoming NaN."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)
