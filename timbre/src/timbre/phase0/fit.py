"""Regress the 128-d folksonomy embedding from 22 acoustic features.

Two models on purpose. Ridge establishes what a *linear* read of the descriptors
buys; the MLP says whether the relationship is nonlinear enough to matter. If
ridge already reaches the MLP's number, that is a finding about the target, not
a shortfall of the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

from ..config import Phase0Config
from .data import l2_normalize


@dataclass
class FitResult:
    name: str
    prediction: np.ndarray  # (n_test, tag_dim), L2-normalised
    mean_cosine: float  # against the true held-out embedding
    seconds: float
    detail: dict
    model: Any  # kept so the winner can be persisted for the joint demo


def _cosine(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((l2_normalize(pred) * l2_normalize(truth)).sum(axis=1).mean())


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Phase0Config,
) -> FitResult:
    t0 = time.perf_counter()
    # RidgeCV's default generalised cross-validation is an exact leave-one-out
    # for this estimator, so no separate validation split is spent on alpha.
    model = RidgeCV(alphas=np.asarray(cfg.ridge_alphas))
    model.fit(x_train, y_train)
    pred = l2_normalize(model.predict(x_test).astype(np.float32))
    return FitResult(
        name="ridge",
        prediction=pred,
        mean_cosine=_cosine(pred, y_test),
        seconds=round(time.perf_counter() - t0, 1),
        detail={"alpha": float(model.alpha_)},
        model=model,
    )


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Phase0Config,
) -> FitResult:
    t0 = time.perf_counter()
    model = MLPRegressor(
        hidden_layer_sizes=cfg.mlp_hidden,
        activation="relu",
        solver="adam",
        batch_size=cfg.mlp_batch_size,
        max_iter=cfg.mlp_max_iter,
        early_stopping=cfg.mlp_early_stopping,
        n_iter_no_change=10,
        random_state=cfg.seed,
    )
    model.fit(x_train, y_train)
    pred = l2_normalize(model.predict(x_test).astype(np.float32))
    return FitResult(
        name="mlp",
        prediction=pred,
        mean_cosine=_cosine(pred, y_test),
        seconds=round(time.perf_counter() - t0, 1),
        detail={
            "hidden": list(cfg.mlp_hidden),
            "batch_size": cfg.mlp_batch_size,
            "n_iter": int(model.n_iter_),
            "hit_iter_cap": int(model.n_iter_) >= cfg.mlp_max_iter,
            "best_validation_score": round(float(model.best_validation_score_), 4),
        },
        model=model,
    )
