"""A persisted audio-to-folksonomy predictor.

Phase 0 answers a yes/no question, but the model it fits is reusable: given
Spotify's descriptors for a track Cadence has never seen on a playlist, it emits
a point in the same 128-d tag space Cadence retrieves in. That is exactly the
interface Phase 1's CNN would later implement, with a mel spectrogram in place
of eleven numbers -- so the joint demo can be built now and re-pointed later.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ARTIFACTS
from .phase0.features import encode


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


@dataclass
class TagPredictor:
    model: Any
    mu: np.ndarray
    sigma: np.ndarray
    kind: str
    mean_cosine: float

    def predict_matrix(self, x_raw: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
        x = ((x - self.mu) / self.sigma).astype(np.float32)
        return _l2(self.model.predict(x).astype(np.float32))

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict from a Cadence tracks frame -- the demo's entry point."""
        x_raw, _ = encode(frame)
        return self.predict_matrix(x_raw)

    def save(self, path: Path | None = None) -> Path:
        path = path or ARTIFACTS / "tag_predictor.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> TagPredictor:
        path = path or ARTIFACTS / "tag_predictor.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run `timbre phase0` first to fit the predictor."
            )
        obj = pickle.loads(path.read_bytes())
        assert isinstance(obj, cls)
        return obj
