"""HTTP API.

The heavy objects (catalog, embeddings, indexes) load once at startup and are
shared across requests; loading them per request would dominate the latency
budget by two orders of magnitude.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import ARTIFACTS
from ..types import GeneratedPlaylist, PlaylistIntent

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ..catalog import Catalog
    from ..engine import CadenceEngine
    from ..models.reranker import Reranker
    from ..planner.base import get_planner

    catalog = Catalog.load()
    reranker = None
    path = ARTIFACTS / "reranker.pkl"
    if path.exists():
        reranker = Reranker.load(path)
    engine = CadenceEngine(
        catalog,
        planner=get_planner(os.environ.get("CADENCE_LLM_PROVIDER", "offline")),
        reranker=reranker,
    )
    # Warm the lazily-built lookup tables so the first request is not an outlier.
    _ = engine.known_tags, engine.head_threshold, catalog.artist_ids
    _ = catalog._artist_name_index, catalog._track_name_index, catalog.tag_matrix_csc
    _state["engine"] = engine
    yield
    _state.clear()


app = FastAPI(
    title="Cadence",
    version="0.1.0",
    description="Natural-language playlist generation with grounded retrieval.",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    n_tracks: int | None = Field(None, ge=1, le=100)
    write_copy: bool = True


class ExplainResponse(BaseModel):
    intent: PlaylistIntent
    channel_sizes: dict[str, int]
    tags: list[str]
    seeds_resolved: dict
    mask_applied: dict[str, int]
    timings_ms: dict[str, float]


def _engine():
    engine = _state.get("engine")
    if engine is None:  # pragma: no cover - only before startup completes
        raise HTTPException(503, "engine not ready")
    return engine


@app.get("/health")
def health() -> dict:
    engine = _state.get("engine")
    return {
        "status": "ok" if engine else "starting",
        "n_tracks": len(engine.catalog) if engine else 0,
    }


@app.get("/info")
def info() -> dict:
    return _engine().catalog.meta


@app.post("/generate", response_model=GeneratedPlaylist)
def generate(req: GenerateRequest) -> GeneratedPlaylist:
    return _engine().generate(req.query, n_tracks=req.n_tracks, write_copy=req.write_copy)


@app.post("/explain", response_model=ExplainResponse)
def explain(req: GenerateRequest) -> ExplainResponse:
    """Return the retrieval trace without assembling a playlist.

    This is the debugging surface: it answers "why did the engine even consider
    these tracks", which is a different question from "why is this track here".
    """
    engine = _engine()
    plan = engine.planner.plan(req.query, engine.known_tags)
    trace = engine.retrieve(plan.intent)
    return ExplainResponse(
        intent=plan.intent,
        channel_sizes=trace.channel_sizes,
        tags=[engine.catalog.tag_vocab[c] for c in trace.tag_cols],
        seeds_resolved=trace.seed_detail,
        mask_applied=trace.mask_applied,
        timings_ms={k: round(v, 2) for k, v in trace.timings_ms.items()},
    )


@app.get("/tracks/{index}")
def track(index: int) -> dict:
    engine = _engine()
    if not (0 <= index < len(engine.catalog)):
        raise HTTPException(404, "unknown track index")
    return {
        "track": engine.catalog.track(index).model_dump(),
        "top_tags": engine.catalog.top_tags(index, k=8),
    }
