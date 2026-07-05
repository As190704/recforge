# api/dependencies.py
"""
Model loading at API startup.

All models are loaded once into memory when the FastAPI application starts.
This avoids reloading on every request (which would be extremely slow for
FAISS indexes and ALS matrices).

Pattern: FastAPI lifespan context manager populates a module-level state dict,
which is then accessed by route handlers via a dependency function.
"""

import pandas as pd
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.popularity import PopularityRecommender
from models.hybrid import HybridRecommender

PROCESSED_DIR = Path("data/processed")
RESULTS_PATH = Path("evaluation/results.json")

# Global state — populated at startup, read-only after that
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models into memory at startup, release at shutdown."""
    print("Loading models...")

    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")

    content_rec = ContentBasedRecommender().load()
    collab_rec = CollaborativeRecommender().load()
    popularity_rec = PopularityRecommender().load()

    hybrid_rec = HybridRecommender(
        content_rec=content_rec,
        collab_rec=collab_rec,
        popularity_rec=popularity_rec,
        alpha=0.6,
    )

    # Load evaluation metrics if available
    eval_metrics = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            eval_metrics = json.load(f)

    _state["content_rec"] = content_rec
    _state["collab_rec"] = collab_rec
    _state["popularity_rec"] = popularity_rec
    _state["hybrid_rec"] = hybrid_rec
    _state["ratings"] = ratings
    _state["items"] = items
    _state["eval_metrics"] = eval_metrics

    print("All models loaded. API ready.")
    yield
    print("Shutting down.")
    _state.clear()


def get_state() -> dict:
    return _state