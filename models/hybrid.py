# models/hybrid.py
"""
Hybrid recommendation engine.

Blends content-based and collaborative filtering scores using a
tunable alpha parameter:

    hybrid_score = alpha * collab_score_normalized
                 + (1 - alpha) * content_score_normalized

alpha = 1.0  → pure collaborative filtering
alpha = 0.0  → pure content-based filtering
alpha = 0.5  → equal weight (default)

For cold-start users (no history), falls back to the popularity model.
For known users with history, both content and collaborative signals
are used. Content signal is derived from the user's mean item embedding.

Score normalization: min-max per recommendation batch so scores
from different models are on the same [0, 1] scale before blending.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.popularity import PopularityRecommender

PROCESSED_DIR = Path("data/processed")


def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1]. Returns 0.5 if all values are equal."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


class HybridRecommender:
    def __init__(
        self,
        content_rec: ContentBasedRecommender,
        collab_rec: CollaborativeRecommender,
        popularity_rec: PopularityRecommender,
        alpha: float = 0.6,   # weight on collaborative signal
        candidate_pool: int = 50,  # how many candidates to pull from each model
    ):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.popularity_rec = popularity_rec
        self.alpha = alpha
        self.candidate_pool = candidate_pool

    def _get_user_history(self, user_id: int, ratings: pd.DataFrame) -> list:
        """Return list of item_ids the user has already interacted with."""
        return ratings[ratings["user_id"] == user_id]["item_id"].tolist()

    def recommend(
        self,
        user_id: int,
        ratings: pd.DataFrame,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Main recommendation method.

        Returns a DataFrame with columns:
          item_id, title, genres, hybrid_score, collab_score_norm,
          content_score_norm, explanation
        """
        user_history = self._get_user_history(user_id, ratings)
        is_cold_start = len(user_history) == 0 or user_id not in self.collab_rec.user_id_map

        if is_cold_start:
            return self._cold_start_recommend(n=n, exclude_ids=user_history)

        return self._warm_recommend(user_id=user_id, user_history=user_history, n=n)

    def _cold_start_recommend(self, n: int, exclude_ids: list) -> pd.DataFrame:
        """
        For new users: return popularity-based recommendations with explanations.
        """
        results = self.popularity_rec.recommend(n=n, exclude_item_ids=exclude_ids)
        results["hybrid_score"] = _minmax_normalize(results["score"])
        results["collab_score_norm"] = 0.0
        results["content_score_norm"] = 0.0
        results["explanation"] = results["item_id"].apply(
            lambda iid: self.popularity_rec.explain(int(iid))
        )
        results["recommendation_type"] = "popularity"
        return results[["item_id", "title", "genres", "hybrid_score",
                         "collab_score_norm", "content_score_norm",
                         "explanation", "recommendation_type"]]

    def _warm_recommend(self, user_id: int, user_history: list, n: int) -> pd.DataFrame:
        """
        For known users: blend content and collaborative signals.
        """
        k = self.candidate_pool

        # --- Collaborative candidates ---
        collab_df = self.collab_rec.recommend(
            user_id=user_id,
            n=k,
            exclude_item_ids=user_history,
        )

        # --- Content candidates from user taste profile ---
        profile_vec = self.content_rec.embed_items(user_history)
        if profile_vec is not None:
            content_df = self.content_rec.recommend_for_profile(
                profile_vec=profile_vec,
                n=k,
                exclude_ids=user_history,
            )
        else:
            content_df = pd.DataFrame(columns=["item_id", "title", "genres", "content_score"])

        # --- Merge candidates ---
        # Union of item IDs from both sources
        all_item_ids = set()
        if not collab_df.empty:
            all_item_ids.update(collab_df["item_id"].tolist())
        if not content_df.empty:
            all_item_ids.update(content_df["item_id"].tolist())

        # Build a merged score DataFrame
        collab_scores = {}
        if not collab_df.empty:
            for _, row in collab_df.iterrows():
                collab_scores[row["item_id"]] = row["collab_score"]

        content_scores = {}
        if not content_df.empty:
            for _, row in content_df.iterrows():
                content_scores[row["item_id"]] = row["content_score"]

        # Fill missing scores with 0
        rows = []
        for iid in all_item_ids:
            rows.append({
                "item_id": iid,
                "collab_score": collab_scores.get(iid, 0.0),
                "content_score": content_scores.get(iid, 0.0),
            })
        merged = pd.DataFrame(rows)

        # Normalize each signal to [0, 1] before blending
        # This is critical: raw ALS scores and cosine similarities are on different scales
        merged["collab_score_norm"] = _minmax_normalize(merged["collab_score"])
        merged["content_score_norm"] = _minmax_normalize(merged["content_score"])

        # Blend: alpha controls how much we trust collaborative vs content signal
        merged["hybrid_score"] = (
            self.alpha * merged["collab_score_norm"]
            + (1 - self.alpha) * merged["content_score_norm"]
        )

        merged = merged.sort_values("hybrid_score", ascending=False).head(n)

        # Join item metadata
        item_meta = self.content_rec.items[["item_id", "title", "genres"]]
        merged = merged.merge(item_meta, on="item_id", how="left")

        # Generate explanations
        merged["explanation"] = merged["item_id"].apply(
            lambda iid: self.collab_rec.explain(
                user_id=user_id,
                item_id=int(iid),
                user_history=user_history,
            )
        )
        merged["recommendation_type"] = "hybrid"

        return merged[["item_id", "title", "genres", "hybrid_score",
                        "collab_score_norm", "content_score_norm",
                        "explanation", "recommendation_type"]].reset_index(drop=True)


if __name__ == "__main__":
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")

    # Load all sub-models
    content_rec = ContentBasedRecommender().load()
    collab_rec = CollaborativeRecommender().load()
    popularity_rec = PopularityRecommender().load()

    hybrid = HybridRecommender(
        content_rec=content_rec,
        collab_rec=collab_rec,
        popularity_rec=popularity_rec,
        alpha=0.6,
    )

    print("Warm user (user_id=1) recommendations:")
    result = hybrid.recommend(user_id=1, ratings=ratings, n=5)
    for _, row in result.iterrows():
        print(f"  [{row['hybrid_score']:.3f}] {row['title']} — {row['explanation']}")

    print("\nCold start (user_id=99999) recommendations:")
    result = hybrid.recommend(user_id=99999, ratings=ratings, n=5)
    for _, row in result.iterrows():
        print(f"  [{row['hybrid_score']:.3f}] {row['title']} — {row['explanation']}")