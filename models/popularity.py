# models/popularity.py
"""
Popularity-based recommender.

Serves as the cold-start fallback for users with no interaction history.
Ranks items by a Bayesian average (Wilson-style) to avoid small-sample bias
where a single 5-star rating beats a 1000-rating blockbuster.

Formula:
    score = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = number of ratings for this item
        m = minimum vote threshold (configurable)
        R = mean rating for this item
        C = global mean rating
"""

import pandas as pd
import numpy as np
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")


class PopularityRecommender:
    def __init__(self, min_votes: int = 50):
        self.min_votes = min_votes
        self.scores: pd.DataFrame = None
        self.items: pd.DataFrame = None

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> "PopularityRecommender":
        """
        Compute Bayesian popularity scores from the ratings DataFrame.
        Stores a sorted DataFrame of (item_id, score, vote_count, mean_rating).
        """
        C = ratings["rating"].mean()  # global mean

        item_stats = (
            ratings.groupby("item_id")["rating"]
            .agg(vote_count="count", mean_rating="mean")
            .reset_index()
        )

        m = self.min_votes
        item_stats["score"] = (
            (item_stats["vote_count"] / (item_stats["vote_count"] + m)) * item_stats["mean_rating"]
            + (m / (item_stats["vote_count"] + m)) * C
        )

        # Merge item metadata for display
        self.scores = (
            item_stats.merge(items[["item_id", "title", "genres"]], on="item_id", how="left")
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        self.items = items
        print(f"PopularityRecommender fitted on {len(self.scores)} items.")
        return self

    def recommend(self, n: int = 10, exclude_item_ids: list = None) -> pd.DataFrame:
        """
        Return top-n popular items, optionally excluding specific item IDs.
        Returns a DataFrame with columns: item_id, title, genres, score, vote_count.
        """
        result = self.scores.copy()
        if exclude_item_ids:
            result = result[~result["item_id"].isin(exclude_item_ids)]
        return result[["item_id", "title", "genres", "score", "vote_count"]].head(n)

    def explain(self, item_id: int) -> str:
        """
        Return a human-readable explanation for why this item appears in trending.
        """
        row = self.scores[self.scores["item_id"] == item_id]
        if row.empty:
            return "This is a popular item."
        r = row.iloc[0]
        return (
            f"'{r['title']}' is trending with {int(r['vote_count'])} ratings "
            f"and an average score of {r['mean_rating']:.1f}/5.0."
        )

    def save(self):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.scores.to_parquet(ARTIFACTS_DIR / "popularity_scores.parquet", index=False)
        print("PopularityRecommender artifacts saved.")

    def load(self) -> "PopularityRecommender":
        self.scores = pd.read_parquet(ARTIFACTS_DIR / "popularity_scores.parquet")
        print("PopularityRecommender artifacts loaded.")
        return self


if __name__ == "__main__":
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")

    rec = PopularityRecommender(min_votes=50)
    rec.fit(ratings, items)
    rec.save()

    print("\nTop 10 trending:")
    print(rec.recommend(n=10).to_string(index=False))
    print()
    sample_id = rec.scores.iloc[0]["item_id"]
    print("Explanation:", rec.explain(int(sample_id)))