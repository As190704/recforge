# models/collaborative.py
"""
Collaborative filtering using implicit ALS (Alternating Least Squares).

We treat ratings >= 3.5 as positive implicit feedback and weight
by the rating value so higher ratings signal stronger preference.

The implicit library expects a (items x users) CSR matrix.
We train, then reconstruct user and item factor matrices to compute
approximate recommendation scores via dot product.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import implicit
import pickle
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")


class CollaborativeRecommender:
    def __init__(self, factors: int = 64, iterations: int = 20, regularization: float = 0.1):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.user_factors: np.ndarray = None
        self.item_factors: np.ndarray = None
        self.user_id_map: dict = {}      # user_id → matrix row index
        self.item_id_map: dict = {}      # item_id → matrix row index
        self.reverse_item_map: dict = {} # matrix row index → item_id
        self.items: pd.DataFrame = None

    def _build_matrix(self, ratings: pd.DataFrame) -> sparse.csr_matrix:
        """
        Build a (users x items) CSR matrix with confidence values.
        Confidence = 1 + alpha * rating (alpha=10 is a common default).
        We use rating directly as the confidence weight here.
        """
        users = sorted(ratings["user_id"].unique())
        items = sorted(ratings["item_id"].unique())

        self.user_id_map = {uid: i for i, uid in enumerate(users)}
        self.item_id_map = {iid: i for i, iid in enumerate(items)}
        self.reverse_item_map = {i: iid for iid, i in self.item_id_map.items()}

        user_indices = ratings["user_id"].map(self.user_id_map).values
        item_indices = ratings["item_id"].map(self.item_id_map).values
        confidences = ratings["rating"].values.astype(np.float32)

        # Shape: (users x items)
        matrix = sparse.csr_matrix(
            (confidences, (user_indices, item_indices)),
            shape=(len(users), len(items)),
        )
        return matrix

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> "CollaborativeRecommender":
        self.items = items
        user_item_matrix = self._build_matrix(ratings)

        # implicit expects (items x users)
        item_user_matrix = user_item_matrix.T.tocsr()

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False,
        )
        print(f"Training ALS: factors={self.factors}, iterations={self.iterations}")
        self.model.fit(item_user_matrix)

        self.user_factors = self.model.user_factors  # shape: (n_users, factors)
        self.item_factors = self.model.item_factors  # shape: (n_items, factors)

        print(f"CollaborativeRecommender trained: {len(self.user_id_map)} users, {len(self.item_id_map)} items.")
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_item_ids: list = None) -> pd.DataFrame:
        """
        Return top-n recommended items for a given user_id.
        Returns DataFrame with: item_id, title, genres, collab_score.
        """
        if user_id not in self.user_id_map:
            return pd.DataFrame()  # unknown user → caller should use popularity fallback

        u_idx = self.user_id_map[user_id]
        # Score all items: dot product of user vector with all item vectors
        scores = self.item_factors @ self.user_factors[u_idx]

        # Build result, excluding items the user has already interacted with
        exclude_indices = set()
        if exclude_item_ids:
            for iid in exclude_item_ids:
                if iid in self.item_id_map:
                    exclude_indices.add(self.item_id_map[iid])

        ranked = np.argsort(scores)[::-1]
        results = []
        for i_idx in ranked:
            if i_idx in exclude_indices:
                continue
            item_id = self.reverse_item_map[i_idx]
            item_row = self.items[self.items["item_id"] == item_id]
            if item_row.empty:
                continue
            r = item_row.iloc[0]
            results.append({
                "item_id": int(item_id),
                "title": r["title"],
                "genres": r["genres"],
                "collab_score": float(scores[i_idx]),
            })
            if len(results) >= n:
                break

        return pd.DataFrame(results)

    def get_user_factor(self, user_id: int) -> np.ndarray | None:
        """Return the latent factor vector for a user, or None if unknown."""
        if user_id not in self.user_id_map:
            return None
        return self.user_factors[self.user_id_map[user_id]]

    def get_item_factor(self, item_id: int) -> np.ndarray | None:
        """Return the latent factor vector for an item, or None if unknown."""
        if item_id not in self.item_id_map:
            return None
        return self.item_factors[self.item_id_map[item_id]]

    def explain(self, user_id: int, item_id: int, user_history: list) -> str:
        """
        Generate a human-readable explanation using the items in the user's history
        whose latent factors are most similar to the recommended item's factors.
        """
        item_vec = self.get_item_factor(item_id)
        if item_vec is None or not user_history:
            return "Recommended based on patterns from users with similar taste."

        # Find the user's history item whose factor is closest to this item's factor
        best_sim = -np.inf
        best_item_id = None
        for hist_id in user_history:
            h_vec = self.get_item_factor(hist_id)
            if h_vec is None:
                continue
            sim = float(np.dot(item_vec, h_vec) / (np.linalg.norm(item_vec) * np.linalg.norm(h_vec) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_item_id = hist_id

        if best_item_id is not None:
            best_row = self.items[self.items["item_id"] == best_item_id]
            rec_row = self.items[self.items["item_id"] == item_id]
            if not best_row.empty and not rec_row.empty:
                return (
                    f"Users who enjoyed '{best_row.iloc[0]['title']}' "
                    f"also rated '{rec_row.iloc[0]['title']}' highly."
                )
        return "Recommended based on patterns from users with similar taste."

    def save(self):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "user_id_map": self.user_id_map,
            "item_id_map": self.item_id_map,
            "reverse_item_map": self.reverse_item_map,
        }
        with open(ARTIFACTS_DIR / "collaborative_model.pkl", "wb") as f:
            pickle.dump(artifact, f)
        self.items.to_parquet(ARTIFACTS_DIR / "collab_items.parquet", index=False)
        print("CollaborativeRecommender artifacts saved.")

    def load(self) -> "CollaborativeRecommender":
        with open(ARTIFACTS_DIR / "collaborative_model.pkl", "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.user_id_map = artifact["user_id_map"]
        self.item_id_map = artifact["item_id_map"]
        self.reverse_item_map = artifact["reverse_item_map"]
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        self.items = pd.read_parquet(ARTIFACTS_DIR / "collab_items.parquet")
        print("CollaborativeRecommender loaded.")
        return self


if __name__ == "__main__":
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")

    rec = CollaborativeRecommender(factors=64, iterations=20)
    rec.fit(ratings, items)
    rec.save()

    print("\nTop 5 recommendations for user_id=1:")
    print(rec.recommend(user_id=1, n=5).to_string(index=False))