# models/content_based.py
"""
Content-based recommender using Sentence Transformers + FAISS.

Each item is represented by an embedding of its description field
(title + genres). At query time, we retrieve the K nearest neighbours
in embedding space using a FAISS index.

Model: all-MiniLM-L6-v2 (384-dim, fast, good quality)
Index: IndexFlatIP (exact inner product search on L2-normalized vectors = cosine similarity)
"""

import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")
MODEL_NAME = "all-MiniLM-L6-v2"


class ContentBasedRecommender:
    def __init__(self):
        self.model = None
        self.index = None
        self.items: pd.DataFrame = None
        self.item_ids: np.ndarray = None  # maps FAISS row index → item_id

    def fit(self, items: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Embed all item descriptions and build a FAISS index.
        items must have columns: item_id, title, genres, description.
        """
        print(f"Loading sentence transformer: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.items = items.reset_index(drop=True)
        self.item_ids = self.items["item_id"].values

        descriptions = self.items["description"].tolist()
        print(f"Embedding {len(descriptions)} items...")
        embeddings = self.model.encode(
            descriptions,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 norm → cosine sim via inner product
        )

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        return self

    def similar_items(self, item_id: int, n: int = 10) -> pd.DataFrame:
        """
        Return n items most similar to item_id (excluding itself).
        Returns DataFrame with: item_id, title, genres, similarity_score.
        """
        if item_id not in self.item_ids:
            raise ValueError(f"item_id {item_id} not found in index.")

        idx = np.where(self.item_ids == item_id)[0][0]
        query_vec = self.index.reconstruct(int(idx)).reshape(1, -1)

        # Retrieve n+1 to exclude the query item itself
        scores, indices = self.index.search(query_vec, n + 1)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, i in zip(scores, indices):
            if i == idx:  # skip self
                continue
            row = self.items.iloc[i]
            results.append({
                "item_id": int(row["item_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "similarity_score": float(score),
            })

        return pd.DataFrame(results[:n])

    def embed_items(self, item_ids: list) -> np.ndarray:
        """
        Return mean embedding for a list of item_ids.
        Used by the hybrid model to represent a user's taste profile.
        """
        indices = [np.where(self.item_ids == iid)[0][0] for iid in item_ids if iid in self.item_ids]
        if not indices:
            return None
        vecs = np.array([self.index.reconstruct(int(i)) for i in indices])
        mean_vec = vecs.mean(axis=0, keepdims=True).astype(np.float32)
        # Re-normalize
        faiss.normalize_L2(mean_vec)
        return mean_vec

    def recommend_for_profile(self, profile_vec: np.ndarray, n: int = 10, exclude_ids: list = None) -> pd.DataFrame:
        """
        Given a pre-computed profile vector, return top-n similar items.
        """
        scores, indices = self.index.search(profile_vec, n + len(exclude_ids or []) + 1)
        scores = scores[0]
        indices = indices[0]

        exclude_set = set(exclude_ids or [])
        results = []
        for score, i in zip(scores, indices):
            if i < 0:
                continue
            row = self.items.iloc[i]
            if int(row["item_id"]) in exclude_set:
                continue
            results.append({
                "item_id": int(row["item_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "content_score": float(score),
            })
            if len(results) >= n:
                break

        return pd.DataFrame(results)

    def explain(self, query_item_id: int, result_item_id: int) -> str:
        """
        Generate a human-readable explanation for a content-based recommendation.
        Compares genres between query and result items.
        """
        query_row = self.items[self.items["item_id"] == query_item_id]
        result_row = self.items[self.items["item_id"] == result_item_id]
        if query_row.empty or result_row.empty:
            return "Similar content to what you viewed."

        q = query_row.iloc[0]
        r = result_row.iloc[0]

        q_genres = set(q["genres"].split("|")) if q["genres"] else set()
        r_genres = set(r["genres"].split("|")) if r["genres"] else set()
        shared = q_genres & r_genres

        if shared:
            shared_str = ", ".join(sorted(shared))
            return (
                f"Recommended because you viewed '{q['title']}'. "
                f"Both are {shared_str} films."
            )
        return f"Similar in style and theme to '{q['title']}'."

    def save(self):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(ARTIFACTS_DIR / "faiss_index.bin"))
        self.items.to_parquet(ARTIFACTS_DIR / "items_with_meta.parquet", index=False)
        np.save(ARTIFACTS_DIR / "item_ids.npy", self.item_ids)
        print("ContentBasedRecommender artifacts saved.")

    def load(self) -> "ContentBasedRecommender":
        self.index = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.bin"))
        self.items = pd.read_parquet(ARTIFACTS_DIR / "items_with_meta.parquet")
        self.item_ids = np.load(ARTIFACTS_DIR / "item_ids.npy")
        print(f"ContentBasedRecommender loaded: {self.index.ntotal} vectors.")
        return self


if __name__ == "__main__":
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")

    rec = ContentBasedRecommender()
    rec.fit(items)
    rec.save()

    # Quick sanity check — find items similar to Toy Story (item_id=1)
    print("\nItems similar to Toy Story (item_id=1):")
    results = rec.similar_items(item_id=1, n=5)
    print(results.to_string(index=False))
    print()
    print("Explanation:", rec.explain(1, int(results.iloc[0]["item_id"])))