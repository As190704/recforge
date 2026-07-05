# evaluation/run_evaluation.py
"""
Runs evaluation for all three models and prints a results table.

Models evaluated:
  1. Popularity baseline
  2. Collaborative filtering (ALS)
  3. Hybrid (content + collaborative)

All models are evaluated on the same leave-last-out test split,
trained on the same training set, at K=10.

Run: python -m evaluation.run_evaluation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tabulate import tabulate

from data.ingest import main as ingest_main
from models.popularity import PopularityRecommender
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from evaluation.metrics import leave_last_out_split, evaluate_recommender

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")
RESULTS_PATH = Path("evaluation/results.json")
K = 10


def main():
    # Load data
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    all_item_ids = items["item_id"].tolist()

    # Split
    train_ratings, test_df = leave_last_out_split(ratings)

    # ---- Train models on training set only ----
    print("\n[1/3] Fitting PopularityRecommender...")
    popularity_rec = PopularityRecommender(min_votes=10)
    popularity_rec.fit(train_ratings, items)

    print("\n[2/3] Fitting ContentBasedRecommender...")
    content_rec = ContentBasedRecommender()
    content_rec.fit(items)  # content model uses item metadata only, not interactions
    content_rec.save()

    print("\n[3/3] Fitting CollaborativeRecommender...")
    collab_rec = CollaborativeRecommender(factors=64, iterations=20)
    collab_rec.fit(train_ratings, items)
    collab_rec.save()

    hybrid_rec = HybridRecommender(
        content_rec=content_rec,
        collab_rec=collab_rec,
        popularity_rec=popularity_rec,
        alpha=0.6,
    )

    # ---- Evaluation functions ----
    def popularity_fn(user_id, exclude_item_ids):
        recs = popularity_rec.recommend(n=K + len(exclude_item_ids), exclude_item_ids=exclude_item_ids)
        return recs["item_id"].tolist()[:K]

    def collab_fn(user_id, exclude_item_ids):
        if user_id not in collab_rec.user_id_map:
            return popularity_fn(user_id, exclude_item_ids)
        recs = collab_rec.recommend(user_id=user_id, n=K, exclude_item_ids=exclude_item_ids)
        return recs["item_id"].tolist()

    def hybrid_fn(user_id, exclude_item_ids):
        recs = hybrid_rec.recommend(user_id=user_id, ratings=train_ratings, n=K)
        return recs["item_id"].tolist()

    # ---- Run evaluation ----
    print(f"\nEvaluating at K={K} on leave-last-out test split...\n")

    results = {}

    print("--- Popularity Baseline ---")
    results["Popularity"] = evaluate_recommender(
        recommend_fn=popularity_fn,
        test_df=test_df,
        all_item_ids=all_item_ids,
        k=K,
    )

    print("\n--- Collaborative Filtering (ALS) ---")
    results["Collaborative (ALS)"] = evaluate_recommender(
        recommend_fn=collab_fn,
        test_df=test_df,
        all_item_ids=all_item_ids,
        k=K,
    )

    print("\n--- Hybrid (Content + Collaborative) ---")
    results["Hybrid"] = evaluate_recommender(
        recommend_fn=hybrid_fn,
        test_df=test_df,
        all_item_ids=all_item_ids,
        k=K,
    )

    # ---- Print results table ----
    table_rows = []
    for model_name, metrics in results.items():
        table_rows.append([
            model_name,
            f"{metrics[f'precision@{K}']:.4f}",
            f"{metrics[f'recall@{K}']:.4f}",
            f"{metrics[f'ndcg@{K}']:.4f}",
            f"{metrics['coverage']:.4f}",
            metrics["users_evaluated"],
        ])

    print("\n" + "=" * 70)
    print(tabulate(
        table_rows,
        headers=["Model", f"Precision@{K}", f"Recall@{K}", f"NDCG@{K}", "Coverage", "Users"],
        tablefmt="github",
    ))
    print("=" * 70)

    # ---- Save results ----
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()