# evaluation/metrics.py
"""
Evaluation metrics for recommendation systems.

All metrics are computed on a held-out test set using leave-last-out splitting:
  - For each user, the most recent interaction is held out as the test item.
  - The model must rank that item within its top-K predictions.

Metrics implemented:
  - Precision@K: fraction of top-K recommendations that are relevant
  - Recall@K: fraction of relevant items retrieved in top-K
  - NDCG@K: normalized discounted cumulative gain (accounts for rank position)
  - Coverage: fraction of all items that appear in at least one recommendation
"""

import numpy as np
import pandas as pd
from typing import Callable


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-k items that are relevant."""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    DCG = sum(rel_i / log2(i+2)) for i in 0..K-1
    Ideal DCG assumes all relevant items are at top positions.
    """
    top_k = recommended[:k]
    dcg = sum(
        (1.0 / np.log2(i + 2)) for i, item in enumerate(top_k) if item in relevant
    )
    # Ideal DCG: all relevant items at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def leave_last_out_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets using leave-last-out.
    For each user, the most recent interaction goes to test.
    Users with fewer than 2 interactions are excluded entirely.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])
    test_rows = ratings.groupby("user_id").tail(1)
    train_rows = ratings.drop(index=test_rows.index)

    # Only keep users with at least 1 training interaction
    valid_users = train_rows["user_id"].unique()
    test_rows = test_rows[test_rows["user_id"].isin(valid_users)]

    print(f"Train: {len(train_rows):,} interactions, {train_rows['user_id'].nunique():,} users")
    print(f"Test:  {len(test_rows):,} interactions, {test_rows['user_id'].nunique():,} users")
    return train_rows.reset_index(drop=True), test_rows.reset_index(drop=True)


def evaluate_recommender(
    recommend_fn: Callable[[int, list], list],
    test_df: pd.DataFrame,
    all_item_ids: list,
    k: int = 10,
    sample_users: int = None,
) -> dict:
    """
    Evaluate a recommendation function against the test set.

    Args:
        recommend_fn: callable(user_id, exclude_item_ids) -> list of item_ids (ordered)
        test_df: DataFrame with columns user_id, item_id (the held-out items)
        all_item_ids: full list of item IDs in the catalogue
        k: cutoff rank
        sample_users: if set, evaluate on a random sample of users (for speed)

    Returns:
        dict with precision, recall, ndcg (all @K) and coverage
    """
    user_test = dict(zip(test_df["user_id"], test_df["item_id"]))
    users = list(user_test.keys())

    if sample_users and sample_users < len(users):
        rng = np.random.default_rng(42)
        users = rng.choice(users, size=sample_users, replace=False).tolist()

    precisions, recalls, ndcgs = [], [], []
    all_recommended = set()

    print(f"Evaluating on {len(users)} users at K={k}...")

    for user_id in users:
        test_item = user_test[user_id]
        relevant = {test_item}

        try:
            recommended = recommend_fn(user_id, exclude_item_ids=[test_item])
        except Exception:
            continue

        if not recommended:
            continue

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        all_recommended.update(recommended[:k])

    coverage = len(all_recommended) / len(all_item_ids) if all_item_ids else 0.0

    results = {
        f"precision@{k}": np.mean(precisions),
        f"recall@{k}": np.mean(recalls),
        f"ndcg@{k}": np.mean(ndcgs),
        "coverage": coverage,
        "users_evaluated": len(precisions),
    }
    return results