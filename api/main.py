# api/main.py
"""
RecForge FastAPI backend.

Endpoints:
  GET /recommend/{user_id}          Personalized hybrid recommendations
  GET /similar/{item_id}            Content-based similar items
  GET /trending                     Popularity-based recommendations
  GET /explain/{user_id}/{item_id}  Explanation for a specific recommendation
  GET /metrics                      Evaluation metrics from the last run
  GET /health                       Health check
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from api.schemas import (
    RecommendationResponse,
    RecommendationItem,
    SimilarItemsResponse,
    TrendingResponse,
    ExplainResponse,
    MetricsResponse,
)
from api.dependencies import lifespan, get_state

app = FastAPI(
    title="RecForge API",
    description="Hybrid recommendation engine: content-based + collaborative filtering",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(get_state())}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Return personalized hybrid recommendations for a user.
    Falls back to popularity if the user is unknown (cold start).
    """
    state = get_state()
    hybrid_rec = state["hybrid_rec"]
    ratings = state["ratings"]

    results = hybrid_rec.recommend(user_id=user_id, ratings=ratings, n=n)

    if results.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}.")

    items_out = []
    for _, row in results.iterrows():
        items_out.append(RecommendationItem(
            item_id=int(row["item_id"]),
            title=str(row["title"]),
            genres=str(row.get("genres", "")),
            score=float(row["hybrid_score"]),
            explanation=str(row.get("explanation", "")),
        ))

    rec_type = str(results.iloc[0].get("recommendation_type", "hybrid"))

    return RecommendationResponse(
        user_id=user_id,
        recommendations=items_out,
        recommendation_type=rec_type,
        n=len(items_out),
    )


@app.get("/similar/{item_id}", response_model=SimilarItemsResponse)
def similar_items(
    item_id: int,
    n: int = Query(default=10, ge=1, le=50),
):
    """
    Return content-based similar items for a given item.
    """
    state = get_state()
    content_rec = state["content_rec"]
    items_df = state["items"]

    if item_id not in content_rec.item_ids:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")

    results = content_rec.similar_items(item_id=item_id, n=n)
    item_title_row = items_df[items_df["item_id"] == item_id]
    item_title = item_title_row.iloc[0]["title"] if not item_title_row.empty else str(item_id)

    items_out = []
    for _, row in results.iterrows():
        explanation = content_rec.explain(
            query_item_id=item_id,
            result_item_id=int(row["item_id"]),
        )
        items_out.append(RecommendationItem(
            item_id=int(row["item_id"]),
            title=str(row["title"]),
            genres=str(row.get("genres", "")),
            score=float(row["similarity_score"]),
            explanation=explanation,
        ))

    return SimilarItemsResponse(
        item_id=item_id,
        title=item_title,
        similar_items=items_out,
        n=len(items_out),
    )


@app.get("/trending", response_model=TrendingResponse)
def trending(
    n: int = Query(default=10, ge=1, le=50),
):
    """
    Return popularity-based trending items.
    """
    state = get_state()
    popularity_rec = state["popularity_rec"]

    results = popularity_rec.recommend(n=n)

    items_out = []
    for _, row in results.iterrows():
        explanation = popularity_rec.explain(int(row["item_id"]))
        items_out.append(RecommendationItem(
            item_id=int(row["item_id"]),
            title=str(row["title"]),
            genres=str(row.get("genres", "")),
            score=float(row["score"]),
            explanation=explanation,
        ))

    return TrendingResponse(trending_items=items_out, n=len(items_out))


@app.get("/explain/{user_id}/{item_id}", response_model=ExplainResponse)
def explain(user_id: int, item_id: int):
    """
    Return a human-readable explanation for why item_id was recommended to user_id.
    """
    state = get_state()
    collab_rec = state["collab_rec"]
    content_rec = state["content_rec"]
    ratings = state["ratings"]
    items_df = state["items"]

    item_row = items_df[items_df["item_id"] == item_id]
    if item_row.empty:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")
    item_title = item_row.iloc[0]["title"]

    user_history = ratings[ratings["user_id"] == user_id]["item_id"].tolist()

    # Try collaborative explanation first; fall back to content-based
    if user_history and user_id in collab_rec.user_id_map:
        explanation = collab_rec.explain(
            user_id=user_id,
            item_id=item_id,
            user_history=user_history,
        )
    elif user_history:
        # Use the most recently rated item as the reference for content explanation
        last_item = user_history[-1]
        explanation = content_rec.explain(
            query_item_id=last_item,
            result_item_id=item_id,
        )
    else:
        from models.popularity import PopularityRecommender
        explanation = state["popularity_rec"].explain(item_id)

    return ExplainResponse(
        user_id=user_id,
        item_id=item_id,
        item_title=item_title,
        explanation=explanation,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    """
    Return evaluation metrics from the last evaluation run.
    """
    state = get_state()
    eval_metrics = state.get("eval_metrics", {})

    if not eval_metrics:
        raise HTTPException(
            status_code=503,
            detail="Evaluation results not found. Run evaluation/run_evaluation.py first."
        )

    # Return the Hybrid model metrics as the primary metrics
    hybrid_metrics = eval_metrics.get("Hybrid", {})

    return MetricsResponse(
        precision_at_10=hybrid_metrics.get("precision@10", 0.0),
        recall_at_10=hybrid_metrics.get("recall@10", 0.0),
        ndcg_at_10=hybrid_metrics.get("ndcg@10", 0.0),
        coverage=hybrid_metrics.get("coverage", 0.0),
        users_evaluated=hybrid_metrics.get("users_evaluated", 0),
    )