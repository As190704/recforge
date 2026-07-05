# api/schemas.py
"""
Pydantic schemas for all API request and response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    genres: str
    score: float = Field(..., description="Hybrid or content similarity score in [0, 1]")
    explanation: str


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    recommendation_type: str = Field(
        ..., description="'hybrid', 'popularity', or 'cold_start'"
    )
    n: int


class SimilarItemsResponse(BaseModel):
    item_id: int
    title: str
    similar_items: List[RecommendationItem]
    n: int


class TrendingResponse(BaseModel):
    trending_items: List[RecommendationItem]
    n: int


class ExplainResponse(BaseModel):
    user_id: int
    item_id: int
    item_title: str
    explanation: str


class MetricsResponse(BaseModel):
    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float
    coverage: float
    users_evaluated: int