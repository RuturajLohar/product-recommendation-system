from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict
from ..schemas.recommendation import RecResponse, ItemDetail
from ..services.recommender import RecommenderService
from ..db.session import get_db

router = APIRouter()

@router.get("/user/{user_id}", response_model=RecResponse)
async def get_user_recommendations(
    user_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get personalized recommendations for a specific user"""
    service = RecommenderService(db)
    results = await service.get_personalized_recs(user_id, limit)
    return results

@router.get("/item/{asin}", response_model=RecResponse)
async def get_similar_items(
    asin: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get items similar to a specific product (Content-based)"""
    service = RecommenderService(db)
    results = await service.get_similar_items(asin, limit)
    return results

@router.get("/trending", response_model=RecResponse)
async def get_trending_items(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get globally trending items based on recent interactions"""
    service = RecommenderService(db)
    results = await service.get_trending(limit)
    return results

@router.get("/bundle/{asin}", response_model=RecResponse)
async def get_bundle_recommendations(
    asin: str,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get 'Frequently Bought Together' items for a bundle"""
    service = RecommenderService(db)
    results = await service.get_bundles(asin, limit)
    return results
