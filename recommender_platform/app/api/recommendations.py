from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from ..schemas.recommendation import RecResponse
from ..services.recommender import RecommenderService
from ..core.cache import get_json, set_json
from ..core.config import settings
from ..db.session import get_db

router = APIRouter()

@router.get("/user/{user_id}", response_model=RecResponse)
async def get_user_recommendations(
    user_id: str,
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    """Get personalized recommendations for a specific user"""
    cache_key = f"recs:user:{user_id}:limit:{limit}"
    cached = get_json(cache_key)
    if cached is not None:
        return cached

    service = RecommenderService(db, request.app.state.engine, request.app.state.ranker)
    results = await service.get_personalized_recs(user_id, limit)
    set_json(cache_key, results, 60)
    return results

@router.get("/item/{asin}", response_model=RecResponse)
async def get_similar_items(
    asin: str,
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    """Get items similar to a specific product (Content-based)"""
    service = RecommenderService(db, request.app.state.engine, request.app.state.ranker)
    results = await service.get_similar_items(asin, limit)
    return results

@router.get("/trending", response_model=RecResponse)
async def get_trending_items(
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    """Get globally trending items based on recent interactions"""
    cache_key = f"recs:trending:limit:{limit}"
    cached = get_json(cache_key)
    if cached is not None:
        return cached

    service = RecommenderService(db, request.app.state.engine, request.app.state.ranker)
    results = await service.get_trending(limit)
    set_json(cache_key, results, settings.CACHE_TTL)
    return results

@router.get("/bundle/{asin}", response_model=RecResponse)
async def get_bundle_recommendations(
    asin: str,
    request: Request,
    limit: int = 5,
    db: Session = Depends(get_db),
):
    """Get 'Frequently Bought Together' items for a bundle"""
    service = RecommenderService(db, request.app.state.engine, request.app.state.ranker)
    results = await service.get_bundles(asin, limit)
    return results
