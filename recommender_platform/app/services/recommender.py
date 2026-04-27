import numpy as np
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from ..db import models
from ..ml.engine.hybrid import HybridEngine
from ..ml.ranking.ranker import XGBRanker

class RecommenderService:
    def __init__(self, db: Session):
        self.db = db
        self.engine = HybridEngine(qdrant_host="qdrant")
        self.ranker = XGBRanker()
        
    async def get_personalized_recs(self, user_id: str, limit: int) -> Dict:
        # 1. Fetch user features
        user = self.db.query(models.User).filter(models.User.external_id == user_id).first()
        if not user:
            return await self.get_trending(limit)
            
        # 2. Get user history for ranking features
        history = self.db.query(models.Item)\
            .join(models.Interaction)\
            .filter(models.Interaction.user_id == user.id)\
            .order_by(models.Interaction.timestamp.desc()).all()
            
        if not history:
            return await self.get_trending(limit)
            
        # 3. Retrieval
        query_item = history[0]
        embedding = self.engine.get_embedding(query_item.title)
        candidates = self.engine.search_candidates(embedding, limit=50)
        
        # 4. Ranking (XGBoost)
        user_hist_dicts = [{"category": h.category} for h in history]
        query_dict = {"price": query_item.price, "category": query_item.category}
        
        ranked_recs = self.ranker.rank(query_dict, candidates, user_hist_dicts)
        
        return {
            "user_id": user_id,
            "recommendations": ranked_recs[:limit],
            "strategy": "xgboost_ranked_hybrid"
        }

    async def get_similar_items(self, asin: str, limit: int) -> Dict:
        item = self.db.query(models.Item).filter(models.Item.asin == asin).first()
        if not item:
            return {"error": "Item not found"}
            
        embedding = self.engine.get_embedding(item.title)
        candidates = self.engine.search_candidates(embedding, limit=limit)
        
        return {
            "seed_item": {"asin": asin},
            "recommendations": candidates,
            "strategy": "content_similarity"
        }

    async def get_trending(self, limit: int) -> Dict:
        # Simple popularity based on interaction counts in last 7 days
        trending_items = self.db.query(models.Item)\
            .join(models.Interaction)\
            .group_by(models.Item.id)\
            .order_by(models.Item.stars.desc())[:limit]
            
        recs = [{
            "asin": item.asin, 
            "title": item.title, 
            "price": item.price,
            "stars": item.stars,
            "category": item.category
        } for item in trending_items]
        
        return {
            "recommendations": recs,
            "strategy": "global_trending"
        }

    async def get_bundles(self, asin: str, limit: int) -> Dict:
        # Placeholder for Association Rule Mining / Co-occurrence logic
        return await self.get_similar_items(asin, limit)
