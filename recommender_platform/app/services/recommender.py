import numpy as np
from typing import Any, Dict, List
from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from ..db import models
from ..ml.engine.hybrid import HybridEngine
from ..ml.ranking.ranker import XGBRanker
from ..services.llm import llm_service
from ..core.config import settings

class RecommenderService:
    def __init__(self, db: Session, engine: HybridEngine, ranker: XGBRanker):
        self.db = db
        self.engine = engine
        self.ranker = ranker
        self.llm = llm_service

    def _normalize_candidate(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure all response fields exist for the frontend/schema.
        out = dict(cand)
        out.setdefault("reviews", 0)
        out.setdefault("bought_in_last_month", 0)
        out.setdefault("is_best_seller", False)
        out.setdefault("product_url", None)
        out.setdefault("img_url", None)
        out.setdefault("product_id", out.get("asin"))
        out.setdefault("brand", None)
        out.setdefault("subcategory", None)
        out.setdefault("description", None)
        out.setdefault("features", [])
        out.setdefault("specifications", {})
        out.setdefault("keywords_tags", "")
        out.setdefault("rating", out.get("stars", 0))
        out.setdefault("review_count", out.get("reviews", 0))
        out.setdefault("popularity", 0.0)
        out.setdefault("best_seller", out.get("is_best_seller", False))
        out.setdefault("availability", "in_stock")
        out.setdefault("image_url", out.get("img_url"))
        return out

    @staticmethod
    def _item_content(item: models.Item) -> str:
        return " | ".join(
            value
            for value in (item.title, item.category, item.description)
            if value
        )

    def _demo_explanation(self, item: Dict[str, Any], user_interest: str | None = None) -> str:
        category = item.get("category") or "this category"
        stars = item.get("stars") or 0
        if user_interest:
            return f"Recommended because it matches your recent interest and is strong in {category}."
        if stars and float(stars) >= 4.5:
            return f"Popular {category} pick with a strong {stars}-star rating."
        return f"Recommended from similar products and trending signals in {category}."
        
    async def get_personalized_recs(self, user_id: str, limit: int) -> Dict:
        user = self.db.query(models.User).filter(models.User.external_id == user_id).first()
        if not user:
            return await self.get_trending(limit)
            
        history = (
            self.db.query(models.Item)
            .join(models.Interaction)
            .filter(models.Interaction.user_id == user.id)
            .order_by(models.Interaction.timestamp.desc())
            .limit(10)
            .all()
        )
            
        if not history:
            return await self.get_trending(limit)
            
        # Retrieval from mean-pooled user embedding
        titles = [h.title for h in history if h.title]
        embedding = self.engine.get_user_embedding(titles)
        candidates = self.engine.search_candidates(embedding, limit=50)
        history_asins = {h.asin for h in history if h.asin}
        candidates = [c for c in candidates if c.get("asin") not in history_asins]
        
        # Ranking (heuristic fallback if model not trained)
        user_hist_dicts = [{"category": h.category or "Unknown"} for h in history]
        query_item = history[0]
        query_dict = {"price": float(query_item.price or 0.0), "category": query_item.category or "Unknown"}
        
        ranked_recs = self.ranker.rank(query_dict, candidates, user_hist_dicts)

        # Diversity rerank (MMR) using vectors returned from Qdrant
        cand_vecs = [c.get("_vector") for c in ranked_recs if c.get("_vector") is not None]
        cand_no_vec = [c for c in ranked_recs if c.get("_vector") is None]
        diverse: List[Dict[str, Any]] = []
        if cand_vecs:
            candidate_vectors = np.asarray(cand_vecs, dtype=np.float32)
            diverse = self.engine.mmr_rerank(
                query_vector=np.asarray(embedding, dtype=np.float32),
                candidates=[c for c in ranked_recs if c.get("_vector") is not None],
                candidate_vectors=candidate_vectors,
                top_k=limit,
                lambda_param=0.7,
            )

        final = (diverse + cand_no_vec)[:limit]
        
        # Phase 5: LLM Reranking (Groq) & Explanation (Gemini)
        if settings.USE_LLM_RERANKING and settings.GROQ_API_KEY:
            user_pref = f"User has viewed: {', '.join(titles[:3])}"
            final = await self.llm.rerank_with_groq(query=titles[0], candidates=final, user_context=user_pref)
            final = final[:limit]

        final = [self._normalize_candidate(c) for c in final]

        # Add Gemini Explanations for top 3
        if settings.GEMINI_API_KEY:
            for item in final[:3]:
                item["explanation_text"] = await self.llm.explain_with_gemini(
                    product_title=item["title"], 
                    user_interest=titles[0]
                )
        for item in final:
            if not item.get("explanation_text") or item.get("explanation_text") in {
                "Fits your interest in this category.",
                "Recommended based on your recent activity.",
            }:
                item["explanation_text"] = self._demo_explanation(item, titles[0] if titles else None)
        
        return {
            "user_id": user_id,
            "recommendations": final,
            "strategy": "llm_enhanced_hybrid" if settings.USE_LLM_RERANKING else "xgboost_ranked_hybrid"
        }

    async def get_similar_items(self, asin: str, limit: int) -> Dict:
        item = self.db.query(models.Item).filter(models.Item.asin == asin).first()
        if not item:
            return {"error": "Item not found"}
            
        embedding = self.engine.get_embedding(self._item_content(item))
        candidates = self.engine.search_candidates(embedding, limit=limit + 10)
        candidates = [c for c in candidates if c.get("asin") != asin][:limit]
        candidates = [self._normalize_candidate(c) for c in candidates]
        for cand in candidates:
            cand["explanation_text"] = self._demo_explanation(cand)
        
        return {
            "seed_item": {"asin": asin},
            "recommendations": candidates,
            "strategy": "content_similarity"
        }

    async def get_trending(self, limit: int) -> Dict:
        is_best_seller_int = case((models.Item.is_best_seller.is_(True), 1), else_=0)
        score_expr = (
            0.55 * func.ln(func.coalesce(models.Item.bought_in_last_month, 0) + 1)
            + 0.25 * func.coalesce(models.Item.stars, 0)
            + 0.15 * func.ln(func.coalesce(models.Item.reviews, 0) + 1)
            + 0.05 * is_best_seller_int
        ).label("trend_score")

        rows = (
            self.db.query(models.Item, score_expr)
            .order_by(desc(score_expr), desc(models.Item.stars))
            .limit(limit)
            .all()
        )

        recs: List[Dict[str, Any]] = []
        for item, score in rows:
            recs.append(
                {
                    "asin": item.asin,
                    "title": item.title,
                    "price": item.price,
                    "stars": item.stars,
                    "category": item.category,
                    "reviews": item.reviews or 0,
                    "bought_in_last_month": item.bought_in_last_month or 0,
                    "is_best_seller": bool(item.is_best_seller),
                    "product_url": item.product_url,
                    "img_url": item.img_url,
                    "score": float(score) if score is not None else None,
                    "explanation_text": self._demo_explanation(
                        {"category": item.category, "stars": item.stars}
                    ),
                }
            )
        
        return {
            "recommendations": recs,
            "strategy": "global_trending"
        }

    async def get_bundles(self, asin: str, limit: int) -> Dict:
        # v1: similar items, but favor best sellers until co-purchase data exists.
        base = await self.get_similar_items(asin, limit=limit * 3)
        recs = base.get("recommendations", [])
        recs = sorted(recs, key=lambda r: (r.get("is_best_seller", False), r.get("score", 0.0)), reverse=True)[:limit]
        return {"seed_item": {"asin": asin}, "recommendations": recs, "strategy": "bundle_v1_best_seller_filtered"}
