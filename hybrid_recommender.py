"""
Hybrid Recommender
===================
Upgraded recommendation engine that combines:
1. Content-based scoring (SBERT + TF-IDF) — from existing app.py
2. Collaborative filtering (LightFM) — NEW
3. Popularity signal — existing
4. Recency signal — NEW

This module imports and wraps the existing classes from app.py,
then extends them with CF and recency scoring.

Architecture:
    User Request
        ↓
    FAISS Candidate Retrieval (top-200)
        ↓
    Content Scoring (SBERT + TF-IDF + category + price + rating)
        ↓
    CF Scoring (LightFM user-item predictions)
        ↓
    Recency Scoring (timestamp decay)
        ↓
    Hybrid Ensemble (weighted combination)
        ↓
    MMR Diversity Reranking
        ↓
    Final Recommendations
"""

import numpy as np
import pandas as pd
import datetime
from typing import Optional, Dict, List
from sklearn.metrics.pairwise import cosine_similarity

# Import existing components from app.py
from app import (
    Config,
    DataProcessor,
    EmbeddingSystem,
    FAISSSystem,
    AdvancedRecommender,
    config,
)
from collaborative_filtering import CollaborativeFilter


class HybridConfig:
    """Configuration for the hybrid recommender."""

    # Hybrid weights (must sum to ~1.0)
    HYBRID_WEIGHTS = {
        "content": 0.45,      # SBERT + TF-IDF + category + price + rating
        "collaborative": 0.30, # LightFM CF score
        "popularity": 0.15,   # Popularity score
        "recency": 0.10,      # How recently user interacted with similar items
    }

    # Content sub-weights (aligned with improved app.py ensemble)
    CONTENT_WEIGHTS = {
        "content_sbert": 0.35,
        "content_tfidf": 0.15,
        "category": 0.15,
        "popularity": 0.05,
        "price": 0.05,
        "rating": 0.10,
        "review_count": 0.05,
        "best_seller": 0.05,
    }


class HybridRecommender:
    """
    Production-grade hybrid recommender combining content-based
    and collaborative filtering approaches.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        faiss_system: FAISSSystem,
        emb_system: EmbeddingSystem,
        cf_model: CollaborativeFilter,
        interactions_df: Optional[pd.DataFrame] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings.astype(np.float32)
        self.faiss_system = faiss_system
        self.emb_system = emb_system
        self.cf_model = cf_model
        self.interactions_df = interactions_df

        self.hybrid_weights = hybrid_weights or HybridConfig.HYBRID_WEIGHTS
        self.content_weights = HybridConfig.CONTENT_WEIGHTS

        # Build the content-only recommender (reuses existing logic)
        self.content_recommender = AdvancedRecommender(
            df, embeddings, faiss_system, emb_system
        )

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        if scores.size == 0:
            return scores
        minv = float(scores.min())
        maxv = float(scores.max())
        if maxv - minv < 1e-9:
            return np.ones_like(scores) * 0.5
        return (scores - minv) / (maxv - minv)

    def _compute_content_scores(
        self, q_idx: int, cand_indices: List[int]
    ) -> np.ndarray:
        """Compute the content-based ensemble score for candidates.
        Uses all improved signals: soft category similarity, Bayesian rating,
        review confidence, best_seller, and symmetric price similarity."""
        q_row = self.df.iloc[q_idx]
        q_title = str(q_row["title"])
        q_emb = self.embeddings[q_idx : q_idx + 1]
        cand_embs = self.embeddings[cand_indices]

        # SBERT
        sbert_sims = (cand_embs @ q_emb.T).flatten().astype(np.float32)

        # TF-IDF
        tfidf_sims = self.content_recommender._compute_tfidf_similarity(
            q_title, cand_indices
        )

        # Price (symmetric log-ratio)
        price_sims = np.array(
            [
                self.content_recommender._price_similarity(
                    q_row["price"], self.df.iloc[i]["price"]
                )
                for i in cand_indices
            ],
            dtype=np.float32,
        )

        # Bayesian rating (discounts high stars with low reviews)
        bayesian_ratings = np.array(
            [
                float(self.df.iloc[i].get("rating_bayesian",
                      self.df.iloc[i].get("rating_normalized", 0.5)))
                for i in cand_indices
            ],
            dtype=np.float32,
        )

        # Soft category similarity (embedding-based)
        q_cat = str(q_row["categoryName"])
        cat_sims = np.array(
            [
                self.content_recommender._category_similarity(
                    q_cat, str(self.df.iloc[i]["categoryName"])
                )
                for i in cand_indices
            ],
            dtype=np.float32,
        )

        # Review confidence
        review_conf = np.array(
            [float(self.df.iloc[i].get("review_confidence", 0.5)) for i in cand_indices],
            dtype=np.float32,
        )

        # Best seller
        best_seller = np.array(
            [float(self.df.iloc[i].get("best_seller", 0.0)) for i in cand_indices],
            dtype=np.float32,
        )

        # Normalize
        sbert_norm = self._normalize_scores(sbert_sims)
        tfidf_norm = self._normalize_scores(tfidf_sims)
        price_norm = self._normalize_scores(price_sims)
        rating_norm = self._normalize_scores(bayesian_ratings)
        cat_norm = self._normalize_scores(cat_sims)
        review_norm = self._normalize_scores(review_conf)

        w = self.content_weights
        content_score = (
            w["content_sbert"] * sbert_norm
            + w["content_tfidf"] * tfidf_norm
            + w["category"] * cat_norm
            + w["price"] * price_norm
            + w["rating"] * rating_norm
            + w.get("review_count", 0.05) * review_norm
            + w.get("best_seller", 0.05) * best_seller
        )

        return content_score

    def _compute_cf_scores(
        self, user_id: int, cand_indices: List[int]
    ) -> np.ndarray:
        """Get collaborative filtering scores from LightFM."""
        if self.cf_model is None or self.cf_model.model is None:
            return np.ones(len(cand_indices), dtype=np.float32) * 0.5

        scores = self.cf_model.predict_user_items(user_id, cand_indices)
        return scores

    def _compute_recency_scores(
        self, user_id: int, cand_indices: List[int]
    ) -> np.ndarray:
        """
        Compute recency-based scores.

        Items similar to what the user interacted with recently get a boost.
        Uses exponential decay: score = exp(-days_since_interaction / 30).
        Optimized: pre-builds category-to-latest-timestamp map.
        """
        if self.interactions_df is None:
            return np.ones(len(cand_indices), dtype=np.float32) * 0.5

        user_history = self.interactions_df[
            self.interactions_df["user_id"] == user_id
        ].copy()

        if user_history.empty:
            return np.ones(len(cand_indices), dtype=np.float32) * 0.5

        user_history["timestamp"] = pd.to_datetime(user_history["timestamp"])
        max_ts = user_history["timestamp"].max()

        # Pre-build category -> latest interaction timestamp map (O(n) instead of O(n*m))
        cat_latest: Dict[str, pd.Timestamp] = {}
        for _, row in user_history.iterrows():
            item_idx = int(row["item_idx"])
            if item_idx >= len(self.df):
                continue
            cat = self.df.iloc[item_idx]["categoryName"]
            ts = row["timestamp"]
            if cat not in cat_latest or ts > cat_latest[cat]:
                cat_latest[cat] = ts

        # Score each candidate in O(1) per candidate
        recency_scores = np.zeros(len(cand_indices), dtype=np.float32)
        for i, cand_idx in enumerate(cand_indices):
            cand_cat = self.df.iloc[cand_idx]["categoryName"]
            if cand_cat in cat_latest:
                days_ago = (max_ts - cat_latest[cand_cat]).total_seconds() / 86400
                recency_scores[i] = float(np.exp(-days_ago / 30.0))

        return recency_scores

    def _compute_popularity_scores(self, cand_indices: List[int]) -> np.ndarray:
        """Get popularity scores for candidates."""
        return np.array(
            [float(self.df.iloc[i].get("popularity_score", 0.5)) for i in cand_indices],
            dtype=np.float32,
        )

    def _mmr(
        self,
        candidate_ids: List[int],
        relevance_scores: np.ndarray,
        top_k: int = 10,
        lamb: float = 0.65,
    ) -> List[int]:
        """MMR diversity reranking."""
        if len(candidate_ids) == 0:
            return []

        cand_embs = self.embeddings[candidate_ids]
        sim_matrix = cosine_similarity(cand_embs)

        selected = []
        idx0 = int(np.argmax(relevance_scores))
        selected.append(idx0)

        while len(selected) < min(top_k, len(candidate_ids)):
            best_score = -float("inf")
            best_idx = -1

            for i in range(len(candidate_ids)):
                if i in selected:
                    continue
                max_sim_to_selected = max(sim_matrix[i, s] for s in selected)
                mmr = lamb * relevance_scores[i] - (1 - lamb) * max_sim_to_selected
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            if best_idx == -1:
                break
            selected.append(best_idx)

        return [candidate_ids[i] for i in selected]

    def recommend_for_user(
        self,
        user_id: int,
        seed_asin: Optional[str] = None,
        max_results: int = 10,
        mmr_lambda: float = 0.65,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Get personalized recommendations for a user.

        Args:
            user_id: User ID (from interaction data).
            seed_asin: Optional seed item ASIN. If None, uses user's last
                       interacted item.
            max_results: Number of recommendations.
            mmr_lambda: MMR diversity parameter.
            weights: Override hybrid weights.

        Returns:
            Dictionary with recommendations and explanations.
        """
        w = weights or self.hybrid_weights

        # Determine seed item
        if seed_asin:
            matches = self.df.index[self.df["asin"] == seed_asin].tolist()
            if not matches:
                return {"error": f"ASIN '{seed_asin}' not found."}
            q_idx = matches[0]
        elif self.interactions_df is not None:
            user_history = self.interactions_df[
                self.interactions_df["user_id"] == user_id
            ]
            if user_history.empty:
                # Cold start: return popular items
                return self._cold_start_recommendations(max_results)
            # Use the most recently interacted item as seed
            user_history = user_history.sort_values("timestamp", ascending=False)
            q_idx = int(user_history.iloc[0]["item_idx"])
        else:
            return {"error": "No seed_asin provided and no interaction data."}

        q_row = self.df.iloc[q_idx]

        # 1. FAISS candidate retrieval
        candidate_pool = max(200, max_results * 20)
        distances, indices = self.faiss_system.search(
            self.embeddings[q_idx : q_idx + 1], k=candidate_pool
        )
        cand_indices = [
            int(i) for i in indices[0] if int(i) != q_idx and int(i) != -1
        ]

        if not cand_indices:
            return {"error": "No candidates found."}

        # 2. Compute all signal scores
        content_scores = self._compute_content_scores(q_idx, cand_indices)
        cf_scores = self._compute_cf_scores(user_id, cand_indices)
        popularity_scores = self._compute_popularity_scores(cand_indices)
        recency_scores = self._compute_recency_scores(user_id, cand_indices)

        # 3. Normalize all
        content_norm = self._normalize_scores(content_scores)
        cf_norm = self._normalize_scores(cf_scores)
        pop_norm = self._normalize_scores(popularity_scores)
        recency_norm = self._normalize_scores(recency_scores)

        # 4. Hybrid ensemble
        hybrid_scores = (
            w.get("content", 0.45) * content_norm
            + w.get("collaborative", 0.30) * cf_norm
            + w.get("popularity", 0.15) * pop_norm
            + w.get("recency", 0.10) * recency_norm
        )

        if np.allclose(hybrid_scores, 0.0):
            hybrid_scores = content_norm

        # 5. MMR diversity
        selected_ids = self._mmr(cand_indices, hybrid_scores, top_k=max_results, lamb=mmr_lambda)

        # 6. Build result
        recs = []
        for idx in selected_ids:
            pos = cand_indices.index(idx)
            p = self.df.iloc[idx]
            explanation = {
                "content": float(content_norm[pos]),
                "collaborative": float(cf_norm[pos]),
                "popularity": float(pop_norm[pos]),
                "recency": float(recency_norm[pos]),
                "hybrid_score": float(hybrid_scores[pos]),
            }
            recs.append(
                {
                    "asin": p["asin"],
                    "title": p["title"],
                    "price": float(p["price"]),
                    "stars": float(p["stars"]),
                    "category": p["categoryName"],
                    "image_url": p.get("imgUrl", ""),
                    "score": float(explanation["hybrid_score"]),
                    "explanation": explanation,
                }
            )

        return {
            "user_id": user_id,
            "seed_item": {
                "asin": q_row["asin"],
                "title": q_row["title"],
                "price": float(q_row["price"]),
                "stars": float(q_row["stars"]),
                "category": q_row["categoryName"],
            },
            "recommendations": recs,
            "diversity_score": (
                len(set(r["category"] for r in recs)) / len(recs) if recs else 0.0
            ),
            "weights_used": w,
            "strategy": "hybrid_content_cf_recency",
        }

    def recommend_item_to_item(
        self, asin: str, max_results: int = 10, mmr_lambda: float = 0.65
    ) -> Dict:
        """
        Pure item-to-item recommendation (no user context).
        Delegates to the existing content-based recommender.
        """
        return self.content_recommender.recommend(
            asin, max_results=max_results, mmr_lambda=mmr_lambda
        )

    def _cold_start_recommendations(self, max_results: int) -> Dict:
        """Fallback for users with no interaction history."""
        popular_items = (
            self.df.nlargest(max_results, "popularity_score")
            if "popularity_score" in self.df.columns
            else self.df.head(max_results)
        )
        recs = []
        for _, p in popular_items.iterrows():
            recs.append(
                {
                    "asin": p["asin"],
                    "title": p["title"],
                    "price": float(p["price"]),
                    "stars": float(p["stars"]),
                    "category": p["categoryName"],
                    "score": float(p.get("popularity_score", 0.5)),
                    "explanation": {"strategy": "cold_start_popularity"},
                }
            )
        return {
            "recommendations": recs,
            "strategy": "cold_start",
        }
