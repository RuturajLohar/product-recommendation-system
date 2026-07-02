import numpy as np
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ...core.config import settings
from ...core.qdrant import create_qdrant_client

class HybridEngine:
    def __init__(self):
        self.sbert = SentenceTransformer(settings.SBERT_MODEL)
        self.q_client = create_qdrant_client()
        self.collection_name = settings.QDRANT_COLLECTION
        
    def get_embedding(self, text: str) -> np.ndarray:
        return self.sbert.encode(text).tolist()

    def get_user_embedding(self, titles: List[str]) -> List[float]:
        """Recency-weighted user profile embedding.
        Most recent title (index 0) gets highest weight via exponential decay.
        This prevents a single outlier interaction from dominating the profile."""
        if not titles:
            return self.get_embedding("")
        vecs = np.asarray(self.sbert.encode(titles))
        n = len(titles)
        decay_rate = 0.8  # each older title gets 80% of previous weight
        weights = np.array([decay_rate ** i for i in range(n)], dtype=np.float32)
        weights /= weights.sum()
        weighted_vec = (vecs.T @ weights).astype(np.float32)
        return weighted_vec.tolist()

    def search_candidates(self, vector: List[float], limit: int = 100) -> List[Dict]:
        """Fetch candidates from Qdrant"""
        search_result = self.q_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        candidates: List[Dict[str, Any]] = []
        for hit in (search_result.points or []):
            payload = hit.payload or {}
            # Preserve score and vector for ranking/diversity.
            payload = dict(payload)
            payload["score"] = float(hit.score) if hit.score is not None else 0.0
            payload["_vector"] = hit.vector
            candidates.append(payload)
        return candidates

    def mmr_rerank(self, 
                   query_vector: np.ndarray, 
                   candidates: List[Dict], 
                   candidate_vectors: np.ndarray,
                   top_k: int = 10, 
                   lambda_param: float = 0.5) -> List[Dict]:
        """Maximal Marginal Relevance for Diversity"""
        selected_indices = []
        candidate_indices = list(range(len(candidates)))
        
        # Relevance scores (cosine sim)
        relevance = cosine_similarity([query_vector], candidate_vectors)[0]
        
        while len(selected_indices) < min(top_k, len(candidates)):
            best_mmr = -1
            best_idx = -1
            
            for i in candidate_indices:
                if i in selected_indices: continue
                
                # Max similarity to already selected items
                if not selected_indices:
                    max_sim = 0
                else:
                    max_sim = max(cosine_similarity([candidate_vectors[i]], candidate_vectors[selected_indices])[0])
                
                mmr_score = lambda_param * relevance[i] - (1 - lambda_param) * max_sim
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            
            selected_indices.append(best_idx)
            
        return [candidates[i] for i in selected_indices]

    def hybrid_score(self, 
                     content_scores: np.ndarray, 
                     cf_scores: np.ndarray, 
                     popularity: np.ndarray,
                     weights: Dict[str, float]) -> np.ndarray:
        """Weighted ensemble of scores"""
        final_scores = (
            weights['content'] * content_scores +
            weights['cf'] * cf_scores +
            weights['popularity'] * popularity
        )
        return final_scores
