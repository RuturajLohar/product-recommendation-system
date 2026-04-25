import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from lightfm import LightFM
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sklearn.metrics.pairwise import cosine_similarity

class HybridEngine:
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 model_name: str = 'all-MiniLM-L12-v2'):
        self.sbert = SentenceTransformer(model_name)
        self.q_client = QdrantClient(host=qdrant_host, port=6333)
        self.collection_name = "products"
        self.lfm_model = LightFM(loss='warp')
        
    def get_embedding(self, text: str) -> np.ndarray:
        return self.sbert.encode(text).tolist()

    def search_candidates(self, vector: List[float], limit: int = 100) -> List[Dict]:
        """Fetch candidates from Qdrant"""
        search_result = self.q_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=True
        )
        return [hit.payload for hit in search_result]

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
