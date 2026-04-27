"""
Universal Recommender Wrapper
============================
A "Plug-and-Play" layer that makes the recommendation engine compatible 
with any raw dataset (not just the Amazon one).

Features:
- Schema Mapping: Map raw column names (e.g. 'movie_id') to system concepts ('id').
- Feature Resilience: Automatically disables signals (price, rating, category) if missing.
- Auto-Discovery: Attempts to guess column roles if no schema is provided.
- One-Click Fit: Single method to load, embed, index, and train.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, Any, List

# Import our core components
from app import DataProcessor, EmbeddingSystem, FAISSSystem
from collaborative_filtering import CollaborativeFilter
from hybrid_recommender import HybridRecommender, HybridConfig
from persistence import PersistenceManager

class UniversalRecommender:
    """The 'Plug-and-Play' interface for the recommendation system."""
    
    DEFAULT_SCHEMA = {
        "id": ["asin", "id", "uuid", "item_id", "product_id"],
        "text": ["title", "name", "description", "text_content"],
        "category": ["categoryName", "category", "genre", "type"],
        "price": ["price", "cost", "amount"],
        "rating": ["stars", "rating", "score"],
        "popularity": ["boughtInLastMonth", "sales", "views", "popularity"]
    }

    def __init__(self, schema: Optional[Dict[str, str]] = None, artifact_dir: str = "universal_artifacts"):
        self.schema = schema or {}
        self.artifact_dir = artifact_dir
        self.persistence = PersistenceManager(artifact_dir)
        
        # State
        self.df = None
        self.recommender = None
        self.is_fitted = False
        self.active_features = []

    def _auto_discover_schema(self, df: pd.DataFrame):
        """Tries to map columns if not provided."""
        discovered = {}
        for role, candidates in self.DEFAULT_SCHEMA.items():
            if role in self.schema:
                discovered[role] = self.schema[role]
                continue
                
            for col in df.columns:
                if col.lower() in [c.lower() for c in candidates]:
                    discovered[role] = col
                    break
        
        self.schema = discovered
        print(f"🔍 Discovered Schema: {self.schema}")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes raw data into the format expected by the engine."""
        processed = pd.DataFrame()
        
        # Mandatory: ID and Text
        if "id" not in self.schema or "text" not in self.schema:
            raise ValueError("Schema must at least map 'id' and 'text' columns.")
            
        processed['asin'] = df[self.schema['id']].astype(str)
        processed['title'] = df[self.schema['text']].astype(str)
        
        # Optional Features
        if "category" in self.schema:
            processed['categoryName'] = df[self.schema['category']].fillna("Unknown")
            self.active_features.append("category")
        else:
            processed['categoryName'] = "Unknown"

        if "price" in self.schema:
            processed['price'] = pd.to_numeric(df[self.schema['price']], errors='coerce').fillna(0.0)
            self.active_features.append("price")
        else:
            processed['price'] = 0.0

        if "rating" in self.schema:
            processed['stars'] = pd.to_numeric(df[self.schema['rating']], errors='coerce').fillna(3.0)
            self.active_features.append("rating")
        else:
            processed['stars'] = 3.0

        if "popularity" in self.schema:
            processed['popularity_score'] = pd.to_numeric(df[self.schema['popularity']], errors='coerce').fillna(0.0)
            self.active_features.append("popularity")
        else:
            processed['popularity_score'] = 0.0
            
        return processed

    def fit(self, data: Any, interactions: Optional[pd.DataFrame] = None):
        """
        Plug-and-play fitting process.
        'data' can be a CSV path or a Pandas DataFrame.
        """
        print("🚀 Universal Fit Started...")
        
        # 1. Load
        if isinstance(data, str):
            df_raw = pd.read_csv(data)
        else:
            df_raw = data
            
        # 2. Map & Prepare
        self._auto_discover_schema(df_raw)
        self.df = self._prepare_data(df_raw)
        
        # 3. Process & Embed
        processor = DataProcessor()
        self.df = processor.create_features(self.df) # Internal engineering
        
        emb_sys = EmbeddingSystem()
        embeddings = emb_sys.create_embeddings(self.df)
        emb_sys.fit_tfidf(self.df)
        
        # 4. Index
        faiss_sys = FAISSSystem(embeddings.shape[1])
        faiss_sys.build_index(embeddings)
        
        # 5. Optional CF
        cf_model = None
        if interactions is not None:
            cf_model = CollaborativeFilter(n_components=64)
            cf_model.train(interactions, n_items_total=len(self.df))
            self.active_features.append("collaborative")

        # 6. Build Hybrid Engine with Adaptive Weights
        # If features are missing, we zero out their weights and re-normalize
        base_weights = HybridConfig.HYBRID_WEIGHTS.copy()
        if "collaborative" not in self.active_features:
            base_weights["collaborative"] = 0.0
        if "popularity" not in self.active_features:
            base_weights["popularity"] = 0.0
            
        # Re-normalize weights
        total = sum(base_weights.values())
        final_weights = {k: v/total for k, v in base_weights.items()}

        self.recommender = HybridRecommender(
            df=self.df,
            embeddings=embeddings,
            faiss_system=faiss_sys,
            emb_system=emb_sys,
            cf_model=cf_model,
            interactions_df=interactions,
            hybrid_weights=final_weights
        )
        
        self.is_fitted = True
        print(f"✅ Universal Model Fitted! Active signals: {self.active_features}")

    def recommend(self, item_id: str, user_id: Optional[int] = None, max_results: int = 10):
        """One-click recommendation."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend().")
            
        if user_id is not None:
            return self.recommender.recommend_for_user(user_id=user_id, seed_asin=item_id, max_results=max_results)
        else:
            # Fallback to item-to-item if no user provided
            return self.recommender.recommend_item_to_item(asin=item_id, max_results=max_results)

# ==================== PLUG-AND-PLAY EXAMPLE ====================
if __name__ == "__main__":
    # Example: Using a completely different raw dataset (Mock Movies)
    mock_movies = pd.DataFrame({
        "movie_id": ["M1", "M2", "M3", "M4"],
        "title": ["Inception", "Interstellar", "The Dark Knight", "The Lion King"],
        "genre": ["Sci-Fi", "Sci-Fi", "Action", "Animation"],
        "vote_average": [8.8, 8.6, 9.0, 8.5]
    })
    
    # Just plug in the data and define the mapping
    # Notice we don't have 'price' or 'popularity' here - it will adapt!
    engine = UniversalRecommender(schema={
        "id": "movie_id",
        "text": "title",
        "category": "genre",
        "rating": "vote_average"
    })
    
    engine.fit(mock_movies)
    
    print("\n🎬 Movie Recommendation Result:")
    results = engine.recommend(item_id="M1", max_results=2)
    for r in results['recommendations']:
        print(f"   - {r['title']} (Score: {r['score']:.3f})")
