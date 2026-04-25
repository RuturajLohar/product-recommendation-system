import xgboost as xgb
import pandas as pd
import numpy as np
from typing import List, Dict

class XGBRanker:
    def __init__(self, model_path: str = None):
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        if model_path:
            self.model.load_model(model_path)

    def feature_engineering(self, 
                            query_item: Dict, 
                            candidates: List[Dict], 
                            user_history: List[Dict]) -> pd.DataFrame:
        """
        Creates a feature matrix for the ranking model.
        """
        features = []
        user_categories = [h['category'] for h in user_history]
        
        for cand in candidates:
            row = {
                "similarity_score": cand.get('score', 0), # Retrieval score
                "price_diff": abs(cand['price'] - query_item['price']),
                "stars": cand['stars'],
                "popularity": cand.get('popularity_score', 0),
                "category_match": 1 if cand['category'] == query_item['category'] else 0,
                "user_pref_match": 1 if cand['category'] in user_categories else 0
            }
            features.append(row)
            
        return pd.DataFrame(features)

    def rank(self, 
             query_item: Dict, 
             candidates: List[Dict], 
             user_history: List[Dict]) -> List[Dict]:
        """
        Ranks candidates using the trained XGBoost model.
        """
        if not candidates:
            return []
            
        X = self.feature_engineering(query_item, candidates, user_history)
        
        # In a real system, we'd use model.predict(X)
        # For this demo, we'll use a sophisticated weighted heuristic if model isn't trained
        # but the structure for model prediction is here.
        try:
            scores = self.model.predict(X)
        except:
            # Fallback to heuristic scoring if model not fit
            scores = (X['similarity_score'] * 0.4 + 
                      X['stars'] * 0.2 + 
                      X['popularity'] * 0.2 + 
                      X['category_match'] * 0.2)
        
        # Attach scores and sort
        for i, cand in enumerate(candidates):
            cand['ranking_score'] = float(scores[i])
            
        return sorted(candidates, key=lambda x: x['ranking_score'], reverse=True)
