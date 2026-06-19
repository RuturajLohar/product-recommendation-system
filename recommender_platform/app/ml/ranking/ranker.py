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
        Now includes 8 signals: similarity, price_diff, stars, popularity,
        category_match, user_pref_match, review_confidence, best_seller,
        and price_affinity (distance from user's median spend).
        """
        features = []
        user_categories = [h['category'] for h in user_history]
        
        # Compute user's median price from history for price affinity
        user_prices = [float(h.get('price', 0)) for h in user_history if h.get('price')]
        median_user_price = np.median(user_prices) if user_prices else float(query_item.get('price', 0))
        
        for cand in candidates:
            cand_price = float(cand.get('price', 0))
            cand_stars = float(cand.get('stars', 0))
            cand_reviews = int(cand.get('reviews', 0))
            
            # Price affinity: how close to user's typical spend (symmetric log-ratio)
            p_a = max(cand_price, 0.01)
            p_b = max(median_user_price, 0.01)
            price_affinity = float(np.exp(-abs(np.log(p_a) - np.log(p_b))))
            
            # Review confidence: log-scaled review count
            review_confidence = np.log1p(max(cand_reviews, 0)) / 12.0  # ~12 = log1p(100000)
            review_confidence = min(review_confidence, 1.0)
            
            # Best seller flag
            best_seller = 1.0 if cand.get('is_best_seller', False) else 0.0

            row = {
                "similarity_score": cand.get('score', 0),
                "price_diff": abs(cand_price - float(query_item.get('price', 0))),
                "stars": cand_stars,
                "popularity": cand.get('popularity_score', 0),
                "category_match": 1 if cand.get('category') == query_item.get('category') else 0,
                "user_pref_match": 1 if cand.get('category') in user_categories else 0,
                "review_confidence": review_confidence,
                "price_affinity": price_affinity,
                "best_seller": best_seller,
            }
            features.append(row)
            
        return pd.DataFrame(features)

    def rank(self, 
             query_item: Dict, 
             candidates: List[Dict], 
             user_history: List[Dict]) -> List[Dict]:
        """
        Ranks candidates using the trained XGBoost model.
        Falls back to a feature-rich heuristic if model isn't trained.
        """
        if not candidates:
            return []
            
        X = self.feature_engineering(query_item, candidates, user_history)
        
        try:
            scores = self.model.predict(X)
        except Exception:
            # Heuristic fallback with all 8 signals
            scores = (
                X['similarity_score'] * 0.25 +
                X['stars'] / 5.0 * 0.10 +
                X['popularity'] * 0.10 +
                X['category_match'] * 0.15 +
                X['user_pref_match'] * 0.15 +
                X['review_confidence'] * 0.10 +
                X['price_affinity'] * 0.10 +
                X['best_seller'] * 0.05
            )
        
        # Attach scores and sort
        for i, cand in enumerate(candidates):
            cand['ranking_score'] = float(scores.iloc[i] if hasattr(scores, 'iloc') else scores[i])
            
        return sorted(candidates, key=lambda x: x['ranking_score'], reverse=True)

