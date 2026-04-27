import pandas as pd
import numpy as np
from app import DataProcessor, EmbeddingSystem, FAISSSystem
from interaction_simulator import InteractionSimulator
from collaborative_filtering import CollaborativeFilter
from evaluation import RecommendationEvaluator
from persistence import PersistenceManager
from hybrid_recommender import HybridRecommender

print("🧪 Starting Sanity Check...")

# 1. Load tiny data
df = pd.DataFrame({
    'asin': ['A1', 'A2', 'A3', 'A4', 'A5'],
    'title': ['iPhone Case Red', 'iPhone Case Blue', 'Samsung Galaxy S21', 'MacBook Pro Charger', 'USB-C Cable'],
    'categoryName': ['Mobile', 'Mobile', 'Mobile', 'Laptops', 'Accessories'],
    'price': [19.99, 19.99, 799.0, 79.0, 9.99],
    'stars': [4.5, 4.0, 4.8, 4.2, 4.1],
    'popularity_score': [0.9, 0.8, 0.95, 0.7, 0.6]
})

# 2. Embeddings
emb_sys = EmbeddingSystem()
embeddings = emb_sys.create_embeddings(df)
emb_sys.fit_tfidf(df)

# 3. FAISS
faiss_sys = FAISSSystem(embeddings.shape[1])
faiss_sys.build_index(embeddings)

# 4. Interaction Simulation
sim = InteractionSimulator(seed=42)
interactions = sim.generate(df, n_users=10, interactions_per_user=(2, 5))

# 5. CF (NMF)
cf = CollaborativeFilter(n_components=4, epochs=10)
cf.train(interactions, n_items_total=len(df))

# 6. Hybrid Recommender
hybrid = HybridRecommender(df, embeddings, faiss_sys, emb_sys, cf, interactions)

# 7. Test Recommendation
result = hybrid.recommend_for_user(user_id=0, max_results=3)
print("\n🎯 Sample Recommendation for User 0:")
print(result)

# 8. Evaluation
evaluator = RecommendationEvaluator()
evaluator.add_result(actual=[1, 2], predicted=[1, 0, 3])
metrics = evaluator.compute_all(k=3)
print("\n📊 Sample Metrics:")
print(metrics)

print("\n✅ Sanity Check Passed!")
