"""
End-to-End Pipeline
====================
Orchestrates the full recommendation system pipeline:

1. Load & clean data
2. Generate embeddings (SBERT + TF-IDF)
3. Build FAISS index
4. Simulate user interactions
5. Train collaborative filtering (LightFM)
6. Build hybrid recommender
7. Evaluate: baseline (content-only) vs hybrid
8. Persist all artifacts
9. Print sample recommendations + metrics comparison

Usage:
    python run_pipeline.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Import existing components
from app import Config, DataProcessor, EmbeddingSystem, FAISSSystem, AdvancedRecommender, config

# Import new components
from interaction_simulator import InteractionSimulator
from collaborative_filtering import CollaborativeFilter
from evaluation import RecommendationEvaluator
from persistence import PersistenceManager
from hybrid_recommender import HybridRecommender


# ==================== CONFIGURATION ====================
DATA_PATH = "amz_uk_processed_data.csv"
EMBED_CACHE = "product_embeddings_hybrid.pkl"
ARTIFACT_DIR = "artifacts"
INTERACTIONS_PATH = os.path.join(ARTIFACT_DIR, "interactions.csv")
CF_MODEL_PATH = os.path.join(ARTIFACT_DIR, "lightfm_model.pkl")

N_USERS = 500
EVAL_K = 10
TEST_SPLIT = 0.2


def run_pipeline():
    total_start = time.time()

    persistence = PersistenceManager(ARTIFACT_DIR)

    # ==================== PHASE 1: DATA ====================
    print("\n" + "=" * 60)
    print("📦 PHASE 1: DATA LOADING & PREPROCESSING")
    print("=" * 60)

    processor = DataProcessor()
    df = processor.load_and_clean_data(DATA_PATH)
    print(f"   Products: {len(df)}")
    print(f"   Categories: {df['categoryName'].nunique()}")
    print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

    # ==================== PHASE 2: EMBEDDINGS ====================
    print("\n" + "=" * 60)
    print("🧠 PHASE 2: EMBEDDING GENERATION")
    print("=" * 60)

    emb_system = EmbeddingSystem()
    embeddings = emb_system.create_embeddings(df, cache_path=EMBED_CACHE)
    emb_system.fit_tfidf(df)
    print(f"   Embedding shape: {embeddings.shape}")

    # ==================== PHASE 3: FAISS INDEX ====================
    print("\n" + "=" * 60)
    print("🔍 PHASE 3: FAISS INDEX")
    print("=" * 60)

    faiss_sys = FAISSSystem(embeddings.shape[1])
    faiss_sys.build_index(embeddings)

    # ==================== PHASE 4: INTERACTION SIMULATION ====================
    print("\n" + "=" * 60)
    print("🧪 PHASE 4: INTERACTION SIMULATION")
    print("=" * 60)

    simulator = InteractionSimulator(seed=42)
    interactions_df = simulator.generate(
        df, n_users=N_USERS, interactions_per_user=(15, 60), save_path=INTERACTIONS_PATH
    )

    # ==================== PHASE 5: TRAIN/TEST SPLIT ====================
    print("\n" + "=" * 60)
    print("✂️  PHASE 5: TRAIN/TEST SPLIT")
    print("=" * 60)

    # Sort by timestamp, split chronologically (more realistic than random)
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(interactions_df) * (1 - TEST_SPLIT))
    train_interactions = interactions_df.iloc[:split_idx].copy()
    test_interactions = interactions_df.iloc[split_idx:].copy()

    print(f"   Train: {len(train_interactions)} interactions")
    print(f"   Test:  {len(test_interactions)} interactions")

    # Build test ground truth: for each user, items they interacted with in test set
    test_ground_truth = {}
    for user_id, group in test_interactions.groupby("user_id"):
        test_ground_truth[int(user_id)] = group["item_idx"].unique().tolist()

    print(f"   Test users: {len(test_ground_truth)}")

    # ==================== PHASE 6: COLLABORATIVE FILTERING ====================
    print("\n" + "=" * 60)
    print("🤝 PHASE 6: COLLABORATIVE FILTERING (NMF)")
    print("=" * 60)

    cf_model = CollaborativeFilter(n_components=64, epochs=30)
    cf_model.train(train_interactions, n_items_total=len(df))
    cf_model.save(CF_MODEL_PATH)

    # ==================== PHASE 7: BUILD RECOMMENDERS ====================
    print("\n" + "=" * 60)
    print("🏗️  PHASE 7: BUILD RECOMMENDERS")
    print("=" * 60)

    # Baseline: content-only (existing system)
    baseline_recommender = AdvancedRecommender(df, embeddings, faiss_sys, emb_system)

    # Hybrid: content + CF + recency
    hybrid_recommender = HybridRecommender(
        df=df,
        embeddings=embeddings,
        faiss_system=faiss_sys,
        emb_system=emb_system,
        cf_model=cf_model,
        interactions_df=train_interactions,
    )

    print("   ✅ Baseline recommender built (content-only)")
    print("   ✅ Hybrid recommender built (content + CF + recency)")

    # ==================== PHASE 8: EVALUATION ====================
    print("\n" + "=" * 60)
    print("📊 PHASE 8: EVALUATION (Baseline vs Hybrid)")
    print("=" * 60)

    baseline_evaluator = RecommendationEvaluator()
    hybrid_evaluator = RecommendationEvaluator()

    # Evaluate on test users
    eval_users = list(test_ground_truth.keys())[:100]  # Cap at 100 users for speed
    print(f"   Evaluating on {len(eval_users)} users...")

    for user_id in eval_users:
        actual_items = test_ground_truth[user_id]
        if len(actual_items) < 2:
            continue

        # Determine seed: use a training interaction for this user
        user_train = train_interactions[train_interactions["user_id"] == user_id]
        if user_train.empty:
            continue
        seed_idx = int(user_train.iloc[-1]["item_idx"])
        seed_asin = df.iloc[seed_idx]["asin"]

        # --- Baseline: content-only ---
        baseline_result = baseline_recommender.recommend(
            seed_asin, max_results=EVAL_K
        )
        if "error" not in baseline_result:
            baseline_predicted = [
                df.index[df["asin"] == r["asin"]].tolist()[0]
                for r in baseline_result["recommendations"]
                if not df.index[df["asin"] == r["asin"]].empty
            ]
            baseline_evaluator.add_result(actual_items, baseline_predicted)

        # --- Hybrid ---
        hybrid_result = hybrid_recommender.recommend_for_user(
            user_id=user_id, seed_asin=seed_asin, max_results=EVAL_K
        )
        if "error" not in hybrid_result:
            hybrid_predicted = [
                df.index[df["asin"] == r["asin"]].tolist()[0]
                for r in hybrid_result["recommendations"]
                if not df.index[df["asin"] == r["asin"]].empty
            ]
            hybrid_evaluator.add_result(actual_items, hybrid_predicted)

    baseline_metrics = baseline_evaluator.print_report(k=EVAL_K, label="BASELINE (Content-Only)")
    hybrid_metrics = hybrid_evaluator.print_report(k=EVAL_K, label="HYBRID (Content + CF + Recency)")

    # ==================== PHASE 9: COMPARISON TABLE ====================
    print("\n" + "=" * 60)
    print("📈 METRICS COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(
        {
            "Metric": ["Precision@10", "Recall@10", "nDCG@10", "MRR"],
            "Baseline": [
                baseline_metrics["precision@k"],
                baseline_metrics["recall@k"],
                baseline_metrics["ndcg@k"],
                baseline_metrics["mrr"],
            ],
            "Hybrid": [
                hybrid_metrics["precision@k"],
                hybrid_metrics["recall@k"],
                hybrid_metrics["ndcg@k"],
                hybrid_metrics["mrr"],
            ],
        }
    )
    comparison["Delta"] = comparison["Hybrid"] - comparison["Baseline"]
    comparison["Lift %"] = (
        (comparison["Delta"] / comparison["Baseline"].clip(lower=1e-9)) * 100
    ).round(1)
    print(comparison.to_string(index=False))

    # ==================== PHASE 10: SAMPLE RECOMMENDATIONS ====================
    print("\n" + "=" * 60)
    print("🎯 SAMPLE RECOMMENDATIONS")
    print("=" * 60)

    sample_users = eval_users[:3]
    for user_id in sample_users:
        user_train = train_interactions[train_interactions["user_id"] == user_id]
        if user_train.empty:
            continue

        result = hybrid_recommender.recommend_for_user(user_id=user_id, max_results=5)
        if "error" in result:
            continue

        seed = result.get("seed_item", {})
        print(f"\n👤 User {user_id}")
        print(f"   Seed: {seed.get('title', 'N/A')[:60]} (${seed.get('price', 0):.2f})")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Diversity: {result['diversity_score']:.2f}")
        for i, rec in enumerate(result["recommendations"]):
            expl = rec["explanation"]
            print(
                f"   {i+1}. {rec['title'][:55]:55s} "
                f"${rec['price']:7.2f} ⭐{rec['stars']:.1f} "
                f"| score={rec['score']:.3f} "
                f"(C={expl['content']:.2f} CF={expl['collaborative']:.2f} "
                f"P={expl['popularity']:.2f} R={expl['recency']:.2f})"
            )

    # ==================== PHASE 11: PERSIST ARTIFACTS ====================
    print("\n" + "=" * 60)
    print("💾 PHASE 11: PERSISTENCE")
    print("=" * 60)

    persistence.save_faiss_index(faiss_sys.index)
    persistence.save_tfidf(emb_system.tfidf_vectorizer, emb_system.tfidf_matrix)
    persistence.save_embeddings(embeddings)
    persistence.list_artifacts()

    # ==================== SUMMARY ====================
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"✅ PIPELINE COMPLETE in {total_time:.1f}s")
    print("=" * 60)
    print(f"   Products:     {len(df)}")
    print(f"   Users:        {N_USERS}")
    print(f"   Interactions: {len(interactions_df)}")
    print(f"   CF Model:     NMF (d={cf_model.n_components})")
    print(f"   Artifacts:    {ARTIFACT_DIR}/")
    print()


if __name__ == "__main__":
    run_pipeline()
