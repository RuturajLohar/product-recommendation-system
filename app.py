
import os
import re
import pickle
import warnings
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import faiss
import gradio as gr
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
class Config:
    SAMPLE_SIZE = 5000
    MIN_PRICE = 0.01
    EMBEDDING_DIM = 384
    BATCH_SIZE = 64
    RANDOM_STATE = 42

    # default ensemble weights (you can tune these)
    ENSEMBLE_WEIGHTS = {
        "content_sbert": 0.35,
        "content_tfidf": 0.15,
        "category": 0.15,
        "popularity": 0.05,
        "price": 0.05,
        "rating": 0.10,
        "review_count": 0.05,
        "best_seller": 0.05,
    }

    # Bayesian rating damping factor — require this many reviews for full confidence
    RATING_DAMPING = 50

config = Config()

# ==================== DATA PROCESSOR ====================
class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # price
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
        df['price_log'] = np.log1p(df['price'].clip(lower=0.0))
        if df['price'].std() == 0 or np.isnan(df['price'].std()):
            df['price_zscore'] = 0.0
        else:
            df['price_zscore'] = (df['price'] - df['price'].mean()) / (df['price'].std() + 1e-9)

        # stars (rating)
        df['stars'] = pd.to_numeric(df.get('stars', None), errors='coerce')
        median_stars = df['stars'].median() if not np.isnan(df['stars'].median()) else 4.0
        df['stars'] = df['stars'].fillna(median_stars)
        df['rating_normalized'] = (df['stars'] / 5.0).clip(0.0, 1.0)

        # reviews — for Bayesian rating and review confidence
        df['reviews'] = pd.to_numeric(df.get('reviews', 0), errors='coerce').fillna(0).astype(int)

        # Bayesian damped rating: (R*v + C*m) / (v + m)
        # Discounts high ratings with few reviews
        global_mean = df['stars'].mean()
        m = config.RATING_DAMPING
        df['rating_bayesian'] = (
            (df['stars'] * df['reviews'].clip(lower=0) + global_mean * m) /
            (df['reviews'].clip(lower=0) + m)
        ) / 5.0
        df['rating_bayesian'] = df['rating_bayesian'].clip(0.0, 1.0)

        # Review confidence — log-scaled review count (tames outliers)
        raw_reviews = df['reviews'].clip(lower=0).astype(float)
        df['review_confidence'] = np.log1p(raw_reviews)
        max_rc = df['review_confidence'].max()
        if max_rc > 0:
            df['review_confidence'] = df['review_confidence'] / max_rc
        else:
            df['review_confidence'] = 0.5

        # Best seller flag
        if 'isBestSeller' in df.columns:
            df['best_seller'] = df['isBestSeller'].apply(
                lambda x: 1.0 if str(x).strip().lower() in ('true', '1', 'yes') else 0.0
            )
        else:
            df['best_seller'] = 0.0

        # Popularity — log-scaled before normalizing (tames viral outliers)
        if 'boughtInLastMonth' in df.columns:
            raw_pop = pd.to_numeric(df['boughtInLastMonth'].fillna(0), errors='coerce')
            df['popularity_score'] = np.log1p(raw_pop.clip(lower=0))
            max_pop = df['popularity_score'].max()
            if max_pop > 0:
                df['popularity_score'] = df['popularity_score'] / max_pop
            else:
                df['popularity_score'] = 0.5
        else:
            df['popularity_score'] = 0.5

        return df

    def load_and_clean_data(self, path: str) -> pd.DataFrame:
        print("📥 Loading data...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at: {path}")

        df = pd.read_csv(path)
        original_len = len(df)

        # required columns
        required_cols = ['asin', 'title', 'categoryName', 'price']
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' not in CSV.")

        df = df.dropna(subset=['asin', 'title', 'price'])
        df['title'] = df['title'].astype(str)
        df = df[df['price'].astype(float) > config.MIN_PRICE]
        df = df[df['title'].str.len() > 5]

        if len(df) > config.SAMPLE_SIZE:
            df = df.sample(n=config.SAMPLE_SIZE, random_state=config.RANDOM_STATE).reset_index(drop=True)

        df['title'] = df['title'].apply(self.clean_text)
        df['categoryName'] = df['categoryName'].fillna('Unknown')
        df['imgUrl'] = df.get('imgUrl', '')
        df = self.create_features(df)
        print(f"✅ Loaded {len(df)} / {original_len} products (sampled up to {config.SAMPLE_SIZE}).")
        return df.reset_index(drop=True)

# ==================== EMBEDDING & TF-IDF SYSTEM ====================
class EmbeddingSystem:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L12-v2'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 Loading SentenceTransformer on device: {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print("⚠️ Model load error, attempting CPU fallback:", e)
            self.model = SentenceTransformer(model_name, device='cpu')

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    @staticmethod
    def _build_rich_text(row) -> str:
        """Build enriched text for embedding: title + category + price tier + rating.
        This lets FAISS retrieval capture category and price-tier semantics at the vector level."""
        parts = [str(row.get('title', ''))]
        cat = row.get('categoryName', '')
        if cat and str(cat) != 'Unknown':
            parts.append(f"category: {cat}")
        price = float(row.get('price', 0))
        if price > 0:
            tier = "budget" if price < 25 else "mid-range" if price < 100 else "premium" if price < 300 else "luxury"
            parts.append(f"price: {tier}")
        stars = float(row.get('stars', 0))
        if stars > 0:
            parts.append(f"rating: {stars:.1f} stars")
        return ". ".join(parts)

    def create_embeddings(self, df: pd.DataFrame, cache_path: Optional[str] = None) -> np.ndarray:
        if cache_path and os.path.exists(cache_path):
            try:
                print("📂 Loading cached embeddings...")
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                    embeddings = np.array(embeddings, dtype=np.float32)
                    faiss.normalize_L2(embeddings)
                    return embeddings
            except Exception:
                print("⚠️ Failed to load cache, will re-create embeddings.")

        print("🔤 Creating enriched SBERT embeddings (title + category + price tier + rating)...")
        texts = df.apply(self._build_rich_text, axis=1).tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.model.device if hasattr(self.model, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embeddings, f)
            except Exception:
                print("⚠️ Could not write embeddings cache.")

        return embeddings

    def fit_tfidf(self, df: pd.DataFrame, max_features: int = 30000, ngram_range=(1,2)):
        print("🧾 Fitting TF-IDF vectorizer on titles...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
        texts = df['title'].fillna('').astype(str).tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print(f"✅ TF-IDF fitted (vocab size: {len(self.tfidf_vectorizer.vocabulary_)})")

# ==================== FAISS SYSTEM ====================
class FAISSSystem:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # works with normalized vectors (cosine)

    def build_index(self, embeddings: np.ndarray):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Embeddings shape mismatch.")
        self.index.add(embeddings)
        print(f"📚 Built FAISS index with {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, k: int = 10):
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

# ==================== ADVANCED RECOMMENDER (SBERT + TFIDF + MMR) ====================
class AdvancedRecommender:
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray, faiss_system: FAISSSystem, emb_system: EmbeddingSystem):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings.astype(np.float32)
        self.faiss_system = faiss_system
        self.emb_system = emb_system
        self.recommendation_cache: Dict[str, Any] = {}

        # Pre-compute category embeddings for soft category similarity
        categories = self.df['categoryName'].unique().tolist()
        print(f"🏷️  Computing category embeddings for {len(categories)} categories...")
        cat_embs = emb_system.model.encode(categories, convert_to_numpy=True)
        cat_embs = cat_embs.astype(np.float32)
        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(cat_embs, axis=1, keepdims=True) + 1e-9
        cat_embs = cat_embs / norms
        self.category_embedding_map: Dict[str, np.ndarray] = dict(zip(categories, cat_embs))

    @staticmethod
    def _price_similarity(a_price, b_price):
        """Symmetric log-ratio price similarity. Returns 1.0 when equal, decays smoothly.
        Fixes the old asymmetric formula where price_sim(5,100) != price_sim(100,5)."""
        a = max(float(a_price) if not np.isnan(a_price) else 0.01, 0.01)
        b = max(float(b_price) if not np.isnan(b_price) else 0.01, 0.01)
        log_ratio = abs(np.log(a) - np.log(b))
        return float(np.exp(-log_ratio))

    @staticmethod
    def _rating_similarity(a_star, b_star):
        return 1.0 - min(1.0, abs(float(a_star) - float(b_star)) / 5.0)

    def _category_similarity(self, cat_a: str, cat_b: str) -> float:
        """Soft category similarity via SBERT embedding cosine distance.
        'Hi-Fi Speakers' vs 'Headphones' now scores ~0.7 instead of 0.0."""
        if str(cat_a) == str(cat_b):
            return 1.0
        emb_a = self.category_embedding_map.get(str(cat_a))
        emb_b = self.category_embedding_map.get(str(cat_b))
        if emb_a is None or emb_b is None:
            return 0.0
        return float(np.dot(emb_a, emb_b))

    @staticmethod
    def _normalize_scores(scores: np.ndarray):
        if scores.size == 0:
            return scores
        minv = float(scores.min())
        maxv = float(scores.max())
        if maxv - minv < 1e-9:
            return np.ones_like(scores) * 0.5
        return (scores - minv) / (maxv - minv)

    def _compute_tfidf_similarity(self, query_text: str, candidate_indices: List[int]):
        if self.emb_system.tfidf_vectorizer is None or self.emb_system.tfidf_matrix is None or len(candidate_indices) == 0:
            return np.zeros(len(candidate_indices), dtype=np.float32)
        q_vec = self.emb_system.tfidf_vectorizer.transform([query_text])
        cand_mat = self.emb_system.tfidf_matrix[candidate_indices]
        sims = cosine_similarity(q_vec, cand_mat).flatten()
        return np.array(sims, dtype=np.float32)

    def _mmr(self, candidate_ids: List[int], relevance_scores: np.ndarray, similarity_matrix: np.ndarray, top_k: int = 10, lamb: float = 0.65):
        """MMR diversity reranking. Fixed O(n²) bug with candidate_ids.index()."""
        if len(candidate_ids) == 0:
            return []

        n = len(candidate_ids)
        selected_positions = []  # positions in the candidate list
        remaining = set(range(n))

        # Start with highest-relevance candidate
        idx0 = int(np.argmax(relevance_scores))
        selected_positions.append(idx0)
        remaining.discard(idx0)

        while len(selected_positions) < min(top_k, n) and remaining:
            best_score = -float('inf')
            best_pos = -1
            for r in remaining:
                # Max similarity to already-selected items
                sim_to_selected = max(similarity_matrix[r, s] for s in selected_positions)
                mmr_score = lamb * relevance_scores[r] - (1 - lamb) * sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_pos = r
            if best_pos == -1:
                break
            selected_positions.append(best_pos)
            remaining.discard(best_pos)

        return [candidate_ids[p] for p in selected_positions]

    def recommend(self, asin_input: str, weights: Optional[Dict[str, float]] = None, max_results: int = 10, mmr_lambda: float = 0.65):
        if weights is None:
            weights = config.ENSEMBLE_WEIGHTS
        cache_key = f"{asin_input}_{str(weights)}_{max_results}_{mmr_lambda}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]

        matches = self.df.index[self.df['asin'] == asin_input].tolist()
        if not matches:
            sample_products = self.df[['asin', 'title']].head(10).to_dict('records')
            return {'error': f"Product ASIN '{asin_input}' not found.", 'available_products': sample_products}
        q_idx = matches[0]
        q_row = self.df.iloc[q_idx]
        q_title = str(q_row['title'])

        # 1) FAISS candidate retrieval (large pool)
        candidate_pool = max(200, max_results * 20)
        distances, indices = self.faiss_system.search(self.embeddings[q_idx:q_idx+1], k=candidate_pool)
        cand_indices = [int(i) for i in indices[0] if int(i) != q_idx and int(i) != -1]
        if len(cand_indices) == 0:
            return {'error': 'No candidates found by FAISS.', 'available_products': []}

        # 2) compute SBERT cosine (dot product because normalized)
        q_emb = self.embeddings[q_idx:q_idx+1]
        cand_embs = self.embeddings[cand_indices]
        sbert_sims = (cand_embs @ q_emb.T).flatten().astype(np.float32)

        # 3) TF-IDF lexical similarity
        tfidf_sims = self._compute_tfidf_similarity(q_title, cand_indices)

        # 4) metadata similarities — all improved
        price_sims = np.array([self._price_similarity(q_row['price'], self.df.iloc[i]['price']) for i in cand_indices], dtype=np.float32)
        rating_sims = np.array([self._rating_similarity(q_row['stars'], self.df.iloc[i]['stars']) for i in cand_indices], dtype=np.float32)

        # Soft category similarity (embedding-based, not binary)
        q_cat = str(q_row['categoryName'])
        category_sims = np.array([self._category_similarity(q_cat, str(self.df.iloc[i]['categoryName'])) for i in cand_indices], dtype=np.float32)

        popularity_vals = np.array([float(self.df.iloc[i].get('popularity_score', 0.5)) for i in cand_indices], dtype=np.float32)

        # Bayesian rating for candidates (discounts high stars with low review count)
        bayesian_ratings = np.array([float(self.df.iloc[i].get('rating_bayesian', self.df.iloc[i].get('rating_normalized', 0.5))) for i in cand_indices], dtype=np.float32)

        # Review confidence (log-scaled review count)
        review_conf = np.array([float(self.df.iloc[i].get('review_confidence', 0.5)) for i in cand_indices], dtype=np.float32)

        # Best seller flag
        best_seller = np.array([float(self.df.iloc[i].get('best_seller', 0.0)) for i in cand_indices], dtype=np.float32)

        # 5) normalize
        sbert_norm = self._normalize_scores(sbert_sims)
        tfidf_norm = self._normalize_scores(tfidf_sims)
        price_norm = self._normalize_scores(price_sims)
        rating_norm = self._normalize_scores(bayesian_ratings)
        pop_norm = self._normalize_scores(popularity_vals)
        category_norm = self._normalize_scores(category_sims)
        review_norm = self._normalize_scores(review_conf)
        # best_seller is already 0/1, no normalization needed

        # 6) ensemble score — now includes all signals
        ensemble_scores = (
            weights.get('content_sbert', 0.35) * sbert_norm +
            weights.get('content_tfidf', 0.15) * tfidf_norm +
            weights.get('category', 0.15) * category_norm +
            weights.get('popularity', 0.05) * pop_norm +
            weights.get('price', 0.05) * price_norm +
            weights.get('rating', 0.10) * rating_norm +
            weights.get('review_count', 0.05) * review_norm +
            weights.get('best_seller', 0.05) * best_seller
        )
        if np.allclose(ensemble_scores, 0.0):
            ensemble_scores = sbert_norm

        # 7) build candidate similarity matrix for MMR (SBERT-based)
        try:
            cand_sim_matrix = cosine_similarity(cand_embs)
        except Exception:
            cand_sim_matrix = np.zeros((len(cand_indices), len(cand_indices)), dtype=np.float32)

        # 8) apply MMR to select top-k
        selected_ids = self._mmr(cand_indices, ensemble_scores, cand_sim_matrix, top_k=max_results, lamb=mmr_lambda)

        # Build index lookup for O(1) position finding
        cand_idx_to_pos = {idx: pos for pos, idx in enumerate(cand_indices)}

        recs = []
        for idx in selected_ids:
            pos = cand_idx_to_pos[idx]
            p = self.df.iloc[idx]
            explanation = {
                'sbert': float(sbert_norm[pos]),
                'tfidf': float(tfidf_norm[pos]),
                'category': float(category_norm[pos]),
                'priceSim': float(price_norm[pos]),
                'ratingSim': float(rating_norm[pos]),
                'popularity': float(pop_norm[pos]),
                'reviewConf': float(review_norm[pos]),
                'bestSeller': float(best_seller[pos]),
                'ensemble_score': float(ensemble_scores[pos])
            }
            recs.append({
                'asin': p['asin'],
                'title': p['title'],
                'price': float(p['price']),
                'stars': float(p['stars']),
                'category': p['categoryName'],
                'image_url': p.get('imgUrl', ''),
                'score': float(explanation['ensemble_score']),
                'explanation': explanation
            })

        result = {
            'input_product': {
                'asin': q_row['asin'],
                'title': q_row['title'],
                'price': float(q_row['price']),
                'stars': float(q_row['stars']),
                'categoryName': q_row['categoryName']
            },
            'recommendations': recs,
            'diversity_score': len(set([r['category'] for r in recs])) / len(recs) if recs else 0.0,
            'weights_used': weights
        }
        self.recommendation_cache[cache_key] = result
        return result

# ==================== GRADIO INTERFACE ====================
class GradioInterface:
    def __init__(self, df: pd.DataFrame, recommender: AdvancedRecommender):
        self.df = df
        self.recommender = recommender

    def create_interface(self):
        first_asin = self.df['asin'].iloc[0] if len(self.df) > 0 else ''
        with gr.Blocks(theme=gr.themes.Soft(), title="🎯 Advanced Product Recommender") as demo:
            gr.Markdown("# 🎯 Advanced Product Recommendation System\nAI-powered hybrid recommendations (SBERT + TF-IDF + MMR)")

            asin_input = gr.Textbox(label="Product ASIN", value=first_asin)
            sample_asins = gr.Dropdown(self.df['asin'].head(50).tolist(), label="Sample Products", value=first_asin)
            sample_asins.change(fn=lambda s: s, inputs=sample_asins, outputs=asin_input)

            with gr.Accordion("🎚️ Strategy Weights", open=False):
                content_sbert_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['content_sbert'], label="SBERT content weight")
                content_tfidf_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['content_tfidf'], label="TF-IDF lexical weight")
                category_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['category'], label="Category similarity weight")
                pop_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['popularity'], label="Popularity weight")
                price_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['price'], label="Price similarity weight")
                rating_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['rating'], label="Bayesian rating weight")
                review_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['review_count'], label="Review confidence weight")
                bestseller_w = gr.Slider(0, 1, value=config.ENSEMBLE_WEIGHTS['best_seller'], label="Best seller weight")

            max_results = gr.Slider(1, 20, value=8, step=1, label="Number of Recommendations")
            mmr_lambda = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="MMR λ (higher -> more relevance, lower -> more diversity)")

            recommend_btn = gr.Button("🎯 Get Recommendations", variant="primary")
            clear_btn = gr.Button("🔄 Clear")

            product_info = gr.Markdown()
            gallery = gr.HTML()
            diversity_score = gr.Number(label="Diversity Score")
            strategy_info = gr.JSON(label="Weights Used")

            def get_recommendations(asin_input, sbert_w, tfidf_w, cat_w, pop_w, pr_w, rt_w, rev_w, bs_w, maxres, mmr_lam):
                weights = {
                    'content_sbert': float(sbert_w),
                    'content_tfidf': float(tfidf_w),
                    'category': float(cat_w),
                    'popularity': float(pop_w),
                    'price': float(pr_w),
                    'rating': float(rt_w),
                    'review_count': float(rev_w),
                    'best_seller': float(bs_w),
                }
                # Normalize weights to sum to 1 for stability
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}

                result = self.recommender.recommend(asin_input, weights=weights, max_results=int(maxres), mmr_lambda=float(mmr_lam))
                if 'error' in result:
                    msg = f"❌ {result['error']}"
                    if 'available_products' in result:
                        msg += "\n\n**Available:**\n" + "\n".join([f"- `{p['asin']}` {p['title'][:60]}..." for p in result['available_products']])
                    return msg, "", 0.0, {}

                ip = result['input_product']
                recs = result['recommendations']
                info = f"""### 📦 Input Product
**ASIN:** `{ip['asin']}`
**Title:** {ip['title']}
**Price:** ${ip['price']:.2f}
**Stars:** {ip['stars']} ⭐
**Category:** {ip['categoryName']}"""

                html = "<div style='display:flex;flex-wrap:wrap;gap:18px;'>"
                for r in recs:
                    img = r['image_url'] if r['image_url'] else f"https://via.placeholder.com/200x200.png?text={r['asin'][-6:]}"
                    expl = r.get('explanation', {})
                    expl_text = (f"SBERT: {expl.get('sbert', 0):.2f} · TF-IDF: {expl.get('tfidf',0):.2f} · "
                                 f"Cat: {expl.get('category',0):.0f} · Price: {expl.get('priceSim',0):.2f} · "
                                 f"Rating: {expl.get('ratingSim',0):.2f}")
                    html += f"""
                        <div style='width:240px;text-align:left;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:10px;background:#0b1020;color:#f5f7fb;'>
                            <img src='{img}' style='width:220px;height:180px;object-fit:cover;border-radius:8px;display:block;margin:0 auto;'/>
                            <p style='font-weight:700;margin:8px 0 4px 0;color:#fff'>{r['title'][:70]}{'...' if len(r['title'])>70 else ''}</p>
                            <p style='margin:0;font-size:14px;'>💰 ${r['price']:.2f} · ⭐ {r['stars']}</p>
                            <p style='margin:4px 0 6px 0;font-size:12px;color:#bcd;'>{r['category']}</p>
                            <p style='font-size:11px;color:#9fb; margin:0;'>{expl_text}</p>
                            <p style='font-size:11px;color:#99a;margin-top:6px;'>score: {r['score']:.3f}</p>
                        </div>
                    """
                html += "</div>"

                return info, html, result['diversity_score'], result['weights_used']

            recommend_btn.click(
                fn=get_recommendations,
                inputs=[asin_input, content_sbert_w, content_tfidf_w, category_w, pop_w, price_w, rating_w, review_w, bestseller_w, max_results, mmr_lambda],
                outputs=[product_info, gallery, diversity_score, strategy_info]
            )

            clear_btn.click(fn=lambda: ["", "", 0.0, {}], outputs=[product_info, gallery, diversity_score, strategy_info])

        return demo

# ==================== MAIN EXECUTION ====================
def main(data_path: str, embed_cache: Optional[str] = None, mount_drive: bool = True):
    print("🚀 Starting Advanced Product Recommender pipeline...")

    # Optional Colab Drive mount
    try:
        if mount_drive:
            from google.colab import drive  # type: ignore
            drive.mount('/content/drive', force_remount=False)
            print("✅ Google Drive mounted.")
    except Exception:
        print("ℹ️ Google Drive mount skipped or unavailable.")

    processor = DataProcessor()
    df = processor.load_and_clean_data(data_path)

    emb_system = EmbeddingSystem()
    embeddings = emb_system.create_embeddings(df, cache_path=embed_cache)
    emb_system.fit_tfidf(df)

    faiss_sys = FAISSSystem(embeddings.shape[1])
    faiss_sys.build_index(embeddings)

    recommender = AdvancedRecommender(df, embeddings, faiss_sys, emb_system)
    interface = GradioInterface(df, recommender)
    demo = interface.create_interface()

    print("✅ System Ready!")
    demo.launch(share=True, inbrowser=False)

# ==================== RUN (update paths if needed) ====================
if __name__ == "__main__":
    # Uploaded files in the Space repo root
    DATA_PATH = "amz_uk_processed_data.csv"        # your uploaded CSV (make sure the name matches exactly)
    EMBED_CACHE = "product_embeddings_hybrid.pkl"  # set to filename if uploaded, else use None
    # Spaces cannot mount Google Drive — disable mount_drive
    main(DATA_PATH, embed_cache=EMBED_CACHE, mount_drive=False)
