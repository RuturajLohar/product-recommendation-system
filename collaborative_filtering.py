"""
Collaborative Filtering Module
================================
Implements collaborative filtering using Scikit-Learn's NMF (Non-negative Matrix Factorization).

Why NMF:
- Robust and widely available via Scikit-Learn.
- No complex C-extensions or compilation issues (unlike LightFM).
- Produces interpretable user and item latent factors.
- Handles implicit feedback via the interaction weight matrix.

Architecture:
- Builds a sparse user-item interaction matrix from interaction logs.
- Decomposes the matrix into User (W) and Item (H) matrices.
- Predicted score is the dot product of user and item latent vectors.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Optional, Tuple, Dict
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF

class CollaborativeFilter:
    """NMF-based collaborative filtering model."""

    def __init__(self, n_components: int = 64, epochs: int = 50):
        """
        Args:
            n_components: Dimensionality of the latent space.
            epochs: Number of iterations for NMF.
        """
        self.n_components = n_components
        self.epochs = epochs
        self.model: Optional[NMF] = None
        
        # Latent matrices
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None

        # Mappings
        self.user_id_map: Dict[int, int] = {}  # external user_id -> internal idx
        self.item_idx_map: Dict[int, int] = {}  # external item_idx -> internal idx
        self.reverse_item_map: Dict[int, int] = {}  # internal idx -> external item_idx

        self.n_users = 0
        self.n_items = 0

    def build_interaction_matrix(
        self, interactions_df: pd.DataFrame, n_items_total: int
    ) -> csr_matrix:
        """
        Build a sparse user-item interaction matrix.
        """
        print("📊 Building interaction matrix...")

        unique_users = sorted(interactions_df["user_id"].unique())
        
        self.user_id_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_idx_map = {iid: i for i, iid in enumerate(range(n_items_total))}
        self.reverse_item_map = {v: k for k, v in self.item_idx_map.items()}

        self.n_users = len(unique_users)
        self.n_items = n_items_total

        # Aggregate weights per (user, item) pair
        agg = (
            interactions_df.groupby(["user_id", "item_idx"])["weight"]
            .sum()
            .reset_index()
        )

        rows = [self.user_id_map[uid] for uid in agg["user_id"]]
        cols = [self.item_idx_map[iid] for iid in agg["item_idx"]]
        vals = agg["weight"].values.astype(np.float32)

        interaction_matrix = coo_matrix(
            (vals, (rows, cols)), shape=(self.n_users, self.n_items)
        ).tocsr()

        density = len(vals) / (self.n_users * self.n_items) * 100
        print(
            f"✅ Interaction matrix: {self.n_users} users × {self.n_items} items "
            f"(density: {density:.3f}%)"
        )
        return interaction_matrix

    def train(self, interactions_df: pd.DataFrame, n_items_total: int):
        """
        Train the NMF model.
        """
        X = self.build_interaction_matrix(interactions_df, n_items_total)

        print(
            f"🧠 Training NMF (components={self.n_components}, "
            f"max_iter={self.epochs})..."
        )

        self.model = NMF(
            n_components=self.n_components,
            init='random',
            random_state=42,
            max_iter=self.epochs,
            alpha_W=0.1,
            alpha_H=0.1,
            l1_ratio=0.5
        )

        # W is (n_users, n_components), H is (n_components, n_items)
        self.user_embeddings = self.model.fit_transform(X)
        self.item_embeddings = self.model.components_.T  # Shape: (n_items, n_components)
        
        print("✅ NMF training complete.")

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get the latent embedding for a user."""
        internal_idx = self.user_id_map.get(user_id)
        if internal_idx is None or self.user_embeddings is None:
            return None
        return self.user_embeddings[internal_idx]

    def get_item_embedding(self, item_idx: int) -> Optional[np.ndarray]:
        """Get the CF latent embedding for an item."""
        internal_idx = self.item_idx_map.get(item_idx)
        if internal_idx is None or self.item_embeddings is None:
            return None
        return self.item_embeddings[internal_idx]

    def predict_user_items(
        self, user_id: int, item_indices: list
    ) -> np.ndarray:
        """
        Predict scores for a user across specific items via dot product.
        """
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            return np.zeros(len(item_indices), dtype=np.float32)

        scores = []
        for idx in item_indices:
            item_emb = self.get_item_embedding(idx)
            if item_emb is not None:
                scores.append(np.dot(user_emb, item_emb))
            else:
                scores.append(0.0)
                
        return np.array(scores, dtype=np.float32)

    def predict_all_items_for_user(self, user_id: int) -> np.ndarray:
        """
        Predict scores for a user across ALL items.
        """
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None or self.item_embeddings is None:
            return np.zeros(self.n_items, dtype=np.float32)

        return np.dot(self.item_embeddings, user_emb)

    def get_user_history(
        self, user_id: int, interactions_df: pd.DataFrame
    ) -> list:
        """Get the list of item indices a user has interacted with."""
        user_rows = interactions_df[interactions_df["user_id"] == user_id]
        return user_rows["item_idx"].unique().tolist()

    def save(self, path: str):
        """Save the trained model and mappings."""
        data = {
            "user_embeddings": self.user_embeddings,
            "item_embeddings": self.item_embeddings,
            "user_id_map": self.user_id_map,
            "item_idx_map": self.item_idx_map,
            "reverse_item_map": self.reverse_item_map,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_components": self.n_components,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 CF model saved to {path}")

    def load(self, path: str):
        """Load a previously trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.user_embeddings = data["user_embeddings"]
        self.item_embeddings = data["item_embeddings"]
        self.user_id_map = data["user_id_map"]
        self.item_idx_map = data["item_idx_map"]
        self.reverse_item_map = data["reverse_item_map"]
        self.n_users = data["n_users"]
        self.n_items = data["n_items"]
        self.n_components = data["n_components"]
        print(f"📂 CF model loaded from {path}")
