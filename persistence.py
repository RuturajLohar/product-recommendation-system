"""
Persistence Layer
==================
Save and load all model artifacts to/from disk.

Artifacts managed:
- FAISS index (binary format via faiss.write_index / read_index)
- TF-IDF vectorizer (pickle)
- TF-IDF matrix (pickle)
- LightFM model (handled by collaborative_filtering.py)
- Embeddings (numpy .npy format — faster than pickle for arrays)
"""

import os
import pickle
import numpy as np
import faiss
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer


class PersistenceManager:
    """Handles saving and loading of all model artifacts."""

    def __init__(self, artifact_dir: str = "artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.artifact_dir, filename)

    # ==================== FAISS ====================
    def save_faiss_index(self, index: faiss.Index, filename: str = "faiss_index.bin"):
        path = self._path(filename)
        faiss.write_index(index, path)
        print(f"💾 FAISS index saved to {path} ({index.ntotal} vectors)")

    def load_faiss_index(self, filename: str = "faiss_index.bin") -> Optional[faiss.Index]:
        path = self._path(filename)
        if not os.path.exists(path):
            print(f"⚠️ FAISS index not found at {path}")
            return None
        index = faiss.read_index(path)
        print(f"📂 FAISS index loaded from {path} ({index.ntotal} vectors)")
        return index

    # ==================== TF-IDF ====================
    def save_tfidf(
        self,
        vectorizer: TfidfVectorizer,
        matrix,
        vec_filename: str = "tfidf_vectorizer.pkl",
        mat_filename: str = "tfidf_matrix.pkl",
    ):
        vec_path = self._path(vec_filename)
        mat_path = self._path(mat_filename)
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(mat_path, "wb") as f:
            pickle.dump(matrix, f)
        print(f"💾 TF-IDF saved: vectorizer → {vec_path}, matrix → {mat_path}")

    def load_tfidf(
        self,
        vec_filename: str = "tfidf_vectorizer.pkl",
        mat_filename: str = "tfidf_matrix.pkl",
    ):
        vec_path = self._path(vec_filename)
        mat_path = self._path(mat_filename)
        if not os.path.exists(vec_path) or not os.path.exists(mat_path):
            print("⚠️ TF-IDF artifacts not found.")
            return None, None
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(mat_path, "rb") as f:
            matrix = pickle.load(f)
        print(f"📂 TF-IDF loaded from {vec_path}")
        return vectorizer, matrix

    # ==================== EMBEDDINGS ====================
    def save_embeddings(self, embeddings: np.ndarray, filename: str = "embeddings.npy"):
        path = self._path(filename)
        np.save(path, embeddings)
        print(f"💾 Embeddings saved to {path} (shape: {embeddings.shape})")

    def load_embeddings(self, filename: str = "embeddings.npy") -> Optional[np.ndarray]:
        path = self._path(filename)
        if not os.path.exists(path):
            print(f"⚠️ Embeddings not found at {path}")
            return None
        embeddings = np.load(path)
        print(f"📂 Embeddings loaded from {path} (shape: {embeddings.shape})")
        return embeddings

    # ==================== GENERIC PICKLE ====================
    def save_object(self, obj, filename: str):
        path = self._path(filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"💾 Object saved to {path}")

    def load_object(self, filename: str):
        path = self._path(filename)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"📂 Object loaded from {path}")
        return obj

    def list_artifacts(self):
        """List all saved artifacts."""
        files = os.listdir(self.artifact_dir)
        if not files:
            print("📁 No artifacts found.")
            return []
        print(f"📁 Artifacts in {self.artifact_dir}/:")
        for f in sorted(files):
            size_mb = os.path.getsize(self._path(f)) / (1024 * 1024)
            print(f"   {f} ({size_mb:.2f} MB)")
        return files
