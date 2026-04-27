"""
Evaluation Framework
=====================
Implements standard recommendation system evaluation metrics.

Metrics implemented:
- Precision@K: Fraction of recommended items that are relevant.
- Recall@K: Fraction of relevant items that are recommended.
- nDCG@K: Normalized Discounted Cumulative Gain — measures ranking quality.
- MRR: Mean Reciprocal Rank — how early the first relevant item appears.

Usage:
    evaluator = RecommendationEvaluator()
    evaluator.add_result(actual=[1,2,3], predicted=[2,5,3,7,1])
    metrics = evaluator.compute_all(k=5)
"""

import numpy as np
from typing import List, Dict


class RecommendationEvaluator:
    """Collects prediction results and computes ranking metrics."""

    def __init__(self):
        self.results = []  # List of (actual_items, predicted_items) tuples

    def add_result(self, actual: List[int], predicted: List[int]):
        """
        Add a single evaluation result.

        Args:
            actual: List of relevant item indices (ground truth).
            predicted: Ranked list of recommended item indices.
        """
        self.results.append((actual, predicted))

    def reset(self):
        """Clear all stored results."""
        self.results = []

    @staticmethod
    def precision_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Precision@K = |{relevant items in top-K}| / K

        Measures: "Of the K items recommended, how many are relevant?"
        """
        if k <= 0:
            return 0.0
        predicted_k = predicted[:k]
        if not predicted_k:
            return 0.0
        relevant = set(actual)
        hits = sum(1 for p in predicted_k if p in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Recall@K = |{relevant items in top-K}| / |{all relevant items}|

        Measures: "Of all relevant items, how many did we recommend in top-K?"
        """
        if not actual or k <= 0:
            return 0.0
        predicted_k = predicted[:k]
        relevant = set(actual)
        hits = sum(1 for p in predicted_k if p in relevant)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain @ K.

        Measures: "How good is the ranking?" Items appearing earlier get
        more credit. A perfect ranking has nDCG = 1.0.
        """
        if not actual or k <= 0:
            return 0.0

        predicted_k = predicted[:k]
        relevant = set(actual)

        # DCG: sum of 1/log2(rank+1) for relevant items
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG: best possible ranking
        ideal_hits = min(len(actual), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def mrr(actual: List[int], predicted: List[int]) -> float:
        """
        Mean Reciprocal Rank.

        Measures: "How early does the first relevant item appear?"
        MRR = 1/rank_of_first_relevant_item
        """
        relevant = set(actual)
        for i, item in enumerate(predicted):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def compute_all(self, k: int = 10) -> Dict[str, float]:
        """
        Compute all metrics averaged across all stored results.

        Args:
            k: The cutoff for @K metrics.

        Returns:
            Dictionary with metric names and values.
        """
        if not self.results:
            return {
                "precision@k": 0.0,
                "recall@k": 0.0,
                "ndcg@k": 0.0,
                "mrr": 0.0,
                "k": k,
                "n_queries": 0,
            }

        precisions = []
        recalls = []
        ndcgs = []
        mrrs = []

        for actual, predicted in self.results:
            precisions.append(self.precision_at_k(actual, predicted, k))
            recalls.append(self.recall_at_k(actual, predicted, k))
            ndcgs.append(self.ndcg_at_k(actual, predicted, k))
            mrrs.append(self.mrr(actual, predicted))

        return {
            "precision@k": float(np.mean(precisions)),
            "recall@k": float(np.mean(recalls)),
            "ndcg@k": float(np.mean(ndcgs)),
            "mrr": float(np.mean(mrrs)),
            "k": k,
            "n_queries": len(self.results),
        }

    def print_report(self, k: int = 10, label: str = ""):
        """Print a formatted metrics report."""
        metrics = self.compute_all(k)
        header = f"📊 Evaluation Report{f' ({label})' if label else ''}"
        print(f"\n{'='*50}")
        print(header)
        print(f"{'='*50}")
        print(f"  K              = {metrics['k']}")
        print(f"  Queries        = {metrics['n_queries']}")
        print(f"  Precision@{k:<4} = {metrics['precision@k']:.4f}")
        print(f"  Recall@{k:<7} = {metrics['recall@k']:.4f}")
        print(f"  nDCG@{k:<9} = {metrics['ndcg@k']:.4f}")
        print(f"  MRR            = {metrics['mrr']:.4f}")
        print(f"{'='*50}\n")
        return metrics
