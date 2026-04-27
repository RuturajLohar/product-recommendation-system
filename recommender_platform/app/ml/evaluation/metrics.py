import numpy as np
from typing import List

class EvaluationFramework:
    @staticmethod
    def precision_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        if not actual: return 0.0
        predicted_k = predicted[:k]
        relevant = [p for p in predicted_k if p in actual]
        return len(relevant) / k

    @staticmethod
    def recall_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        if not actual: return 0.0
        predicted_k = predicted[:k]
        relevant = [p for p in predicted_k if p in actual]
        return len(relevant) / len(actual)

    @staticmethod
    def ndcg_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        def dcg(scores):
            return sum([s / np.log2(i + 2) for i, s in enumerate(scores)])
        
        predicted_k = predicted[:k]
        relevance_scores = [1 if p in actual else 0 for p in predicted_k]
        
        actual_relevance = sorted([1] * min(len(actual), k) + [0] * max(0, k - len(actual)), reverse=True)
        
        idcg = dcg(actual_relevance)
        if idcg == 0: return 0.0
        
        return dcg(relevance_scores) / idcg

    @staticmethod
    def mrr(actual: List[int], predicted: List[int]) -> float:
        for i, p in enumerate(predicted):
            if p in actual:
                return 1.0 / (i + 1)
        return 0.0
