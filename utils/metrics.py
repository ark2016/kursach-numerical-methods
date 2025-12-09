"""
Метрики @k для multi-label классификации:
- Precision@k
- Recall@k
- F1@k
- MAP@k (Mean Average Precision)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Coverage@k
- Hit Rate@k
"""

import numpy as np
from typing import Union


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Precision@k
    """
    n_samples = y_true.shape[0]
    precisions = []

    for i in range(n_samples):
        # Индексы top-k меток по убыванию скоров
        top_k_indices = np.argsort(y_scores[i])[::-1][:k]
        # Сколько из top-k реально релевантны
        n_relevant_in_top_k = np.sum(y_true[i, top_k_indices])
        precisions.append(n_relevant_in_top_k / k)

    return np.mean(precisions)


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Recall@k
    """
    n_samples = y_true.shape[0]
    recalls = []

    for i in range(n_samples):
        n_true_labels = np.sum(y_true[i])
        if n_true_labels == 0:
            recalls.append(0.0)
            continue

        top_k_indices = np.argsort(y_scores[i])[::-1][:k]
        n_relevant_in_top_k = np.sum(y_true[i, top_k_indices])
        recalls.append(n_relevant_in_top_k / n_true_labels)

    return np.mean(recalls)


def f1_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    F1@k: гармоническое среднее Precision@k и Recall@k.
    """
    p = precision_at_k(y_true, y_scores, k)
    r = recall_at_k(y_true, y_scores, k)

    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Average Precision@k для одного sample.
    AP@k = (1/min(k, |relevant|)) * sum_{i=1}^{k} P@i * rel(i)
    """
    n_relevant = np.sum(y_true)
    if n_relevant == 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1][:k]
    ap_sum = 0.0
    n_hits = 0

    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            n_hits += 1
            precision_at_i = n_hits / (i + 1)
            ap_sum += precision_at_i

    return ap_sum / min(k, n_relevant)


def map_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Mean Average Precision@k: среднее AP@k по всем samples.
    """
    n_samples = y_true.shape[0]
    ap_scores = []

    for i in range(n_samples):
        ap_scores.append(average_precision_at_k(y_true[i], y_scores[i], k))

    return np.mean(ap_scores)


def dcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain@k для одного sample.
    DCG@k = sum_{i=1}^{k} rel(i) / log2(i + 1)
    """
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    dcg = 0.0

    for i, idx in enumerate(sorted_indices):
        rel = y_true[idx]
        dcg += rel / np.log2(i + 2)  # log2(i + 2) т.к. i начинается с 0

    return dcg


def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain@k: нормализованный DCG.
    """
    n_samples = y_true.shape[0]
    ndcg_scores = []

    for i in range(n_samples):
        dcg = dcg_at_k(y_true[i], y_scores[i], k)
        # Ideal DCG: сортируем по истинной релевантности
        ideal_dcg = dcg_at_k(y_true[i], y_true[i].astype(float), k)

        if ideal_dcg == 0:
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(dcg / ideal_dcg)

    return np.mean(ndcg_scores)


def hit_rate_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Hit Rate@k: доля samples, для которых хотя бы одна релевантная метка попала в top-k.
    """
    n_samples = y_true.shape[0]
    hits = 0

    for i in range(n_samples):
        top_k_indices = np.argsort(y_scores[i])[::-1][:k]
        if np.sum(y_true[i, top_k_indices]) > 0:
            hits += 1

    return hits / n_samples


def coverage_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Coverage@k: доля уникальных классов, которые появились в top-k хотя бы для одного sample.
    """
    n_samples, n_classes = y_scores.shape
    covered_classes = set()

    for i in range(n_samples):
        top_k_indices = np.argsort(y_scores[i])[::-1][:k]
        covered_classes.update(top_k_indices)

    return len(covered_classes) / n_classes


def compute_all_metrics_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: list[int] = [1, 3, 5, 10]
) -> dict:
    """
    Вычисляет все метрики @k для заданных значений k.
    """
    results = {}

    for k in k_values:
        if k > y_scores.shape[1]:
            continue

        results[f"precision_at_{k}"] = precision_at_k(y_true, y_scores, k)
        results[f"recall_at_{k}"] = recall_at_k(y_true, y_scores, k)
        results[f"f1_at_{k}"] = f1_at_k(y_true, y_scores, k)
        results[f"map_at_{k}"] = map_at_k(y_true, y_scores, k)
        results[f"ndcg_at_{k}"] = ndcg_at_k(y_true, y_scores, k)
        results[f"hit_rate_at_{k}"] = hit_rate_at_k(y_true, y_scores, k)

    return results


