"""Evaluation metrics for deepfake detection (dependency-free numpy)."""
from __future__ import annotations

import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (tie-aware)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = scores[labels]
    negatives = scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([negatives, positives]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks for ties.
    combined = np.concatenate([negatives, positives])
    sorted_scores = combined[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    rank_sum_positive = ranks[len(negatives):].sum()
    u_statistic = rank_sum_positive - len(positives) * (len(positives) + 1) / 2
    return float(u_statistic / (len(positives) * len(negatives)))


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Returns (EER, threshold at EER)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(scores)
    best_gap, eer, eer_threshold = np.inf, 1.0, 0.5
    for threshold in thresholds:
        predictions = scores >= threshold
        false_accept = float((predictions & ~labels).sum() / max((~labels).sum(), 1))
        false_reject = float((~predictions & labels).sum() / max(labels.sum(), 1))
        gap = abs(false_accept - false_reject)
        if gap < best_gap:
            best_gap = gap
            eer = (false_accept + false_reject) / 2
            eer_threshold = float(threshold)
    return eer, eer_threshold


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (probabilities >= edges[i]) & (probabilities < edges[i + 1])
        if not mask.any():
            continue
        confidence = probabilities[mask].mean()
        accuracy = labels[mask].mean()
        ece += (mask.mean()) * abs(confidence - accuracy)
    return float(ece)
