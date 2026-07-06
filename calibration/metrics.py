"""
Calibration metrics for probabilistic classifiers.

Implements:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Brier score (multi-class, one-vs-rest formulation)
    - Negative log-likelihood (NLL)
    - Per-bin statistics used to build reliability diagrams

References:
    Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    Guo et al. (2017), "On Calibration of Modern Neural Networks." ICML.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinStats:
    bin_edges: np.ndarray          # (n_bins + 1,)
    bin_confidence: np.ndarray     # (n_bins,) mean predicted confidence per bin
    bin_accuracy: np.ndarray       # (n_bins,) empirical accuracy per bin
    bin_count: np.ndarray          # (n_bins,) number of samples per bin


def _get_confidence_and_correctness(probs: np.ndarray, labels: np.ndarray):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)
    return confidences, correctness


def compute_bin_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> BinStats:
    """Bucket predictions into equal-width confidence bins.

    Args:
        probs: (N, C) predicted class probabilities (rows sum to 1).
        labels: (N,) ground-truth integer labels.
        n_bins: number of equal-width bins over [0, 1].
    """
    confidences, correctness = _get_confidence_and_correctness(probs, labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_confidence = np.zeros(n_bins)
    bin_accuracy = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    # Rightmost bin is closed on both ends so confidence == 1.0 is included.
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True), 0, n_bins - 1)

    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        bin_count[b] = count
        if count > 0:
            bin_confidence[b] = confidences[mask].mean()
            bin_accuracy[b] = correctness[mask].mean()

    return BinStats(bin_edges, bin_confidence, bin_accuracy, bin_count)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: weighted average gap between confidence and accuracy across bins."""
    stats = compute_bin_stats(probs, labels, n_bins)
    n = stats.bin_count.sum()
    gaps = np.abs(stats.bin_confidence - stats.bin_accuracy)
    weights = stats.bin_count / n
    return float(np.sum(weights * gaps))


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """MCE: worst-case gap between confidence and accuracy across any bin."""
    stats = compute_bin_stats(probs, labels, n_bins)
    occupied = stats.bin_count > 0
    if not occupied.any():
        return 0.0
    gaps = np.abs(stats.bin_confidence - stats.bin_accuracy)
    return float(np.max(gaps[occupied]))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error between probs and one-hot labels."""
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Mean NLL of the true class probability."""
    true_class_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))


def summarize(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Convenience wrapper returning all metrics + accuracy in one dict."""
    predictions = probs.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    return {
        "accuracy": accuracy,
        "ece": expected_calibration_error(probs, labels, n_bins),
        "mce": maximum_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }
