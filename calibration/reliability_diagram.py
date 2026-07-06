"""
Reliability diagrams and confidence histograms for visualizing calibration.

A reliability diagram bins predictions by confidence and plots empirical
accuracy against mean confidence per bin. A perfectly calibrated model
produces bars that sit exactly on the y = x diagonal; bars below the
diagonal indicate over-confidence (the common failure mode in deep
networks), bars above indicate under-confidence.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .metrics import compute_bin_stats, expected_calibration_error


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    ax=None,
):
    """Plot a single reliability diagram (accuracy vs. confidence per bin)."""
    stats = compute_bin_stats(probs, labels, n_bins)
    ece = expected_calibration_error(probs, labels, n_bins)

    bin_centers = (stats.bin_edges[:-1] + stats.bin_edges[1:]) / 2
    width = 1.0 / n_bins

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Perfect calibration reference line.
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

    # Accuracy bars.
    ax.bar(
        bin_centers,
        stats.bin_accuracy,
        width=width,
        edgecolor="black",
        color="#4C72B0",
        alpha=0.85,
        label="Accuracy",
    )

    # Gap between confidence and accuracy, shown as a red overlay.
    gap = stats.bin_confidence - stats.bin_accuracy
    ax.bar(
        bin_centers,
        gap,
        bottom=stats.bin_accuracy,
        width=width,
        edgecolor="black",
        color="#C44E52",
        alpha=0.5,
        label="Confidence gap",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_confidence_histogram(probs: np.ndarray, n_bins: int = 15, title: str = "Confidence Histogram", ax=None):
    """Plot the distribution of the model's top-class confidence values."""
    confidences = probs.max(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.hist(confidences, bins=n_bins, range=(0, 1), color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.axvline(confidences.mean(), color="#C44E52", linestyle="--", label=f"Mean = {confidences.mean():.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_before_after(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: str | None = None,
):
    """Side-by-side reliability diagrams: before vs. after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_reliability_diagram(probs_before, labels, n_bins, title="Before Calibration", ax=axes[0])
    plot_reliability_diagram(probs_after, labels, n_bins, title="After Calibration (Temperature Scaling)", ax=axes[1])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
