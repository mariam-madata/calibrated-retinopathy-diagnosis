"""
Unit tests for the calibration framework. Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import (
    TemperatureScaler,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    negative_log_likelihood,
)


def test_ece_perfect_calibration_is_zero():
    # Construct probabilities exactly equal to empirical accuracy per bin
    # by making every prediction correct with the same confidence as accuracy.
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    probs = np.zeros((10, 2))
    probs[np.arange(10), labels] = 1.0  # fully confident, always correct
    ece = expected_calibration_error(probs, labels)
    assert ece < 1e-8


def test_ece_detects_overconfidence():
    # Model is 100% confident but only 50% correct -> large ECE.
    labels = np.array([0, 1, 0, 1])
    probs = np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01]])
    ece = expected_calibration_error(probs, labels)
    assert ece > 0.4  # confidence ~0.99, accuracy 0.5 -> gap ~0.49


def test_mce_at_least_as_large_as_ece():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=200)
    probs = rng.dirichlet(alpha=[1, 1, 1], size=200)
    ece = expected_calibration_error(probs, labels)
    mce = maximum_calibration_error(probs, labels)
    assert mce >= ece - 1e-9


def test_brier_score_zero_for_perfect_confident_correct_predictions():
    labels = np.array([0, 1, 2])
    probs = np.eye(3)
    assert brier_score(probs, labels) < 1e-8


def test_brier_score_positive_for_wrong_predictions():
    labels = np.array([0, 1])
    probs = np.array([[0.0, 1.0], [1.0, 0.0]])  # both predictions wrong & confident
    score = brier_score(probs, labels)
    assert score > 1.9  # max possible per-sample squared error is 2.0


def test_nll_lower_for_better_calibrated_confident_correct_predictions():
    labels = np.array([0, 0, 0])
    probs_good = np.array([[0.9, 0.1]] * 3)
    probs_bad = np.array([[0.5, 0.5]] * 3)
    assert negative_log_likelihood(probs_good, labels) < negative_log_likelihood(probs_bad, labels)


def test_temperature_scaling_does_not_change_argmax():
    torch.manual_seed(0)
    logits = torch.randn(50, 5) * 3  # exaggerated, over-confident logits
    labels = torch.randint(0, 5, (50,))

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    original_preds = logits.argmax(dim=1)
    calibrated_probs = scaler.calibrate(logits)
    calibrated_preds = calibrated_probs.argmax(dim=1)

    assert torch.equal(original_preds, calibrated_preds)


def test_temperature_scaling_reduces_ece_on_overconfident_logits():
    torch.manual_seed(1)
    n = 300
    labels = torch.randint(0, 5, (n,))
    # Build over-confident logits: correct class gets a large boost only
    # ~60% of the time, but the magnitude is large enough to always look
    # very confident either way.
    logits = torch.randn(n, 5)
    for i in range(n):
        if torch.rand(1).item() < 0.6:
            logits[i, labels[i]] += 6.0
        else:
            wrong = (labels[i] + 1) % 5
            logits[i, wrong] += 6.0

    probs_before = torch.softmax(logits, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, labels.numpy())

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    probs_after = scaler.calibrate(logits).numpy()
    ece_after = expected_calibration_error(probs_after, labels.numpy())

    assert ece_after < ece_before
    assert scaler.temperature.item() > 1.0  # should learn to "soften" over-confidence
