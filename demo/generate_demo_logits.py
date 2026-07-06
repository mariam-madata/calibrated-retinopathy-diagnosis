"""
Generate SYNTHETIC validation/test logits for demonstrating the calibration
framework end-to-end without requiring GPU access or the original APTOS
dataset/checkpoint.

IMPORTANT: These are NOT real model outputs. They are constructed to mimic
the *shape* of miscalibration typically seen in over-confident deep
networks, using the per-class F1 scores and class frequencies reported for
the real 5-class ResNet18 diabetic retinopathy model in
`diabetic-retinopathy-detection/README.md` (No_DR 0.98, Moderate 0.77,
Mild 0.67, Proliferate_DR 0.54, Severe 0.34; overall test accuracy 83%) as
a realistic target, and the APTOS 2019 class distribution as a realistic
class prior. This lets the calibration pipeline (temperature scaling, ECE/
MCE/Brier, reliability diagrams) be demonstrated and unit-tested immediately.

To replace with real numbers: run `scripts/extract_logits_from_model.py`
against your actual trained checkpoint and the APTOS dataset (e.g. on
Colab), then point the demo notebook/scripts at the resulting .npy files
instead of these synthetic ones.
"""

from __future__ import annotations

import numpy as np
import torch

CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]  # alphabetical (ImageFolder order)

# Approximate per-class recall targets, tuned so overall accuracy lands near
# the real model's reported 83% test accuracy, using the reported F1 scores
# as a guide for relative class difficulty.
_CLASS_ACCURACY_TARGET = {
    "No_DR": 0.985,
    "Moderate": 0.80,
    "Mild": 0.70,
    "Proliferate_DR": 0.58,
    "Severe": 0.38,
}

# Approximate APTOS 2019 class prior (rounded), used only to make the
# synthetic sample realistic in composition.
_CLASS_PRIOR = {
    "No_DR": 0.493,
    "Moderate": 0.273,
    "Mild": 0.101,
    "Proliferate_DR": 0.081,
    "Severe": 0.053,
}


def _sample_labels(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = np.array([_CLASS_PRIOR[c] for c in CLASS_NAMES])
    probs = probs / probs.sum()
    return rng.choice(len(CLASS_NAMES), size=n_samples, p=probs)


def _make_logits_for_labels(labels: np.ndarray, seed: int, overconfidence: float = 6.0) -> np.ndarray:
    """Construct logits whose argmax matches the target per-class accuracy,
    and whose magnitude is large enough to produce the over-confident,
    poorly-calibrated softmax outputs typical of an uncalibrated deep net.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    n_classes = len(CLASS_NAMES)
    logits = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_classes))

    for i, true_label in enumerate(labels):
        class_name = CLASS_NAMES[true_label]
        target_acc = _CLASS_ACCURACY_TARGET[class_name]
        is_correct = rng.random() < target_acc
        winning_class = true_label if is_correct else rng.choice(
            [c for c in range(n_classes) if c != true_label]
        )
        # Push the winning class's logit up so the model is confidently
        # correct (or confidently wrong) -- this over-confidence is exactly
        # what temperature scaling is designed to correct.
        logits[i, winning_class] += overconfidence + rng.normal(0, 0.5)

    return logits


def generate_split(n_samples: int, seed: int):
    """Return (logits, labels) as torch tensors for one data split."""
    labels = _sample_labels(n_samples, seed)
    logits = _make_logits_for_labels(labels, seed + 1)
    return torch.tensor(logits, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def generate_demo_dataset(val_size: int = 2000, test_size: int = 2000, seed: int = 42):
    """Generate a val/test split. Sizes are larger than the original 15%/15%
    held-out splits (~549 images each) purely so the reliability-diagram bin
    statistics are stable enough for a clear demonstration; this has no
    bearing on the real model's actual dataset size.
    """
    val_logits, val_labels = generate_split(val_size, seed=seed)
    test_logits, test_labels = generate_split(test_size, seed=seed + 100)
    return {
        "val_logits": val_logits,
        "val_labels": val_labels,
        "test_logits": test_logits,
        "test_labels": test_labels,
        "class_names": CLASS_NAMES,
    }


if __name__ == "__main__":
    import os

    data = generate_demo_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "cached_logits")
    os.makedirs(out_dir, exist_ok=True)
    for key in ["val_logits", "val_labels", "test_logits", "test_labels"]:
        np.save(os.path.join(out_dir, f"{key}.npy"), data[key].numpy())
    print(f"Synthetic demo logits written to {out_dir}/")
    print("Reminder: these are SYNTHETIC, not real model outputs. See module docstring.")
