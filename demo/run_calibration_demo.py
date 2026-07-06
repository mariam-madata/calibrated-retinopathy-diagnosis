"""
End-to-end calibration demo:
    1. Load logits (synthetic by default; swap in real logits from
       scripts/extract_logits_from_model.py when available)
    2. Compute pre-calibration metrics (accuracy, ECE, MCE, Brier, NLL)
    3. Fit temperature scaling on the validation split
    4. Compute post-calibration metrics on the held-out test split
    5. Save before/after reliability diagrams and a results table

Run with:
    python demo/run_calibration_demo.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import TemperatureScaler, summarize, plot_before_after  # noqa: E402
from demo.generate_demo_logits import generate_demo_dataset  # noqa: E402


def main(use_cached: bool = False):
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_logits")
    if use_cached and os.path.exists(cache_dir):
        val_logits = torch.tensor(np.load(os.path.join(cache_dir, "val_logits.npy")), dtype=torch.float32)
        val_labels = torch.tensor(np.load(os.path.join(cache_dir, "val_labels.npy")), dtype=torch.long)
        test_logits = torch.tensor(np.load(os.path.join(cache_dir, "test_logits.npy")), dtype=torch.float32)
        test_labels = torch.tensor(np.load(os.path.join(cache_dir, "test_labels.npy")), dtype=torch.long)
        source = "cached_logits/ (replace with real extract_logits_from_model.py output when available)"
    else:
        data = generate_demo_dataset()
        val_logits, val_labels = data["val_logits"], data["val_labels"]
        test_logits, test_labels = data["test_logits"], data["test_labels"]
        source = "SYNTHETIC demo logits (see demo/generate_demo_logits.py)"

    print(f"Logit source: {source}\n")

    # --- Pre-calibration ---
    probs_before = F.softmax(test_logits, dim=1).numpy()
    metrics_before = summarize(probs_before, test_labels.numpy())

    # --- Fit temperature scaling on validation split only ---
    scaler = TemperatureScaler()
    final_nll = scaler.fit(val_logits, val_labels)
    learned_T = scaler.temperature.item()

    # --- Post-calibration, evaluated on held-out test split ---
    probs_after = scaler.calibrate(test_logits).numpy()
    metrics_after = summarize(probs_after, test_labels.numpy())

    print(f"Learned temperature T = {learned_T:.4f} (validation NLL at convergence: {final_nll:.4f})\n")
    print(f"{'Metric':<12}{'Before':>12}{'After':>12}")
    for key in ["accuracy", "ece", "mce", "brier", "nll"]:
        print(f"{key:<12}{metrics_before[key]:>12.4f}{metrics_after[key]:>12.4f}")

    # --- Save reliability diagrams ---
    fig_path = os.path.join(results_dir, "reliability_before_after.png")
    plot_before_after(probs_before, probs_after, test_labels.numpy(), save_path=fig_path)
    print(f"\nSaved reliability diagrams to {fig_path}")

    # --- Save results table ---
    results = {
        "logit_source": source,
        "learned_temperature": learned_T,
        "before": metrics_before,
        "after": metrics_after,
    }
    results_path = os.path.join(results_dir, "calibration_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results table to {results_path}")


if __name__ == "__main__":
    main()
