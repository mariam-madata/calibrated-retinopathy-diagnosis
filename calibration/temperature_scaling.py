"""
Temperature scaling for post-hoc calibration of a trained classifier.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    "On Calibration of Modern Neural Networks." ICML.

Temperature scaling learns a single scalar T > 0 that rescales a model's
logits (z -> z / T) before the softmax. It does not change the model's
argmax predictions (and therefore not its accuracy) -- it only reshapes
the confidence distribution so that predicted probabilities better match
empirical accuracy. T is fit by minimizing negative log-likelihood (NLL)
on a held-out validation set, with the underlying classifier's weights
kept completely frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Wraps a trained classifier and learns a single temperature parameter.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)          # learn T on validation set
        calibrated_probs = scaler.calibrate(test_logits)
        T = scaler.temperature.item()
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        # Stored in log-space internally is unnecessary here; T is constrained
        # to be positive via a clamp during optimization instead, which keeps
        # the parameter directly interpretable.
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits (same shape as input)."""
        return logits / self._safe_temperature()

    def _safe_temperature(self) -> torch.Tensor:
        # Guard against T collapsing to <= 0 during optimization.
        return torch.clamp(self.temperature, min=1e-3)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit the temperature parameter on validation logits/labels using LBFGS.

        Args:
            val_logits: (N, C) raw, un-normalized logits from the frozen model
                on a held-out validation split (never seen during model training).
            val_labels: (N,) ground-truth integer class labels.
            lr: learning rate for the LBFGS optimizer.
            max_iter: maximum LBFGS iterations.

        Returns:
            The final NLL loss value after fitting.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)
        return float(final_loss.detach())

    @torch.no_grad()
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated class probabilities (softmax of scaled logits)."""
        return F.softmax(self.forward(logits), dim=1)
