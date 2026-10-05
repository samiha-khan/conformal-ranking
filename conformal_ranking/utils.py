"""Shared helper, identical to torchcp.utils.common.calculate_conformal_value
(reimplemented, not imported, so this package has no dependency on torchcp
existing being installed -- useful during development, and avoidable
later if/when this is folded into torchcp directly).
"""
from __future__ import annotations

import math
import warnings

import torch


def calculate_conformal_value(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    n = scores.shape[0]
    if n == 0:
        warnings.warn("No calibration scores; threshold set to +inf.")
        return torch.tensor(float("inf"))
    quantile_level = math.ceil((n + 1) * (1 - alpha)) / n
    if quantile_level > 1:
        warnings.warn("Quantile level exceeds 1 (too few calibration points for this alpha); threshold set to +inf.")
        return torch.tensor(float("inf"))
    return torch.kthvalue(scores, math.ceil((n + 1) * (1 - alpha)), dim=0).values.to(scores.device)
