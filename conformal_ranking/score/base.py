"""Abstract base class for ranking nonconformity score functions.

Mirrors torchcp.classification.score.base.BaseScore's interface (the
library this project is meant to eventually contribute to), so a
predictor built on top of this can be a near-identical port of
torchcp.classification.predictor.split.SplitPredictor once it has
somewhere to go upstream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseScore(ABC):
    @abstractmethod
    def __call__(self, item_scores: torch.Tensor, item_id: torch.Tensor | None = None) -> torch.Tensor:
        """Computes the nonconformity score(s).

        Args:
            item_scores: (batch, num_candidates) relevance/logit scores a
                ranking model assigned to each candidate item.
            item_id: (batch,) the index, into the candidate dimension, of
                the item the user actually interacted with. If None,
                returns scores for every candidate instead of just the
                true one.

        Returns:
            (batch,) scores if item_id is given, else (batch, num_candidates).
        """
        raise NotImplementedError
