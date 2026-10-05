"""Adaptive Prediction Sets (Romano et al., 2020), applied to ranking/
recommendation instead of classification.

TorchCP already implements APS for classification
(torchcp.classification.score.aps.APS): given softmax probabilities over
a fixed, small label set, the nonconformity score of a label is the
cumulative probability mass of all labels ranked at or above it (with a
randomized tie-break within the true label's own probability mass, so
ties don't bias the score deterministically toward the worst case).

The math transfers directly to ranking: a sequential/collaborative-
filtering recommender's predicted relevance scores over a candidate item
set, softmax-normalized, are structurally identical to classification
logits over a label set. "Which item will this user interact with next"
is a multi-class decision over a (possibly huge) candidate set, not
fundamentally different from "which class is this image," so the APS
guarantee (marginal coverage >= 1 - alpha under exchangeability) carries
over without modification. What's new here is the application, not the
method -- this is deliberately NOT a claimed reproduction of any specific
recsys-conformal paper (CPFT, DCR, etc.), since none of those papers had
a public code release to reproduce against (checked directly; see
project README). It's an honest, separately-motivated extension of an
established, already-in-TorchCP technique to a domain TorchCP doesn't
cover yet.
"""
from __future__ import annotations

import torch

from conformal_ranking.score.base import BaseScore


class RankingAPS(BaseScore):
    """
    Reference:
        Romano, Y., Sesia, M., & Candès, E. (2020). Classification with
        Valid and Adaptive Coverage. NeurIPS 2020.
        https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf

    Args:
        randomized: whether to randomize the tie-break within the true
            item's own probability mass (recommended; without it, the
            achieved coverage is conservative -- i.e. higher than
            1 - alpha but at the cost of larger prediction sets).
    """

    def __init__(self, randomized: bool = True):
        self.randomized = randomized

    def __call__(self, item_scores: torch.Tensor, item_id: torch.Tensor | None = None) -> torch.Tensor:
        if item_scores.dim() != 2:
            raise ValueError("item_scores must be 2D: (batch, num_candidates)")
        probs = torch.softmax(item_scores, dim=-1)
        if item_id is None:
            return self._all_items(probs)
        return self._single_item(probs, item_id)

    def _sort_sum(self, probs: torch.Tensor):
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _all_items(self, probs: torch.Tensor) -> torch.Tensor:
        indices, ordered, cumsum = self._sort_sum(probs)
        u = torch.rand(probs.shape, device=probs.device) if self.randomized else torch.zeros_like(probs)
        ordered_scores = cumsum - ordered * u
        _, unsort = torch.sort(indices, dim=-1)
        # Cumulative softmax mass is mathematically bounded by 1; any excess
        # is float32 rounding noise from softmax + cumsum, not a real score
        # above 1. Clamping reflects the true bound instead of leaking the
        # rounding error into downstream quantile/threshold calculations.
        return ordered_scores.gather(dim=-1, index=unsort).clamp(max=1.0)

    def _single_item(self, probs: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        indices, ordered, cumsum = self._sort_sum(probs)
        u = torch.rand(indices.shape[0], device=probs.device) if self.randomized else torch.zeros(indices.shape[0], device=probs.device)
        idx = torch.where(indices == item_id.view(-1, 1))
        return (cumsum[idx] - u * ordered[idx]).clamp(max=1.0)
