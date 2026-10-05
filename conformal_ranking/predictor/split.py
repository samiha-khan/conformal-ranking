"""Split conformal prediction for ranking/recommendation, structured as
a near-identical port of torchcp.classification.predictor.split.SplitPredictor
(see that file's docstring for why): calibrate a threshold q_hat on a
held-out set of (item_scores, true_item_id) pairs, then at prediction
time include every candidate item whose score is <= q_hat in the
returned set. Under exchangeability between calibration and test data,
the true item lands in that set with probability >= 1 - alpha.
"""
from __future__ import annotations

import torch

from conformal_ranking.score.base import BaseScore
from conformal_ranking.utils import calculate_conformal_value


class SplitRankingPredictor:
    def __init__(self, score_function: BaseScore, alpha: float = 0.1):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        self.score_function = score_function
        self.alpha = alpha
        self.q_hat: torch.Tensor | None = None

    def calibrate(self, cal_item_scores: torch.Tensor, cal_item_ids: torch.Tensor, alpha: float | None = None) -> None:
        """
        Args:
            cal_item_scores: (n_cal, num_candidates)
            cal_item_ids: (n_cal,) index of the true item within each row's candidates
        """
        alpha = self.alpha if alpha is None else alpha
        self.cal_scores = self.score_function(cal_item_scores, cal_item_ids)
        self.q_hat = calculate_conformal_value(self.cal_scores, alpha)

    def predict(self, item_scores: torch.Tensor, q_hat: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            item_scores: (batch, num_candidates)

        Returns:
            (batch, num_candidates) boolean tensor; True where that
            candidate is included in the user's prediction set.
        """
        q_hat = self.q_hat if q_hat is None else q_hat
        if q_hat is None:
            raise ValueError("Call calibrate() first, or pass q_hat explicitly.")
        all_scores = self.score_function(item_scores)
        return all_scores <= q_hat

    def evaluate(self, item_scores: torch.Tensor, item_ids: torch.Tensor) -> dict:
        pred_sets = self.predict(item_scores)
        covered = pred_sets[torch.arange(len(item_ids)), item_ids]
        return {
            "coverage_rate": covered.float().mean().item(),
            "average_set_size": pred_sets.sum(dim=-1).float().mean().item(),
        }
