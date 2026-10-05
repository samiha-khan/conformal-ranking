"""Correctness checks on synthetic data, before touching anything real.

Same discipline as calibration/conformal.py's own test suite in
perception-validation-suite: verify the marginal coverage guarantee
actually holds empirically under exchangeable data before trusting the
implementation on anything that matters.
"""
from __future__ import annotations

import torch

from conformal_ranking.predictor.split import SplitRankingPredictor
from conformal_ranking.score.aps import RankingAPS


def _synthetic_scores(n: int, num_candidates: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(n, num_candidates, generator=g)
    item_ids = torch.randint(0, num_candidates, (n,), generator=g)
    return scores, item_ids


def test_single_item_score_is_between_zero_and_one():
    scores, item_ids = _synthetic_scores(200, 10, seed=0)
    aps = RankingAPS(randomized=False)
    s = aps(scores, item_ids)
    assert torch.all(s >= 0) and torch.all(s <= 1)


def test_all_items_score_matches_single_item_score_for_the_true_item():
    scores, item_ids = _synthetic_scores(50, 8, seed=1)
    aps = RankingAPS(randomized=False)
    single = aps(scores, item_ids)
    all_scores = aps(scores)
    picked = all_scores[torch.arange(len(item_ids)), item_ids]
    torch.testing.assert_close(single, picked)


def test_highest_scored_item_has_the_smallest_nonconformity_score():
    scores, _ = _synthetic_scores(100, 6, seed=2)
    aps = RankingAPS(randomized=False)
    all_scores = aps(scores)
    best_item = scores.argmax(dim=-1)
    smallest_score_item = all_scores.argmin(dim=-1)
    torch.testing.assert_close(best_item, smallest_score_item)


def test_empirical_coverage_matches_target_on_exchangeable_data():
    """The actual guarantee this whole method rests on: calibrate on one
    i.i.d. draw, evaluate on a fresh i.i.d. draw from the SAME
    distribution, coverage should land at or above 1 - alpha (allowing
    for finite-sample noise, same tolerance used in
    perception-validation-suite's analogous classification test)."""
    num_candidates = 20
    alpha = 0.1
    cal_scores, cal_ids = _synthetic_scores(4000, num_candidates, seed=10)
    test_scores, test_ids = _synthetic_scores(4000, num_candidates, seed=11)

    predictor = SplitRankingPredictor(RankingAPS(randomized=True), alpha=alpha)
    predictor.calibrate(cal_scores, cal_ids)
    result = predictor.evaluate(test_scores, test_ids)

    assert result["coverage_rate"] >= (1 - alpha) - 0.03
    assert 0 < result["average_set_size"] <= num_candidates


def test_lower_alpha_gives_larger_or_equal_prediction_sets():
    """Demanding higher coverage (lower alpha) should never produce
    smaller prediction sets on the same data."""
    scores, ids = _synthetic_scores(2000, 15, seed=3)

    loose = SplitRankingPredictor(RankingAPS(randomized=False), alpha=0.3)
    loose.calibrate(scores, ids)
    tight = SplitRankingPredictor(RankingAPS(randomized=False), alpha=0.05)
    tight.calibrate(scores, ids)

    loose_size = loose.predict(scores).sum(dim=-1).float().mean()
    tight_size = tight.predict(scores).sum(dim=-1).float().mean()
    assert loose_size <= tight_size


def test_non_exchangeable_shift_breaks_the_coverage_guarantee():
    """Mirrors the exact failure mode perception-validation-suite found
    on real data tonight: if test data is NOT exchangeable with
    calibration data (here, a deliberate score shift), the marginal
    coverage guarantee can fail. This is the implementation correctly
    reflecting a real, known limitation of split conformal prediction,
    not a bug -- the method assumes exchangeability, and when that
    assumption breaks, so does the guarantee."""
    num_candidates = 20
    alpha = 0.1
    cal_scores, cal_ids = _synthetic_scores(4000, num_candidates, seed=20)
    test_scores, test_ids = _synthetic_scores(4000, num_candidates, seed=21)
    # Shift test scores so the true item systematically scores lower
    # relative to distractors than it did during calibration.
    test_scores[torch.arange(len(test_ids)), test_ids] -= 3.0

    predictor = SplitRankingPredictor(RankingAPS(randomized=True), alpha=alpha)
    predictor.calibrate(cal_scores, cal_ids)
    result = predictor.evaluate(test_scores, test_ids)

    assert result["coverage_rate"] < (1 - alpha) - 0.03
