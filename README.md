# conformal-ranking

**The question this answers:** a recommender system scores every item for
a user. Normally you just show the single top-scored item and hope it's
right. This instead asks: "how many items would I need to show this user
to be 90% sure their actual favorite is somewhere in that list?" The
answer is different for every user: some are easy to predict and the
list can be short, some are hard and the list has to be longer. This
gives you that per-user list length with a real, checkable statistical
guarantee behind it, not a guess.

The technique is [conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction):
well established for classification and regression, with mature,
widely-used libraries ([MAPIE](https://github.com/scikit-learn-contrib/MAPIE),
[TorchCP](https://github.com/ml-stat-Sustech/TorchCP)). Applying it to
ranking/recommendation specifically is much less settled: the papers
that do it ([CPFT](https://arxiv.org/abs/2402.08976), a DCR paper) have
no public code release anywhere (checked directly against both papers
and GitHub, see "What this isn't" below). This project doesn't
reproduce either of them. It extends a technique TorchCP already has for
classification, [Adaptive Prediction Sets](https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf)
(Romano et al., 2020), to ranking instead, since the math transfers
directly: "which item will this user pick" is structurally the same
decision as "which class is this image," just over a much bigger set of
candidates.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install torch pytest

# Correctness tests on synthetic data (no download needed):
PYTHONPATH=. pytest tests/ -v

# Real demo on MovieLens-100K:
curl -L -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
python examples/movielens_demo.py --data-dir ml-100k
```

## What it actually does

Three pieces, each a near-direct port of TorchCP's own classification
module structure (`score/`, `predictor/`), so a maintainer reviewing
this already knows the shape of it:

- **`conformal_ranking/score/aps.py`**: the nonconformity score. Sorts
  a user's candidate items by predicted relevance, takes the cumulative
  probability mass, and scores the true item by how much mass sits at
  or above it (with a randomized tie-break, same as TorchCP's
  classification APS, so ties don't bias the result toward the
  worst case).
- **`conformal_ranking/predictor/split.py`**: split conformal
  prediction. Calibrates a threshold on a held-out set of
  (item scores, true item) pairs, then at prediction time includes every
  candidate whose score falls at or under that threshold.
- **`tests/test_aps_split.py`**: 6 tests on synthetic data, including
  one that deliberately breaks the method (shifts test-time scores so
  calibration and test data are no longer exchangeable) and checks that
  coverage correctly degrades. That's the same failure mode
  `perception-validation-suite` found on real GTSRB data; reproducing it
  on purpose here confirms the implementation reflects a real limitation
  of split conformal prediction, not a bug.

## The MovieLens demo, and what it actually found

`examples/movielens_demo.py` trains a small matrix-factorization
recommender on MovieLens-100K's standard `u1.base`/`u1.test` split (real
ratings, 943 users, 1,682 movies), then calibrates and evaluates two
ways:

```
=== Full catalog (1,682 movies) ===
  Coverage: 87.8%  |  Average list size: 724 of 1,682 movies

=== Two-stage: top-100 retrieval candidates, then calibrated ===
  Coverage: 95.9%  |  Average list size: 90.7 of 100 candidates
  (157 of 230 eval users' true movie wasn't in their own top-100)
```

An earlier run of this script reported a 90%-coverage set covering
1,558 of 1,682 movies, which is a sign of an undertrained model (15
full-batch gradient steps, barely beating a predict-the-global-mean
baseline), not a real result. Caught by the output not making sense,
fixed by training the model properly (300 steps, real convergence,
train MSE 0.13 vs. a 1.25 global-mean baseline) before trusting any
number from it.

The real finding, after fixing that: **calibrating over the whole
1,682-movie catalog needs an uncomfortably large list (724 movies) to
hit 90% confidence**, because pinning down one specific user's single
highest-rated movie among that many real candidates is genuinely hard.
Restricting to a 100-candidate shortlist first (mirroring how real
production recommenders actually work: cheap retrieval, then a
calibrated rerank, not one calibration pass over the entire catalog)
gets a much more usable set size (91 of 100), but reveals something
more important: for 157 of 230 users, their actual favorite movie
wasn't even in their own top-100 scored candidates. The conformal layer
correctly hit its coverage target on the users retrieval *did* capture
(95.9% vs. a 90% target). **The matrix-factorization baseline itself is
the bottleneck here, not the calibration method.** A stronger retrieval
model would narrow this gap; this project's job was to find and report
that honestly, not to paper over it with a better model it didn't build.

## What this isn't

- **Not a reproduction of CPFT or any specific recsys-conformal paper.**
  Checked directly: CPFT's arXiv page lists no code, and a GitHub search
  by exact paper title returns nothing. Could not reproduce code that
  does not exist; this is a separately-motivated application of an
  already-established technique (APS) instead.
- **Not merged into TorchCP (yet).** Built to match their module
  conventions on purpose, as a first step toward a real contribution,
  but no issue was opened or PR submitted as of this commit.
- **Not a strong recommender.** The matrix-factorization model here is
  deliberately simple (a learned vector per user and movie, dot
  product, nothing fancier). The point of this project is the
  calibration layer on top of a recommender's scores, not the
  recommender itself. The retrieval-miss finding above is really a
  property of this simple baseline, not of the calibration method.

## Project layout

```
conformal_ranking/
  score/base.py, aps.py       the nonconformity score
  predictor/split.py          calibration + prediction-set generation
  utils.py                    the finite-sample-corrected quantile helper
                               (same formula as torchcp.utils.common,
                               reimplemented so this has no torchcp dependency yet)
tests/test_aps_split.py       6 correctness tests on synthetic data
examples/movielens_demo.py    real demo on MovieLens-100K
```

## Status

Core method implemented and tested on synthetic data. Real-data demo
built, run, and debugged (caught and fixed an undertrained-model bug
before trusting its output). Not yet started: opening a scoping
discussion with TorchCP's maintainers before proposing this as a PR.
A maintainer may already have plans for this, or want a different API
shape than what's built here, and finding that out before writing more
code is cheaper than finding it out after.
