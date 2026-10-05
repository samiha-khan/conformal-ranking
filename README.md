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

# Real demo on MovieLens-100K (matrix-factorization baseline):
curl -L -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
python examples/movielens_demo.py --data-dir ml-100k

# A stronger model on the same task (SASRec, real chronological sequences):
python examples/sasrec_movielens_demo.py --data-dir ml-100k
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

## A stronger model: SASRec, and a lesson in not trusting the first run

The matrix-factorization baseline above is deliberately simple. The
actual conformal-for-recsys literature this project extends (CPFT, DCR)
builds on SASRec (Kang & McAuley, 2018): self-attention over a user's
real chronological interaction history, not an unordered rating matrix.
`conformal_ranking/sasrec.py` implements it directly from the original
paper's architecture (neither CPFT nor DCR has public code to follow
instead), and `examples/sasrec_movielens_demo.py` reruns the identical
two-stage conformal evaluation on top of it, using MovieLens-100K's real
timestamps to build real per-user sequences and the paper's own
leave-one-out protocol (last interaction held out as the target).

**First run, 15 epochs:** retrieval-miss rate 69% (328/472), statistically
indistinguishable from the matrix-factorization baseline's 68% (157/230).
That result was not trusted. Train loss was still visibly dropping
(13.56 to 6.32 over 15 epochs), which is a sign of an undertrained model,
the same shape of problem the matrix-factorization baseline had in its
own first, broken run (see above). Rather than report "SASRec doesn't
help" from a model that hadn't finished learning, training was extended.

**Second run, 80 epochs** (same data, same evaluation, only the training
budget changed):

```
=== Full catalog (1,682 movies) ===
  Coverage: 88.8%  |  Average list size: 524 of 1,682 movies

=== Two-stage: top-100 retrieval candidates, then calibrated ===
  Coverage: 91.8%  |  Average list size: 84.0 of 100 candidates
  (229 of 472 eval users' true movie wasn't in their own top-100)
```

A real, earned improvement: the retrieval-miss rate drops from 68%
(matrix factorization) to 49% (properly-trained SASRec), and the
full-catalog set size shrinks from 724 to 524 movies for the same 90%
target. Train loss was still (very slowly) decreasing at epoch 80
(5.55 to 5.52 over the last 5 epochs, against a 7.4-point drop over the
first 20), so there's likely a small amount of further headroom with
more training, but returns are clearly diminishing: this is close to
converged for a model this size on a dataset this small, not a case of
"just train it longer and it'll keep improving at the same rate."

The honest takeaway isn't "SASRec is better" on its own; it's that the
*first* number from a model, SASRec or otherwise, is a hypothesis, not
a result, until there's a convergence check behind it. Two different
models produced the same "undertrained-looking" plateau in this
project's history, for different underlying reasons (15 full-batch
steps vs. 15 real epochs that still weren't enough), and both got caught
by the same discipline rather than two different ad-hoc fixes.

## What this isn't

- **Not a reproduction of CPFT or any specific recsys-conformal paper.**
  Checked directly: CPFT's arXiv page lists no code, and a GitHub search
  by exact paper title returns nothing. Could not reproduce code that
  does not exist; this is a separately-motivated application of an
  already-established technique (APS) instead.
- **Not merged into TorchCP (yet).** Built to match their module
  conventions on purpose, as a first step toward a real contribution,
  but no issue was opened or PR submitted as of this commit.
- **Not a tuned, state-of-the-art recommender.** SASRec at 80 epochs is
  a real improvement over the matrix-factorization baseline, not a
  maximally-optimized one: default hyperparameters, no learning-rate
  schedule, no hyperparameter search. The point of this project is the
  calibration layer on top of a recommender's scores, not pushing
  recommender accuracy as far as it could go. Both models are provided
  specifically to make the calibration layer's behavior comparable
  across a weak and a stronger base model, not as a competing claim
  about which recommender architecture is best.

## Project layout

```
conformal_ranking/
  score/base.py, aps.py       the nonconformity score
  predictor/split.py          calibration + prediction-set generation
  sasrec.py                   SASRec (Kang & McAuley, 2018)
  utils.py                    the finite-sample-corrected quantile helper
                               (same formula as torchcp.utils.common,
                               reimplemented so this has no torchcp dependency yet)
tests/
  test_aps_split.py           6 correctness tests on synthetic data
  test_sasrec.py               4 tests, including a causal-masking check
                               that caught a real NaN bug before training
examples/
  movielens_demo.py           matrix-factorization baseline on MovieLens-100K
  sasrec_movielens_demo.py    SASRec on the same task, real chronological
                               sequences, same two-stage evaluation
```

## Status

Core method implemented and tested on synthetic data (10 tests total).
Two real-data demos built, run, and debugged: a matrix-factorization
baseline (caught and fixed an undertrained-model bug) and SASRec (caught
a real NaN bug in causal attention masking, then an undertrained first
training run, before trusting a comparison between the two models).

Not yet started: opening a scoping discussion with TorchCP's maintainers
before proposing this as a PR. A maintainer may already have plans for
this, or want a different API shape than what's built here, and finding
that out before writing more code is cheaper than finding it out after.
