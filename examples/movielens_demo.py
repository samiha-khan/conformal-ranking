"""A real, runnable demo on MovieLens-100K.

Plain-English version of what this does: a recommender model scores
every movie for a user. Normally you'd just show the user the single
top-scored movie and hope it's right. This instead asks a different,
more honest question: "how many movies would I need to show this user
to be 90% sure their actual next favorite is somewhere in that list?"
The answer is different for every user, because some users are easy to
predict (the model's confident and the list can be short) and some are
hard (the list has to be longer to hit the same 90% guarantee). That's
the whole point of conformal prediction: instead of one global list
length, you get a *calibrated, per-user* list length with a real,
checkable statistical guarantee behind it, not just a vibe.

Steps:
  1. Train a small matrix-factorization model on MovieLens-100K's
     standard u1.base train split (real ratings, real users, real movies).
  2. Score every movie for every user in the held-out test split.
  3. Split the test users into a calibration half and an evaluation
     half. Calibrate the conformal threshold on the first half.
  4. On the second half, measure: did the user's actual highest-rated
     held-out movie land inside their calibrated prediction set? Do
     this for every user and report the real percentage, which should
     land close to the target (90% here) if everything is working.

Run from the conformal-ranking repo root:
    python examples/movielens_demo.py --data-dir /path/to/ml-100k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conformal_ranking.predictor.split import SplitRankingPredictor
from conformal_ranking.score.aps import RankingAPS


def load_movielens(data_dir: Path):
    def read(path):
        rows = []
        with open(path, encoding="latin-1") as f:
            for line in f:
                u, i, r, _ = line.strip().split("\t")
                rows.append((int(u) - 1, int(i) - 1, float(r)))
        return rows

    train = read(data_dir / "u1.base")
    test = read(data_dir / "u1.test")
    n_users = max(r[0] for r in train + test) + 1
    n_items = max(r[1] for r in train + test) + 1
    return train, test, n_users, n_items


class MatrixFactorization(nn.Module):
    """The simplest real recommender model: a learned vector per user,
    a learned vector per movie, and a predicted rating that's just
    their dot product (plus a bias per user/item). Nothing fancier;
    the point of this demo is the calibration layer on top, not model
    novelty."""

    def __init__(self, n_users: int, n_items: int, dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def forward(self, users, items):
        dot = (self.user_emb(users) * self.item_emb(items)).sum(-1)
        return dot + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)

    def score_all_items(self, users):
        """(len(users), n_items) predicted rating for every item."""
        u = self.user_emb(users)
        ub = self.user_bias(users)
        scores = u @ self.item_emb.weight.T
        scores = scores + ub + self.item_bias.weight.squeeze(-1).unsqueeze(0)
        return scores


def train_model(train_rows, n_users, n_items, epochs=300, device="cpu"):
    """Full-batch gradient descent (simplest to reason about for a demo),
    so `epochs` here really does mean gradient steps over the whole
    80K-row training set, not mini-batch epochs. An earlier version of
    this script ran only 15 such steps and the model never converged
    (train MSE stuck at 11.5 on a 1-5 rating scale, barely better than
    predicting the global mean for everyone) -- caught by the resulting
    demo output making no sense (a 90%-coverage prediction set covering
    1,558 of 1,682 movies, i.e. an undertrained model that can't tell
    movies apart), not by the training loss curve alone, which is why
    the demo prints a real baseline comparison below."""
    model = MatrixFactorization(n_users, n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-5)
    users = torch.tensor([r[0] for r in train_rows], device=device)
    items = torch.tensor([r[1] for r in train_rows], device=device)
    ratings = torch.tensor([r[2] for r in train_rows], device=device)
    global_mean = ratings.mean().item()
    baseline_mse = nn.functional.mse_loss(torch.full_like(ratings, global_mean), ratings).item()

    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(users, items)
        loss = nn.functional.mse_loss(pred, ratings)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  step {epoch + 1}/{epochs}  train MSE: {loss.item():.4f}  "
                  f"(global-mean baseline MSE: {baseline_mse:.4f})")
    final_mse = nn.functional.mse_loss(model(users, items), ratings).item()
    if final_mse >= baseline_mse * 0.9:
        print(f"  WARNING: model barely beats predicting the global mean "
              f"({final_mse:.4f} vs {baseline_mse:.4f} MSE) -- undertrained.")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage rate; 0.1 = target 90% coverage")
    args = parser.parse_args()

    print("Loading real MovieLens-100K ratings...")
    train_rows, test_rows, n_users, n_items = load_movielens(args.data_dir)
    print(f"  {len(train_rows)} train ratings, {len(test_rows)} test ratings, "
          f"{n_users} users, {n_items} movies\n")

    print("Training a small matrix-factorization recommender...")
    model = train_model(train_rows, n_users, n_items)

    # For each test user, their "true next item" is their single
    # highest-rated held-out movie (ties broken by first occurrence).
    # This is the real, standard "next-item" framing used in sequential
    # recommendation evaluation, not an invented simplification.
    best_per_user: dict[int, int] = {}
    best_rating_per_user: dict[int, float] = {}
    for u, i, r in test_rows:
        if u not in best_rating_per_user or r > best_rating_per_user[u]:
            best_rating_per_user[u] = r
            best_per_user[u] = i

    test_users = sorted(best_per_user.keys())
    # Calibration/evaluation split of the test USERS (not ratings), so
    # calibration and evaluation never share a user.
    half = len(test_users) // 2
    cal_users, eval_users = test_users[:half], test_users[half:]

    with torch.no_grad():
        cal_scores = model.score_all_items(torch.tensor(cal_users))
        cal_items = torch.tensor([best_per_user[u] for u in cal_users])
        eval_scores = model.score_all_items(torch.tensor(eval_users))
        eval_items = torch.tensor([best_per_user[u] for u in eval_users])

    target_pct = int((1 - args.alpha) * 100)
    print(f"\nCalibrated on {len(cal_users)} users, evaluated on {len(eval_users)} held-out users.")
    print(f"Target: each user's prediction set should contain their real top-rated "
          f"held-out movie at least {target_pct}% of the time.\n")

    print(f"=== Full catalog ({n_items} movies) ===")
    full_predictor = SplitRankingPredictor(RankingAPS(randomized=True), alpha=args.alpha)
    full_predictor.calibrate(cal_scores, cal_items)
    full_result = full_predictor.evaluate(eval_scores, eval_items)
    print(f"  Coverage: {full_result['coverage_rate']:.1%}  |  "
          f"Average list size: {full_result['average_set_size']:.0f} of {n_items} movies")
    print(f"  Picking one specific favorite movie out of the entire catalog is genuinely "
          f"hard; this is the honest cost of a {target_pct}% guarantee with no narrowing step.\n")

    # Real production recommenders don't calibrate over the whole catalog --
    # a cheap retrieval stage narrows to a candidate shortlist first, and
    # only THAT shortlist gets reranked/calibrated. Mirroring that here:
    # keep only each user's top-K highest-scored movies as the candidate
    # set, recompute nonconformity scores within that smaller set, and
    # recalibrate. This is a different, smaller problem (rank within a
    # shortlist you're fairly confident already contains the answer), not
    # the same problem with a smaller catalog pasted on top.
    print(f"=== Two-stage: top-100 retrieval candidates, then calibrated ===")
    K = 100
    cal_topk_scores, cal_topk_items = [], []
    for row_idx, u in enumerate(cal_users):
        topk_vals, topk_idx = cal_scores[row_idx].topk(K)
        true_item = cal_items[row_idx].item()
        if true_item in topk_idx.tolist():
            local_id = topk_idx.tolist().index(true_item)
            cal_topk_scores.append(topk_vals)
            cal_topk_items.append(local_id)
    eval_topk_scores, eval_topk_items, eval_dropped = [], [], 0
    for row_idx, u in enumerate(eval_users):
        topk_vals, topk_idx = eval_scores[row_idx].topk(K)
        true_item = eval_items[row_idx].item()
        if true_item in topk_idx.tolist():
            local_id = topk_idx.tolist().index(true_item)
            eval_topk_scores.append(topk_vals)
            eval_topk_items.append(local_id)
        else:
            eval_dropped += 1

    if cal_topk_scores and eval_topk_scores:
        topk_predictor = SplitRankingPredictor(RankingAPS(randomized=True), alpha=args.alpha)
        topk_predictor.calibrate(torch.stack(cal_topk_scores), torch.tensor(cal_topk_items))
        topk_result = topk_predictor.evaluate(torch.stack(eval_topk_scores), torch.tensor(eval_topk_items))
        print(f"  Coverage: {topk_result['coverage_rate']:.1%}  |  "
              f"Average list size: {topk_result['average_set_size']:.1f} of {K} candidates")
        print(f"  ({eval_dropped} of {len(eval_users)} eval users' true movie wasn't in their own "
              f"top-{K}, i.e. the retrieval stage itself missed -- excluded from this "
              f"stage's numbers since conformal calibration can't fix a retrieval miss, "
              f"only honestly report on what retrieval actually handed it)")
    else:
        print("  Not enough users had their true item inside the top-K candidates to report this.")


if __name__ == "__main__":
    main()
