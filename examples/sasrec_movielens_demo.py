"""A stronger base recommender for the same calibration question
examples/movielens_demo.py asks, using the architecture the
conformal-for-sequential-recsys literature (CPFT, DCR) actually builds
on: SASRec (Kang & McAuley, 2018), not plain matrix factorization.

movielens_demo.py's matrix-factorization model missed 157 of 230
users' true favorite movie in its own top-100 candidates: a retrieval
problem, not a calibration problem (see that script's README section).
This swaps in a real sequential model trained on each user's actual
chronological interaction history (not a random train/test split of
ratings) and reruns the identical two-stage conformal evaluation to see
whether a stronger, more appropriate model closes that gap.

Standard SASRec leave-one-out evaluation protocol: for each user with at
least 3 interactions, sort their ratings by timestamp, hold out the
last one as the prediction target, everything before it is the input
sequence. Trains with next-item prediction at every position in the
sequence (not only the final one), the paper's own training signal.

Run from the conformal-ranking repo root:
    python examples/sasrec_movielens_demo.py --data-dir /path/to/ml-100k
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conformal_ranking.predictor.split import SplitRankingPredictor
from conformal_ranking.sasrec import SASRec
from conformal_ranking.score.aps import RankingAPS

MAX_LEN = 50


def load_sequences(data_dir: Path):
    rows = []
    with open(data_dir / "u.data", encoding="latin-1") as f:
        for line in f:
            u, i, r, t = line.strip().split("\t")
            rows.append((int(u), int(i), int(t)))

    by_user = defaultdict(list)
    for u, i, t in rows:
        by_user[u].append((t, i))
    for u in by_user:
        by_user[u].sort()
        by_user[u] = [i for _, i in by_user[u]]

    n_items = max(i for _, i, _ in rows)
    return by_user, n_items


def build_split(by_user: dict[int, list[int]]):
    """Standard leave-one-out: last item = target, rest = input
    sequence. Users with fewer than 3 interactions are dropped (not
    enough signal for a meaningful train/target split)."""
    train_seqs, targets, user_ids = [], [], []
    for u, items in by_user.items():
        if len(items) < 3:
            continue
        seq = items[:-1][-MAX_LEN:]
        seq = [0] * (MAX_LEN - len(seq)) + seq  # left-pad
        train_seqs.append(seq)
        targets.append(items[-1] - 1)  # 0-indexed into score_candidates' output
        user_ids.append(u)
    return torch.tensor(train_seqs), torch.tensor(targets), user_ids


def train_sasrec(by_user, n_items, epochs=80, device="cpu"):
    model = SASRec(n_items=n_items, max_len=MAX_LEN, dim=64, n_heads=2, n_blocks=2, dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Build (sequence-up-to-t, target-at-t+1) pairs for every position in
    # every user's TRAINING sequence (everything except their held-out
    # last item), not only a single pair per user -- the paper's own
    # "dense" training signal.
    all_seqs, all_targets = [], []
    for u, items in by_user.items():
        if len(items) < 3:
            continue
        history = items[:-1]
        for t in range(1, len(history)):
            seq = history[:t][-MAX_LEN:]
            seq = [0] * (MAX_LEN - len(seq)) + seq
            all_seqs.append(seq)
            all_targets.append(history[t] - 1)

    seqs_t = torch.tensor(all_seqs, device=device)
    targets_t = torch.tensor(all_targets, device=device)
    print(f"  {len(all_seqs)} training (sequence, next-item) pairs from {len(by_user)} users")

    batch_size = 512
    n = seqs_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            scores = model.score_candidates(seqs_t[idx])
            loss = nn.functional.cross_entropy(scores, targets_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}  train loss: {total_loss / n_batches:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    print("Loading real MovieLens-100K as chronological per-user sequences...")
    by_user, n_items = load_sequences(args.data_dir)
    print(f"  {len(by_user)} users, {n_items} movies\n")

    print("Training SASRec (self-attentive sequential recommender)...")
    model = train_sasrec(by_user, n_items)

    print("\nScoring every user's held-out next item...")
    eval_seqs, eval_targets, user_ids = build_split(by_user)
    model.eval()
    with torch.no_grad():
        all_scores = model.score_candidates(eval_seqs)

    half = len(user_ids) // 2
    cal_scores, cal_items = all_scores[:half], eval_targets[:half]
    test_scores, test_items = all_scores[half:], eval_targets[half:]

    target_pct = int((1 - args.alpha) * 100)
    print(f"\nCalibrated on {half} users, evaluated on {len(user_ids) - half} held-out users.\n")

    print(f"=== Full catalog ({n_items} movies) ===")
    full_predictor = SplitRankingPredictor(RankingAPS(randomized=True), alpha=args.alpha)
    full_predictor.calibrate(cal_scores, cal_items)
    full_result = full_predictor.evaluate(test_scores, test_items)
    print(f"  Coverage: {full_result['coverage_rate']:.1%}  |  "
          f"Average list size: {full_result['average_set_size']:.0f} of {n_items} movies")

    print(f"\n=== Two-stage: top-100 retrieval candidates, then calibrated ===")
    K = 100
    cal_topk_scores, cal_topk_items = [], []
    for row in range(len(cal_items)):
        topk_vals, topk_idx = cal_scores[row].topk(K)
        true_item = cal_items[row].item()
        if true_item in topk_idx.tolist():
            cal_topk_scores.append(topk_vals)
            cal_topk_items.append(topk_idx.tolist().index(true_item))
    eval_topk_scores, eval_topk_items, dropped = [], [], 0
    for row in range(len(test_items)):
        topk_vals, topk_idx = test_scores[row].topk(K)
        true_item = test_items[row].item()
        if true_item in topk_idx.tolist():
            eval_topk_scores.append(topk_vals)
            eval_topk_items.append(topk_idx.tolist().index(true_item))
        else:
            dropped += 1

    topk_predictor = SplitRankingPredictor(RankingAPS(randomized=True), alpha=args.alpha)
    topk_predictor.calibrate(torch.stack(cal_topk_scores), torch.tensor(cal_topk_items))
    topk_result = topk_predictor.evaluate(torch.stack(eval_topk_scores), torch.tensor(eval_topk_items))
    print(f"  Coverage: {topk_result['coverage_rate']:.1%}  |  "
          f"Average list size: {topk_result['average_set_size']:.1f} of {K} candidates")
    print(f"  ({dropped} of {len(test_items)} eval users' true movie wasn't in their own top-{K})")

    print(f"\nCompare against examples/movielens_demo.py's matrix-factorization "
          f"baseline: that model missed 157 of 230 users (68%) at top-100. "
          f"This run missed {dropped} of {len(test_items)} ({dropped / len(test_items):.0%}).")


if __name__ == "__main__":
    main()
