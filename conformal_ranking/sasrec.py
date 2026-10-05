"""A compact SASRec (Kang & McAuley, 2018, "Self-Attentive Sequential
Recommendation") implementation: the standard architecture the
conformal-for-recsys literature this project is extending (CPFT, DCR)
actually builds on, not invented for this project. Neither paper has
public code (see README), so this follows the original SASRec paper's
own published architecture and evaluation protocol directly, not a
guess at what those two papers might have done internally.

Architecture: item embeddings + learned positional embeddings, causal
(left-to-right only) multi-head self-attention blocks, a point-wise
feed-forward layer per block, LayerNorm and residual connections around
each sub-layer, same as the paper. Predicts the next item at every
position in a user's interaction sequence (not just the final one),
which is the paper's own training signal: more supervision per sequence
than predicting only the single last item.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, n_items: int, max_len: int = 50, dim: int = 64,
                 n_heads: int = 2, n_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        # item id 0 is reserved as the padding token
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.max_len = max_len

        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        for _ in range(n_blocks):
            self.attn_norms.append(nn.LayerNorm(dim))
            self.attn_layers.append(nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True))
            self.ffn_norms.append(nn.LayerNorm(dim))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim),
            ))
        self.last_norm = nn.LayerNorm(dim)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seqs: (batch, seq_len) item ids, 0 = padding, left-padded
                (earliest interaction first, most recent last).

        Returns:
            (batch, seq_len, dim) a hidden state per position, position
            t's hidden state is this model's prediction of item t+1
            given everything up to and including position t (causal).
        """
        batch, seq_len = seqs.shape
        positions = torch.arange(seq_len, device=seqs.device).unsqueeze(0).expand(batch, -1)
        x = self.item_emb(seqs) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        padding_mask = seqs == 0  # (batch, seq_len), True where padded
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seqs.device, dtype=torch.bool), diagonal=1)

        for attn, attn_norm, ffn, ffn_norm in zip(
            self.attn_layers, self.attn_norms, self.ffn_layers, self.ffn_norms
        ):
            normed = attn_norm(x)
            attn_out, _ = attn(normed, normed, normed, attn_mask=causal_mask,
                                key_padding_mask=padding_mask, need_weights=False)
            # A padding query position whose only causally-visible key is
            # itself (also padding) has an all -inf attention row, which
            # softmaxes to NaN, not zero. Those rows belong to padded
            # positions that get explicitly zeroed out below anyway, so
            # replacing NaN with 0 here is recovering the intended value,
            # not papering over a real numerical problem.
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
            x = x + attn_out
            x = x + ffn(ffn_norm(x))

        x = self.last_norm(x)
        x = x * (~padding_mask).unsqueeze(-1)  # zero out padded positions
        return x

    def score_candidates(self, seqs: torch.Tensor) -> torch.Tensor:
        """Score every real item (excluding the padding item 0) for the
        next position after each sequence. Returns (batch, n_items), the
        n_items dimension indexed 0..n_items-1 (item id - 1, since item
        id 0 is padding and item embeddings are 1-indexed internally)."""
        hidden = self.forward(seqs)
        last_hidden = hidden[:, -1, :]  # (batch, dim), prediction for the position after the sequence
        all_item_embs = self.item_emb.weight[1:]  # drop the padding embedding
        return last_hidden @ all_item_embs.T
