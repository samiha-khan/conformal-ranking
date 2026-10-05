"""Correctness checks on the SASRec implementation, before training it
on anything real. The one that matters most: causality. A self-attention
bug that lets position t see position t+1's item would make next-item
prediction trivially easy during training (the model just copies the
answer) and catastrophically wrong at real inference time (the future
position doesn't exist yet). This kind of bug produces a model that
LOOKS like it's training well (loss drops fast) right up until it's
evaluated on real sequential predictions, which is exactly the kind of
silent failure worth a dedicated test for.
"""
from __future__ import annotations

import torch

from conformal_ranking.sasrec import SASRec


def test_output_shape():
    model = SASRec(n_items=50, max_len=10, dim=16, n_heads=2, n_blocks=1)
    seqs = torch.randint(1, 51, (4, 10))
    out = model(seqs)
    assert out.shape == (4, 10, 16)


def test_score_candidates_shape_excludes_padding_item():
    model = SASRec(n_items=50, max_len=10, dim=16, n_heads=2, n_blocks=1)
    seqs = torch.randint(1, 51, (4, 10))
    scores = model.score_candidates(seqs)
    assert scores.shape == (4, 50)  # not 51: padding item (id 0) excluded


def test_causal_masking_an_earlier_positions_output_is_unaffected_by_later_items():
    """The real correctness property: position t's hidden state must be
    identical regardless of what item is placed at position t+1 or later.
    """
    model = SASRec(n_items=50, max_len=10, dim=16, n_heads=2, n_blocks=2, dropout=0.0)
    model.eval()

    base = torch.randint(1, 51, (1, 10))
    modified = base.clone()
    modified[0, -1] = (base[0, -1] % 50) + 1  # change only the last (future) position

    with torch.no_grad():
        out_base = model(base)
        out_modified = model(modified)

    # Every position except the one we actually changed must be identical.
    torch.testing.assert_close(out_base[:, :-1, :], out_modified[:, :-1, :])
    # Sanity check the test itself isn't vacuous: the changed position's
    # own output (which legitimately depends on its own item embedding)
    # really should differ.
    assert not torch.allclose(out_base[:, -1, :], out_modified[:, -1, :])


def test_padding_positions_produce_zero_hidden_state():
    model = SASRec(n_items=50, max_len=6, dim=8, n_heads=2, n_blocks=1, dropout=0.0)
    model.eval()
    seqs = torch.tensor([[0, 0, 0, 5, 12, 3]])  # left-padded, 3 real items
    with torch.no_grad():
        out = model(seqs)
    torch.testing.assert_close(out[:, :3, :], torch.zeros(1, 3, 8))
    assert not torch.allclose(out[:, 3:, :], torch.zeros(1, 3, 8))
