"""Unit tests for metrics module."""

from lodimp import linalg

import torch


def test_effective_rank():
    """Test effective_rank matches intuition."""
    matrix = torch.diag(torch.tensor([100, 100, 1e-6]))
    actual = linalg.effective_rank(matrix)
    assert torch.allclose(torch.tensor(actual), torch.tensor(2.))
