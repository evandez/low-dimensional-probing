"""Unit tests for metrics module."""

from lodimp.common import linalg

import torch


def test_effective_rank():
    """Test effective_rank matches intuition."""
    matrix = torch.diag(torch.tensor([100, 100, 1e-6]))
    actual = linalg.effective_rank(matrix)
    assert torch.allclose(torch.tensor(actual), torch.tensor(2.))


def test_truncate():
    """Test truncate matches intuition."""
    matrix = torch.eye(3)
    actual = linalg.truncate(matrix, 2)
    assert torch.matrix_rank(actual) == 2
