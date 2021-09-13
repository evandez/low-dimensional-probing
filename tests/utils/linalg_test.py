"""Unit tests for metrics module."""
from lodimp.utils import linalg

import torch
import torch.linalg


def test_effective_rank():
    """Test effective_rank matches intuition."""
    matrix = torch.diag(torch.tensor([100, 100, 1e-6]))
    actual = linalg.effective_rank(matrix)
    assert torch.allclose(torch.tensor(actual), torch.tensor(2.))


def test_truncate():
    """Test truncate matches intuition."""
    matrix = torch.eye(3)
    actual = linalg.truncate(matrix, 2)
    assert torch.linalg.matrix_rank(actual) == 2


def test_rowspace():
    """Test rowspace returns projection onto rowspace."""
    matrix = torch.tensor([[3., 1., 0.], [0., 2., 0.], [1., 0., 0.]])
    expected = torch.eye(3)
    expected[-1, -1] = 0
    actual = linalg.rowspace(matrix)

    assert actual.allclose(expected, atol=1e-7)
    assert actual.mm(actual).allclose(actual, atol=1e-7)


def test_rowspace_close_to_zero():
    """Test rowspace returns zeros when all elements close to 0."""
    matrix = torch.ones(10, 10) / 1e10
    actual = linalg.rowspace(matrix)
    assert actual.equal(torch.zeros_like(actual))
