"""Defines various performance metrics."""

import torch
from torch import distributions


def effective_rank(matrix: torch.Tensor) -> float:
    """Return the effective rank of the matrix.

    Effective rank is intuitively the expected dimensionality of the range of
    the matrix. See "THE EFFECTIVE RANK: A MEASURE OF EFFECTIVE DIMENSIONALITY"
    for detailed treatment.

    Args:
        matrix (torch.Tensor): Size (M, N) matrix for which to compute
            effective rank.

    Returns:
        float: The effective rank.

    """
    if len(matrix.shape) != 2:
        raise ValueError(f'expected 2D matrix, got shape {matrix.shape}')
    _, s, _ = torch.svd(matrix, compute_uv=False)
    return torch.exp(distributions.Categorical(logits=s).entropy()).item()


def truncate(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Truncate the rank of the matrix by zeroing out singular values.

    Args:
        matrix (torch.Tensor): The matrix to truncate.
        rank (int): The rank to truncate the matrix to.

    Returns:
        torch.Tensor: The truncated matrix.

    """
    if rank <= 0:
        raise ValueError(f'rank must be positive, got {rank}')
    if len(matrix.shape) != 2:
        raise ValueError(f'cannot truncate tensor of shape {matrix.shape}')
    original = min(matrix.shape)
    if original < rank:
        raise ValueError(f'matrix rank {original} < target rank {rank}')

    u, s, v = torch.svd(matrix)
    s[rank:] = 0
    return u.mm(torch.diag(s)).mm(v.t())
