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
