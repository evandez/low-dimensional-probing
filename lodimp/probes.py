"""Defines probe architectures."""

from typing import Optional

import torch
from torch import nn


class Bilinear(nn.Module):
    """Bilinear classifier in a low-dimensional subspace.

    This classifier measures the pairwise "compatibility" between
    input vectors.
    """

    def __init__(self, dimension: int):
        """Initialize the architecture.

        Args:
            dimension (int): Dimensionality of input vectors.

        """
        super().__init__()
        self.dimension = dimension
        self.compat = nn.Bilinear(dimension, dimension, 1)
        # Softmax is implicit in loss functions.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute pairwise compatibility between representations.

        Args:
            inputs (torch.Tensor): Matrix of representations.
                Must have shape (N, dimension), where N is the
                number of representations to compare.

        Returns:
            torch.Tensor: Size (N, N) matrix representing pairwise
                compatibilities.

        Raises:
            ValueError: If input is misshapen.

        """
        dims = len(inputs.shape)
        if dims != 2:
            raise ValueError(f'expected 2D tensor, got {dims}D')

        length, dimension = inputs.shape
        if dimension != self.dimension:
            raise ValueError(
                f'expected dimension {self.dimension}, got {dimension}')

        # For performance, we avoid copying data and construct two views
        # of the projected representations so that the entire bilinear
        # operation happens as one big matrix multiplication.
        left = inputs.repeat(1, length).view(length**2, self.dimension)
        right = inputs.repeat(length, 1)
        return self.compat(left, right).view(length, length)


class MLP(nn.Sequential):
    """MLP probe in a low-dimensional subspace of the representation."""

    def __init__(self,
                 input_dimension: int,
                 classes: int,
                 hidden_dimension: Optional[int] = None):
        """Initilize the architecture.

        Args:
            input_dimension (int): Dimensionality of input vectors.
            classes (int): Number of classes to predict for each vector.
            hidden_dimension (int, optional): Dimensionality of MLP hidden
                layer. Defaults to projected_dimension.

        """
        hidden_dimension = hidden_dimension or input_dimension
        super().__init__(nn.Linear(input_dimension, hidden_dimension),
                         nn.ReLU(), nn.Linear(hidden_dimension, classes))
