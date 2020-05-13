"""Defines probe architectures."""

from typing import Optional

import torch
from torch import nn


class Projection(nn.Linear):
    """Defines a linear projection."""


class ProjectedLinear(nn.Module):
    """Linear probe in a low-dimensional subspace of representation."""

    def __init__(self, input_dimension: int, projected_dimension: int,
                 classes: int):
        """Initilize the architecture.

        Args:
            input_dimension (int): Dimensionality of input vectors.
            projected_dimension (int): Dimensionality of projected space.
            classes (int): Number of classes to predict for each vector.

        """
        super().__init__()

        self.input_dimension = input_dimension
        self.projected_dimension = projected_dimension
        self.classes = classes

        self.project = Projection(input_dimension, projected_dimension)
        self.classify = nn.Linear(projected_dimension, classes)
        # Softmax is implicit in loss functions.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the inputs to low dimension and predict labels for them.

        Args:
            inputs (torch.Tensor): Shape (batch_size, input_dimension) tensor.

        Returns:
            torch.Tensor: Shape (batch_size, classes) tensor containing class
                logits for each sample in batch.

        Raises:
            ValueError: If input is misshapen.

        """
        if len(inputs.shape) != 2:
            raise ValueError(f'expected 2D tensor, got {len(inputs.shape)}D')

        if inputs.shape[-1] != self.input_dimension:
            raise ValueError(f'expected dimension {self.input_dimension}, '
                             f'got {inputs.shape[-1]}')

        return self.classify(self.project(inputs))


class ProjectedBilinear(nn.Module):
    """Bilinear classifier in a low-dimensional subspace.

    This classifier measures the pairwise "compatibility" between
    input vectors.
    """

    def __init__(self, input_dimension: int, projected_dimension: int):
        """Initialize the architecture.

        Args:
            input_dimension (int): Dimensionality of input vectors.
            projected_dimension (int): Dimensionality of projected space.

        """
        super().__init__()

        self.input_dimension = input_dimension
        self.projected_dimension = projected_dimension

        self.project = Projection(input_dimension, projected_dimension)
        self.compat = nn.Bilinear(projected_dimension, projected_dimension, 1)
        # Softmax is implicit in loss functions.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute pairwise compatibility between representations.

        Args:
            inputs (torch.Tensor): Matrix of representations.
                Must have shape (N, input_dimension), where N is the
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
        if dimension != self.input_dimension:
            raise ValueError(
                f'expected dimension {self.input_dimension}, got {dimension}')

        projected = self.project(inputs)

        # For performance, we avoid copying data and construct two views
        # of the projected representations so that the entire bilinear
        # operation happens as one big matrix multiplication.
        left = projected.repeat(1, length).view(length**2,
                                                self.projected_dimension)
        right = projected.repeat(length, 1)
        return self.compat(left, right).view(length, length)


class ProjectedMLP(nn.Module):
    """MLP probe in a low-dimensional subspace of the representation."""

    def __init__(self,
                 input_dimension: int,
                 projected_dimension: int,
                 classes: int,
                 hidden_dimension: Optional[int] = None):
        """Initilize the architecture.

        Args:
            input_dimension (int): Dimensionality of input vectors.
            projected_dimension (int): Dimensionality of projected space.
            classes (int): Number of classes to predict for each vector.
            hidden_dimension (int, optional): Dimensionality of MLP hidden
                layer. Defaults to projected_dimension.

        """
        super().__init__()

        self.input_dimension = input_dimension
        self.projected_dimension = projected_dimension
        self.classes = classes
        self.hidden_dimension = hidden_dimension or projected_dimension

        self.project = Projection(input_dimension, projected_dimension)
        self.classify = nn.Sequential(
            nn.Linear(projected_dimension, self.hidden_dimension),
            nn.ReLU(),
            nn.Linear(self.hidden_dimension, classes),
            # Softmax is implicit in loss functions.
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the inputs to low dimension and predict labels with an MLP.

        Args:
            inputs (torch.Tensor): Shape (batch_size, input_dimension) tensor.

        Returns:
            torch.Tensor: Shape (batch_size, classes) tensor containing class
                logits for each sample in batch.

        Raises:
            ValueError: If input is misshapen.

        """
        if len(inputs.shape) != 2:
            raise ValueError(f'expected 2D tensor, got {len(inputs.shape)}D')

        if inputs.shape[-1] != self.input_dimension:
            raise ValueError(f'expected dimension {self.input_dimension}, '
                             f'got {inputs.shape[-1]}')

        return self.classify(self.project(inputs))
