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
        self.project = Projection(input_dimension, projected_dimension)
        self.classify = nn.Linear(projected_dimension, classes)
        # Softmax is implicit in loss functions.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the inputs to low dimension and predict labels for them.

        Args:
            inputs (torch.Tensor): Shape (B, H) tensor, where B is batch size
                and H is hidden_dimension.

        Returns:
            torch.Tensor: Shape (B, C) tensor containing class logits for each
                sample in batch.

        """
        return self.classify(self.project(inputs))


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
        self.project = Projection(input_dimension, projected_dimension)
        hidden_dimension = hidden_dimension or projected_dimension
        self.classify = nn.Sequential(
            nn.Linear(projected_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, classes),
            # Softmax is implicit in loss functions.
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the inputs to low dimension and predict labels with an MLP.

        Args:
            inputs (torch.Tensor): Shape (B, H) tensor, where B is batch size
                and H is hidden_dimension.

        Returns:
            torch.Tensor: Shape (B, C) tensor containing class logits for each
                sample in batch.

        """
        return self.classify(self.project(inputs))
