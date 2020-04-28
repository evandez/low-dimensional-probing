"""Defines probe architectures."""

import torch
from torch import nn


class Projection(nn.Module):
    """Probe for low-dimensional subspace of representation."""

    def __init__(self, input_dimension: int, hidden_dimension: int,
                 classes: int):
        """Initilize the architecture.

        Args:
            input_dimension (int): Dimensionality of input vectors.
            hidden_dimension (int): Dimensionality of projected space.
            classes (int): Number of classes to predict for each vector.

        """
        super().__init__()
        self.project = nn.Linear(input_dimension, hidden_dimension)
        self.classify = nn.Linear(hidden_dimension, classes)
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
