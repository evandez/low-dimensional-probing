"""Defines probe architectures."""

from torch import nn


class Projection(nn.Sequential):
    """Probe for low-dimensional subspace of representation."""

    def __init__(self, input_dimension: int, hidden_dimension: int,
                 classes: int):
        """Initilize the architecture.

        Args:
            input_dimension (int): Dimensionality of input vectors.
            hidden_dimension (int): Dimensionality of projected space.
            classes (int): Number of classes to predict for each vector.

        """
        super(Projection, self).__init__(
            nn.Linear(input_dimension, hidden_dimension),
            nn.Linear(hidden_dimension, classes),
            # Softmax is implicit in loss functions.
        )
