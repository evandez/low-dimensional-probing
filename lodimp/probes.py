"""Defines probe architectures."""

from torch import nn


class MLP(nn.Sequential):
    """A simple multi-layer perceptron probe."""

    def __init__(self,
                 input_dimension: int,
                 classes: int,
                 hidden_dimension: int = 1000,
                 hidden_layers: int = 2):
        """Initialize the architecture.

        Args:
            input_dimension (int): Dimensionality of input vectors.
            classes (int): Number of classes to predict for each word.
            hidden_dimension (int, optional): Dimensionality of hidden layers.
                Defaults to 1000.
            hidden_layers (int, optional): Number of hidden layers.
                Defaults to 2.

        """
        if hidden_layers <= 0:
            raise ValueError(f'hidden_layers not positive: {hidden_layers}')

        architecture = [
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
        ]
        for _ in range(hidden_layers - 1):
            architecture += [
                nn.Linear(hidden_dimension, hidden_dimension),
                nn.ReLU(),
            ]
        architecture += [
            nn.Linear(hidden_dimension, classes),
            nn.ReLU(),
        ]

        super(MLP, self).__init__(*architecture)
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.classes = classes
        self.hidden_layers = hidden_layers
