"""Defines model architectures."""

from typing import Optional, Sequence

import torch
from torch import nn


class Projection(nn.Module):
    """A linear projection composed of one or more distinct linear projections.

    Importantly, only the last projection in the composition has trainable
    parameters. The other projections are assumed to be fixed.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 compose: Optional[Sequence[nn.Linear]] = None):
        """Initialize the model architecture.

        Args:
            in_features (int): Number of input features, i.e. dimensionality of
                the input space.
            out_features (int): Number of output features, i.e. dimensionality
                of the output space.
            compose (Optional[Sequence[nn.Linear]], optional): Linear
                projections to apply before the final projection. The first
                projection in this sequence must expect in_features features.
                Importantly, the projections specified in this argument will
                NEVER have gradients because they are assumed to be fixed.
                By default, only one linear projection is used.

        Raises:
            ValueError: If any of the composed projections have mismatching
                input and output shapes.

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.preprocess: Optional[nn.Sequential] = None
        if compose is not None:
            for module in compose:
                if module.in_features != in_features:
                    raise ValueError(f'cannot compose {in_features}d '
                                     f'with {module.in_features}d')
                in_features = module.out_features
            self.preprocess = nn.Sequential(*compose)

        self.project = nn.Linear(in_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the inputs.

        Args:
            inputs (torch.Tensor): Tensor of shape (*, in_features), where
                * is an arbitrary shape.

        Raises:
            ValueError: If the last dimension of the input does not match
                expected number of input features.

        Returns:
            torch.Tensor: Shape (*, out_features) projected tensor.

        """
        features = inputs.shape[-1]
        if features != self.in_features:
            raise ValueError(f'expected {self.in_features} input features, '
                             f'got {features}')

        if self.preprocess is not None:
            with torch.no_grad():
                inputs = self.preprocess(inputs)

        return self.project(inputs)

    def extend(self, out_features: int) -> 'Projection':
        """Extend this projection by composing with it a new projection.

        Args:
            out_features (int): Number of output features for the linear
                projection that will be applied to the output of this one.

        Returns:
            Projection: The extended projection.

        """
        compose = []
        if self.preprocess is not None:
            for module in self.preprocess:
                assert isinstance(module, nn.Linear), 'non-linear composition?'
                compose.append(module)
        compose.append(self.project)
        return Projection(self.in_features, out_features, compose=compose)


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
            hidden_dimension (Optional[int], optional): Dimensionality of
                MLP hidden layer. Defaults to input_dimension.

        """
        hidden_dimension = hidden_dimension or input_dimension
        super().__init__(nn.Linear(input_dimension, hidden_dimension),
                         nn.ReLU(), nn.Linear(hidden_dimension, classes))

        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.classes = classes


class PairwiseBilinear(nn.Module):
    """Pairwise bilinear classifier.

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


class PairwiseMLP(MLP):
    """MLP that computes pairwise compatibilities between representations."""

    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: Optional[int] = None):
        """Initialize the network.

        Args:
            input_dimension (int): Dimensionality of input representations.
            hidden_dimension (Optional[int], optional): Dimensionality of MLP
                hidden layer. Defaults to input_dimension.

        """
        super().__init__(input_dimension * 2,
                         1,
                         hidden_dimension=hidden_dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute pairwise compatibilities for given matrix.

        Args:
            inputs (torch.Tensor): Shape (N, input_dimension) matrix,
                where N is the number of representations to compare.

        Raises:
            ValueError: If input is misshapen.

        Returns:
            torch.Tensor: Size (N, N) matrix representing pairwise
                compatibilities.

        """
        dims = len(inputs.shape)
        if dims != 2:
            raise ValueError(f'expected 2D tensor, got {dims}D')

        length, dimension = inputs.shape
        expected_dimension = self.hidden_dimension // 2
        if dimension != expected_dimension:
            raise ValueError(
                f'expected {expected_dimension}d, got {dimension}d')

        lefts = inputs.repeat(1, length).view(length**2, dimension)
        rights = inputs.repeat(length, 1)
        pairs = torch.cat((lefts, rights), dim=-1)
        return super().forward(pairs).view(length, length)
