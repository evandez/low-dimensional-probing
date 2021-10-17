"""Defines probe architectures.

Probes map representations to labels, potentially after projecting the
representations into a subspace.
"""
from typing import Optional

from ldp.models import projections

import torch
from torch import nn


class Linear(nn.Module):
    """A linear probe."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 project: Optional[projections.Projection] = None):
        """Initialize model architecture.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            project (Optional[projections.Projection], optional): Apply this
                transformation to inputs first. By default, no transformation.

        """
        super().__init__()
        self.project = project
        self.classify = nn.Linear(in_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Linearly transform inputs, potentially after a projection.

        Args:
            inputs (torch.Tensor): Model inputs. If projecting,
                shape must be coercible to (*, project.in_features)
                and the projected inputs must be coercible to (*, in_features).
                Otherwise must have shape (*, in_features) to start.

        Returns:
            torch.Tensor: Transformation outputs of shape (N, out_features),
                where N is the same as it was described above.

        """
        if self.project is not None:
            inputs = inputs.view(-1, self.project.in_features)
            inputs = self.project(inputs)
        inputs = inputs.view(-1, self.classify.in_features)
        return self.classify(inputs)


class PairwiseBilinear(nn.Module):
    """Pairwise bilinear classifier.

    This classifier measures the pairwise "compatibility" between pairs of
    input vectors.
    """

    def __init__(self,
                 features: int,
                 project: Optional[projections.PairwiseProjection] = None):
        """Initialize the architecture.

        Args:
            features (int): Number of features in input vectors.
            project (Optional[projections.PairwiseProjection], optional): If
                set, transform each pair of inputs with either one projection
                for both, or separate projections for each. By default, no
                transformation.

        """
        super().__init__()

        if project is not None:
            # Our pairwise models will not support shape coercion, so enforce
            # output features are what the bilinear module expects.
            if project.out_features != features:
                raise ValueError(f'projection must output {features} '
                                 f'features, got {project.out_features}')

        self.features = features

        self.project = project
        self.compat = nn.Bilinear(features, features, 1)
        # Softmax is implicit in loss functions.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute pairwise compatibility between representations.

        Args:
            inputs (torch.Tensor): Matrix of representations.
                Must have shape (N, features), where N is the
                number of representations to compare.

        Returns:
            torch.Tensor: Size (N, N) matrix representing pairwise
                compatibilities.

        Raises:
            ValueError: If input is misshapen.

        """
        if len(inputs.shape) != 2:
            raise ValueError(f'expected 2D tensor, got {len(inputs.shape)}D')

        # For performance, we frame this operation as a large matrix multiply.
        length, features = inputs.shape
        left = inputs.repeat(1, length).view(length**2, features)
        right = inputs.repeat(length, 1)
        if self.project is not None:
            left, right = self.project(left, right)
        return self.compat(left, right).view(length, length)


class MLP(nn.Module):
    """A simple MLP probe.

    The MLP contains two hidden layers: the first is a linear layer with ReLU,
    the latter is just linear.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: Optional[int] = None,
                 project: Optional[projections.Projection] = None):
        """Initilize the architecture.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            hidden_features (Optional[int], optional): Number of
                features in MLP hidden layer. Defaults to in_feaures.
            project (Optional[Projection], optional): Apply this
                transformation to inputs first. By default, no transformation.

        """
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features

        self.project = project
        self.mlp = nn.Sequential(
            nn.Linear(in_features, self.hidden_features), nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Send inputs through MLP, projecting first if necessary.

        Args:
            inputs (torch.Tensor): Model inputs. If projecting,
                shape must be coercible to (*, project.in_features)
                and the projected inputs must be coercible to (*, in_features).
                Otherwise must be coercible to (*, in_features).

        Returns:
            torch.Tensor: MLP output.

        """
        if self.project is not None:
            inputs = inputs.view(-1, self.project.in_features)
            inputs = self.project(inputs)
        inputs = inputs.view(-1, self.in_features)
        return self.mlp(inputs)


class PairwiseMLP(nn.Module):
    """MLP that computes pairwise compatibilities between representations.

    Unlike BiMLP, the PairwiseMLP computes compatibilities between ALL
    pairs of representations. The former only computes compatibilities
    between given pairs.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 project: Optional[projections.PairwiseProjection] = None):
        """Initialize the network.

        Args:
            in_features (int): Number of features in inputs.
            hidden_features (Optional[int], optional): Number of features in
                MLP hidden layer. Defaults to the same as MLP's input features.
            project (Optional[projections.PairwiseProjection], optional): See
                PairwiseBilinear.__init__ docstring.

        """
        super().__init__()

        if project is not None:
            # Our pairwise models will not support shape coercion, so enforce
            # output features are what the bilinear module expects.
            if project.out_features != in_features:
                raise ValueError(f'projection must output {in_features} '
                                 f'features, got {project.out_features}')

        self.in_features = in_features
        self.hidden_features = hidden_features or in_features

        self.project = project
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2, self.hidden_features), nn.ReLU(),
            nn.Linear(self.hidden_features, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute pairwise compatibilities for given matrix.

        Args:
            inputs (torch.Tensor): Shape (N, in_features) matrix,
                where N is the number of representations to compare.

        Raises:
            ValueError: If input is misshapen.

        Returns:
            torch.Tensor: Size (N, N) matrix representing pairwise
                compatibilities.

        """
        if len(inputs.shape) != 2:
            raise ValueError(f'expected 2D tensor, got {len(inputs.shape)}D')

        length, features = inputs.shape
        lefts = inputs.repeat(1, length).view(length**2, features)
        rights = inputs.repeat(length, 1)
        if self.project is not None:
            lefts, rights = self.project(lefts, rights)
        pairs = torch.cat((lefts, rights), dim=-1)
        return self.mlp(pairs).view(length, length)
