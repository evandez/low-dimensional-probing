"""Defines model architectures."""

from typing import Optional, Tuple

import torch
from torch import nn


class Projection(nn.Module):
    """A linear projection, potentially composed with other projections.

    Importantly, only the last projection in the composition has trainable
    parameters. The other projections are assumed to be fixed.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 compose: Optional['Projection'] = None):
        """Initialize the model architecture.

        Args:
            in_features (int): Number of input features, i.e. dimensionality of
                the input space.
            out_features (int): Number of output features, i.e. dimensionality
                of the output space.
            compose (Optional[Projection], optional): Projection to apply
                before this one. Its in_features must equal the one specified
                in this constructor. Importantly, this projection will
                NEVER have gradients.

        Raises:
            ValueError: If the composed projection misaligns with this one.

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.compose = compose
        if compose is not None:
            if in_features != compose.in_features:
                raise ValueError(
                    f'composed projection expects {compose.in_features} '
                    f'features, but new one expects {in_features}')
            in_features = compose.out_features

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

        if self.compose is not None:
            with torch.no_grad():
                inputs = self.compose(inputs)

        return self.project(inputs)


class Linear(nn.Module):
    """A linear probe."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 project: Optional[Projection] = None):
        """Initialize model architecture.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            project (Optional[Projection], optional): Apply this
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


class MLP(nn.Module):
    """A 2-layer MLP probe."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: Optional[int] = None,
                 project: Optional[Projection] = None):
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
                Otherwise must have shape (*, in_features) to start.

        Returns:
            torch.Tensor: MLP output.

        """
        if self.project is not None:
            inputs = inputs.view(-1, self.project.in_features)
            inputs = self.project(inputs)
            inputs = inputs.view(-1, self.in_features)
        return self.mlp(inputs)


class PairwiseProjection(nn.Module):
    """Projection applied to pairs of tensors.

    Can apply the same projection to both, or separate projections
    for each.
    """

    def __init__(self, left: Projection, right: Optional[Projection] = None):
        """Initialize the projection.

        Args:
            left (Projection): Transformation to apply to left inputs.
            right (Optional[Projection], optional): Transformation to apply to
                right inputs. Defaults to same as left transformation.

        """
        super().__init__()

        if right is not None:
            # We won't experiment with pairwise projections in which the
            # projections do not share the same rank, so disallow this.
            left_shape = (left.in_features, left.out_features)
            right_shape = (right.in_features, right.out_features)
            if left_shape != right_shape:
                raise ValueError(f'projections must have same shape, '
                                 f'got {left_shape} vs. {right_shape}')

        self.in_features = left.in_features
        self.out_features = left.out_features

        self.left = left
        self.right = right or left

    def forward(
        self,
        lefts: torch.Tensor,
        rights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project the given tensor pairs.

        Args:
            lefts (torch.Tensor): Left elements in the pair.
            rights (torch.Tensor): Right elements in the pair.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Projected left/right elements.

        """
        return self.left(lefts), self.right(rights)


class PairwiseBilinear(nn.Module):
    """Pairwise bilinear classifier.

    This classifier measures the pairwise "compatibility" between pairs of
    input vectors.
    """

    def __init__(self,
                 features: int,
                 project: Optional[PairwiseProjection] = None):
        """Initialize the architecture.

        Args:
            features (int): Number of features in input vectors.
            project (Optional[PairwiseProjection], optional): Transform each
                pair of inputs with either one projection for both, or separate
                projections for each. By default, no transformation.

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

        # For performance, we avoid copying data and construct two views
        # of the projected representations so that the entire bilinear
        # operation happens as one big matrix multiplication.
        length, features = inputs.shape
        left = inputs.repeat(1, length).view(length**2, features)
        right = inputs.repeat(length, 1)
        if self.project is not None:
            left, right = self.project(left, right)
        return self.compat(left, right).view(length, length)


class PairwiseMLP(nn.Module):
    """MLP that computes pairwise compatibilities between representations."""

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 project: Optional[PairwiseProjection] = None):
        """Initialize the network.

        Args:
            in_features (int): Number of features in inputs.
            hidden_features (Optional[int], optional): Number of features in
                MLP hidden layer. Defaults to the same as MLP's input features.
            project (Optional[PairwiseProjection], optional): See
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
