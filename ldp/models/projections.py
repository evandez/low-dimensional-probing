"""Defines linear projections."""
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
        self.right = right

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
        right = self.left if self.right is None else self.right
        return self.left(lefts), right(rights)
