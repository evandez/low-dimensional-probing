"""Unit tests for projection module."""

from lodimp.common.models import projections

import pytest
import torch
from torch import optim

BATCH_SIZE = 5
IN_FEATURES = 100
PROJ_FEATURES = 50


@pytest.fixture
def proj():
    """Returns a projetion for testing."""
    return projections.Projection(IN_FEATURES, PROJ_FEATURES)


def test_projection_init(proj):
    """Test Projection.__init__ sets fields properly."""
    assert proj.in_features == IN_FEATURES
    assert proj.out_features == PROJ_FEATURES
    assert proj.compose is None


def test_projection_init_compose(proj):
    """Test Projection.__init__ sets fields properly when composed."""
    composed = projections.Projection(IN_FEATURES,
                                      PROJ_FEATURES - 1,
                                      compose=proj)
    assert composed.in_features == IN_FEATURES
    assert composed.out_features == PROJ_FEATURES - 1
    assert composed.compose is not None
    assert composed.compose.in_features == IN_FEATURES
    assert composed.compose.out_features == PROJ_FEATURES


def test_projection_forward(proj):
    """Test Projection.forward outputs well shaped tensor."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = proj(inputs)
    assert actual.shape == (BATCH_SIZE, PROJ_FEATURES)


def test_projection_forward_compose(proj):
    """Test Projection.forward outputs well shaped tensor when composed."""
    composed = projections.Projection(IN_FEATURES,
                                      PROJ_FEATURES - 1,
                                      compose=proj)
    # Create an optimizer for later. We will verify composed projection
    # is not optimized.
    optimizer = optim.SGD(composed.parameters(), lr=1)

    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = composed(inputs)
    assert actual.shape == (BATCH_SIZE, PROJ_FEATURES - 1)

    snapshot = composed.compose.project.weight.data.clone()
    loss = actual.sum()
    loss.backward()
    optimizer.step()
    assert composed.compose.project.weight.data.equal(snapshot)


def test_projection_forward_bad_in_features(proj):
    """Test Projection.forward dies when given misshapen input."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES - 1)
    with pytest.raises(ValueError, match=f'.*{IN_FEATURES - 1}.*'):
        proj(inputs)


@pytest.fixture
def pairwise_proj(proj):
    """Returns a PairwiseProjection for testing."""
    return projections.PairwiseProjection(
        proj,
        right=projections.Projection(IN_FEATURES, PROJ_FEATURES),
    )


def test_pairwise_projection_init(pairwise_proj, proj):
    """Test PairwiseProjection.__init__ sets properties correctly."""
    assert pairwise_proj.in_features == IN_FEATURES
    assert pairwise_proj.out_features == PROJ_FEATURES
    assert pairwise_proj.left == proj
    assert pairwise_proj.left != pairwise_proj.right


def test_pairwise_projection_init_shared(proj):
    """Test PairwiseProjection.__init__ defaults to left for right."""
    pairwise_proj = projections.PairwiseProjection(proj)
    assert pairwise_proj.left == proj
    assert pairwise_proj.right is None


@pytest.mark.parametrize(
    'right',
    (
        projections.Projection(IN_FEATURES - 1, PROJ_FEATURES),
        projections.Projection(IN_FEATURES, PROJ_FEATURES - 1),
    ),
)
def test_pairwise_projection_init_mismatched_projections(proj, right):
    """Test PairwiseProjection.__init__ explodes when proj sizes mismatch."""
    with pytest.raises(ValueError, match='.*same shape.*'):
        projections.PairwiseProjection(proj, right=right)


def test_pairwise_projection_forward(proj, pairwise_proj):
    """Test PairwiseProjection.forward applies projection to both inputs."""
    lefts, rights = torch.randn(2, BATCH_SIZE, IN_FEATURES)
    actuals = pairwise_proj(lefts, rights)
    assert len(actuals) == 2
    assert actuals[0].equal(proj(lefts))
    assert not actuals[0].equal(actuals[1])
