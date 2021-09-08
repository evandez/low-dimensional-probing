"""Unit tests for probes module."""

from lodimp.common.models import probes, projections

import pytest
import torch

BATCH_SIZE = 5
IN_FEATURES = 100
PROJ_FEATURES = 50
OUT_FEATURES = 25


@pytest.fixture
def projection():
    """Returns a projections.Projection for constructing probes."""
    return projections.Projection(IN_FEATURES, PROJ_FEATURES)


@pytest.fixture
def linear(projection):
    """Returns a Linear probe for testing."""
    return probes.Linear(PROJ_FEATURES, OUT_FEATURES, project=projection)


def test_linear_forward(linear):
    """Test Linear.forward returns outputs of correct shape."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = linear(inputs)
    assert actual.shape == (BATCH_SIZE, OUT_FEATURES)


def test_linear_forward_coerced(projection):
    """Test Linear.forward coerces projection inputs to correct shape."""
    model = probes.Linear(PROJ_FEATURES * 2, OUT_FEATURES, project=projection)
    inputs = torch.randn(BATCH_SIZE, 2, IN_FEATURES)
    actual = model(inputs)
    assert actual.shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.fixture
def pairwise_projection(projection):
    """Returns a projections.PairwiseProjection for constructing probes."""
    return projections.PairwiseProjection(
        projection,
        right=projections.Projection(IN_FEATURES, PROJ_FEATURES),
    )


@pytest.fixture
def pairwise_bilinear(pairwise_projection):
    """Returns a PairwiseBilinear probe for testing."""
    return probes.PairwiseBilinear(PROJ_FEATURES, project=pairwise_projection)


def test_pairwise_bilinear_init(pairwise_bilinear):
    """Test PairwiseBilinear.__init__ sets state correctly."""
    assert pairwise_bilinear.features == PROJ_FEATURES
    assert pairwise_bilinear.project is not None


def test_pairwise_bilinear_init_bad_features(pairwise_projection):
    """Test PairwiseBilinear.__init__ dies when given bad features."""
    bad = PROJ_FEATURES - 1
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        probes.PairwiseBilinear(bad, project=pairwise_projection)


def test_pairwise_bilinear_forward(pairwise_bilinear):
    """Test PairwiseBilinear.forward outputs correct shape."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = pairwise_bilinear(inputs)
    assert actual.shape == (BATCH_SIZE, BATCH_SIZE)


@pytest.mark.parametrize('shape', (
    (BATCH_SIZE, 2, IN_FEATURES),
    (IN_FEATURES,),
))
def test_pairwise_bilinear_forward_non_2d_input(pairwise_bilinear, shape):
    """Test PairwiseBilinear.forward dies when given non-2D input."""
    inputs = torch.randn(*shape)
    with pytest.raises(ValueError, match=f'.*got {len(shape)}D.*'):
        pairwise_bilinear(inputs)


@pytest.fixture
def mlp(projection):
    """Returns an MLP probe for testing."""
    return probes.MLP(PROJ_FEATURES, OUT_FEATURES, project=projection)


def test_mlp_init(mlp):
    """Test MLP.__init__ sets state correctly."""
    assert mlp.in_features == PROJ_FEATURES
    assert mlp.hidden_features == PROJ_FEATURES
    assert mlp.out_features == OUT_FEATURES
    assert mlp.project is not None


def test_mlp_forward(mlp):
    """Test MLP.forward returns outputs of correct shape."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = mlp(inputs)
    assert actual.shape == (BATCH_SIZE, OUT_FEATURES)


def test_mlp_forward_coerced(projection):
    """Test MLP.forward coerces projection inputs to correct shape."""
    mlp = probes.MLP(PROJ_FEATURES * 2, OUT_FEATURES, project=projection)
    inputs = torch.randn(BATCH_SIZE, 2, IN_FEATURES)
    actual = mlp(inputs)
    assert actual.shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.fixture
def pairwise_mlp(pairwise_projection):
    """Returns a PairwiseMLP probe for testing."""
    return probes.PairwiseMLP(PROJ_FEATURES, project=pairwise_projection)


def test_pairwise_mlp_init(pairwise_mlp):
    """Test PairwiseMLP.__init__ sets state correctly."""
    assert pairwise_mlp.in_features == PROJ_FEATURES
    assert pairwise_mlp.hidden_features == PROJ_FEATURES
    assert pairwise_mlp.project is not None


def test_pairwise_mlp_init_bad_features(pairwise_projection):
    """Test PairwiseMLP.__init__ dies when given bad features."""
    bad = PROJ_FEATURES - 1
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        probes.PairwiseMLP(bad, project=pairwise_projection)


def test_pairwise_mlp_forward(pairwise_mlp):
    """Test PairwiseMLP.forward outputs correct shape."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = pairwise_mlp(inputs)
    assert actual.shape == (BATCH_SIZE, BATCH_SIZE)


@pytest.mark.parametrize('shape', (
    (BATCH_SIZE, 2, IN_FEATURES),
    (IN_FEATURES,),
))
def test_pairwise_mlp_forward_non_2d_input(pairwise_mlp, shape):
    """Test PairwiseMLP.forward dies when given non-2D input."""
    inputs = torch.randn(*shape)
    with pytest.raises(ValueError, match=f'.*got {len(shape)}D.*'):
        pairwise_mlp(inputs)
