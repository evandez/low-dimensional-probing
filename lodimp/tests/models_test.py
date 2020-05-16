"""Unit tests for probes module."""

from lodimp import models

import pytest
import torch

BATCH_SIZE = 5
IN_FEATURES = 100
PROJ_FEATURES = 50


@pytest.fixture
def projection():
    """Returns a projetion for testing."""
    return models.Projection(IN_FEATURES, PROJ_FEATURES)


def test_projection_init(projection):
    """Test Projection.__init__ sets fields properly."""
    assert projection.in_features == IN_FEATURES
    assert projection.out_features == PROJ_FEATURES
    assert projection.compose is None


def test_projection_init_compose(projection):
    """Test Projection.__init__ sets fields properly when composed."""
    composed = models.Projection(IN_FEATURES,
                                 PROJ_FEATURES - 1,
                                 compose=projection)
    assert composed.in_features == IN_FEATURES
    assert composed.out_features == PROJ_FEATURES - 1
    assert composed.compose is not None
    assert composed.compose.in_features == IN_FEATURES
    assert composed.compose.out_features == PROJ_FEATURES


def test_projection_forward(projection):
    """Test Projection.forward outputs well shaped tensor."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = projection(inputs)
    assert actual.shape == (BATCH_SIZE, PROJ_FEATURES)


def test_projection_forward_compose(projection):
    """Test Projection.forward outputs well shaped tensor when composed."""
    composed = models.Projection(IN_FEATURES,
                                 PROJ_FEATURES - 1,
                                 compose=projection)
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = composed(inputs)
    assert actual.shape == (BATCH_SIZE, PROJ_FEATURES - 1)


def test_projection_forward_bad_in_features(projection):
    """Test Projection.forward dies when given misshapen input."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES - 1)
    with pytest.raises(ValueError, match=f'.*{IN_FEATURES - 1}.*'):
        projection(inputs)


OUT_FEATURES = 25


@pytest.fixture
def linear(projection):
    """Returns a Linear probe for testing."""
    return models.Linear(PROJ_FEATURES, OUT_FEATURES, project=projection)


def test_linear_forward(linear):
    """Test Linear.forward returns outputs of correct shape."""
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES)
    actual = linear(inputs)
    assert actual.shape == (BATCH_SIZE, OUT_FEATURES)


def test_linear_forward_coerced(projection):
    """Test Linear.forward coerces projection inputs to correct shape."""
    model = models.Linear(PROJ_FEATURES * 2, OUT_FEATURES, project=projection)
    inputs = torch.randn(BATCH_SIZE, 2, IN_FEATURES)
    actual = model(inputs)
    assert actual.shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.fixture
def mlp(projection):
    """Returns an MLP probe for testing."""
    return models.MLP(PROJ_FEATURES, OUT_FEATURES, project=projection)


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
    mlp = models.MLP(PROJ_FEATURES * 2, OUT_FEATURES, project=projection)
    inputs = torch.randn(BATCH_SIZE, 2, IN_FEATURES)
    actual = mlp(inputs)
    assert actual.shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.fixture
def pairwise_projection(projection):
    """Returns a PairwiseProjection for testing."""
    return models.PairwiseProjection(
        projection,
        right=models.Projection(IN_FEATURES, PROJ_FEATURES),
    )


def test_pairwise_projection_init(pairwise_projection, projection):
    """Test PairwiseProjection.__init__ sets properties correctly."""
    assert pairwise_projection.in_features == IN_FEATURES
    assert pairwise_projection.out_features == PROJ_FEATURES
    assert pairwise_projection.left == projection
    assert pairwise_projection.left != pairwise_projection.right


def test_pairwise_projection_init_shared(projection):
    """Test PairwiseProjection.__init__ defaults to left for right."""
    pairwise_projection = models.PairwiseProjection(projection)
    assert pairwise_projection.left == pairwise_projection.right


@pytest.mark.parametrize(
    'right',
    (
        models.Projection(IN_FEATURES - 1, PROJ_FEATURES),
        models.Projection(IN_FEATURES, PROJ_FEATURES - 1),
    ),
)
def test_pairwise_projection_init_mismatched_projections(projection, right):
    """Test PairwiseProjection.__init__ explodes when proj sizes mismatch."""
    with pytest.raises(ValueError, match='.*same shape.*'):
        models.PairwiseProjection(projection, right=right)


def test_pairwise_projection_forward(projection, pairwise_projection):
    """Test PairwiseProjection.forward applies projection to both inputs."""
    lefts, rights = torch.randn(2, BATCH_SIZE, IN_FEATURES)
    actuals = pairwise_projection(lefts, rights)
    assert len(actuals) == 2
    assert actuals[0].equal(projection(lefts))
    assert not actuals[0].equal(actuals[1])


@pytest.fixture
def pairwise_bilinear(pairwise_projection):
    """Returns a PairwiseBilinear probe for testing."""
    return models.PairwiseBilinear(PROJ_FEATURES, project=pairwise_projection)


def test_pairwise_bilinear_init(pairwise_bilinear):
    """Test PairwiseBilinear.__init__ sets state correctly."""
    assert pairwise_bilinear.features == PROJ_FEATURES
    assert pairwise_bilinear.project is not None


def test_pairwise_bilinear_init_bad_features(pairwise_projection):
    """Test PairwiseBilinear.__init__ dies when given bad features."""
    bad = PROJ_FEATURES - 1
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        models.PairwiseBilinear(bad, project=pairwise_projection)


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
def pairwise_mlp(pairwise_projection):
    """Returns a PairwiseMLP probe for testing."""
    return models.PairwiseMLP(PROJ_FEATURES, project=pairwise_projection)


def test_pairwise_mlp_init(pairwise_mlp):
    """Test PairwiseMLP.__init__ sets state correctly."""
    assert pairwise_mlp.in_features == PROJ_FEATURES
    assert pairwise_mlp.hidden_features == PROJ_FEATURES
    assert pairwise_mlp.project is not None


def test_pairwise_mlp_init_bad_features(pairwise_projection):
    """Test PairwiseMLP.__init__ dies when given bad features."""
    bad = PROJ_FEATURES - 1
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        models.PairwiseMLP(bad, project=pairwise_projection)


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
