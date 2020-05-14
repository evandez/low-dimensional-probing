"""Unit tests for probes module."""

from lodimp import models

import pytest
import torch
from torch import nn

INPUT_DIM = 100
OUTPUT_DIM = 50
BATCH_SIZE = 5


def test_projection_init_bad_composition():
    """Test Projection.__init__ dies when given misshapen composition."""
    bad = INPUT_DIM - 1
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        models.Projection(
            INPUT_DIM,
            OUTPUT_DIM,
            compose=(nn.Linear(bad, OUTPUT_DIM),),
        )


def test_projection_forward():
    """Test Projection.forward outputs correct dimensions."""
    projection = models.Projection(INPUT_DIM, OUTPUT_DIM)
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    actual = projection(inputs)
    assert actual.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_projection_forward_composed():
    """Test Projection.forward outputs correct dimensions when composed."""
    projection = models.Projection(
        INPUT_DIM,
        OUTPUT_DIM,
        compose=(nn.Linear(INPUT_DIM, INPUT_DIM - 1),),
    )
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    actual = projection(inputs)
    assert actual.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_projection_extend():
    """Test Projection.extend composes projections correctly."""
    dimension = OUTPUT_DIM - 1
    projection = models.Projection(INPUT_DIM, OUTPUT_DIM).extend(dimension)
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    actual = projection(inputs)
    assert actual.shape == (BATCH_SIZE, dimension)


CLASSES = 10
SEQ_LENGTH = 15


@pytest.fixture
def mlp():
    """Returns a MLP probe for testing."""
    return models.MLP(INPUT_DIM, CLASSES)


def test_mlp_forward(mlp):
    """Tests MLP.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    actual = mlp(inputs)
    assert actual.shape == (BATCH_SIZE, CLASSES)


@pytest.fixture
def pairwise_bilinear():
    """Returns a PairwiseBilinear probe for testing."""
    return models.PairwiseBilinear(INPUT_DIM)


def test_bilinear_forward(pairwise_bilinear):
    """Test PairwiseBilinear.forward returns tensor of correct dimension."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM)
    actual = pairwise_bilinear(inputs)
    assert actual.shape == (SEQ_LENGTH, SEQ_LENGTH)


def test_pairwise_bilinear_forward_bad_dimensionality(pairwise_bilinear):
    """Test PairwiseBilinear.forward dies when given not a 2D tensor."""
    inputs = torch.randn(1, SEQ_LENGTH, INPUT_DIM)
    with pytest.raises(ValueError, match='.*3D.*'):
        pairwise_bilinear(inputs)


def test_pairwise_bilinear_forward_bad_shape(pairwise_bilinear):
    """Test PairwiseBilinear.forward dies when given misshapen tensor."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM - 1)
    with pytest.raises(ValueError, match=f'.*got {INPUT_DIM - 1}.*'):
        pairwise_bilinear(inputs)


@pytest.fixture
def pairwise_mlp():
    """Returns a PairwiseMLP probe for testing."""
    return models.PairwiseMLP(INPUT_DIM)


def test_pairwise_mlp_forward(pairwise_mlp):
    """Test PairwiseMLP.forward returns tensor of correct dimension."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM)
    actual = pairwise_mlp(inputs)
    assert actual.shape == (SEQ_LENGTH, SEQ_LENGTH)


def test_pairwise_mlp_forward_bad_dimensionality(pairwise_mlp):
    """Test PairwiseMLP.forward dies when given not a 2D tensor."""
    inputs = torch.randn(1, SEQ_LENGTH, INPUT_DIM)
    with pytest.raises(ValueError, match='.*3D.*'):
        pairwise_mlp(inputs)


def test_pairwise_mlp_forward_bad_shape(pairwise_mlp):
    """Test PairwiseMLP.forward dies when given misshapen tensor."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM - 1)
    with pytest.raises(ValueError, match=f'.*got {INPUT_DIM - 1}.*'):
        pairwise_mlp(inputs)
