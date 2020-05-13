"""Unit tests for probes module."""

from lodimp import probes

import pytest
import torch

INPUT_DIM = 100
PROJECTION_DIM = 10
CLASSES = 10
BATCH_SIZE = 5
SEQ_LENGTH = 15


@pytest.fixture
def projected_linear():
    """Returns a ProjectedLinear probe for testing."""
    return probes.ProjectedLinear(INPUT_DIM, PROJECTION_DIM, CLASSES)


def test_projected_linear_forward(projected_linear):
    """Test ProjectedLinear.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    actual = projected_linear(inputs)
    assert actual.shape == (BATCH_SIZE, CLASSES)


def test_projected_linear_forward_bad_dimensionality(projected_linear):
    """Test ProjectedLinear.forward dies when given not a 2D tensor."""
    inputs = torch.randn(1, BATCH_SIZE, INPUT_DIM)
    with pytest.raises(ValueError, match='.*3D.*'):
        projected_linear(inputs)


def test_projected_linear_forward_bad_shape(projected_linear):
    """Test ProjectedLinear.forward dies when given misshapen tensor."""
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM - 1)
    with pytest.raises(ValueError, match=f'.*got {INPUT_DIM - 1}.*'):
        projected_linear(inputs)


@pytest.fixture
def projected_bilinear():
    """Returns a ProjectedBilinear probe for testing."""
    return probes.ProjectedBilinear(INPUT_DIM, PROJECTION_DIM)


def test_projected_bilinear_forward(projected_bilinear):
    """Test ProjectedBilinear.forward returns tensor of correct dimension."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM)
    actual = projected_bilinear(inputs)
    assert actual.shape == (SEQ_LENGTH, SEQ_LENGTH)


def test_projected_bilinear_forward_bad_dimensionality(projected_bilinear):
    """Test ProjectedBilinear.forward dies when given not a 2D tensor."""
    inputs = torch.randn(1, SEQ_LENGTH, INPUT_DIM)
    with pytest.raises(ValueError, match='.*3D.*'):
        projected_bilinear(inputs)


def test_projected_bilinear_forward_bad_shape(projected_bilinear):
    """Test ProjectedBilinear.forward dies when given misshapen tensor."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM - 1)
    with pytest.raises(ValueError, match=f'.*got {INPUT_DIM - 1}.*'):
        projected_bilinear(inputs)


@pytest.fixture
def projected_mlp():
    """Returns a ProjectedMLP probe for testing."""
    return probes.ProjectedMLP(INPUT_DIM, PROJECTION_DIM, CLASSES)


def test_projected_mlp_forward(projected_mlp):
    """Tests ProjectedMLP.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    actual = projected_mlp(inputs)
    assert actual.shape == (BATCH_SIZE, CLASSES)


def test_projected_mlp_forward_bad_dimensionality(projected_mlp):
    """Test ProjectedMLP.forward dies when given not a 2D tensor."""
    inputs = torch.randn(1, SEQ_LENGTH, INPUT_DIM)
    with pytest.raises(ValueError, match='.*3D.*'):
        projected_mlp(inputs)


def test_projected_mlp_forward_bad_shape(projected_mlp):
    """Test ProjectedMLP.forward dies when given misshapen tensor."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM - 1)
    with pytest.raises(ValueError, match=f'.*got {INPUT_DIM - 1}.*'):
        projected_mlp(inputs)
