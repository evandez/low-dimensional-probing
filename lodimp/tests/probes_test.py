"""Unit tests for probes module."""

from lodimp import probes

import pytest
import torch

INPUT_DIM = 100
CLASSES = 10
BATCH_SIZE = 5
SEQ_LENGTH = 15


@pytest.fixture
def bilinear():
    """Returns a Bilinear probe for testing."""
    return probes.Bilinear(INPUT_DIM)


def test_bilinear_forward(bilinear):
    """Test Bilinear.forward returns tensor of correct dimension."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM)
    actual = bilinear(inputs)
    assert actual.shape == (SEQ_LENGTH, SEQ_LENGTH)


def test_bilinear_forward_bad_dimensionality(bilinear):
    """Test Bilinear.forward dies when given not a 2D tensor."""
    inputs = torch.randn(1, SEQ_LENGTH, INPUT_DIM)
    with pytest.raises(ValueError, match='.*3D.*'):
        bilinear(inputs)


def test_bilinear_forward_bad_shape(bilinear):
    """Test Bilinear.forward dies when given misshapen tensor."""
    inputs = torch.randn(SEQ_LENGTH, INPUT_DIM - 1)
    with pytest.raises(ValueError, match=f'.*got {INPUT_DIM - 1}.*'):
        bilinear(inputs)


@pytest.fixture
def mlp():
    """Returns a MLP probe for testing."""
    return probes.MLP(INPUT_DIM, CLASSES)


def test_mlp_forward(mlp):
    """Tests MLP.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    actual = mlp(inputs)
    assert actual.shape == (BATCH_SIZE, CLASSES)
