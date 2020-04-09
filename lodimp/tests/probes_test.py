"""Unit tests for probes module."""

import pytest
import torch

from lodimp import probes

INPUT_DIMENSIONS = 100
CLASSES = 10
BATCH_SIZE = 5
SEQ_LENGTH = 15


@pytest.fixture
def mlp():
    """Returns MLP for testing."""
    return probes.MLP(INPUT_DIMENSIONS, CLASSES)


def test_mlp_forward(mlp):
    """Test MLP.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIMENSIONS)
    actual = mlp(inputs)
    assert actual.shape == (BATCH_SIZE, SEQ_LENGTH, CLASSES)
