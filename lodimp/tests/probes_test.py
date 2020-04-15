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
def projection():
    """Returns a Projection for testing."""
    return probes.Projection(INPUT_DIM, PROJECTION_DIM, CLASSES)


def test_projection_forward(projection):
    """Tests Projection.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    actual = projection(inputs)
    assert actual.shape == (BATCH_SIZE, SEQ_LENGTH, CLASSES)
