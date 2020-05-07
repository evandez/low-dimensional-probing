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
    """Tests ProjectedLinear.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    actual = projected_linear(inputs)
    assert actual.shape == (BATCH_SIZE, SEQ_LENGTH, CLASSES)


@pytest.fixture
def projected_mlp():
    """Returns a ProjectedMLP probe for testing."""
    return probes.ProjectedMLP(INPUT_DIM, PROJECTION_DIM, CLASSES)


def test_projected_mlp_forward(projected_mlp):
    """Tests ProjectedMLP.forward returns tensor of correct dimension."""
    inputs = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    actual = projected_mlp(inputs)
    assert actual.shape == (BATCH_SIZE, SEQ_LENGTH, CLASSES)
