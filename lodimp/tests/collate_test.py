"""Unit tests for collate module."""

from lodimp import collate

import pytest
import torch

DIM = 10
LABELS = 5
LENGTHS = [1, 2, 3]


@pytest.fixture
def samples():
    """Returns fake sequences for testing."""
    return [(torch.randn(length, DIM), torch.randint(LABELS, size=(length,)))
            for length in LENGTHS]


def test_pack(samples):
    """Test pack correctly packs sequences."""
    sequences = collate.pack(samples)
    assert len(sequences) == 2
    inputs, labels = sequences
    assert inputs.shape == (sum(LENGTHS), DIM)
    assert labels.shape == (sum(LENGTHS),)


def test_pack_no_samples():
    """Test pack dies when given no samples."""
    with pytest.raises(ValueError, match='.*empty.*'):
        collate.pack([])


def test_pack_empty_samples():
    """Test pack dies when given empty samples."""
    with pytest.raises(ValueError, match='.*empty.*'):
        collate.pack([[], []])


def test_pack_different_sample_sizes():
    """Test pack dies when given samples of different sizes."""
    with pytest.raises(ValueError, match='.*lengths.*'):
        collate.pack([
            (torch.tensor(1),),
            (torch.tensor(1), torch.tensor(2)),
        ])


def test_pack_different_sequence_lengths():
    """Test pack dies when individual sample has mismatched sequence length."""
    with pytest.raises(ValueError, match='.*lengths.*'):
        collate.pack([(torch.ones(5), torch.ones(4))])
