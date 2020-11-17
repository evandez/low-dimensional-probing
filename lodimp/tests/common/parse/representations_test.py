"""Unit tests for representations module."""

import pathlib
import tempfile

from lodimp.common.parse import representations

import h5py
import numpy as np
import pytest
import torch

REP_LAYERS = 3
REP_DIMENSION = 1024
SEQ_LENGTHS = (1, 2, 3)


@pytest.fixture
def reps():
    """Returns fake representations for testing."""
    return [
        np.random.randn(REP_LAYERS, length, REP_DIMENSION)
        for length in SEQ_LENGTHS
    ]


@pytest.yield_fixture
def path(reps):
    """Yields the path to a fake representations h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'representations.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('sentence_to_index', data=0)
            for index, rep in enumerate(reps):
                handle.create_dataset(str(index), data=rep)
        yield path


@pytest.fixture
def representation_dataset(path):
    """Returns a RepresentationDataset for testing."""
    return representations.RepresentationDataset(path)


def test_representation_dataset_getitem(representation_dataset, reps):
    """Test RepresentationDataset.__getitem__ returns correct shape."""
    for index, expected in enumerate(reps):
        actual = representation_dataset[index]
        assert actual.equal(torch.tensor(expected))


def test_representation_dataset_len(representation_dataset):
    """Test RepresentationDataset.__len__ returns correct length."""
    assert len(representation_dataset) == len(SEQ_LENGTHS)


def test_representation_dataset_dimension(representation_dataset):
    """Test RepresentationDataset.dimension returns correct dimension."""
    assert representation_dataset.dimension == REP_DIMENSION


def test_representation_dataset_length(representation_dataset):
    """Test RepresentationDataset.length returns sequence lengths."""
    for index, expected in enumerate(SEQ_LENGTHS):
        assert representation_dataset.length(index) == expected


def test_representation_dataset_layer(representation_dataset):
    """Test RepresentationDataset.layer returns correct view."""
    for layer in range(REP_LAYERS):
        actual = representation_dataset.layer(layer)
        assert actual.dataset == representation_dataset
        assert actual.layer == layer


LAYER = 0


@pytest.fixture
def representation_layer_dataset(representation_dataset):
    """Returns a RepresentationLayerDataset for testing."""
    return representations.RepresentationLayerDataset(representation_dataset,
                                                      LAYER)


def test_representation_layer_dataset_getitem(representation_layer_dataset,
                                              reps):
    """Test RepresentationLayerDataset.__getitem__ returns correct layer."""
    for index, expected in enumerate(reps):
        actual = representation_layer_dataset[index]
        assert actual.equal(torch.tensor(expected[LAYER]))


def test_representation_layer_dataset_len(representation_layer_dataset):
    """Test RepresentationLayerDataset.__len__ returns number of samples."""
    assert len(representation_layer_dataset) == len(SEQ_LENGTHS)


def test_representation_layer_dataset_init_bad_layer(representation_dataset):
    """Test RepresentationLayerDataset.__init__ dies when given bad layer."""
    with pytest.raises(IndexError, match='.*3 out of bounds.*'):
        representations.RepresentationLayerDataset(representation_dataset,
                                                   REP_LAYERS)
