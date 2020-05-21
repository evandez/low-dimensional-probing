"""Unit tests for representations module."""

import pathlib
import tempfile

from lodimp import datasets

import h5py
import numpy as np
import pytest

REP_LAYERS = 3
REP_DIMENSION = 1024
SEQ_LENGTHS = (1, 2, 3)


@pytest.yield_fixture
def representations_path():
    """Yields the path to a fake ELMo h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'elmo.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('sentence_to_index', data=0)
            for index, length in enumerate(SEQ_LENGTHS):
                data = np.zeros((REP_LAYERS, length, REP_DIMENSION))
                data[:] = index
                handle.create_dataset(str(index), data=data)
        yield path


def test_representations_dataset_init_bad_layer(representations_path):
    """Test RepresentationsDataset.__init__ dies when given bad layer."""
    with pytest.raises(IndexError, match='.*got 3.*'):
        datasets.RepresentationsDataset(representations_path, REP_LAYERS)


def test_representations_dataset_dimension(representations_path):
    """Test RepresentationsDataset.dimension returns correct dimension."""
    for layer in range(REP_LAYERS):
        dataset = datasets.RepresentationsDataset(representations_path, layer)
        assert dataset.dimension == REP_DIMENSION


def test_representations_dataset_getitem(representations_path):
    """Test RepresentationsDataset.__getitem__ returns correct shape."""
    for layer in range(REP_LAYERS):
        dataset = datasets.RepresentationsDataset(representations_path, layer)
        for index, length in enumerate(SEQ_LENGTHS):
            assert dataset[index].shape == (length, REP_DIMENSION)
            assert (dataset[index] == index).all()


def test_representations_dataset_len(representations_path):
    """Test RepresentationsDataset.__len__ returns correct length."""
    for layer in range(REP_LAYERS):
        dataset = datasets.RepresentationsDataset(representations_path, layer)
        assert len(dataset) == len(SEQ_LENGTHS)
