"""Unit tests for datasets module."""

import pathlib
import tempfile

import h5py
import numpy as np
import pytest

from lodimp import datasets

ELMO_LAYERS = 3
ELMO_DIMENSION = 1024
SEQ_LENGTHS = (1, 2, 3)


@pytest.yield_fixture
def elmo_path():
    """Yields the path to a fake ELMo h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'elmo.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('sentence_to_index', data=0)
            for index, length in enumerate(SEQ_LENGTHS):
                data = np.zeros((ELMO_LAYERS, length, ELMO_DIMENSION))
                data[:] = index
                handle.create_dataset(str(index), data=data)
        yield path


def test_elmo_init_bad_layer(elmo_path):
    """Test ELMoRepresentationsDataset.__init__ dies when given bad layer."""
    with pytest.raises(ValueError, match='.*layer.*'):
        datasets.ELMoRepresentationsDataset(elmo_path, 3)


def test_elmo_representaitons_dataset_getitem(elmo_path):
    """Test ELMoRepresentationsDataset.__len__ returns correct shape."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        for index, length in enumerate(SEQ_LENGTHS):
            assert dataset[index].shape == (length, ELMO_DIMENSION)
            assert (dataset[index] == index).all()


def test_elmo_representations_dataset_len(elmo_path):
    """Test ELMoRepresentationsDataset.__len__ returns correct length."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        assert len(dataset) == len(SEQ_LENGTHS)
