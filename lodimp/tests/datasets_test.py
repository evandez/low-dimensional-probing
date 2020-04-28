"""Unit tests for datasets module."""

import pathlib
import tempfile

from lodimp import datasets, ptb

import h5py
import numpy as np
import pytest
import torch

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


PTB_SAMPLES = [
    ptb.Sample(
        ['The', 'company', 'expects', 'earnings', '.'],
        ['DT', 'NN', 'VBZ', 'NNS', '.'],
    ),
    ptb.Sample(
        ['He', 'was', 'named', 'chief', '.'],
        ['PRP', 'VBD', 'VBN', 'JJ', '.'],
    ),
]


def task(sample):
    """A fake task."""
    labels = []
    for xpos in sample.xpos:
        labels.append(int(xpos == '.'))
    return torch.tensor(labels, dtype=torch.uint8)


@pytest.fixture
def ptb_dataset():
    """Returns a PTBDataset for testing."""
    return datasets.PTBDataset(PTB_SAMPLES, task)


def test_ptb_dataset_getitem(ptb_dataset):
    """Test whether PTBDataset.__getitem__ returns correct labels."""
    for index in (0, 1):
        item = ptb_dataset[index]
        assert item[-1] == 1
        assert not item[:-1].any()


def test_ptb_dataset_len(ptb_dataset):
    """Test whether PTBDataset.__len__ returns correct length."""
    assert len(ptb_dataset) == len(PTB_SAMPLES)


LONG_DATASET = ['foo', 'bar', 'baz']
SHORT_DATASET = [1, 2]


@pytest.fixture
def zipped_datasets():
    """Returns ZippedDatasets for testing."""
    return datasets.ZippedDatasets(LONG_DATASET, SHORT_DATASET)


def test_zipped_datasets_getitem(zipped_datasets):
    """Test ZippedDatasets.__getitem__ returns zipped items."""
    assert list(zipped_datasets) == list(zip(LONG_DATASET, SHORT_DATASET))


def test_zipped_datasets_len(zipped_datasets):
    """Test ZippedDatasets.__len__ returns length of smallest dataset."""
    assert len(zipped_datasets) == len(SHORT_DATASET)


@pytest.fixture
def collated_dataset(zipped_datasets):
    """Returns CollatedDataset for testing."""
    return datasets.CollatedDataset(zipped_datasets)


def test_collated_dataset_iter(collated_dataset):
    """Test CollatedDataset.__iter__ only returns one batch."""
    batches = [batch for batch in collated_dataset]
    assert len(batches) == 1

    batch, = batches
    assert len(batch) == 2

    names, labels = batch
    assert len(names) == 2
    assert labels.shape == (2,)


def test_collated_dataset_getitem(collated_dataset):
    """Test CollatedDataset.__getitem__ behaves like a regular dataset."""
    for index in range(len(SHORT_DATASET)):
        items = collated_dataset[index]
        assert len(items) == 2
        name, label = items
        assert name == LONG_DATASET[index]
        assert label == SHORT_DATASET[index]


def test_collated_dataset_len(collated_dataset):
    """Test CollatedDataset.__len__ returns correct length."""
    assert len(collated_dataset) == len(SHORT_DATASET)
