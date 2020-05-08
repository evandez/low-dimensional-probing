"""Unit tests for datasets module."""

import pathlib
import tempfile

from lodimp import datasets

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


def test_elmo_representations_dataset_dimension(elmo_path):
    """Test ELMoRepresentationsDataset.dimension returns correct dimension."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        assert dataset.dimension == ELMO_DIMENSION


def test_elmo_representations_dataset_length(elmo_path):
    """Test ELMoRepresentationsDataset.length returns correct seq lengths."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        for index, expected in enumerate(SEQ_LENGTHS):
            assert dataset.length(index) == expected


def test_elmo_representaitons_dataset_getitem(elmo_path):
    """Test ELMoRepresentationsDataset.__getitem__ returns correct shape."""
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


NSAMPLES = 5
NFEATURES = 10
NLABELS = 3


@pytest.fixture
def features():
    """Returns fake features for testing."""
    return torch.randn(NSAMPLES, NFEATURES)


@pytest.fixture
def labels():
    """Returns fake labels for testing."""
    return torch.randint(NLABELS, size=(NSAMPLES,))


@pytest.yield_fixture
def task_dataset(features, labels):
    """Yields the path to a fake ELMo h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'task.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('features', data=features)
            dataset = handle.create_dataset('labels', data=labels)
            dataset.attrs['nlabels'] = NLABELS
        yield datasets.TaskDataset(path)


def test_task_dataset_getitem(task_dataset, features, labels):
    """Test TaskDataset.__getitem__ returns all (feature, label) pairs."""
    for (af, al), ef, el in zip(task_dataset, features, labels):
        assert torch.tensor(af).equal(ef)
        assert torch.tensor(al).equal(el)


def test_task_dataset_getitem_bad_index(task_dataset):
    """Test TaskDataset.__getitem__ explodes when given a bad index."""
    for bad in (-1, NSAMPLES):
        with pytest.raises(IndexError, match=f'.*bounds: {bad}.*'):
            task_dataset[bad]


def test_task_dataset_len(task_dataset):
    """Test TaskDataset.__len__ returns correct length."""
    assert len(task_dataset) == NSAMPLES


def test_task_dataset_nfeatures(task_dataset):
    """Test TaskDataset.nfeatures returns correct number of features."""
    assert task_dataset.nfeatures == NFEATURES


def test_task_dataset_nlabels(task_dataset):
    """Test TaskDataset.nlabels returns correct number of labels."""
    assert task_dataset.nlabels == NLABELS


class FakeDataset(torch.utils.data.Dataset):
    """A very dumb dataset."""

    def __init__(self, features, labels):
        """Store simple features and labels.

        Args:
            features: Feature tensor.
            labels: Label tensor.

        """
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        """Return the (feature, label) at the given index.

        Args:
            index: Index of sample to retrieve.

        Returns:
            The (feature, label) pair.

        """
        return self.features[index], self.labels[index]

    def __len__(self):
        """Returns number of samples in the dataset."""
        return len(self.features)


LENGTH = 5
DIMENSION = 10


@pytest.fixture
def collated_dataset():
    """Returns CollatedDataset for testing."""
    dataset = FakeDataset(torch.ones(LENGTH, DIMENSION), torch.ones(LENGTH))
    return datasets.CollatedDataset(dataset)


def test_collated_dataset_iter(collated_dataset):
    """Test CollatedDataset.__iter__ only returns one batch."""
    batches = [batch for batch in collated_dataset]
    assert len(batches) == 1

    batch, = batches
    assert len(batch) == 2

    features, labels = batch
    assert features.shape == (LENGTH, DIMENSION)
    assert labels.shape == (LENGTH,)


def test_collated_dataset_len(collated_dataset):
    """Test CollatedDataset.__len__ returns correct length."""
    assert len(collated_dataset) == 1
