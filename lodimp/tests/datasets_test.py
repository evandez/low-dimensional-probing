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
        assert af.equal(ef)
        assert al.equal(el)


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


@pytest.fixture
def chunked_task_dataset(task_dataset):
    """Returns a ChunkedTaskDataset for testing."""
    return datasets.ChunkedTaskDataset(task_dataset,
                                       chunks=2,
                                       device=torch.device('cpu'))


def test_chunked_task_dataset_iter(chunked_task_dataset, features, labels):
    """Test ChunkedTaskDataset.__iter__ yields all chunks."""
    chunks = list(iter(chunked_task_dataset))
    assert len(chunks) == 2

    first, second = chunks
    assert len(first) == 2
    assert len(second) == 2

    af, al = first
    assert af.equal(features[:3])
    assert al.equal(labels[:3])

    af, al = second
    assert af.equal(features[3:])
    assert al.equal(labels[3:])


def test_chunked_dataset_len(chunked_task_dataset):
    """Test ChunkedTaskDataset.__len__ returns correct length."""
    assert len(chunked_task_dataset) == 2
