"""Unit tests for datasets module."""

import pathlib
import tempfile

from lodimp import datasets

import h5py
import pytest
import torch

BREAKS = (0, 3)
NSAMPLES = 5
NGRAMS = 2
NDIMS = 10
NTAGS = 3


@pytest.fixture
def breaks():
    """Returns fake sentence breaks for testing."""
    return torch.tensor(BREAKS)


@pytest.fixture
def representations():
    """Returns fake representations for testing."""
    return torch.randn(NSAMPLES, NGRAMS, NDIMS)


@pytest.fixture
def labels():
    """Returns fake labels for testing."""
    return torch.randint(NTAGS, size=(NSAMPLES,))


@pytest.yield_fixture
def task_dataset(breaks, representations, labels):
    """Yields the path to a fake h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'task.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('breaks', data=breaks)
            handle.create_dataset('reps', data=representations)
            dataset = handle.create_dataset('tags', data=labels)
            dataset.attrs['ntags'] = NTAGS
        yield datasets.TaskDataset(path)


def test_task_dataset_getitem(task_dataset, representations, labels):
    """Test TaskDataset.__getitem__ returns all (reps, label) pairs."""
    for (af, al), ef, el in zip(task_dataset, representations, labels):
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


def test_task_dataset_ngrams(task_dataset):
    """Test TaskDataset.ngrams returns correct number of reps per sample."""
    assert task_dataset.ngrams == NGRAMS


def test_task_dataset_ndims(task_dataset):
    """Test TaskDataset.ndims returns correct number of features."""
    assert task_dataset.ndims == NDIMS


def test_task_dataset_ntags(task_dataset):
    """Test TaskDataset.ntags returns correct number of labels."""
    assert task_dataset.ntags == NTAGS


@pytest.fixture
def test_task_dataset_ngrams_unigram_dataset(breaks, representations, labels):
    """Test TaskDataset.ngrams returns 1 when dataset is unigram."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'task.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('breaks', data=breaks)
            handle.create_dataset('reps', data=representations)
            handle.create_dataset('tags', data=labels)
        dataset = datasets.TaskDataset(path)
        assert dataset.ngrams == 1


@pytest.fixture
def sentence_task_dataset(task_dataset):
    """Returns a SentenceTaskDataset for testing."""
    return datasets.SentenceTaskDataset(task_dataset)


def test_sentence_task_dataset_iter(sentence_task_dataset, representations,
                                    labels):
    """Test SentenceTaskDataset.__iter__ yields all samples."""
    batches = list(sentence_task_dataset)
    assert len(batches) == 2
    first, second = batches

    assert len(first) == 2
    ar, al = first
    assert ar.equal(representations[0:3])
    assert al.equal(labels[0:3])

    assert len(second) == 2
    ar, al = second
    assert ar.equal(representations[3:])
    assert al.equal(labels[3:])


def test_sentence_task_dataset_len(sentence_task_dataset):
    """Test SentenceTaskDataset.__len__ returns number of sentences."""
    assert len(sentence_task_dataset) == len(BREAKS)


@pytest.fixture
def chunked_task_dataset(task_dataset):
    """Returns a ChunkedTaskDataset for testing."""
    return datasets.ChunkedTaskDataset(task_dataset,
                                       chunks=2,
                                       device=torch.device('cpu'))


def test_chunked_task_dataset_iter(chunked_task_dataset, representations,
                                   labels):
    """Test ChunkedTaskDataset.__iter__ yields all chunks."""
    chunks = list(iter(chunked_task_dataset))
    assert len(chunks) == 2

    first, second = chunks
    assert len(first) == 2
    assert len(second) == 2

    af, al = first
    assert af.equal(representations[:3])
    assert al.equal(labels[:3])

    af, al = second
    assert af.equal(representations[3:])
    assert al.equal(labels[3:])


def test_chunked_task_dataset_iter_predefined_chunks(task_dataset,
                                                     representations, labels):
    """Test ChunkedTaskDataset.__iter__ uses predefined chunks when given."""
    chunks = (0, 2, 4)
    chunked_task_dataset = datasets.ChunkedTaskDataset(task_dataset,
                                                       chunks=chunks)
    actuals = tuple(chunked_task_dataset)
    assert len(actuals) == len(chunks)

    expecteds = (
        (representations[:2], labels[:2]),
        (representations[2:4], labels[2:4]),
        (representations[4:], labels[4:]),
    )
    for actual, expected in zip(actuals, expecteds):
        assert len(actual) == 2
        for (at, et) in zip(actual, expected):
            assert at.equal(et)


def test_chunked_dataset_len(chunked_task_dataset):
    """Test ChunkedTaskDataset.__len__ returns correct length."""
    assert len(chunked_task_dataset) == 2
