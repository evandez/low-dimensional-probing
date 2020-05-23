"""Unit and functional tests for the learning module."""

import pathlib
import tempfile

from lodimp.common import learning

import h5py
import pytest
import torch
import wandb
from torch import nn

BREAKS = (0, 3)
NSAMPLES = 5
NDIMS = 10
NTAGS = 3

BREAKS_KEY = 'test-breaks'
REPS_KEY = 'test-reps'
TAGS_KEY = 'test-tags'


@pytest.fixture
def breaks():
    """Returns fake sentence breaks for testing."""
    return torch.tensor(BREAKS)


@pytest.fixture
def representations():
    """Returns fake representations for testing."""
    return torch.randn(NSAMPLES, NDIMS)


@pytest.fixture
def tags():
    """Returns fake tags for testing."""
    return torch.randint(NTAGS, size=(NSAMPLES,))


@pytest.yield_fixture
def path(breaks, representations, tags):
    """Yields the path to a fake h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'task.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset(BREAKS_KEY, data=breaks)
            handle.create_dataset(REPS_KEY, data=representations)
            handle.create_dataset(TAGS_KEY, data=tags)
        yield path


@pytest.fixture
def task_dataset(path):
    """Returns a TaskDataset for testing."""
    return learning.TaskDataset(path,
                                breaks_key=BREAKS_KEY,
                                reps_key=REPS_KEY,
                                tags_key=TAGS_KEY)


def test_task_dataset_getitem(task_dataset, representations, tags):
    """Test TaskDataset.__getitem__ returns all (reps, label) pairs."""
    for index, (er, et) in enumerate(zip(representations, tags)):
        ar, at = task_dataset[index]
        assert ar.equal(er)
        assert at.equal(et)


def test_task_dataset_getitem_bad_index(task_dataset):
    """Test TaskDataset.__getitem__ explodes when given a bad index."""
    for bad in (-1, NSAMPLES):
        with pytest.raises(IndexError, match=f'.*bounds: {bad}.*'):
            task_dataset[bad]


def test_task_dataset_iter(task_dataset, representations, tags):
    """Test TaskDataset.__iter__ yields all (reps, label) pairs."""
    actuals = tuple(iter(task_dataset))
    for (ar, at), (er, et) in zip(actuals, zip(representations, tags)):
        assert ar.shape == (1, NDIMS)
        assert at.shape == (1,)
        assert ar.squeeze().equal(er)
        assert at.squeeze().equal(et)


def test_task_dataset_len(task_dataset):
    """Test TaskDataset.__len__ returns correct length."""
    assert len(task_dataset) == NSAMPLES


def test_task_dataset_dimension(task_dataset):
    """Test TaskDataset.ndims returns correct number of features."""
    assert task_dataset.dimension == NDIMS


@pytest.fixture
def sentence_iterable_task_dataset(path):
    """Returns a SentenceTaskDataset for testing."""
    return learning.SentenceIterableTaskDataset(path,
                                                breaks_key=BREAKS_KEY,
                                                reps_key=REPS_KEY,
                                                tags_key=TAGS_KEY)


def test_sentence_iterable_task_dataset_getitem(sentence_iterable_task_dataset,
                                                representations, tags):
    """Test SentenceIterableTaskDataset.__getitem__ returns correct samples."""
    expecteds = (
        (representations[BREAKS[0]:BREAKS[1]], tags[BREAKS[0]:BREAKS[1]]),
        (representations[BREAKS[1]:], tags[BREAKS[1]:]),
    )
    for index, expected in enumerate(expecteds):
        batch = sentence_iterable_task_dataset[index]
        assert len(batch) == 2

        ar, at = batch
        er, et = expected
        assert ar.equal(er)
        assert at.equal(et)


@pytest.mark.parametrize('index', (-1, len(BREAKS)))
def test_sentence_iterable_task_dataset_getitem_bad_index(
        sentence_iterable_task_dataset, index):
    """Test SentenceIterableDataset.__getitem__ dies when given bad index."""
    with pytest.raises(IndexError, match='sentence index out of bounds.*'):
        sentence_iterable_task_dataset[index]


def test_sentence_iterable_task_dataset_iter(sentence_iterable_task_dataset,
                                             representations, tags):
    """Test SentenceIterableTaskDataset.__iter__ yields all samples."""
    batches = tuple(iter(sentence_iterable_task_dataset))
    assert len(batches) == 2
    first, second = batches

    assert len(first) == 2
    ar, at = first
    assert ar.equal(representations[BREAKS[0]:BREAKS[1]])
    assert at.equal(tags[BREAKS[0]:BREAKS[1]])

    assert len(second) == 2
    ar, at = second
    assert ar.equal(representations[BREAKS[1]:])
    assert at.equal(tags[BREAKS[1]:])


def test_sentence_iterable_task_dataset_len(sentence_iterable_task_dataset):
    """Test SentenceIterableTaskDataset.__len__ returns number of sentences."""
    assert len(sentence_iterable_task_dataset) == len(BREAKS)


@pytest.fixture
def in_memory_task_dataset(path):
    """Returns a InMemoryTaskDataset for testing."""
    return learning.InMemoryTaskDataset(path,
                                        device=torch.device('cpu'),
                                        breaks_key=BREAKS_KEY,
                                        reps_key=REPS_KEY,
                                        tags_key=TAGS_KEY)


def test_in_memory_task_dataset_getitem(
    in_memory_task_dataset,
    representations,
    tags,
):
    """Test InMemoryTaskDataset.__getitem__ returns the entire dataset."""
    batch = in_memory_task_dataset[0]
    assert len(batch) == 2
    ar, at = batch
    assert ar.equal(representations)
    assert at.equal(tags)


@pytest.mark.parametrize('index', (-1, 1))
def test_in_memory_task_dataset_getitem_bad_index(in_memory_task_dataset,
                                                  index):
    """Test InMemoryTaskDataset.__getitem__ dies when index not 0."""
    with pytest.raises(IndexError, match='.*must be 0.*'):
        in_memory_task_dataset[index]


def test_in_memory_task_dataset_iter(in_memory_task_dataset, representations,
                                     tags):
    """Test InMemoryTaskDataset.__iter__ yields all chunks."""
    batches = tuple(iter(in_memory_task_dataset))
    assert len(batches) == 1

    batch, = batches
    assert len(batch) == 2

    ar, at = batch
    assert ar.equal(representations)
    assert at.equal(tags)


def test_in_memory_task_dataset_len(in_memory_task_dataset):
    """Test InMemoryTaskDataset.__len__ returns correct length."""
    assert len(in_memory_task_dataset) == 1


PATIENCE = 4


def test_early_stopping_init_decreasing():
    """Test EarlyStopping.__init__ records when value should decrease."""
    early_stopping = learning.EarlyStopping(patience=PATIENCE, decreasing=True)
    assert early_stopping.patience == PATIENCE
    assert early_stopping.decreasing is True
    assert early_stopping.best == float('inf')
    assert early_stopping.num_bad == 0


def test_early_stopping_init_increasing():
    """Test EarlyStopping.__init__ records when value should increase."""
    early_stopping = learning.EarlyStopping(
        patience=PATIENCE,
        decreasing=False,
    )
    assert early_stopping.patience == PATIENCE
    assert early_stopping.decreasing is False
    assert early_stopping.best == float('-inf')
    assert early_stopping.num_bad == 0


def test_early_stopping_call_decreasing():
    """Test EarlyStopping.__call__ returns when value does not decrease."""
    early_stopping = learning.EarlyStopping(patience=PATIENCE, decreasing=True)
    assert not early_stopping(-1)
    for i in range(PATIENCE):
        assert not early_stopping(i)
    assert early_stopping(0)


def test_early_stopping_call_increasing():
    """Test EarlyStopping.__call__ reports when value does not increases."""
    early_stopping = learning.EarlyStopping(
        patience=PATIENCE,
        decreasing=False,
    )
    assert not early_stopping(PATIENCE + 1)
    for i in range(PATIENCE):
        assert not early_stopping(i)
    assert early_stopping(0)


EPOCHS = 2


def test_train(task_dataset, mocker):
    """Test train runs without crashing."""
    wandb_log = mocker.patch.object(wandb, 'log')

    probe = nn.Linear(NDIMS, NTAGS)
    before = probe.weight.data.clone()

    learning.train(probe, task_dataset, epochs=EPOCHS, also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)
    assert wandb_log.call_args_list == [
        mocker.call({'train accuracy': mocker.ANY}),
    ] * len(task_dataset) * EPOCHS


def test_train_with_early_stopping(task_dataset, mocker):
    """Test train stops early."""
    wandb_log = mocker.patch.object(wandb, 'log')

    early_stopping = learning.EarlyStopping(patience=PATIENCE)
    # Cannot possible go lower! So we should stop after PATIENCE steps.
    early_stopping(float('-inf'))

    probe = nn.Linear(NDIMS, NTAGS)
    before = probe.weight.data.clone()

    learning.train(probe,
                   task_dataset,
                   epochs=EPOCHS,
                   stopper=early_stopping,
                   also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)
    assert wandb_log.call_args_list == [
        mocker.call({'train accuracy': mocker.ANY}),
    ] * (PATIENCE + 1)


def test_train_with_dev_dataset(task_dataset, mocker):
    """Test train runs without crashing."""
    wandb_log = mocker.patch.object(wandb, 'log')

    probe = nn.Linear(NDIMS, NTAGS)
    before = probe.weight.data.clone()

    learning.train(probe,
                   task_dataset,
                   dev_dataset=task_dataset,
                   epochs=EPOCHS,
                   also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)

    expected = []
    for _ in range(EPOCHS):
        expected.extend([mocker.call({'train accuracy': mocker.ANY})] *
                        len(task_dataset))
        expected.extend([mocker.call({'dev accuracy': mocker.ANY})])
    assert wandb_log.call_args_list == expected


def test_train_with_early_stopping_and_dev_dataset(task_dataset, mocker):
    """Test train stops early."""
    wandb_log = mocker.patch.object(wandb, 'log')

    early_stopping = learning.EarlyStopping(patience=0)
    # Cannot possible go lower! So we should stop after PATIENCE steps.
    early_stopping(float('-inf'))

    probe = nn.Linear(NDIMS, NTAGS)
    before = probe.weight.data.clone()

    learning.train(probe,
                   task_dataset,
                   dev_dataset=task_dataset,
                   epochs=EPOCHS,
                   stopper=early_stopping,
                   also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)

    expected = [
        mocker.call({'train accuracy': mocker.ANY}),
    ] * len(task_dataset)
    expected.append(mocker.call({'dev accuracy': mocker.ANY}))
    assert wandb_log.call_args_list == expected


def test_test(task_dataset, tags):
    """Test test returns expected accuracy."""
    tag = tags[0]
    expected = tags.eq(tag).sum().item() / NSAMPLES

    class FakeModule(nn.Module):
        """Always returns the same prediction."""

        def forward(self, reps):
            """Just returns the tag."""
            assert reps.shape == (1, NDIMS)
            logits = torch.zeros(1, NTAGS)
            logits[0, tag] = 1
            return logits

    actual = learning.test(FakeModule(),
                           task_dataset,
                           device=torch.device('cpu'))
    assert actual == expected
