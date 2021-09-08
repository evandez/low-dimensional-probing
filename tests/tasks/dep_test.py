"""Unit tests for the tasks/dep module."""
import pathlib
import tempfile

from lodimp.parse import ptb, representations
from lodimp.tasks import dep

import h5py
import pytest
import torch

WORDS = (
    'word-a',
    'word-b',
    'word-c',
    'word-d',
    'word-e',
)

SAMPLES = (
    ptb.Sample(
        (WORDS[0], WORDS[1], WORDS[2]),
        ('irrelevant', 'irrelevant', 'irrelevant'),
        (2, 0, -1),
        ('irrelevant', 'irrelevant', 'irrelevant'),
    ),
    ptb.Sample(
        (WORDS[3], WORDS[4]),
        ('irrelevant', 'irrelevant'),
        (-1, 0),
        ('irrelevant', 'irrelevant'),
    ),
)
SEQ_LENGTHS = tuple(len(sample.sentence) for sample in SAMPLES)

INDEXED_EDGES = (torch.tensor([2, 0, 2]), torch.tensor([0, 0]))


@pytest.fixture
def dep_indexer():
    """Returns a DEPIndexer for testing."""
    return dep.DEPIndexer()


def test_dep_indexer_call(dep_indexer):
    """Test DEPIndexer.__call__ captures all arcs."""
    for sample, expected in zip(SAMPLES, INDEXED_EDGES):
        actual = dep_indexer(sample)
        assert torch.equal(actual, expected)


@pytest.fixture
def control_dep_indexer():
    """Returns a ControlDEPIndexer for testing."""
    return dep.ControlDEPIndexer(SAMPLES)


def test_control_dep_indexer_init(control_dep_indexer):
    """Test ControlDEPIndexer.__init__ sets state correctly."""
    assignments = sum((
        len(control_dep_indexer.attach_to_self),
        len(control_dep_indexer.attach_to_first),
        len(control_dep_indexer.attach_to_last),
    ))
    assert assignments == len(WORDS)


def test_control_dep_indexer_call(control_dep_indexer):
    """Test ControlDEPIndexer.__call__ returns reasonable labels."""
    for sample in SAMPLES:
        labels = control_dep_indexer(sample)
        for index, label in enumerate(labels):
            assert label == index or label == 0 or label == len(
                sample.sentence) - 1


def test_control_dep_indexer_call_deterministic(control_dep_indexer):
    """Test ControlDEPIndexer.__call__ is deterministic."""
    expecteds = [control_dep_indexer(sample) for sample in SAMPLES]
    actuals = [control_dep_indexer(sample) for sample in SAMPLES]
    for actual, expected in zip(actuals, expecteds):
        assert actual.equal(expected)


N_LAYERS = 3
N_DIMS_PER_REP = 10


@pytest.fixture
def reps():
    """Returns fake representations for testing."""
    return tuple(
        torch.randn(N_LAYERS, seq_length, N_DIMS_PER_REP)
        for seq_length in SEQ_LENGTHS)


LAYER = 0


@pytest.yield_fixture
def rep_layer_dataset(reps):
    """Yields the path to a standard h5 file containing representations."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'reps.h5'
        with h5py.File(path, 'w') as file:
            for index, reps in enumerate(reps):
                file.create_dataset(str(index), data=reps)
            file.create_dataset('ignore', data=0)
        yield representations.RepresentationDataset(path).layer(LAYER)


@pytest.fixture
def dep_task_dataset(rep_layer_dataset):
    """Returns a DEPTaskDataset for testing."""
    return dep.DEPTaskDataset(rep_layer_dataset, SAMPLES)


def test_dep_task_dataset_iter(dep_task_dataset, rep_layer_dataset):
    """Test DEPTaskDataset.__iter__ yields all samples."""
    actuals = list(iter(dep_task_dataset))
    expecteds = list(zip(list(rep_layer_dataset), INDEXED_EDGES))
    for actual, expected in zip(actuals, expecteds):
        assert len(actual) == 2
        assert len(expected) == 2, 'sanity check!'

        actual_reps, actual_tags = actual
        expected_reps, expected_tags = expected
        assert actual_reps.equal(expected_reps)
        assert actual_tags.equal(expected_tags)


def test_dep_task_dataset_getitem(dep_task_dataset, rep_layer_dataset):
    """Test DEPTaskDataset.__getitem__ returns correct samples."""
    expecteds = list(zip(list(rep_layer_dataset), INDEXED_EDGES))
    for index, expected in enumerate(expecteds):
        assert len(expected) == 2, 'sanity check!'
        expected_reps, expected_tags = expected

        actual = dep_task_dataset[index]
        assert len(actual) == 2

        actual_reps, actual_tags = actual
        assert actual_reps.equal(expected_reps)
        assert actual_tags.equal(expected_tags)


def test_dep_task_dataset_len(dep_task_dataset):
    """Test DEPTaskDataset.__len__ returns number of sentences."""
    return len(dep_task_dataset) == len(SEQ_LENGTHS)


def test_dep_task_dataset_sample_representations_shape(dep_task_dataset):
    """Test DEPTaskDataset.sample_representations_shape returns rep dim."""
    assert dep_task_dataset.sample_representations_shape == (N_DIMS_PER_REP,)


def test_dep_task_dataset_sample_features_shape(dep_task_dataset):
    """Test DEPTaskDataset.sample_features_shape returns ()."""
    assert dep_task_dataset.sample_features_shape == ()


def test_pos_task_dataset_count_samples(dep_task_dataset):
    """Test DEPTaskDataset.count_samples returns number of words."""
    assert dep_task_dataset.count_samples() == sum(SEQ_LENGTHS)


def test_dep_task_dataset_count_unique_features(dep_task_dataset):
    """Test DEPTaskDataset.count_samples returns None."""
    assert dep_task_dataset.count_unique_features() is None


def test_dep_task_dataset_init_bad_lengths(rep_layer_dataset):
    """Test DEPTaskDataset.__init__ dies when given different reps/annos."""
    with pytest.raises(ValueError, match='.*1 annotation.*'):
        dep.DEPTaskDataset(rep_layer_dataset, SAMPLES[:1])
