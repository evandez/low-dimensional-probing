"""Unit tests for the tasks/pos module."""

import pathlib
import tempfile

from lodimp.common.data import ptb, representations
from lodimp.tasks import pos

import h5py
import pytest
import torch

TAGS = (
    'tag-a',
    'tag-b',
    'tag-c',
    'tag-d',
)

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
        (TAGS[0], TAGS[1], TAGS[2]),
        (-1, -1, -1),
        ('irrelevant', 'irrelevant', 'irrelevant'),
    ),
    ptb.Sample(
        (WORDS[3], WORDS[4]),
        (TAGS[3], TAGS[0]),
        (-1, -1),
        ('irrelevant', 'irrelevant'),
    ),
)
SEQ_LENGTHS = tuple(len(sample.sentence) for sample in SAMPLES)

INDEXED_TAGS = (
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 1]),
)


@pytest.fixture
def pos_indexer():
    """Returns a POSIndexer for testing."""
    return pos.POSIndexer(SAMPLES)


def test_pos_task_init(pos_indexer):
    """Test POSTask.__init__ assigns tags as expected."""
    assert pos_indexer.indexer == {
        pos.UNK: 0,
        TAGS[0]: 1,
        TAGS[1]: 2,
        TAGS[2]: 3,
        TAGS[3]: 4,
    }


def test_pos_indexer_init_distinguished():
    """Test POSIndexer.__init__ indexes correctly when given tags=..."""
    pos_indexer = pos.POSIndexer(SAMPLES, distinguish=(*TAGS[:2],))
    assert pos_indexer.indexer == {pos.UNK: 0, TAGS[0]: 1, TAGS[1]: 2}


def test_pos_indexer_call(pos_indexer):
    """Test POSIndexer.__call__ tags samples correctly."""
    for sample, expected in zip(SAMPLES, INDEXED_TAGS):
        actual = pos_indexer(sample)
        assert torch.equal(actual, expected)


def test_pos_indexer_call_unknown_tag(pos_indexer):
    """Test POSIndexer.__call__ handles unknown tags correctly."""
    actual = pos_indexer(ptb.Sample(('foo',), ('blah',), (-1,), ('root',)))
    assert actual.equal(torch.tensor([0]))


def test_pos_indexer_len(pos_indexer):
    """Test POSTask.__len__ returns correct number of tags."""
    assert len(pos_indexer) == len(TAGS) + 1


@pytest.fixture
def control_pos_indexer():
    """Returns a ControlPOSIndexer for testing."""
    return pos.ControlPOSIndexer(SAMPLES)


def test_control_pos_indexer_init(control_pos_indexer):
    """Test ControlPOSIndexer.__init__ sets state correctly."""
    assert list(control_pos_indexer.dist) == [0.4, 0.2, 0.2, 0.2]
    assert control_pos_indexer.tags.keys() == set(WORDS)


def test_control_pos_indexer_call(control_pos_indexer):
    """Test ControlPOSIndexer.__call__ tags sensibly."""
    for sample in SAMPLES:
        actual = control_pos_indexer(sample)
        assert len(actual) == len(sample.sentence)


def test_control_pos_indexer_call_unknown_tag(control_pos_indexer):
    """Test ControlPOSIndexer.__call__ handles unknown tags correctly."""
    actual = control_pos_indexer(
        ptb.Sample(('foo',), ('blah',), (-1,), ('root',)))
    assert actual.equal(torch.tensor([0]))


def test_control_pos_task_len(control_pos_indexer):
    """Test ControlPOSIndexer.__len__ returns correct number of tags."""
    assert len(control_pos_indexer) == len(TAGS) + 1


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
def pos_task_dataset(rep_layer_dataset, pos_indexer):
    """Returns a POSTaskDataset for testing."""
    return pos.POSTaskDataset(rep_layer_dataset, SAMPLES, pos_indexer)


def test_pos_task_dataset_iter(pos_task_dataset, rep_layer_dataset):
    """Test POSTaskDataset.__iter__ yields all samples."""
    actuals = list(iter(pos_task_dataset))
    expecteds = list(zip(list(rep_layer_dataset), INDEXED_TAGS))
    for actual, expected in zip(actuals, expecteds):
        assert len(actual) == 2
        assert len(expected) == 2, 'sanity check!'

        actual_reps, actual_tags = actual
        expected_reps, expected_tags = expected
        assert actual_reps.equal(expected_reps)
        assert actual_tags.equal(expected_tags)


def test_pos_task_dataset_getitem(pos_task_dataset, rep_layer_dataset):
    """Test POSTaskDataset.__getitem__ returns correct samples."""
    expecteds = list(zip(list(rep_layer_dataset), INDEXED_TAGS))
    for index, expected in enumerate(expecteds):
        assert len(expected) == 2, 'sanity check!'
        expected_reps, expected_tags = expected

        actual = pos_task_dataset[index]
        assert len(actual) == 2

        actual_reps, actual_tags = actual
        assert actual_reps.equal(expected_reps)
        assert actual_tags.equal(expected_tags)


def test_pos_task_dataset_len(pos_task_dataset):
    """Test POSTaskDataset.__len__ returns number of sentences."""
    return len(pos_task_dataset) == len(SEQ_LENGTHS)


def test_pos_task_dataset_sample_representations_shape(pos_task_dataset):
    """Test POSTaskDataset.sample_representations_shape returns rep dim."""
    assert pos_task_dataset.sample_representations_shape == (N_DIMS_PER_REP,)


def test_pos_task_dataset_sample_features_shape(pos_task_dataset):
    """Test POSTaskDataset.sample_features_shape returns ()."""
    assert pos_task_dataset.sample_features_shape == ()


def test_pos_task_dataset_count_samples(pos_task_dataset):
    """Test POSTaskDataset.count_samples returns number of words."""
    assert pos_task_dataset.count_samples() == sum(SEQ_LENGTHS)


def test_pos_task_dataset_count_unique_features(pos_task_dataset):
    """Test POSTaskDataset.count_samples returns number of words."""
    assert pos_task_dataset.count_unique_features() == len(TAGS) + 1


def test_pos_task_dataset_init_bad_lengths(rep_layer_dataset, pos_indexer):
    """Test POSTaskDataset.__init__ dies when given different reps/annos."""
    with pytest.raises(ValueError, match='.*1 annotation.*'):
        pos.POSTaskDataset(rep_layer_dataset, SAMPLES[:1], pos_indexer)
