"""Unit tests for tasks/dlp module."""

import pathlib
import tempfile

from lodimp.common.data import ptb, representations
from lodimp.tasks import dlp

import h5py
import pytest
import torch

ROOT = 'root'
LABELS = (
    'label-a',
    'label-b',
    ROOT,
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
        ('irrelevant', 'irrelevant', 'irrelevant'),
        (2, 0, -1),
        (LABELS[0], LABELS[1], ROOT),
    ),
    ptb.Sample(
        (WORDS[3], WORDS[4]),
        ('irrelevant', 'irrelevant'),
        (-1, 0),
        (ROOT, LABELS[0]),
    ),
)
SEQ_LENGTHS = tuple(len(sample.sentence) for sample in SAMPLES)

INDEXED_LABELS = (
    torch.tensor([
        [0, 0, 1],
        [2, 0, 0],
        [0, 0, 3],
    ]),
    torch.tensor([
        [3, 0],
        [1, 0],
    ]),
)


@pytest.fixture
def dlp_indexer():
    """Returns a DLPIndexer for testing."""
    return dlp.DLPIndexer(SAMPLES)


def test_dlp_indexer_init(dlp_indexer):
    """Test DLPIndexer.__init__ maps labels to integers correctly."""
    assert dlp_indexer.indexer == {
        dlp.UNK: 0,
        LABELS[0]: 1,
        LABELS[1]: 2,
        LABELS[2]: 3,
    }


def test_dlp_indexer_call(dlp_indexer):
    """Test DLPIndexer.__call__ returns correct label matrix."""
    for sample, expected in zip(SAMPLES, INDEXED_LABELS):
        actual = dlp_indexer(sample)
        assert torch.equal(actual, expected)


def test_dlp_indexer_len(dlp_indexer):
    """Test DLPIndexer.__len__ returns number of labels."""
    assert len(dlp_indexer) == len(LABELS) + 1


@pytest.fixture
def control_dlp_indexer():
    """Returns a ControlDLPIndexer for testing."""
    return dlp.ControlDLPIndexer(SAMPLES)


def test_control_dlp_indexer_init(control_dlp_indexer):
    """Test ControlDLPIndexer.__init__ maps labels to integers."""
    assert 0 not in control_dlp_indexer.rels
    assert len(control_dlp_indexer.dist) == 3
    assert len(control_dlp_indexer.rels) == 5


def test_control_dlp_indexer_call(control_dlp_indexer):
    """Test ControlDLPIndexer.__call__ returns reasonable labels."""
    for sample, expected in zip(SAMPLES, INDEXED_LABELS):
        actual = control_dlp_indexer(sample)
        assert actual.shape == expected.shape
        assert actual[expected == 0].eq(0).all()
        assert not actual[expected != 0].eq(0).any()


def test_control_dlp_indexer_call_deterministic(control_dlp_indexer):
    """Test ControlDLPIndexer.__call__ is deterministic."""
    expecteds = [control_dlp_indexer(sample) for sample in SAMPLES]
    actuals = [control_dlp_indexer(sample) for sample in SAMPLES]
    for actual, expected in zip(actuals, expecteds):
        assert actual.equal(expected)


def test_control_dlp_indexer_len(control_dlp_indexer):
    """Test ControlDLPIndexer.__len__ returns correct length."""
    task = dlp.ControlDLPIndexer(SAMPLES)
    assert len(task) == len(LABELS) + 1


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
def dlp_task_dataset(rep_layer_dataset):
    """Returns a DLPTaskDataset for testing."""
    return dlp.DLPTaskDataset(rep_layer_dataset, SAMPLES)


@pytest.fixture
def rep_pairs(rep_layer_dataset):
    """Returns the expected representation pairs for each sentence."""
    return (
        torch.stack((
            torch.stack((rep_layer_dataset[0][0], rep_layer_dataset[0][2])),
            torch.stack((rep_layer_dataset[0][1], rep_layer_dataset[0][0])),
            torch.stack((rep_layer_dataset[0][2], rep_layer_dataset[0][2])),
        )),
        torch.stack((
            torch.stack((rep_layer_dataset[1][0], rep_layer_dataset[1][0])),
            torch.stack((rep_layer_dataset[1][1], rep_layer_dataset[1][0])),
        )),
    )


FLATTENED_INDEXED_LABELS = (torch.tensor((1, 2, 3)), torch.tensor((3, 1)))


def test_dlp_task_dataset_iter(dlp_task_dataset, rep_pairs):
    """Test DLPTaskDataset.__iter__ yields all samples."""
    actuals = list(iter(dlp_task_dataset))

    expecteds = list(zip(rep_pairs, FLATTENED_INDEXED_LABELS))
    for actual, expected in zip(actuals, expecteds):
        assert len(actual) == 2
        assert len(expected) == 2, 'sanity check!'

        actual_reps, actual_tags = actual
        expected_reps, expected_tags = expected
        assert actual_reps.equal(expected_reps)
        assert actual_tags.equal(expected_tags)


def test_dlp_task_dataset_getitem(dlp_task_dataset, rep_pairs):
    """Test DLPTaskDataset.__getitem__ returns correct samples."""
    expecteds = list(zip(rep_pairs, FLATTENED_INDEXED_LABELS))
    for index, expected in enumerate(expecteds):
        assert len(expected) == 2, 'sanity check!'
        expected_reps, expected_tags = expected

        actual = dlp_task_dataset[index]
        assert len(actual) == 2

        actual_reps, actual_tags = actual
        assert actual_reps.equal(expected_reps)
        assert actual_tags.equal(expected_tags)


def test_dlp_task_dataset_len(dlp_task_dataset):
    """Test DLPTaskDataset.__len__ returns number of sentences."""
    return len(dlp_task_dataset) == len(SEQ_LENGTHS)


def test_dlp_task_dataset_sample_representations_shape(dlp_task_dataset):
    """Test DLPTaskDataset.sample_representations_shape returns rep dim."""
    assert dlp_task_dataset.sample_representations_shape == (2, N_DIMS_PER_REP)


def test_dlp_task_dataset_sample_features_shape(dlp_task_dataset):
    """Test DLPTaskDataset.sample_features_shape returns ()."""
    assert dlp_task_dataset.sample_features_shape == ()


def test_dlp_task_dataset_count_samples(dlp_task_dataset):
    """Test DLPTaskDataset.count_samples returns number of words."""
    assert dlp_task_dataset.count_samples() == sum(SEQ_LENGTHS)


def test_dlp_task_dataset_count_unique_features(dlp_task_dataset):
    """Test DLPTaskDataset.count_samples returns number of words."""
    assert dlp_task_dataset.count_unique_features() == len(LABELS) + 1


def test_dlp_task_dataset_init_bad_lengths(rep_layer_dataset):
    """Test DLPTaskDataset.__init__ dies when given different reps/annos."""
    with pytest.raises(ValueError, match='.*1 annotation.*'):
        dlp.DLPTaskDataset(rep_layer_dataset, SAMPLES[:1])
