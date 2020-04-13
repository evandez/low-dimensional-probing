"""Unit tests for the tasks module."""

from lodimp import ptb, tasks

import torch

SAMPLES = [
    ptb.Sample(['foo', 'bar', 'baz'], ['A', 'B', 'C']),
    ptb.Sample(['boof', 'foo'], ['D', 'A']),
]

REAL_TAGS = [
    torch.tensor([0, 1, 2]),
    torch.tensor([3, 0]),
]


def test_ptb_real_pos_init():
    """Test PTBRealPOS.__init__ assigns tags as expected."""
    task = tasks.PTBRealPOS(SAMPLES)
    assert task.indexer == {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }


def test_ptb_real_pos_call():
    """Test PTBRealPOS.__call__ tags samples correctly."""
    task = tasks.PTBRealPOS(SAMPLES)
    for sample, expected in zip(SAMPLES, REAL_TAGS):
        actual = task(sample)
        assert torch.equal(actual, expected)


def test_ptb_control_pos_init():
    """Test PTBControlPOS.__init__ sets state correctly."""
    task = tasks.PTBControlPOS(SAMPLES)
    assert list(task.dist) == [0.4, 0.2, 0.2, 0.2]
    assert task.tags.keys() == {'foo', 'bar', 'baz', 'boof'}


def test_ptb_control_pos_call():
    """Test PTBControlPOs.__call__ tags sensibly."""
    task = tasks.PTBControlPOS(SAMPLES)
    for sample in SAMPLES:
        actual = task(sample)
        assert len(actual) == len(sample.sentence)
