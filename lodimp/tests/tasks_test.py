"""Unit tests for the tasks module."""

from lodimp import ptb, tasks

import torch

SAMPLES = [
    ptb.Sample(
        ('foo', 'bar', 'baz'),
        ('A', 'B', 'C'),
        (-1, 0, 0),
        ('root', 'first', 'second'),
    ),
    ptb.Sample(
        ('boof', 'foo'),
        ('D', 'A'),
        (1, -1),
        ('first', 'root'),
    ),
]

POS_TAGS = (
    torch.tensor([0, 1, 2]),
    torch.tensor([3, 0]),
)


def test_pos_task_init():
    """Test POSTask.__init__ assigns tags as expected."""
    task = tasks.POSTask(SAMPLES)
    assert task.indexer == {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'UNK': 4,
    }


def test_pos_task_init_tags():
    """Test POSTask.__init__ indexes correctly when given tags=..."""
    task = tasks.POSTask(SAMPLES, tags={'A', 'B', 'C'})
    assert task.indexer == {'A': 0, 'B': 1, 'C': 2, 'UNK': 3}


def test_pos_task_call():
    """Test POSTask.__call__ tags samples correctly."""
    task = tasks.POSTask(SAMPLES)
    for sample, expected in zip(SAMPLES, POS_TAGS):
        actual = task(sample)
        assert torch.equal(actual, expected)


def test_pos_task_call_unknown_tag():
    """Test POSTask.__call__ handles unknown tags correctly."""
    task = tasks.POSTask(SAMPLES)
    actual = task(ptb.Sample(('foo',), ('blah',), (-1,), ('root',)))
    assert actual.equal(torch.tensor([4]))


def test_pos_task_len():
    """Test POSTask.__len__ returns correct number of tags."""
    task = tasks.POSTask(SAMPLES)
    assert len(task) == 5


def test_control_pos_task_init():
    """Test ControlPOSTask.__init__ sets state correctly."""
    task = tasks.ControlPOSTask(SAMPLES)
    assert list(task.dist) == [0.4, 0.2, 0.2, 0.2]
    assert task.tags.keys() == {'foo', 'bar', 'baz', 'boof'}


def test_control_pos_task_call():
    """Test ControlPOSTask.__call__ tags sensibly."""
    task = tasks.ControlPOSTask(SAMPLES)
    for sample in SAMPLES:
        actual = task(sample)
        assert len(actual) == len(sample.sentence)


def test_control_pos_task_len():
    """Test ControlPOSTask.__len__ returns correct number of tags."""
    task = tasks.ControlPOSTask(SAMPLES)
    assert len(task) == 4


DEPENDENCY_ARCS = (torch.tensor([
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
]), torch.tensor([
    [0, 1],
    [0, 0],
]))


def test_dependency_arc_task_call():
    """Test DependencyArcTask.__call__ captures all arcs."""
    task = tasks.DependencyArcTask(SAMPLES)
    for sample, expected in zip(SAMPLES, DEPENDENCY_ARCS):
        actual = task(sample)
        assert torch.equal(actual, expected)


def test_dependency_arc_task_len():
    """Test DependencyArcTask.__len__ returns 1."""
    task = tasks.DependencyArcTask(SAMPLES)
    assert len(task) == 1


DEPENDENCY_LABELS = (
    torch.tensor([
        [1, 3, 3],
        [0, 3, 3],
        [2, 3, 3],
    ]),
    torch.tensor([
        [3, 0],
        [3, 1],
    ]),
)


def test_dependency_label_task_init():
    """Test DependencyLabelTask.__init__ maps labels to integers correctly."""
    task = tasks.DependencyLabelTask(SAMPLES)
    assert task.indexer == {
        'first': 0,
        'root': 1,
        'second': 2,
        'unk': 3,
    }


def test_dependency_label_task_init_relations():
    """Test DependencyLabelTask.__init__ filters relations when provided."""
    task = tasks.DependencyLabelTask(SAMPLES, relations={'first'})
    assert task.indexer == {
        'first': 0,
        'unk': 1,
    }


def test_dependency_label_task_call():
    """Test DependencyLabelTask.__call__ returns correct label matrix."""
    task = tasks.DependencyLabelTask(SAMPLES)
    for sample, expected in zip(SAMPLES, DEPENDENCY_LABELS):
        actual = task(sample)
        assert torch.equal(actual, expected)


def test_dependency_label_task_len():
    """Test DependencyLabelTask.__len__ returns number of labels."""
    task = tasks.DependencyLabelTask(SAMPLES)
    assert len(task) == 4
