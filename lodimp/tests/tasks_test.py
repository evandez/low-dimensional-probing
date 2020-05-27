"""Unit tests for the tasks module."""

from lodimp import tasks
from lodimp.common.data import ptb

import torch

SAMPLES = [
    ptb.Sample(
        ('foo', 'bar', 'baz'),
        ('A', 'B', 'C'),
        (2, 0, -1),
        ('root', 'first', 'second'),
    ),
    ptb.Sample(
        ('boof', 'biff'),
        ('D', 'A'),
        (-1, 0),
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
    assert task.tags.keys() == {'foo', 'bar', 'baz', 'boof', 'biff'}


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


DEPENDENCY_ARCS = (torch.tensor([2, 0, 2]), torch.tensor([0, 0]))


def test_dependency_arc_task_call():
    """Test DependencyArcTask.__call__ captures all arcs."""
    task = tasks.DependencyArcTask(SAMPLES)
    for sample, expected in zip(SAMPLES, DEPENDENCY_ARCS):
        actual = task(sample)
        assert torch.equal(actual, expected)


def test_control_dependency_arc_task_init():
    """Test ControlDependencyArcTask.__init__ sets state correctly."""
    task = tasks.ControlDependencyArcTask(SAMPLES)

    assignments = sum((
        len(task.attach_to_self),
        len(task.attach_to_first),
        len(task.attach_to_last),
    ))
    assert assignments == 5


def test_control_dependency_arc_task_call():
    """Test ControlDependencyArcTask.__call__ returns reasonable labels."""
    task = tasks.ControlDependencyArcTask(SAMPLES)

    for sample in SAMPLES:
        labels = task(sample)
        for index, label in enumerate(labels):
            assert label == index or label == 0 or label == len(
                sample.sentence) - 1


def test_control_dependency_arc_task_call_deterministic():
    """Test ControlDependencyArcTask.__call__ is deterministic."""
    task = tasks.ControlDependencyArcTask(SAMPLES)
    expecteds = [task(sample) for sample in SAMPLES]
    actuals = [task(sample) for sample in SAMPLES]
    for actual, expected in zip(actuals, expecteds):
        assert actual.equal(expected)


DEPENDENCY_LABELS = (
    torch.tensor([
        [0, 0, 2],
        [1, 0, 0],
        [0, 0, 3],
    ]),
    torch.tensor([
        [1, 0],
        [2, 0],
    ]),
)


def test_dependency_label_task_init():
    """Test DependencyLabelTask.__init__ maps labels to integers correctly."""
    task = tasks.DependencyLabelTask(SAMPLES)
    assert task.indexer == {
        'unk': 0,
        'first': 1,
        'root': 2,
        'second': 3,
    }


def test_dependency_label_task_init_relations():
    """Test DependencyLabelTask.__init__ filters relations when provided."""
    task = tasks.DependencyLabelTask(SAMPLES, relations={'first'})
    assert task.indexer == {
        'unk': 0,
        'first': 1,
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


def test_control_dependency_label_task_init():
    """Test ControlDependencyLabelTask.__init__ maps labels to integers."""
    task = tasks.ControlDependencyLabelTask(SAMPLES)
    assert 0 not in task.rels
    assert len(task.dist) == 3
    assert len(task.rels) == 5


def test_control_dependency_label_task_call():
    """Test ControlDependencyLabelTask.__call__ returns reasonable labels."""
    task = tasks.ControlDependencyLabelTask(SAMPLES)
    for sample, expected in zip(SAMPLES, DEPENDENCY_LABELS):
        actual = task(sample)
        assert actual.shape == expected.shape
        assert actual[expected == 0].eq(0).all()
        assert not actual[expected != 0].eq(0).any()


def test_control_dependency_label_task_call_deterministic():
    """Test ControlDependencyLabelTask.__call__ is deterministic."""
    task = tasks.ControlDependencyLabelTask(SAMPLES)
    expecteds = [task(sample) for sample in SAMPLES]
    actuals = [task(sample) for sample in SAMPLES]
    for actual, expected in zip(actuals, expecteds):
        assert actual.equal(expected)


def test_control_dependency_label_task_len():
    """Test ControlDependencyLabelTask.__len__ returns correct length."""
    task = tasks.ControlDependencyLabelTask(SAMPLES)
    assert len(task) == 4
