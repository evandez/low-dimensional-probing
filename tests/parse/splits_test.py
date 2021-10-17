"""Unit tests for splits module."""
import pathlib
import tempfile

from ldp.parse import splits

import pytest


@pytest.yield_fixture
def paths():
    """Yields fake (representation, annotation) path pair for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        root = pathlib.Path(tempdir)

        representations = root / 'train_reps.h5'
        representations.touch()

        annotations = root / 'train_annotations.conll'
        annotations.touch()

        yield representations, annotations


def test_ensure(paths):
    """Test ensure returns correct Split tuple."""
    representations, annotations = paths
    actual = splits.ensure(representations, annotations)
    assert actual.representations == representations
    assert actual.annotations == annotations


def test_ensure_str_paths(paths):
    """Test ensure returns correct Split tuple when given string paths."""
    representations, annotations = paths
    actual = splits.ensure(str(representations), str(annotations))
    assert actual.representations == representations
    assert actual.annotations == annotations


@pytest.mark.parametrize('delete', (0, 1))
def test_ensure_bad_paths(paths, delete):
    """Test ensure explodes when one or more paths does not exist."""
    deleted = paths[delete]
    deleted.unlink()
    with pytest.raises(FileNotFoundError, match=f'.*not found: {deleted}.*'):
        splits.ensure(*paths)


@pytest.fixture
def representations_by_split(paths):
    """Returns dict mapping split key to fake representations paths."""
    train_reps, _ = paths
    test_reps = train_reps.parent / 'test_reps.h5'
    test_reps.touch()
    return {splits.TRAIN: train_reps, splits.TEST: test_reps}


@pytest.fixture
def annotations_by_split(paths):
    """Returns dict mapping split key to fake annotation paths."""
    _, train_annotations = paths
    test_annotations = train_annotations.parent / 'test_annotations.conll'
    test_annotations.touch()
    return {splits.TRAIN: train_annotations, splits.TEST: test_annotations}


def test_join(representations_by_split, annotations_by_split):
    """Test join produces correct splits in basic case."""
    actual = splits.join(representations_by_split, annotations_by_split)
    assert actual.keys() == {splits.TRAIN, splits.TEST}
    for key in (splits.TRAIN, splits.TEST):
        split = actual[key]
        assert split.representations == representations_by_split[key]
        assert split.annotations == annotations_by_split[key]


def test_join_no_validate(representations_by_split, annotations_by_split):
    """Test join produces correct splits even when it does not validate."""
    representations_by_split[splits.TRAIN].unlink()
    actual = splits.join(representations_by_split,
                         annotations_by_split,
                         validate=False)
    assert actual.keys() == {splits.TRAIN, splits.TEST}
    for key in (splits.TRAIN, splits.TEST):
        split = actual[key]
        assert split.representations == representations_by_split[key]
        assert split.annotations == annotations_by_split[key]


@pytest.mark.parametrize('validate', (True, False))
def test_join_mismatched_keys(representations_by_split, annotations_by_split,
                              validate):
    """Test join dies when reps/anno dictionaries have different keys."""
    del representations_by_split[splits.TRAIN]
    with pytest.raises(ValueError, match='reps have splits.*'):
        splits.join(representations_by_split,
                    annotations_by_split,
                    validate=validate)


def test_join_with_root(representations_by_split, annotations_by_split):
    """Test join adjusts paths correctly when given root."""
    root = representations_by_split[splits.TRAIN].parent
    for key in (splits.TRAIN, splits.TEST):
        representations_by_split[key] = representations_by_split[key].name
        annotations_by_split[key] = annotations_by_split[key].name
    actual = splits.join(representations_by_split,
                         annotations_by_split,
                         root=root)
    assert actual.keys() == {splits.TRAIN, splits.TEST}
    for key in (splits.TRAIN, splits.TEST):
        split = actual[key]
        assert split.representations == root / representations_by_split[key]
        assert split.annotations == root / annotations_by_split[key]


def test_join_bad_root(representations_by_split, annotations_by_split):
    """Test join validates root when validate=True."""
    root = 'fake!'
    with pytest.raises(FileNotFoundError, match=f'.*root {root}.*'):
        splits.join(representations_by_split, annotations_by_split, root=root)


def test_join_bad_root_no_validate():
    """Test join does not die when given bad root and validate=False."""
    root = pathlib.Path('root')
    reps = pathlib.Path('train.h5')
    annos = pathlib.Path('annotations.h5')
    actual = splits.join({splits.TRAIN: reps}, {splits.TRAIN: annos},
                         root=root,
                         validate=False)
    assert actual == {splits.TRAIN: splits.Split(root / reps, root / annos)}
