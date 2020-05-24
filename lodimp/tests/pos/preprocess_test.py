"""Unit and functional tests for the preprocess module."""

import pathlib
import tempfile

from lodimp.common.data import ptb, representations, splits
from lodimp.pos import preprocess

import h5py
import numpy as np
import pytest
import torch

CONLLX = '''\
1       The     _       DET     DT      _       2       det     _       _
2       company _       NOUN    NN      _       3       nsubj   _       _
3       expects _       VERB    VBZ     _       0       root    _       _
4       earnings _       NOUN    NNS     _       16      nsubj   _       _

1       He      _       PRON    PRP     _       3       nsubjpass       _     _
2       was     _       AUX     VBD     _       3       auxpass _       _
3       named   _       VERB    VBN     _       0       root    _       _
4       chief   _       ADJ     JJ      _       6       amod    _       _

1       He      _       PRON    PRP     _       3       nsubjpass       _     _
2       took    _       VERB    VBD     _       0       root    _       _
'''

INDEXER = {
    'UNK': 0,
    'DT': 1,
    'JJ': 2,
    'NN': 3,
    'NNS': 4,
    'PRP': 5,
    'VBD': 6,
    'VBN': 7,
    'VBZ': 8,
}

POS_INDEXES = (
    torch.tensor([1, 3, 8, 4], dtype=torch.long),
    torch.tensor([5, 6, 7, 2], dtype=torch.long),
    torch.tensor([5, 6], dtype=torch.long),
)


@pytest.yield_fixture
def annotations_path():
    """Yields the path to a fake PTB .conllx file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'annotations.conllx'
        with path.open('w') as file:
            file.write(CONLLX)
        yield path


@pytest.fixture
def samples(annotations_path):
    """Returns a sequence of fake ptb.Sample objects."""
    return ptb.load(annotations_path)


@pytest.fixture
def pos_task(samples):
    """Returns a POSTask for testing."""
    return preprocess.POSTask(samples)


def test_pos_task_init(pos_task):
    """Test POSTask.__init__ assigns tags as expected."""
    assert pos_task.indexer == INDEXER


RESTRICTED_TAGS = ('DT', 'VBZ')
RESTRICTED_INDEXER = {'UNK': 0, 'DT': 1, 'VBZ': 2}
RESTRICTED_POS_INDEXES = (
    torch.tensor([1, 0, 2, 0], dtype=torch.long),
    torch.tensor([0, 0, 0, 0], dtype=torch.long),
    torch.tensor([0, 0], dtype=torch.long),
)


def test_pos_task_init_tags(samples):
    """Test POSTask.__init__ indexes correctly when given tags=..."""
    task = preprocess.POSTask(samples, tags=RESTRICTED_TAGS)
    assert task.indexer == RESTRICTED_INDEXER


def test_pos_task_call(samples, pos_task):
    """Test POSTask.__call__ tags samples correctly."""
    for sample, expected in zip(samples, POS_INDEXES):
        actual = pos_task(sample)
        assert torch.equal(actual, expected)


def test_pos_task_call_unknown_tag(pos_task):
    """Test POSTask.__call__ handles unknown tags correctly."""
    actual = pos_task(ptb.Sample(('foo',), ('blah',), (-1,), ('root',)))
    assert actual.equal(torch.tensor([0]))


def test_pos_task_len(pos_task):
    """Test POSTask.__len__ returns correct number of tags."""
    assert len(pos_task) == len(INDEXER)


@pytest.fixture
def pos_tags_dataset(samples, pos_task):
    """Returns a POSTagsDataset for testing."""
    return preprocess.POSTagsDataset(samples, pos_task)


def test_pos_tags_dataset_getitem(pos_tags_dataset):
    """Test POSTagsDataset.__getitem__ returns correct tags in order."""
    for index, expected in enumerate(POS_INDEXES):
        actual = pos_tags_dataset[index]
        assert actual.equal(expected)


def test_pos_tags_dataset_len(pos_tags_dataset):
    """Test POSTagsDataset.__len__ returns number of samples."""
    assert len(pos_tags_dataset) == len(POS_INDEXES)


REPS_DIMENSION = 1024
REPS_LAYERS = 3
SEQ_LENGTHS = (4, 4, 2)

BREAKS = (0, 4, 8)


@pytest.fixture
def breaks():
    """Returns fake breaks for testing."""
    return torch.tensor(BREAKS, dtype=torch.int)


@pytest.fixture
def reps():
    """Returns fake representations for tesing."""
    return [
        torch.randn(REPS_LAYERS, length, REPS_DIMENSION)
        for length in SEQ_LENGTHS
    ]


@pytest.yield_fixture
def representations_path(reps):
    """Yields the path to a fake representations hdf5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'representations.hdf5'
        with h5py.File(path, 'w') as file:
            for index, data in enumerate(reps):
                file.create_dataset(str(index), data=data)
            file.create_dataset('sentence_to_index',
                                data=np.array(['foo'.encode()]))
        yield path


@pytest.fixture
def representation_dataset(representations_path):
    """Returns a fake RepresentationDataset."""
    return representations.RepresentationDataset(representations_path)


LAYER = 0


@pytest.fixture
def representation_layer_dataset(representation_dataset):
    """Returns a fake RepresentationLayerDataset."""
    return representation_dataset.layer(LAYER)


BREAKS_KEY = 'breaks'
REPS_KEY = 'reps'
TAGS_KEY = 'tags'


def test_collate(representation_layer_dataset, pos_tags_dataset, breaks, reps):
    """Test collate constructs correct hdf5 file in the basic case."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'out.hdf5'
        preprocess.collate(representation_layer_dataset,
                           pos_tags_dataset,
                           out,
                           breaks_key=BREAKS_KEY,
                           reps_key=REPS_KEY,
                           tags_key=TAGS_KEY)
        with h5py.File(out, 'r') as actual:
            assert len(actual.keys()) == 3

            actual_breaks = torch.tensor(actual[BREAKS_KEY][:])
            assert actual_breaks.equal(breaks)

            actual_reps = torch.tensor(actual[REPS_KEY][:])
            expected_reps = torch.cat([rep[LAYER] for rep in reps])
            assert actual_reps.allclose(expected_reps)

            actual_tags = torch.tensor(actual[TAGS_KEY][:], dtype=torch.long)
            expected_tags = torch.cat(POS_INDEXES)
            assert actual_tags.equal(expected_tags)


def test_collate_out_exists(representation_layer_dataset, pos_tags_dataset):
    """Test collate dies when out path exists and force=False."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'out.hdf5'
        out.touch()
        with pytest.raises(FileExistsError, match=f'.*{out} exists.*'):
            preprocess.collate(representation_layer_dataset,
                               pos_tags_dataset,
                               out,
                               force=False)


def test_collate_out_exists_force(representation_layer_dataset,
                                  pos_tags_dataset):
    """Test collate does not die when out path exists and force=True."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'out.hdf5'
        out.touch()
        preprocess.collate(representation_layer_dataset,
                           pos_tags_dataset,
                           out,
                           breaks_key=BREAKS_KEY,
                           reps_key=REPS_KEY,
                           tags_key=TAGS_KEY,
                           force=True)

        # Just do a sanity check.
        with h5py.File(out, 'r') as actual:
            assert len(actual.keys()) == 3
            assert actual[BREAKS_KEY].shape == (len(SEQ_LENGTHS),)
            assert actual[REPS_KEY].shape == (sum(SEQ_LENGTHS), REPS_DIMENSION)
            assert actual[TAGS_KEY].shape == (sum(SEQ_LENGTHS),)


@pytest.fixture
def data(representations_path, annotations_path):
    """Returns a fake split dict for testing."""
    return {
        splits.TRAIN:
            splits.Split(representations=representations_path,
                         annotations=annotations_path)
    }


def test_run(data, reps, breaks):
    """Test run finishes preprocessing end to end."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'preprocessed'
        preprocess.run(data, out, reps_key=REPS_KEY, tags_key=TAGS_KEY)

        for layer in range(REPS_LAYERS):
            path = out / str(layer)
            assert path.exists()
            file = path / f'{splits.TRAIN}.h5'
            assert file.exists() and file.is_file()

            with h5py.File(file, 'r') as actual:
                actual_breaks = torch.tensor(actual[BREAKS_KEY][:])
                assert actual_breaks.equal(breaks)

                actual_reps = torch.tensor(actual[REPS_KEY][:])
                expected_reps = torch.cat([rep[layer] for rep in reps])
                assert actual_reps.equal(expected_reps)

                actual_tags = torch.tensor(actual[TAGS_KEY][:],
                                           dtype=torch.long)
                expected_tags = torch.cat(POS_INDEXES)
                assert actual_tags.equal(expected_tags)


def test_run_with_layers(data, reps, breaks):
    """Test run only outputs specified layers."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'preprocessed'
        preprocess.run(data,
                       out,
                       layers=(LAYER,),
                       breaks_key=BREAKS_KEY,
                       reps_key=REPS_KEY,
                       tags_key=TAGS_KEY)

        path = out / str(LAYER)
        assert path.exists()
        file = path / f'{splits.TRAIN}.h5'
        assert file.exists() and file.is_file()
        with h5py.File(file, 'r') as actual:
            actual_breaks = torch.tensor(actual[BREAKS_KEY][:])
            assert actual_breaks.equal(breaks)

            actual_reps = torch.tensor(actual[REPS_KEY][:])
            expected_reps = torch.cat([rep[LAYER] for rep in reps])
            assert actual_reps.equal(expected_reps)

            actual_tags = torch.tensor(actual[TAGS_KEY][:], dtype=torch.long)
            expected_tags = torch.cat(POS_INDEXES)
            assert actual_tags.equal(expected_tags)

        for layer in set(range(REPS_LAYERS)) - {LAYER}:
            path = out / str(layer)
            assert not path.exists()


def test_run_with_tags(data, reps, breaks):
    """Test run restricts to given tags."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'preprocessed'
        preprocess.run(data,
                       out,
                       tags=RESTRICTED_TAGS,
                       breaks_key=BREAKS_KEY,
                       reps_key=REPS_KEY,
                       tags_key=TAGS_KEY)

        for layer in range(REPS_LAYERS):
            path = out / str(layer)
            assert path.exists()
            file = path / f'{splits.TRAIN}.h5'
            assert file.exists() and file.is_file()

            with h5py.File(file, 'r') as actual:
                actual_breaks = torch.tensor(actual[BREAKS_KEY][:])
                assert actual_breaks.equal(breaks)

                actual_reps = torch.tensor(actual[REPS_KEY][:])
                expected_reps = torch.cat([rep[layer] for rep in reps])
                assert actual_reps.equal(expected_reps)

                actual_tags = torch.tensor(actual[TAGS_KEY][:],
                                           dtype=torch.long)
                expected_tags = torch.cat(RESTRICTED_POS_INDEXES)
                assert actual_tags.equal(expected_tags)


def test_run_out_exists(data):
    """Test run dies when out exists and force=False."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir)
        out.touch()
        with pytest.raises(FileExistsError, match=f'{out} exists.*'):
            preprocess.run(data, out)


def test_run_out_exists_force(data):
    """Test run dies when out exists and force=False."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'preprocessed'
        out.mkdir()
        preprocess.run(data, out, force=True)

        # Just some sanity checks.
        for layer in range(REPS_LAYERS):
            path = out / str(layer)
            assert path.exists()

            file = path / f'{splits.TRAIN}.h5'
            assert file.exists()
