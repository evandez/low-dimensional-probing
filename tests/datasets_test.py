"""Unit tests for the datasets module."""
import pathlib
import tempfile

from lodimp import datasets

import h5py
import pytest
import torch

BREAKS_KEY = 'test-breaks'
REPS_KEY = 'test-reps'
FEATS_KEY = 'test-feats'

N_SAMPLES = 5
N_DIMS_PER_REP = 10
N_REPS_PER_SAMPLE = 2
N_UNIQUE_FEATS = 3

SEQ_LENGTHS = (3, 2)


@pytest.fixture
def representations():
    """Returns batches of fake representations for testing."""
    return tuple(
        torch.randn(seq_length, N_REPS_PER_SAMPLE, N_DIMS_PER_REP)
        for seq_length in SEQ_LENGTHS)


@pytest.fixture
def collated_representations(representations):
    """Returns same as representations(), but collated."""
    return torch.cat(representations)


@pytest.fixture
def features():
    """Returns batches of fake feature tensors for testing."""
    return tuple(
        torch.randint(N_UNIQUE_FEATS, size=(seq_length,))
        for seq_length in SEQ_LENGTHS)


@pytest.fixture
def collated_features(features):
    """Returns same as features(), but collated."""
    return torch.cat(features)


@pytest.fixture
def collated_breaks():
    """Returns fake sentence breaks for testing."""
    breaks = [0]
    for seq_length in SEQ_LENGTHS:
        breaks.append(breaks[-1] + seq_length)
    return torch.tensor(breaks[:-1], dtype=torch.int)


class FakeTaskDataset(datasets.TaskDataset):
    """A fake task dataset that iterates over pre-defined reps/feats pairs."""

    def __init__(self, reps_batches, feats_batches):
        """Initialize the fake dataset."""
        assert len(reps_batches) == len(feats_batches)
        self.reps_batches = reps_batches
        self.feats_batches = feats_batches

    def __iter__(self):
        """Yields all of the predefined (reps, feats) pairs."""
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        """Returns the index'th batch."""
        assert index >= 0 and index < len(self)
        return self.reps_batches[index], self.feats_batches[index]

    def __len__(self) -> int:
        """Returns number of (reps, feats) pairs."""
        return len(self.reps_batches)

    @property
    def sample_representations_shape(self):
        """Returns the shape of the representations component of a sample."""
        return (N_REPS_PER_SAMPLE, N_DIMS_PER_REP)

    @property
    def sample_features_shape(self):
        """Returns the shape of the representations component of a sample."""
        return ()

    def count_samples(self):
        """Returns the number of samples (not batches) in the dataset."""
        return sum(SEQ_LENGTHS)

    def count_unique_features(self):
        """Returns the number of unique features in the dataset."""
        return N_UNIQUE_FEATS


@pytest.fixture
def task_dataset(representations, features):
    """Returns something a fake TaskDataset for testing."""
    return FakeTaskDataset(representations, features)


def test_task_dataset_collate(task_dataset, collated_breaks,
                              collated_representations, collated_features):
    """Test TaskDataset.collate constructs correct hdf5 file in basic case."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'out.hdf5'
        task_dataset.collate(out,
                             breaks_key=BREAKS_KEY,
                             representations_key=REPS_KEY,
                             features_key=FEATS_KEY)
        with h5py.File(out, 'r') as actual:
            assert len(actual.keys()) == 3

            actual_breaks = torch.tensor(actual[BREAKS_KEY][:])
            assert actual_breaks.equal(collated_breaks)

            actual_reps = torch.tensor(actual[REPS_KEY][:])
            assert actual_reps.allclose(collated_representations)

            actual_feats = torch.tensor(actual[FEATS_KEY][:], dtype=torch.long)
            assert actual_feats.equal(collated_features)


def test_task_dataset_collate_out_exists(task_dataset):
    """Test TaskDatset.collate dies when out path exists and force=False."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'out.hdf5'
        out.touch()
        with pytest.raises(FileExistsError, match=f'.*{out} exists.*'):
            task_dataset.collate(out, force=False)


def test_task_dataset_collate_out_exists_force(task_dataset):
    """Test collate does not die when out path exists and force=True."""
    with tempfile.TemporaryDirectory() as tempdir:
        out = pathlib.Path(tempdir) / 'out.hdf5'
        out.touch()
        task_dataset.collate(out,
                             breaks_key=BREAKS_KEY,
                             representations_key=REPS_KEY,
                             features_key=FEATS_KEY,
                             force=True)

        # Just do a sanity check.
        with h5py.File(out, 'r') as actual:
            assert len(actual.keys()) == 3
            assert actual[BREAKS_KEY].shape == (len(SEQ_LENGTHS),)
            assert actual[REPS_KEY].shape == (
                sum(SEQ_LENGTHS),
                N_REPS_PER_SAMPLE,
                N_DIMS_PER_REP,
            )
            assert actual[FEATS_KEY].shape == (sum(SEQ_LENGTHS),)


@pytest.yield_fixture
def collated_path(
    collated_breaks,
    collated_representations,
    collated_features,
):
    """Yields the path to a fake output of TaskDataset.collate(...)."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'task.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset(BREAKS_KEY, data=collated_breaks)
            handle.create_dataset(REPS_KEY, data=collated_representations)
            feats = handle.create_dataset(FEATS_KEY, data=collated_features)
            feats.attrs[datasets.H5_UNIQUE_FEATURES_KEY] = N_UNIQUE_FEATS
        yield path


@pytest.fixture
def collated_task_dataset(collated_path):
    """Returns a CollatedTaskDataset for testing."""
    return datasets.CollatedTaskDataset(collated_path,
                                        representations_key=REPS_KEY,
                                        features_key=FEATS_KEY)


def test_collated_task_dataset_init(collated_task_dataset):
    """Test CollatedTaskDataset.__init__ sets state as expected."""
    assert collated_task_dataset.representations_cache is None
    assert collated_task_dataset.features_cache is None


def test_collated_task_dataset_init_cached(collated_path,
                                           collated_representations,
                                           collated_features):
    """Test CollatedTaskDataset.__init__ creates caches when device given."""
    device = torch.device('cpu')
    dataset = datasets.CollatedTaskDataset(collated_path,
                                           representations_key=REPS_KEY,
                                           features_key=FEATS_KEY,
                                           device=device)

    assert dataset.representations_cache is not None
    assert dataset.representations_cache.equal(collated_representations)
    assert dataset.representations_cache.device == device

    assert dataset.features_cache is not None
    assert dataset.features_cache.equal(collated_features)
    assert dataset.features_cache.device == device


@pytest.mark.parametrize('device', (None, torch.device('cpu')))
def test_collated_task_dataset_getitem(
    collated_path,
    device,
    collated_representations,
    collated_features,
):
    """Test CollatedTaskDataset.__getitem__ returns all (reps, feats) pairs."""
    dataset = datasets.CollatedTaskDataset(collated_path,
                                           representations_key=REPS_KEY,
                                           features_key=FEATS_KEY,
                                           device=device)
    for index, (er, ef) in enumerate(
            zip(collated_representations, collated_features)):
        ar, af = dataset[index]
        assert ar.equal(er)
        assert af.equal(ef)


@pytest.mark.parametrize('index', (-1, N_SAMPLES))
def test_collated_task_dataset_getitem_bad_index(collated_task_dataset, index):
    """Test TaskDataset.__getitem__ explodes when given a bad index."""
    with pytest.raises(IndexError, match=f'.*bounds: {index}.*'):
        collated_task_dataset[index]


def test_collated_task_dataset_iter(collated_task_dataset,
                                    collated_representations,
                                    collated_features):
    """Test TaskDataset.__iter__ yields all (reps, feats) pairs."""
    actuals = tuple(iter(collated_task_dataset))
    expecteds = zip(collated_representations, collated_features)
    for (ar, af), (er, ef) in zip(actuals, expecteds):
        assert ar.shape == (1, N_REPS_PER_SAMPLE, N_DIMS_PER_REP)
        assert af.shape == (1,)
        assert ar.squeeze().equal(er)
        assert af.squeeze().equal(ef)


def test_collated_task_dataset_len(collated_task_dataset):
    """Test CollatedTaskDataset.__len__ returns correct length."""
    assert len(collated_task_dataset) == N_SAMPLES


def test_collated_task_dataset_sample_representations_shape(
        collated_task_dataset):
    """Check CollatedTaskDataset.sample_representations_shape return value."""
    assert collated_task_dataset.sample_representations_shape == (
        N_REPS_PER_SAMPLE, N_DIMS_PER_REP)


def test_collated_task_dataset_sample_features_shape(collated_task_dataset):
    """Check CollatedTaskDataset.sample_features_shape return value."""
    assert collated_task_dataset.sample_features_shape == ()


def test_collated_task_dataset_count_samples(collated_task_dataset):
    """Test CollatedTaskDataset.count_samples returns dataset length."""
    assert collated_task_dataset.count_samples() == N_SAMPLES


def test_collated_task_dataset_count_unique_features(collated_task_dataset):
    """Test CollatedTaskDataset.count_unique_features returns correct value."""
    assert collated_task_dataset.count_unique_features() == N_UNIQUE_FEATS


@pytest.fixture
def sentence_batching_collated_task_dataset(collated_path):
    """Returns a SentenceBatchingCollatedTaskDataset for testing."""
    return datasets.SentenceBatchingCollatedTaskDataset(
        collated_path,
        breaks_key=BREAKS_KEY,
        representations_key=REPS_KEY,
        features_key=FEATS_KEY,
        device=torch.device('cpu'))


@pytest.mark.parametrize('device', (None, torch.device('cpu')))
def test_sentence_batching_collated_task_dataset_getitem(
        collated_path, device, collated_representations, collated_features):
    """Check SentenceBatchingCollatedTaskDataset.__getitem__ returns values."""
    dataset = datasets.SentenceBatchingCollatedTaskDataset(
        collated_path,
        breaks_key=BREAKS_KEY,
        representations_key=REPS_KEY,
        features_key=FEATS_KEY,
        device=device)

    assert len(SEQ_LENGTHS) == 2, 'sanity check for constant'
    expected_split = SEQ_LENGTHS[0]
    expecteds = (
        (
            collated_representations[:expected_split],
            collated_features[:expected_split],
        ),
        (
            collated_representations[expected_split:],
            collated_features[expected_split:],
        ),
    )
    for index, expected in enumerate(expecteds):
        batch = dataset[index]
        assert len(batch) == 2

        ar, af = batch
        er, ef = expected
        assert ar.equal(er)
        assert af.equal(ef)


@pytest.mark.parametrize('index', (-1, len(SEQ_LENGTHS)))
def test_sentence_batching_collated_task_dataset_getitem_bad_index(
        sentence_batching_collated_task_dataset, index):
    """Test SentenceBatchingDataset.__getitem__ dies when given bad index."""
    with pytest.raises(IndexError, match='sentence index out of bounds.*'):
        sentence_batching_collated_task_dataset[index]


def test_sentence_batching_task_dataset_iter(
        sentence_batching_collated_task_dataset, collated_representations,
        collated_features):
    """Test SentenceBatchingCollatedTaskDataset.__iter__ yields all samples."""
    batches = tuple(iter(sentence_batching_collated_task_dataset))
    assert len(batches) == 2
    first, second = batches

    assert len(first) == 2
    ar, af = first
    expected_split = SEQ_LENGTHS[0]
    assert ar.equal(collated_representations[:expected_split])
    assert af.equal(collated_features[:expected_split])

    assert len(second) == 2
    ar, af = second
    assert ar.equal(collated_representations[expected_split:])
    assert af.equal(collated_features[expected_split:])


def test_sentence_batching_task_dataset_len(
        sentence_batching_collated_task_dataset):
    """Test SentenceBatchingCollatedTaskDataset.__len__ gives num sentences."""
    assert len(sentence_batching_collated_task_dataset) == len(SEQ_LENGTHS)


@pytest.fixture
def non_batching_collated_task_dataset(collated_path):
    """Returns a NonBatchingCollatedTaskDataset for testing."""
    return datasets.NonBatchingCollatedTaskDataset(
        collated_path,
        representations_key=REPS_KEY,
        features_key=FEATS_KEY,
    )


@pytest.mark.parametrize('device', (None, torch.device('cpu')))
def test_non_batching_collated_task_dataset_getitem(
    collated_path,
    device,
    collated_representations,
    collated_features,
):
    """Test NonBatchingCollatedTaskDataset.__getitem__ returns full dataset."""
    dataset = datasets.NonBatchingCollatedTaskDataset(
        collated_path,
        representations_key=REPS_KEY,
        features_key=FEATS_KEY,
    )
    batch = dataset[0]
    assert len(batch) == 2
    ar, af = batch
    assert ar.equal(collated_representations)
    assert af.equal(collated_features)


@pytest.mark.parametrize('index', (-1, 1))
def test_non_batching_collated_task_dataset_getitem_bad_index(
        non_batching_collated_task_dataset, index):
    """Test NonBatchingCollatedTaskDataset.__getitem__ dies when index != 0."""
    with pytest.raises(IndexError, match='.*must be 0.*'):
        non_batching_collated_task_dataset[index]


def test_non_batching_task_dataset_iter(non_batching_collated_task_dataset,
                                        collated_representations,
                                        collated_features):
    """Test NonBatchingCollatedTaskDataset.__iter__ yields all chunks."""
    batches = tuple(iter(non_batching_collated_task_dataset))
    assert len(batches) == 1

    batch, = batches
    assert len(batch) == 2

    ar, at = batch
    assert ar.equal(collated_representations)
    assert at.equal(collated_features)


def test_non_batching_task_dataset_len(non_batching_collated_task_dataset):
    """Test NonBatchingCollatedTaskDataset.__len__ returns 1."""
    assert len(non_batching_collated_task_dataset) == 1
