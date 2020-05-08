"""Unit tests for datasets module."""

import pathlib
import tempfile

from lodimp import datasets, ptb

import h5py
import numpy as np
import pytest
import torch

SAMPLES = (
    ptb.Sample(
        ('The', 'company', 'expects', 'earnings', '.'),
        ('DT', 'NN', 'VBZ', 'NNS', '.'),
        (1, 2, -1, 15, 1),
        ('det', 'nsubj', 'root', 'nsubj', 'punct'),
    ),
    ptb.Sample(
        ('He', 'was', 'named', 'chief', '.'),
        ('PRP', 'VBD', 'VBN', 'JJ', '.'),
        (3, 3, 0, 6),
        ('nsubjpass', 'auxpass', 'root', 'amod'),
    ),
)
ELMO_LAYERS = 3
ELMO_DIMENSION = 1024
SEQ_LENGTHS = (*[len(sample.sentence) for sample in SAMPLES],)


@pytest.yield_fixture
def elmo_path():
    """Yields the path to a fake ELMo h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'elmo.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('sentence_to_index', data=0)
            for index, length in enumerate(SEQ_LENGTHS):
                data = np.zeros((ELMO_LAYERS, length, ELMO_DIMENSION))
                data[:] = index
                handle.create_dataset(str(index), data=data)
        yield path


def test_elmo_init_bad_layer(elmo_path):
    """Test ELMoRepresentationsDataset.__init__ dies when given bad layer."""
    with pytest.raises(ValueError, match='.*layer.*'):
        datasets.ELMoRepresentationsDataset(elmo_path, 3)


def test_elmo_representations_dataset_dimension(elmo_path):
    """Test ELMoRepresentationsDataset.dimension returns correct dimension."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        assert dataset.dimension == ELMO_DIMENSION


def test_elmo_representations_dataset_length(elmo_path):
    """Test ELMoRepresentationsDataset.length returns correct seq lengths."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        for index, expected in enumerate(SEQ_LENGTHS):
            assert dataset.length(index) == expected


def test_elmo_representaitons_dataset_getitem(elmo_path):
    """Test ELMoRepresentationsDataset.__getitem__ returns correct shape."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        for index, length in enumerate(SEQ_LENGTHS):
            assert dataset[index].shape == (length, ELMO_DIMENSION)
            assert (dataset[index] == index).all()


def test_elmo_representations_dataset_len(elmo_path):
    """Test ELMoRepresentationsDataset.__len__ returns correct length."""
    for layer in range(ELMO_LAYERS):
        dataset = datasets.ELMoRepresentationsDataset(elmo_path, layer)
        assert len(dataset) == len(SEQ_LENGTHS)


NSAMPLES = 5
NFEATURES = 10
NLABELS = 3


@pytest.fixture
def features():
    """Returns fake features for testing."""
    return torch.randn(NSAMPLES, NFEATURES)


@pytest.fixture
def labels():
    """Returns fake labels for testing."""
    return torch.randint(NLABELS, size=(NSAMPLES,))


@pytest.yield_fixture
def task_dataset(features, labels):
    """Yields the path to a fake ELMo h5 file."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'task.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('features', data=features)
            dataset = handle.create_dataset('labels', data=labels)
            dataset.attrs['nlabels'] = NLABELS
        yield datasets.TaskDataset(path)


def test_task_dataset_getitem(task_dataset, features, labels):
    """Test TaskDataset.__getitem__ returns all (feature, label) pairs."""
    for (af, al), ef, el in zip(task_dataset, features, labels):
        assert torch.tensor(af).equal(ef)
        assert torch.tensor(al).equal(el)


def test_task_dataset_getitem_bad_index(task_dataset):
    """Test TaskDataset.__getitem__ explodes when given a bad index."""
    for bad in (-1, NSAMPLES):
        with pytest.raises(IndexError, match=f'.*bounds: {bad}.*'):
            task_dataset[bad]


def test_task_dataset_len(task_dataset):
    """Test TaskDataset.__len__ returns correct length."""
    assert len(task_dataset) == NSAMPLES


def test_task_dataset_nfeatures(task_dataset):
    """Test TaskDataset.nfeatures returns correct number of features."""
    assert task_dataset.nfeatures == NFEATURES


def test_task_dataset_nlabels(task_dataset):
    """Test TaskDataset.nlabels returns correct number of labels."""
    assert task_dataset.nlabels == NLABELS


class Task:
    """A dumb, fake task."""

    def __len__(self):
        """There is only one, binray label in this task: period, no period."""
        return 1

    def __call__(self, sample):
        """Map words to period vs. no period label."""
        labels = []
        for xpos in sample.xpos:
            labels.append(int(xpos == '.'))
        return torch.tensor(labels, dtype=torch.uint8)


@pytest.fixture
def labels_dataset():
    """Returns a LabelsDataset for testing."""
    return datasets.LabelsDataset(SAMPLES, Task())


def test_labels_dataset_getitem(labels_dataset):
    """Test whether LabelsDataset.__getitem__ returns correct labels."""
    for index in (0, 1):
        item = labels_dataset[index]
        assert item[-1] == 1
        assert not item[:-1].any()


def test_labels_dataset_len(labels_dataset):
    """Test whether LabelsDataset.__len__ returns correct length."""
    assert len(labels_dataset) == len(SAMPLES)


@pytest.fixture
def representations_dataset(elmo_path):
    """Returns a RepresentationsDataset for testing."""
    return datasets.ELMoRepresentationsDataset(elmo_path, 0)


@pytest.fixture
def labeled_representations_dataset(representations_dataset, labels_dataset):
    """Returns a LabeledRepresentationsDataset for testing."""
    return datasets.LabeledRepresentationsDataset(representations_dataset,
                                                  labels_dataset)


def test_labeled_representations_dataset_nfeatures(
        labeled_representations_dataset):
    """Test LabeledRepresentationsDataset.nfeatures returns rep dimension."""
    assert labeled_representations_dataset.nfeatures == ELMO_DIMENSION


def test_labeled_representations_dataset_nlabels(
        labeled_representations_dataset):
    """Test LabeledRepresentationsDataset.nlabels forwards to len(labels)."""
    assert labeled_representations_dataset.nlabels == 1


def test_labeled_representations_dataset_init_bad_dataset_lengths(
        representations_dataset):
    """Test LabeledRepresentationsDataset.__init__ check dataset lengths."""
    with pytest.raises(ValueError, match=r'.*2 vs\. 1.*'):
        samples = list(SAMPLES)
        del samples[-1]
        datasets.LabeledRepresentationsDataset(
            representations_dataset, datasets.LabelsDataset(samples, Task()))


@pytest.fixture
def labeled_representation_singles_dataset(representations_dataset,
                                           labels_dataset):
    """Returns a LabeledRepresentationSinglesDataset for testing."""
    return datasets.LabeledRepresentationSinglesDataset(
        representations_dataset,
        labels_dataset,
    )


def test_labeled_representation_singles_dataset_getitem(
        labeled_representation_singles_dataset, representations_dataset,
        labels_dataset):
    """Test LabeledRepresentationSinglesDataset.__getitem__ gives all words."""
    expected_reps = torch.cat(list(representations_dataset))
    expected_labels = torch.cat(list(labels_dataset))
    for index in range(len(labeled_representation_singles_dataset)):
        rep, label = labeled_representation_singles_dataset[index]
        assert torch.equal(rep, expected_reps[index])
        assert torch.equal(label, expected_labels[index])


def test_labeled_representation_singles_dataset_len(
        labeled_representation_singles_dataset):
    """Test LabeledRepresentationSinglesDataset.__len__ gives right length."""
    assert len(labeled_representation_singles_dataset) == sum(SEQ_LENGTHS)


def test_labeled_representation_singles_dataset_init_bad_seq_lengths(
        representations_dataset):
    """Test LabeledRepresentationSinglesDataset.__init__ checks seq lengths."""
    with pytest.raises(ValueError, match=r'.*5 representations but 4.*'):
        samples = list(SAMPLES)
        samples[-1] = ptb.Sample(*[item[:-1] for item in samples[-1]])
        datasets.LabeledRepresentationSinglesDataset(
            representations_dataset, datasets.LabelsDataset(samples, Task()))


def pairwise_task(sample):
    """A fake pairwise task always returning the identity."""
    return torch.eye(len(sample.sentence))


@pytest.fixture
def pairwise_labels_dataset():
    """Returns a pairwise LabelsDataset for testing."""
    return datasets.LabelsDataset(SAMPLES, pairwise_task)


@pytest.fixture
def labeled_representation_pairs_dataset(representations_dataset,
                                         pairwise_labels_dataset):
    """Returns a LabeledRepresentationPairsDataset for testing."""
    return datasets.LabeledRepresentationPairsDataset(representations_dataset,
                                                      pairwise_labels_dataset)


def test_labeled_representation_pairs_dataset_getitem(
        labeled_representation_pairs_dataset, representations_dataset):
    """Test LabeledRepresentationPairsDataset.__getitem__ preserves order."""
    actual_rep, actual_label = labeled_representation_pairs_dataset[0]
    expected_rep = torch.cat(
        (representations_dataset[0][0], representations_dataset[0][0]))
    assert torch.equal(actual_rep, expected_rep)
    assert actual_label == 1

    actual_rep, actual_label = labeled_representation_pairs_dataset[1]
    expected_rep = torch.cat(
        (representations_dataset[0][0], representations_dataset[0][1]))
    assert torch.equal(actual_rep, expected_rep)
    assert actual_label == 0


def test_labeled_representation_pairs_dataset_len(
        labeled_representation_pairs_dataset):
    """Test LabeledRepresentationPairsDataset.__len__ gives correct length."""
    expected = sum(len(sample.sentence)**2 for sample in SAMPLES)
    assert len(labeled_representation_pairs_dataset) == expected


def test_labeled_representation_pairs_dataset_init_bad_seq_lengths(
        representations_dataset):
    """Test LabeledRepresentationsDataset.__init__ checks all seq lengths."""
    with pytest.raises(ValueError, match=r'.*5 representations but size.*'):
        samples = list(SAMPLES)
        samples[-1] = ptb.Sample(*[item[:-1] for item in samples[-1]])
        datasets.LabeledRepresentationPairsDataset(
            representations_dataset,
            datasets.LabelsDataset(samples, pairwise_task))


class FakeDataset(torch.utils.data.Dataset):
    """A very dumb dataset."""

    def __init__(self, features, labels):
        """Store simple features and labels.

        Args:
            features: Feature tensor.
            labels: Label tensor.

        """
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        """Return the (feature, label) at the given index.

        Args:
            index: Index of sample to retrieve.

        Returns:
            The (feature, label) pair.

        """
        return self.features[index], self.labels[index]

    def __len__(self):
        """Returns number of samples in the dataset."""
        return len(self.features)


LENGTH = 5
DIMENSION = 10


@pytest.fixture
def collated_dataset():
    """Returns CollatedDataset for testing."""
    dataset = FakeDataset(torch.ones(LENGTH, DIMENSION), torch.ones(LENGTH))
    return datasets.CollatedDataset(dataset)


def test_collated_dataset_iter(collated_dataset):
    """Test CollatedDataset.__iter__ only returns one batch."""
    batches = [batch for batch in collated_dataset]
    assert len(batches) == 1

    batch, = batches
    assert len(batch) == 2

    features, labels = batch
    assert features.shape == (LENGTH, DIMENSION)
    assert labels.shape == (LENGTH,)


def test_collated_dataset_len(collated_dataset):
    """Test CollatedDataset.__len__ returns correct length."""
    assert len(collated_dataset) == 1
