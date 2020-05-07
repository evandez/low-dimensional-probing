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


def task(sample):
    """A fake task."""
    labels = []
    for xpos in sample.xpos:
        labels.append(int(xpos == '.'))
    return torch.tensor(labels, dtype=torch.uint8)


@pytest.fixture
def labels_dataset():
    """Returns a LabelsDataset for testing."""
    return datasets.LabelsDataset(SAMPLES, task)


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


def test_labeled_representations_dataset_getitem(
        labeled_representations_dataset, representations_dataset,
        labels_dataset):
    """Test LabeledRepresentationsDataset.__getitem__ returns word-by-word."""
    expected_reps = torch.cat(list(representations_dataset))
    expected_labels = torch.cat(list(labels_dataset))
    for index in range(len(labeled_representations_dataset)):
        actual_rep, actual_label = labeled_representations_dataset[index]
        assert torch.equal(actual_rep, expected_reps[index])
        assert torch.equal(actual_label, expected_labels[index])


def test_labeled_representations_dataset_len(labeled_representations_dataset):
    """Test LabeledRepresentationsDataset.__len__ returns correct length."""
    assert len(labeled_representations_dataset) == sum(SEQ_LENGTHS)


def test_labeled_representations_dataset_init_bad_dataset_lengths(
        representations_dataset):
    """Test LabeledRepresentationsDataset.__init__ checks dataset lengths."""
    with pytest.raises(ValueError, match=r'.*2 vs\. 1.*'):
        samples = list(SAMPLES)
        del samples[-1]
        datasets.LabeledRepresentationsDataset(
            representations_dataset, datasets.LabelsDataset(samples, task))


def test_labeled_representations_dataset_init_bad_seq_lengths(
        representations_dataset):
    """Test LabeledRepresentationsDataset.__init__ checks all seq lengths."""
    with pytest.raises(ValueError, match=r'.*5 representations but 4.*'):
        samples = list(SAMPLES)
        samples[-1] = ptb.Sample(*[item[:-1] for item in samples[-1]])
        datasets.LabeledRepresentationsDataset(
            representations_dataset, datasets.LabelsDataset(samples, task))


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


def test_labeled_representation_pairs_dataset_init_bad_dataset_lengths(
        representations_dataset):
    """Test LabeledRepresentationPairsDataset.__init__ checks dataset lens."""
    with pytest.raises(ValueError, match=r'.*2 vs\. 1.*'):
        samples = list(SAMPLES)
        del samples[-1]
        datasets.LabeledRepresentationPairsDataset(
            representations_dataset,
            datasets.LabelsDataset(samples, pairwise_task))


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
