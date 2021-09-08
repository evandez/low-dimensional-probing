"""Unit and functional tests for the learning module."""
from lodimp import datasets, learning

import pytest
import torch
import wandb
from torch import nn

PATIENCE = 4


def test_early_stopping_init_decreasing():
    """Test EarlyStopping.__init__ records when value should decrease."""
    early_stopping = learning.EarlyStopping(patience=PATIENCE, decreasing=True)
    assert early_stopping.patience == PATIENCE
    assert early_stopping.decreasing is True
    assert early_stopping.best == float('inf')
    assert early_stopping.num_bad == 0


def test_early_stopping_init_increasing():
    """Test EarlyStopping.__init__ records when value should increase."""
    early_stopping = learning.EarlyStopping(
        patience=PATIENCE,
        decreasing=False,
    )
    assert early_stopping.patience == PATIENCE
    assert early_stopping.decreasing is False
    assert early_stopping.best == float('-inf')
    assert early_stopping.num_bad == 0


def test_early_stopping_call_decreasing():
    """Test EarlyStopping.__call__ returns when value does not decrease."""
    early_stopping = learning.EarlyStopping(patience=PATIENCE, decreasing=True)
    assert not early_stopping(-1)
    for i in range(PATIENCE):
        assert not early_stopping(i)
    assert early_stopping(0)


def test_early_stopping_call_increasing():
    """Test EarlyStopping.__call__ reports when value does not increases."""
    early_stopping = learning.EarlyStopping(
        patience=PATIENCE,
        decreasing=False,
    )
    assert not early_stopping(PATIENCE + 1)
    for i in range(PATIENCE):
        assert not early_stopping(i)
    assert early_stopping(0)


N_SAMPLES = 5
N_DIMS_PER_REP = 10
N_UNIQUE_FEATS = 3
SEQ_LENGTHS = (3, 2)


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
        return (N_DIMS_PER_REP,)

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
def representations():
    """Returns batches of fake representations for testing."""
    return tuple(
        torch.randn(seq_length, N_DIMS_PER_REP) for seq_length in SEQ_LENGTHS)


@pytest.fixture
def features():
    """Returns batches of fake feature tensors for testing."""
    return tuple(
        torch.randint(N_UNIQUE_FEATS, size=(seq_length,))
        for seq_length in SEQ_LENGTHS)


@pytest.fixture
def task_dataset(representations, features):
    """Returns a fake TaskDataset for testing."""
    return FakeTaskDataset(representations, features)


EPOCHS = 10


def test_train(task_dataset, mocker):
    """Test train runs without crashing."""
    wandb_log = mocker.patch.object(wandb, 'log')

    probe = nn.Linear(N_DIMS_PER_REP, N_UNIQUE_FEATS)
    before = probe.weight.data.clone()

    learning.train(probe, task_dataset, epochs=EPOCHS, also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)
    assert wandb_log.call_args_list == [
        mocker.call({'train accuracy': mocker.ANY}),
    ] * len(task_dataset) * EPOCHS


def test_train_with_early_stopping(task_dataset, mocker):
    """Test train stops early."""
    wandb_log = mocker.patch.object(wandb, 'log')

    early_stopping = learning.EarlyStopping(patience=PATIENCE)
    # Cannot possible go lower! So we should stop after PATIENCE steps.
    early_stopping(float('-inf'))

    probe = nn.Linear(N_DIMS_PER_REP, N_UNIQUE_FEATS)
    before = probe.weight.data.clone()

    learning.train(probe,
                   task_dataset,
                   epochs=EPOCHS,
                   stopper=early_stopping,
                   also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)
    assert wandb_log.call_args_list == [
        mocker.call({'train accuracy': mocker.ANY}),
    ] * (PATIENCE + 1)


def test_train_with_dev_dataset(task_dataset, mocker):
    """Test train runs without crashing."""
    wandb_log = mocker.patch.object(wandb, 'log')

    probe = nn.Linear(N_DIMS_PER_REP, N_UNIQUE_FEATS)
    before = probe.weight.data.clone()

    learning.train(probe,
                   task_dataset,
                   dev_dataset=task_dataset,
                   epochs=EPOCHS,
                   also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)

    expected = []
    for _ in range(EPOCHS):
        expected.extend([mocker.call({'train accuracy': mocker.ANY})] *
                        len(task_dataset))
        expected.extend([mocker.call({'dev accuracy': mocker.ANY})])
    assert wandb_log.call_args_list == expected


def test_train_with_early_stopping_and_dev_dataset(task_dataset, mocker):
    """Test train stops early."""
    wandb_log = mocker.patch.object(wandb, 'log')

    early_stopping = learning.EarlyStopping(patience=0)
    # Cannot possible go lower! So we should stop after PATIENCE steps.
    early_stopping(float('-inf'))

    probe = nn.Linear(N_DIMS_PER_REP, N_UNIQUE_FEATS)
    before = probe.weight.data.clone()

    learning.train(probe,
                   task_dataset,
                   dev_dataset=task_dataset,
                   epochs=EPOCHS,
                   stopper=early_stopping,
                   also_log_to_wandb=True)
    after = probe.weight.data

    assert not before.equal(after)

    expected = [
        mocker.call({'train accuracy': mocker.ANY}),
    ] * len(task_dataset)
    expected.append(mocker.call({'dev accuracy': mocker.ANY}))
    assert wandb_log.call_args_list == expected


def test_test(task_dataset, features):
    """Test test returns expected accuracy."""
    features = torch.cat(features)
    feat = features[0]
    expected = features.eq(feat).sum().item() / N_SAMPLES

    class FakeModule(nn.Module):
        """Always returns the same prediction."""

        def forward(self, reps):
            """Just returns the tag."""
            assert reps.shape[-1] == N_DIMS_PER_REP
            logits = torch.zeros(len(reps), N_UNIQUE_FEATS)
            logits[:, feat] = 1
            return logits

    actual = learning.test(FakeModule(),
                           task_dataset,
                           device=torch.device('cpu'))
    assert actual == expected
