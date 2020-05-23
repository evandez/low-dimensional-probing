"""Defines functions for training models."""

import logging
import pathlib
from typing import Any, Iterator, Optional, Tuple

import h5py
import torch
import wandb
from torch import nn, optim
from torch.utils import data


class TaskDataset(data.IterableDataset):
    """A standard dataset containing representations and tags.

    Because we train many, many probes on the same tasks, we must preprocess
    the representations and tags to make training as fast as possible.
    The key to fast training is to avoid collation and I/O between disk,
    RAM, and the GPU memory, ideally moving the entire dataset to the GPU
    at the beginning of training and then relying on the GPU's finesse with
    large matix multiplications. To faciliate this, we pre-collate
    all representations into a contiguous array and store it in an h5 file.
    We do the same for the tags, storing them in a separate array.

    While this format inherently destroys sentence boundaries, we record the
    start index of each sentence in a third dataset, which we refer to as
    the "breaks" dataset.

    This class wraps the pre-computed h5 file, making few assumptions about
    the format other than that there are three arrays: one for representations,
    one for tags, and one for sentence breaks.
    """

    def __init__(self,
                 path: pathlib.Path,
                 breaks_key: str = 'breaks',
                 reps_key: str = 'reps',
                 tags_key: str = 'tags'):
        """Initialize the task dataset.

        Args:
            path (pathlib.Path): Path to the preprocessed h5 file.
            breaks_key (str, optional): Key for the dataset of sentence break
                points in the h5 file. Defaults to 'breaks'.
            reps_key (str, optional): Key for the dataset of representations
                in the h5 file. Defaults to 'reps'.
            tags_key (str, optional): Key for the dataset of tags in the h5
                file. Defaults to 'tags'.

        Raises:
            KeyError: If an expected dataset is missing.

        """
        self.file = h5py.File(path, 'r')

        for key in (breaks_key, reps_key, tags_key):
            if key not in self.file:
                raise KeyError(f'dataset not in h5 file: {key}')

        self.breaks = self.file[breaks_key]
        self.representations = self.file[reps_key]
        self.tags = self.file[tags_key]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the (representations, label) pair at the given index.

        Args:
            index (int): The index of the pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The (representations, label)
                tensors, the first with shape (1, dimension) and the second
                with shape (1,).

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'sample index out of bounds: {index}')
        reps = torch.tensor(self.representations[index])
        tag = torch.tensor(self.tags[index], dtype=torch.long)
        return reps, tag

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields every sample in the dataset in sequence."""
        for index in range(len(self)):
            reps, tag = self[index]
            # Unsqueeze so the reps appear batch-like.
            yield reps.unsqueeze(0), tag.unsqueeze(0)

    def __len__(self) -> int:
        """Returns the number of samples in this dataset."""
        return len(self.representations)

    @property
    def dimension(self) -> int:
        """Returns the representation dimensionality."""
        return self.representations.shape[-1]


class SentenceIterableTaskDataset(TaskDataset):
    """A TaskDataset that iterates at the sentence level."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns all representations and tags for the index'th sentence.

        Args:
            index (int): Index of the sentence to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Shape (L, *) tensors, where
                L is the length of the index'th sentence and * is whatever
                shape the representations/tags would have if you accessed
                them with PreprocessedDataset.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'sentence index out of bounds: {index}')
        start = self.breaks[index]
        if index < len(self) - 1:
            end = self.breaks[index + 1]
            reps = self.representations[start:end]
            tags = self.tags[start:end]
        else:
            assert index == len(self) - 1
            reps = self.representations[start:]
            tags = self.tags[start:]
        return torch.tensor(reps), torch.tensor(tags, dtype=torch.long)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over (sentence representations, sentence labels) pairs."""
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        """Returns the number of sentences in the dataset."""
        return self.breaks.shape[0]


class InMemoryTaskDataset(TaskDataset):
    """A TaskDataset where data is collated in-memory.

    Iterating over this dataset returns the full collated dataset as if it were
    a single batch.
    """

    def __init__(self,
                 path: pathlib.Path,
                 device: Optional[torch.device] = None,
                 **kwargs: Any):
        """Read the entire dataset into memory.

        Keyword arguments forwarded to TaskDataset.__init__.

        Args:
            path (pathlib.Path): Path to the preprocessed h5 file.
            device (Optional[torch.device], optional): Move data to this
                device. By default, data configured for CPU.

        """
        super().__init__(path, **kwargs)
        device = device or torch.device('cpu')
        self.breaks = torch.tensor(self.breaks[:], device=device)
        self.representations = torch.tensor(self.representations[:],
                                            device=device)
        self.tags = torch.tensor(self.tags[:], dtype=torch.long, device=device)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the collated dataset.

        Args:
            index (int): Index of the dataset. Must be 0. This parameter is
                here for sake of type checking.

        Raises:
            IndexError: If the index is not 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The full (representations, tags)
                dataset.

        """
        if index != 0:
            raise IndexError(f'index must be 0, got {index}')
        return self.representations, self.tags

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields the one and only chunk of data."""
        yield self.representations, self.tags

    def __len__(self) -> int:
        """Returns 1 because there is only one chunk."""
        return 1


class EarlyStopping:
    """Observes a numerical value and determines when it has not improved."""

    def __init__(self, patience: int = 3, decreasing: bool = True):
        """Initialize the early stopping tracker.

        Args:
            patience (int): Allow tracked value to not improve over its
                best value this many times. Defaults to 3.
            decreasing (bool, optional): If True, the tracked value "improves"
                if it decreases. If False, it "improves" if it increases.
                Defaults to True.

        """
        self.patience = patience
        self.decreasing = decreasing
        self.best = float('inf') if decreasing else float('-inf')
        self.num_bad = 0

    def __call__(self, value: float) -> bool:
        """Considers the new tracked value and decides whether to stop.

        Args:
            value (float): The new tracked value.

        Returns:
            bool: True if patience has been exceeded.

        """
        improved = self.decreasing and value < self.best
        improved |= not self.decreasing and value > self.best
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1

        return self.num_bad > self.patience


def train(probe: nn.Module,
          train_dataset: TaskDataset,
          dev_dataset: Optional[TaskDataset] = None,
          stopper: Optional[EarlyStopping] = None,
          device: Optional[torch.device] = None,
          lr: float = 1e-3,
          epochs: int = 25,
          also_log_to_wandb: bool = False) -> None:
    """Train a probe on the given data.

    The probe is always trained against cross entropy and optimized with Adam
    using the default hyperparameters, bar learning rate, which can be set by
    the caller.

    This function makes only three assumptions:
    (1) Probe takes one tensor as input and produces one tensor as output.
    (2) Probe's outputs are logits that can be passed as the first argument
        to `torch.nn.CrossEntropyLoss`.
    (3) Inputs are paired with integral tensors that can be directly passed as
        the second argument to the loss above.

    Args:
        probe (nn.Module): The model to train.
        train_dataset (TaskDataset): The data on which to train.
            Iterates are (tensor, tensor) pairs, the former being model
            inputs that will be fed directly to the probe, and the latter
            integral tags that will be fed directly to the cross entropy loss.
            These tensors will not be reshaped in any way; the probe will have
            to do that internally.
        dev_dataset (Optional[TaskDataset], optional): Same format as
            `train_dataset`. If set, probe will be evaluated on this dataset
            after every epoch. Defaults to None.
        stopper (Optional[EarlyStopping], optional): If set, track the loss
            value and end training when patience is exceeded. Dev loss is used
            if `dev_data` is set, otherwise uses training loss.
            Defaults to None.
        device (Optional[torch.device], optional): Send probe, loss function,
            and all tensors in dataset to this device throughout training.
            By default, device is not changed on any module or tensor.
        lr (float, optional): Learning rate for Adam optimization.
            Defaults to 1e-3.
        epochs (int, optional): Maximum number of passes to make through
            `train_data`. Defaults to 25.
        also_log_to_wandb (bool, optional): Log train and, if applicable, dev
            losses to wandb. Defaults to False.

    """
    log = logging.getLogger(__name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    if device:
        probe = probe.to(device)
        criterion = criterion.to(device)

    for epoch in range(epochs):
        probe.train()
        for iteration, (inputs, tags) in enumerate(train_dataset):
            if device:
                inputs, tags = inputs.to(device), tags.to(device)

            optimizer.zero_grad()
            preds = probe(inputs)
            loss = criterion(preds, tags)
            loss.backward()
            optimizer.step()

            log.info('epoch %d batch %d train loss %f', epoch + 1,
                     iteration + 1, loss.item())

            if also_log_to_wandb:
                wandb.log({'train accuracy': loss})

            if not dev_dataset and stopper and stopper(loss.item()):
                log.info('patience on train loss exceed, training is now over')
                return

        if dev_dataset is not None:
            probe.eval()
            dev_loss, count = 0., 0
            for inputs, tags in dev_dataset:
                if device is not None:
                    inputs, tags = inputs.to(device), tags.to(device)
                preds = probe(inputs)
                dev_loss += criterion(preds, tags).item() * len(inputs)
                count += len(inputs)
            dev_loss /= count

            log.info('epoch %d dev loss %f', epoch + 1, dev_loss)

            if also_log_to_wandb:
                # Technically this creates a new step, when it should use the
                # same step as the last train loss log event. It irks me, but
                # oh well.
                wandb.log({'dev accuracy': dev_loss})

            if stopper and stopper(dev_loss):
                log.info('patience on dev loss exceed, training is now over')
                return


def test(probe: nn.Module,
         dataset: TaskDataset,
         device: Optional[torch.device] = None) -> float:
    """Compute classification accuracy of a probe on the given data.

    This function makes the same assumptions as `train` above. Importantly,
    it assumes the last dimension of the probes outputs are logits, and will
    take the argmax across that dimension to determine class predictions.

    Args:
        probe (nn.Module): The classifier probe.
        dataset (Iterable[Batch]): Inputs to classify paired with labels. See
            `train_dataset` parameter in `train`.
        device (Optional[torch.device], optional): Probe, inputs, and tags
            will be sent to this device. By default, device is not changed
            on any module or tensor.

    Raises:
        ValueError: If `dataset` is empty.

    Returns:
        float: Fraction of correctly classified data points.

    """
    if device:
        probe.to(device)

    total, count = 0, 0
    for inputs, tags in dataset:
        if device:
            inputs, tags = inputs.to(device), tags.to(device)
        count += len(inputs)
        with torch.no_grad():
            total += probe(inputs).argmax(dim=-1).eq(tags).sum().item()
    assert total <= count, 'more correct than counted?'

    # There should never be a case where we hand this function an empty
    # dataset, so explode if that is the case.
    if not count:
        raise ValueError('no data in dataset')

    return total / count
