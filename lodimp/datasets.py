"""Defines datasets for training probes."""

import pathlib
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Type

from lodimp import ptb, tasks

import h5py
import torch
from torch.utils import data


class RepresentationsDataset(data.Dataset):
    """Abstract dataset of word representations."""

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the representations."""
        raise NotImplementedError


class ELMoRepresentationsDataset(RepresentationsDataset):
    """Iterates through a dataset of word representations."""

    def __init__(self, path: pathlib.Path, layer: int):
        """Load the h5 file contained pre-computed ELMo representations.

        Args:
            path (str): Path to h5 file containing pre-computed reps.
            layer (int): Which ELMo layer to use.

        """
        super(ELMoRepresentationsDataset, self).__init__()
        if layer not in (0, 1, 2):
            raise ValueError(f'invalid layer: {layer}')
        self.file = h5py.File(path, 'r')
        self.layer = layer

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the ELMo representations."""
        assert '0' in self.file, 'ELMo reps file has no 0th element?'
        return self.file['0'].shape[-1]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Returns the ELMo represenations for the sentence at the given index.

        Args:
            index (int): Which sentence to retrieve ELMo reps for.

        Returns:
            ELMo representations for the given sentence, given as shape (L, D)
            tensor, where L is length of the sentence and D is the ELMo layer
            dimensionality.

        Raises:
            IndexError: If the index is out of bounds.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        return torch.tensor(self.file[str(index)][self.layer])

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.file.keys()) - 1  # Ignore sentence_to_index.


class LabelsDataset(data.Dataset):
    """Iterates over Penn Treebank for some task.

    A task is a mapping from samples to tags. So in principle this is just
    a dataset of tags, to be used in conjunction with some dataset of
    representations.
    """

    def __init__(self, samples: Sequence[ptb.Sample], task: tasks.Task):
        """Initialize the dataset.

        Args:
            samples (Sequence[ptb.Sample]): The Penn Treebank samples.
            task (tasks.Task): Maps Treebank samples to tags.

        """
        self.samples = samples
        self.task = task

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the tags for the index-th sentence.

        Args:
            index (int): Which tags to return.

        Returns:
            torch.Tensor: Integer-encoded tags.

        Raises:
            IndexError: If the index is out of bounds.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        return self.task(self.samples[index])

    def __len__(self) -> int:
        """Returns the number of tagged sentences in this dataset."""
        return len(self.samples)


class LabeledRepresentationsDataset(data.Dataset):
    """Word representations paired with labels.

    This dataset effectively zips a dataset of word representations and a
    dataset of word labels. The data are assumed to be grouped by sentence,
    but note that this dataset iterates at the word level.
    """

    def __init__(self, reps: RepresentationsDataset, labels: LabelsDataset):
        """Validate the datasets and preprocess for fast access.

        Args:
            reps (data.Dataset): The dataset of word representations.
            labels (LabelsDataset): The dataset of word labels.

        Raises:
            ValueError: If the datasets have different lengths or if any pair
                of sentences in the datasets have mismatched lengths.

        """
        self.reps = reps
        self.labels = labels

        if len(reps) != len(labels):
            raise ValueError(f'rep/label datasets have different sizes: '
                             f'{len(reps)} vs. {len(labels)}')

        self.coordinates = []
        for index in range(len(reps)):
            nreps, nlabels = len(reps[index]), len(labels[index])
            if nreps != nlabels:
                raise ValueError(f'sample {index} has {nreps} representations '
                                 f'but {nlabels} labels')
            self.coordinates += [(index, word) for word in range(nreps)]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch the index'th (rep, label) pair.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The (rep, label) pair.

        Raises:
            IndexError: If the index is out of bounds.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        sentence, word = self.coordinates[index]
        rep = self.reps[sentence][word]
        label = self.labels[sentence][word]
        return rep, label

    def __len__(self) -> int:
        """Returns the number of (rep, label) pairs in the dataset."""
        return len(self.coordinates)


class LabeledRepresentationPairsDataset(data.Dataset):
    """Pairs of word representations with labels.

    This dataset assumes that for a sentence of length W, the labels are
    given by a shape (W, W) matrix denoting some relationship between
    each pair of words, e.g. for dependency edge prediction.

    Note that this dataset includes ALL negative examples, so its length
    will be larger than the total number of words in the original dataset.
    """

    def __init__(self, reps: RepresentationsDataset, labels: LabelsDataset):
        """Validate the datasets and preprocess for fast access.

        Args:
            reps (data.Dataset): The dataset of word representations.
            labels (LabelsDataset): The dataset of word labels.

        Raises:
            ValueError: If the datasets have different lengths or if any pair
                of sentences in the datasets have mismatched lengths.

        """
        self.reps = reps
        self.labels = labels

        if len(reps) != len(labels):
            raise ValueError(f'rep/label datasets have different sizes: '
                             f'{len(reps)} vs. {len(labels)}')

        self.coordinates = []
        for index in range(len(reps)):
            nreps, lshape = len(reps[index]), labels[index].shape
            if lshape != (nreps, nreps):
                raise ValueError(f'sample {index} has {nreps} representations '
                                 f'but size {lshape} label matrix')
            self.coordinates += [
                (index, wi, wj) for wi in range(nreps) for wj in range(nreps)
            ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch the index'th (rep, label) pair.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The (rep, label) pair.

        Raises:
            IndexError: If the index is out of bounds.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        sentence, wi, wj = self.coordinates[index]
        reps = self.reps[sentence]
        rep = torch.cat((reps[wi], reps[wj]))
        label = self.labels[sentence][wi, wj]
        return rep, label

    def __len__(self) -> int:
        """Returns the number of (rep, label) pairs in the dataset."""
        return len(self.coordinates)


# TODO(evandez): Split up or test this function.
def load_elmo_ptb(
        path: pathlib.Path,
        task: Type[tasks.Task],
        layers: Sequence[int] = (0, 1, 2),
        splits: Sequence[str] = ('train', 'dev', 'test'),
        ptb_prefix: str = 'ptb3-wsj-',
        ptb_suffix: str = '.conllx',
        elmo_prefix: str = 'raw.',
        elmo_suffix: str = '.elmo-layers.hdf5',
        **kwargs: Any) -> Sequence[Dict[str, LabeledRepresentationsDataset]]:
    """Load the ELMo-encoded PTB from the given directory.

    Args:
        path (pathlib.Path): Path to data directory.
        task (Type[tasks.Task]): Task used to derive labels from samples.
        layers (Sequence[int], optional): Which ELMo layers to load.
            Defaults to (0, 1, 2).
        splits (Sequence[str], optional): Which dataset splits to look for.
            Defaults to ('train', 'dev', 'test').
        ptb_prefix (str, optional): Expected prefix for PTB files.
            Defaults to 'ptb3-wsj-'.
        ptb_suffix (str, optional): Expected suffix for PTB files.
            Defaults to '.conllx'.
        elmo_prefix (str, optional): Expected prefix for ELMo files.
            Defaults to 'raw.'.
        elmo_suffix (str, optional): Expected suffix for ELMo files.
            Defaults to '.elmo-layers.hdf5'.

    Returns:
        Sequence[Dict[str, LabeledRepresentationsDataset]]: Loaded datasets.
            Grouped first by ELMo layer, then by split.

    """
    elmo_paths, labels = {}, {}
    for split in splits:
        samples = ptb.load(path / f'{ptb_prefix}{split}{ptb_suffix}')
        labels[split] = LabelsDataset(samples, task(samples, **kwargs))
        elmo_paths[split] = path / f'{elmo_prefix}{split}{elmo_suffix}'

    groups = []
    for layer in layers:
        datasets = {}
        for split in splits:
            elmos = ELMoRepresentationsDataset(elmo_paths[split], layer)
            dataset = LabeledRepresentationsDataset(elmos, labels[split])
            datasets[split] = dataset
        groups.append(datasets)

    return (*groups,)


class CollatedDataset(data.IterableDataset):
    """Pre-collate a dataset.

    In other words, treat the entire dataset as one giant batch. Items may
    not be accessed individually.
    """

    def __init__(self,
                 dataset: data.Dataset,
                 device: Optional[torch.device] = None,
                 **kwargs: Any):
        """Run the collation and cache the results.

        Keyword arguments forwarded to torch.utils.data.DataLoader.

        Args:
            dataset (data.Dataset): The dataset to pre-collate.
            device (torch.device): Send data to this device immediately
                so it need not be done repeatedly.

        """
        super().__init__()
        for forbidden in ('batch_size', 'shuffle'):
            if forbidden in kwargs:
                raise ValueError(f'cannot set {forbidden}')

        self.collated, = list(
            data.DataLoader(
                dataset,
                batch_size=len(dataset),
                **kwargs,
            ))

        if device is not None:
            self.collated = [
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in self.collated
            ]

    def __iter__(self) -> Iterable:
        """No need to iterate over a collated dataset. Just yield the data."""
        yield self.collated

    def __len__(self) -> int:
        """Returns the number of samples in the individual dataset."""
        return 1
