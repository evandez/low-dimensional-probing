"""Defines datasets for training probes."""

import pathlib
from typing import Any, Iterable, Optional, Tuple

import h5py
import torch
from torch.utils import data


class RepresentationsDataset(data.Dataset):
    """Abstract dataset of word representations."""

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the representations."""
        raise NotImplementedError

    def length(self, index: int) -> int:
        """Returns the length of the index'th sequence.

        This is generally much faster than reading the sequence from disk.
        Useful for preprocessing.
        """
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

    def length(self, index: int) -> int:
        """Determines the length of the index'th sequence.

        Only looks at file metadata, so this function is fast.

        Args:
            index: The sequence to find the length of.

        Returns:
            The index'th sequence length.

        Raises:
            IndexError: If the index is out of bounds.

        """
        if index < 0 or index >= len(self):
            raise IndexError()
        return self.file[str(index)].shape[1]

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


class TaskDataset(data.Dataset):
    """Iterates through a precomputed task.

    Tasks map word representations to labels. See preprocess.py and tasks.py
    to see how this is done. This dataset simply reads the pre-computed task.
    """

    def __init__(self, path: pathlib.Path):
        """Initialize the task dataset.

        Args:
            path (pathlib.Path): Path to the h5 file containing the task.

        """
        self.file = h5py.File(path, 'r')
        assert 'features' in self.file, 'no features?'
        assert 'labels' in self.file, 'no labels?'

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the (features, label) pair at the given index.

        Args:
            index (int): The index of the pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The (feature, label) tensors.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        features = torch.tensor(self.file['features'][index])
        labels = torch.tensor(self.file['labels'][index])
        return features, labels

    def __len__(self) -> int:
        """Returns the number of samples in this dataset."""
        return len(self.file['features'])

    @property
    def nfeatures(self) -> int:
        """Returns the number of features in each sample."""
        return self.file['features'].shape[-1]

    @property
    def nlabels(self) -> int:
        """Returns the number of unique labels in the dataset."""
        return self.file['labels'].attrs['nlabels']


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
