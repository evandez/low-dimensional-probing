"""Defines datasets for training probes."""

import pathlib
from typing import List, Tuple

import h5py
import torch
from torch.utils import data

from lodimp import ptb, tasks


class ELMoRepresentationsDataset(data.Dataset):
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
        assert '0' in self.file, 'ELMo reps file has no 0th element?'
        self.dimension = self.file['0'].shape[-1]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Returns the ELMo represenations for the sentence at the given index.

        Args:
            index (int): Which sentence to retrieve ELMo reps for.

        Returns:
            ELMo representations for the given sentence, given as shape (L, D)
            tensor, where L is length of the sentence and D is the ELMo layer
            dimensionality.

        """
        if not index < len(self):
            raise IndexError(f'index out of bounds: {index}')
        return torch.tensor(self.file[str(index)][self.layer])

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.file.keys()) - 1  # Ignore sentence_to_index.


class PTBDataset(data.Dataset):
    """Iterates over Penn Treebank for some task.

    A task is a mapping from samples to tags. So in principle this is just
    a dataset of tags, to be used in conjunction with some dataset of
    representations.
    """

    def __init__(self, samples: List[ptb.Sample], task: tasks.Task):
        """Initialize the dataset.

        Args:
            samples (List[ptb.Sample]): The Penn Treebank samples.
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

        """
        return self.task(self.samples[index])

    def __len__(self) -> int:
        """Returns the number of tagged sentences in this dataset."""
        return len(self.samples)


class ZippedDatasets(data.Dataset):
    """Zips one or more datasets."""

    def __init__(self, *datasets: data.Dataset):
        """Initialize and verify datasets."""
        super(ZippedDatasets, self).__init__()
        if not len(datasets):
            raise ValueError('must specify at least one dataset.')

        self.datasets = datasets
        self.length = min([len(dataset) for dataset in datasets])

    def __len__(self) -> int:
        """Returns the length of the zipped dataset."""
        return self.length

    def __getitem__(self, index: int) -> Tuple:
        """Zips the index'th items of each dataset.

        Args:
            index (int): Which items to zip.

        Returns:
            Tuple: The index'th item from each dataset.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        return (*[dataset[index] for dataset in self.datasets],)
