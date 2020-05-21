"""Utilities for interacting with precomputed word representations."""

import pathlib

import h5py
import torch
from torch.utils import data


class RepresentationsDataset(data.Dataset):
    """Iterates through a dataset of word representations."""

    def __init__(self, path: pathlib.Path, layer: int):
        """Load the h5 file contained pre-computed representations.

        Args:
            path (str): Path to h5 file containing pre-computed reps.
            layer (int): Which layer to use.

        """
        super(RepresentationsDataset, self).__init__()
        self.file = h5py.File(path, 'r')
        self.layer = layer

        assert '0' in self.file, 'reps file has no 0th element?'
        layers = self.file['0'].shape[0]
        if layer < 0 or layer >= layers:
            raise IndexError(f'expected layer in [0, {layers}), got {layer}')

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the ELMo representations."""
        assert '0' in self.file, 'reps file has no 0th element?'
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
