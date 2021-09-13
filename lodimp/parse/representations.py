"""Utilities for interacting with precomputed word representations."""
from typing import Any

from lodimp.utils.typing import PathLike

import h5py
import torch
from torch.utils import data


class RepresentationDataset(data.Dataset):
    """Iterates through a dataset of word representations."""

    def __init__(self, path: PathLike, **kwargs: Any):
        """Load the h5 file contained pre-computed representations.

        Keyword arguments are forwarded to h5py.File constructor.

        Args:
            path (PathLike): Path to h5 file containing pre-computed reps.

        """
        super(RepresentationDataset, self).__init__()
        assert 'mode' not in kwargs, 'why are you setting mode?'
        self.file = h5py.File(str(path), 'r', **kwargs)

        assert '0' in self.file, 'reps file has no 0th element?'
        example = self.file['0']
        self.dimension = example.shape[-1]
        self.layers = self.file['0'].shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the represenations for the sentence at the given index.

        Args:
            index (int): Which sentence to retrieve representations for.

        Returns:
            torch.Tensor: Representations for the given sentence, given
                as shape (L, N, D) tensor, where L is the number of
                representation layers, N is length of the sentence and D
                is the representation dimensionality.

        Raises:
            IndexError: If the index is out of bounds.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        return torch.tensor(self.file[str(index)])

    def __len__(self) -> int:
        """Return the number of samples (sentences) in the dataset."""
        return len(self.file.keys()) - 1  # Ignore sentence_to_index.

    def length(self, index: int) -> int:
        """Return the length of the index'th sequence.

        This function exists because it is much faster than using __getitem__.
        It only reads file metadata as opposed to reading in every single
        representation.

        Args:
            index (int): Index of the sample to fetch the length for.

        Returns:
            Number of representations (words, i.e. length of the sentence)
            for the sample.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        return self.file[str(index)].shape[1]

    def layer(self, layer: int) -> 'RepresentationLayerDataset':
        """Create a view of this dataset restricted to one layer.

        Args:
            layer (int): The layer to restrict to.

        Returns:
            RepresentationLayerDataset: The restricted dataset.

        """
        return RepresentationLayerDataset(self, layer)


class RepresentationLayerDataset(data.Dataset):
    """Wrapper around RepresentationDataset that restricts to one layer."""

    def __init__(self, dataset: RepresentationDataset, layer: int):
        """Initialize the dataset.

        Args:
            dataset (RepresentationDataset): Dataset to restrict.
            layer (int): Layer to restrict to.

        Raises:
            IndexError: If the layer number is out of bounds.

        """
        if layer < 0 or layer >= dataset.layers:
            raise IndexError(f'layer {layer} out of bounds')

        self.dataset = dataset
        self.layer = layer

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the represenations layer for the sentence at the index.

        Args:
            index (int): Which sentence to retrieve representations for.

        Returns:
            torch.Tensor: Layer representations for the given sentence, given
                as shape (N, D) tensor, where N is length of the sentence and D
                is the representation dimensionality.

        """
        return self.dataset[index][self.layer]

    def __len__(self) -> int:
        """Return the number of samples (sentences) in the dataset."""
        return len(self.dataset)
