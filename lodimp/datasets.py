"""Defines datasets for training probes."""

import math
import pathlib
from typing import Iterator, Optional, Sequence, Tuple, Union

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

        assert 'breaks' in self.file, 'no sentence breaks?'
        self.breaks = list(self.file['breaks'][:])

        assert 'reps' in self.file, 'no reps?'
        self.representations = self.file['reps']

        assert 'tags' in self.file, 'no tags?'
        self.labels = self.file['tags']

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the (features, label) pair at the given index.

        Args:
            index (int): The index of the pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The (feature, label) tensors.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        features = torch.tensor(self.representations[index])
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        return features, labels

    def __len__(self) -> int:
        """Returns the number of samples in this dataset."""
        return len(self.representations)

    @property
    def ngrams(self) -> int:
        """Returns number of representations in each sample."""
        if len(self.representations.shape) == 2:
            # Shape must be (nsamples, nfeatures), so only unigrams.
            return 1
        else:
            assert len(self.representations.shape) == 3
            return self.representations.shape[1]

    @property
    def ndims(self) -> int:
        """Returns the representation dimensionality."""
        return self.representations.shape[-1]

    @property
    def ntags(self) -> Optional[int]:
        """Returns the number of unique labels in the dataset.

        If this quantity is not defined for the task, returns None.
        """
        return self.labels.attrs.get('ntags')


class SentenceTaskDataset(data.IterableDataset):
    """Wrapper for TaskDataset that iterates at the sentence level."""

    def __init__(self, dataset: TaskDataset):
        """Initialize the dataset."""
        self.dataset = dataset

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over (sentence representations, sentence labels) pairs."""
        starts = self.dataset.breaks
        ends = starts[1:] + [len(self.dataset)]
        for start, end in zip(starts, ends):
            reps = torch.tensor(self.dataset.representations[start:end])
            labels = torch.tensor(self.dataset.labels[start:end],
                                  dtype=torch.long)
            yield reps, labels

    def __len__(self) -> int:
        """Returns the number of sentences in the dataset."""
        return len(self.dataset.breaks)


class ChunkedTaskDataset(TaskDataset, data.IterableDataset):
    """Iterable wrapper around TaskDataset that pre-collates chunks in memory.

    Unlike SentenceTaskDataset above, this class can batch the data
    arbitrarily and, importantly, it loads the entire dataset into memory
    immediately upon construction and then batches it. The intention is for
    this class to be used when running on high-memory, high-throughput devices
    on which I/O and collation become the bottleneck during training.
    """

    def __init__(self,
                 dataset: TaskDataset,
                 chunks: Union[int, Sequence[int]] = 1,
                 device: Optional[torch.device] = None):
        """Chunk the dataset for later iteration.

        Args:
            dataset (TaskDataset): The dataset to chunk.
            chunks (int, optional): Number of chunks to create. Defaults to 1.
            device (Optional[torch.device], optional): If set, send chunks to
                this device. By default chunks are kept on current device.

        """
        if isinstance(chunks, int):
            size = math.ceil(len(dataset) / chunks)
            starts = [index * size for index in range(chunks)]
        else:
            starts = list(chunks)

        # Append the index of the end of the dataset so we include the
        # last chunk with the loop below.
        starts.append(len(dataset))

        self.chunks = []
        for start, end in zip(starts, starts[1:]):
            reps = torch.tensor(dataset.representations[start:end])
            labels = torch.tensor(dataset.labels[start:end], dtype=torch.long)
            if device is not None:
                reps, labels = reps.to(device), labels.to(device)
            chunk = (reps, labels)
            self.chunks.append(chunk)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterates over all chunks."""
        return iter(self.chunks)

    def __len__(self) -> int:
        """Returns the number of chunks in the dataset."""
        return len(self.chunks)
