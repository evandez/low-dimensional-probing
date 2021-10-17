"""Defines task datasets.

A task is a mapping from one or more word representations to linguistic
features. The datasets
"""
import logging
import pathlib
from typing import Any, Iterator, Optional, Sequence, Tuple

from ldp.utils.typing import Device, PathLike

import h5py
import torch
from torch.utils import data

DEFAULT_H5_BREAKS_KEY = 'breaks'
DEFAULT_H5_REPRESENTATIONS_KEY = 'representations'
DEFAULT_H5_FEATURES_KEY = 'features'
H5_UNIQUE_FEATURES_KEY = 'unique-features'

# Old defaults...keeping them here just in case.
# DEFAULT_H5_REPRESENTATIONS_KEY = 'reps'
# DEFAULT_H5_FEATURES_KEY = 'tags'
# H5_UNIQUE_FEATURES_KEY = 'ntags'


class TaskDataset(data.IterableDataset):
    """Abstract dataset of (representation, linguistic features) pairs."""

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield batches of samples.

        Each batch should be a tuple of tensors: (representations, features).
        The representations tensor should have shape
        (batch_size, *sample_representations_shape) and the features tensor
        should have shape (batch_size, *sample_features_shape).

        Typically, one batch is one sentence worth of representations.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the batch of samples at the given index.

        Typically, one batch is one sentence worth of representations.

        Args:
            index (int): Which batch to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First tensor contains
                word representations and has shape
                (batch_size, *sample_representations_shape), second contains
                linguistic features (given as integral values) and has shape
                (batch_size, *shample_features_shape).

        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of batches to iterate over."""
        raise NotImplementedError

    @property
    def sample_representations_shape(self) -> Sequence[int]:
        """Return the shape of the representations tensor for one sample."""
        raise NotImplementedError

    @property
    def sample_features_shape(self) -> Sequence[int]:
        """Return the shape of the features tensor for one sample."""
        raise NotImplementedError

    def count_samples(self) -> int:
        """Return the number of individual samples in the dataset.

        This is usually the total number of words in the dataset. However,
        the representations yielded by __iter__ do not necessarily correspond
        to individual words. E.g., in DLP, they correspond to pairs of words.
        So we use the term "samples" instead.

        This function is not an @property because it may require lengthy
        computation, e.g. if all sentences have to be read from disk to
        determine the number of words in each sentence.
        """
        raise NotImplementedError

    def count_unique_features(self) -> Optional[int]:
        """Return number of unique features, if that quantity makes sense.

        E.g. in part of speech tagging, there are 45 parts of speech, so this
        should return 45. However, for dependency edge prediction, the number
        of outputs is the length of the current sentence, so there is no one
        number that makes sense for this function. Returns None in that case.

        This function is not an @property because it may require lengthy
        computation.
        """
        raise NotImplementedError

    def collate(
        self,
        out: PathLike,
        breaks_key: str = DEFAULT_H5_BREAKS_KEY,
        representations_key: str = DEFAULT_H5_REPRESENTATIONS_KEY,
        features_key: str = DEFAULT_H5_FEATURES_KEY,
        force: bool = False,
    ) -> None:
        """Collate the task data into contiguous arrays and write it to disk.

        The resulting h5 file will have two datasets: one for representations,
        which will be a contiguous array of all representations in order,
        and one for linguistic features represented as integers. The datasets
        will align, in that the i-th representation in the representations
        dataset should be tagged with the i-th feature in the features dataset.

        Args:
            out (PathLike): Path at which to write output file. Must not
                exist, unless force is set to True.
            breaks_key (str, optional): Key to use for the breaks dataset
                in the output h5 file. Defaults to 'breaks'.
            representations_key (str, optional): Key to use for the
                representations dataset in the output h5 file.
                Defaults to 'representations'.
            features_key (str, optional): Same as above, for the dataset of
                linguistic features. Defaults to 'features'.
            force (bool, optional): Overwrite the output file if it exists.
                Defaults to False.

        Raises:
            FileExistsError: If output file exists and `force=False`.
            ValueError: If length of representations does not equal length of
                features, or if any samples therein have different sequence
                lengths.

        """
        out = pathlib.Path(out)
        if out.exists() and not force:
            raise FileExistsError(f'{out} exists, set force=True to overwrite')

        log = logging.getLogger(__name__)
        log.info('counting samples to write, this may take a minute')
        n_samples = self.count_samples()
        log.info('%d sentences to collate, %d reps/features total', len(self),
                 n_samples)

        with h5py.File(out, 'w') as handle:
            breaks_out = handle.create_dataset(breaks_key,
                                               shape=(len(self),),
                                               dtype='i')
            reps_out = handle.create_dataset(
                representations_key,
                shape=(n_samples, *self.sample_representations_shape),
                dtype='f',
            )
            features_out = handle.create_dataset(
                features_key,
                shape=(n_samples, *self.sample_features_shape),
                dtype='i',
            )
            start, end = 0, 0
            for index, (reps, features) in enumerate(self):
                log.info('writing %d of %d', index + 1, len(self))
                end = start + len(reps)
                breaks_out[index] = start
                reps_out[start:end] = reps
                features_out[start:end] = features
                start = end
            assert end == n_samples, 'did not finish writing?'

            log.info('counting unique features, this may take a minute')
            unique_features = self.count_unique_features()
            if unique_features is not None:
                log.info('found %d unique features, will write to metadata',
                         unique_features)
                features_out.attrs[H5_UNIQUE_FEATURES_KEY] = unique_features
                log.info('collation complete')


class CollatedTaskDataset(TaskDataset):
    """Dataset of pre-collated representations/features tensors.

    This class wraps the pre-computed h5 file that is written by
    TaskDataset.collate(...). The collate operation is idempotent, but
    is intended to be used called on custom subclasses of TaskDataset
    that implement the task-specific mapping from representations to features.

    Because we train many, many probes on the same tasks, we must preprocess
    the representations and features to make training as fast as possible.
    The key to fast training is to avoid collation and I/O between disk,
    RAM, and the GPU memory, ideally moving the entire dataset to the GPU
    at the beginning of training and then relying on the GPU's finesse with
    large matix multiplications. To faciliate this, we pre-collate
    all representations into a contiguous array and store it in an h5 file.
    We do the same for the integral features, storing them in a separate array.
    """

    def __init__(self,
                 path: PathLike,
                 representations_key: str = DEFAULT_H5_REPRESENTATIONS_KEY,
                 features_key: str = DEFAULT_H5_FEATURES_KEY,
                 device: Optional[Device] = None):
        """Initialize the task dataset.

        Args:
            path (PathLike): Path to the preprocessed h5 file.
            representations_key (str, optional): Key for the dataset of
                representations in the h5 file. Defaults to 'representations'.
            features_key (str, optional): Key for the dataset of features in
                the h5 file. Defaults to 'features'.
            device (Optional[Device], optional): Move data to this device.
                By default, data configured for CPU.

        Raises:
            KeyError: If an expected dataset is missing.

        """
        self.file = h5py.File(str(path), 'r')

        for key in (representations_key, features_key):
            if key not in self.file:
                raise KeyError(f'dataset not in h5 file: {key}')

        self.representations = self.file[representations_key]
        self.features = self.file[features_key]
        assert len(self.features) == len(self.representations)

        self.representations_cache = None
        self.features_cache = None
        if device is not None:
            self.representations_cache = torch.tensor(self.representations[:],
                                                      device=device)
            self.features_cache = torch.tensor(self.features[:],
                                               dtype=torch.long,
                                               device=device)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the (representations, features) pair at the given index.

        Args:
            index (int): The index of the pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The (representations, features)
                tensors, the first with shape
                (1, *sample_representations_shape) and second with shape
                (1, *sample_features_shape).

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'sample index out of bounds: {index}')

        if self.representations_cache is not None:
            assert self.features_cache is not None, 'features not cached?'
            reps = self.representations_cache[index]
            features = self.features_cache[index]
        else:
            reps = torch.tensor(self.representations[index])
            features = torch.tensor(self.features[index], dtype=torch.long)
        return reps, features

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield every sample in the dataset in sequence."""
        for index in range(len(self)):
            reps, features = self[index]
            # Unsqueeze so the reps appear batch-like.
            yield reps.unsqueeze(0), features.unsqueeze(0)

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.representations)

    @property
    def sample_representations_shape(self) -> Sequence[int]:
        """Return the representation dimensionality."""
        return self.representations.shape[1:]

    @property
    def sample_features_shape(self) -> Sequence[int]:
        """Return the representation dimensionality."""
        return self.features.shape[1:]

    def count_samples(self) -> int:
        """See TaskDataset.count_samples."""
        return len(self)

    def count_unique_features(self) -> Optional[int]:
        """See TaskDataset.count_unique_features."""
        return self.features.attrs.get(H5_UNIQUE_FEATURES_KEY)


class SentenceBatchingCollatedTaskDataset(CollatedTaskDataset):
    """A TaskDataset that iterates at the sentence level.

    While our chosen file layout inherently destroys sentence boundaries,
    we record the start index of each sentence in a third dataset, which we
    refer to as the "breaks" dataset. This class uses it to allow iteration
    of representations/features at the sentence level.
    """

    def __init__(self,
                 path: PathLike,
                 breaks_key: str = DEFAULT_H5_BREAKS_KEY,
                 **kwargs: Any):
        """Initialize the dataset.

        Keyword arguments are forwarded to CollatedTaskDataset.__init__ call.

        Args:
            path (PathLike): Path to the preprocessed h5 file.
            breaks_key (str, optional): Key for the dataset of sentence breaks
                in the h5 file. Defaults to 'breaks'.

        Raises:
            KeyError: If breaks dataset not found.

        """
        super().__init__(path, **kwargs)
        if breaks_key not in self.file:
            raise KeyError(f'dataset not in h5 file: {breaks_key}')

        # We never deliberately expose the breaks dataset, and it should never
        # be so big that it does not fit into memory, so just read it all.
        self.breaks = self.file[breaks_key][:]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all representations and features for the index'th sentence.

        Args:
            index (int): Index of the sentence data to retrieve.
            device (Optional[Device], optional): Move data to this device.
                By default, data configured for CPU.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Shape (L, *) tensors, where
                L is the length of the index'th sentence and * is whatever
                shape the representations/features would have if you accessed
                them with PreprocessedDataset.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'sentence index out of bounds: {index}')

        start = self.breaks[index]
        if index < len(self) - 1:
            end = self.breaks[index + 1]
        else:
            assert index == len(self) - 1
            end = len(self.representations)

        if self.representations_cache is not None:
            assert self.features_cache is not None, 'features not cached?'
            reps = self.representations_cache[start:end]
            features = self.features_cache[start:end]
        else:
            reps = torch.tensor(self.representations[start:end])
            features = torch.tensor(self.features[start:end], dtype=torch.long)

        return reps, features

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over samples batched by sentence."""
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        """Return the number of sentences in the dataset."""
        return self.breaks.shape[0]


class NonBatchingCollatedTaskDataset(CollatedTaskDataset):
    """A TaskDataset where all data is treated as one batch."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the collated dataset.

        Args:
            index (int): Index of the dataset. Must be 0. This parameter is
                here for sake of type checking.

        Raises:
            IndexError: If the index is not 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The full, collated
                (representations, features) dataset.

        """
        if index != 0:
            raise IndexError(f'index must be 0, got {index}')
        if self.representations_cache is not None:
            assert self.features_cache is not None, 'features not cached?'
            reps = self.representations_cache
            features = self.features_cache
        else:
            reps = torch.tensor(self.representations[:])
            features = torch.tensor(self.features[:], dtype=torch.long)
        return reps, features

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield the one and only chunk of data."""
        yield self[0]

    def __len__(self) -> int:
        """Return 1 because there is only one chunk."""
        return 1
