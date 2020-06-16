"""Defines core experiments for part of speech tagging task."""

import collections
import itertools
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

from lodimp.common import tasks
from lodimp.common.data import ptb, representations as reps

import numpy as np
import torch

# Standard unknown symbol.
UNK = 'unk'

# Frequently used POS tags.
NOUNS = ('NN', 'NNS', 'NNP', 'NNPS')
NOUNS_PROPER = ('NNP', 'NNPS')
NOUNS_PLURAL = ('NNS', 'NNPS')
VERBS = ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
VERBS_PRESENT = ('VBZ', 'VBP', 'VBG')
VERBS_PAST = ('VBD', 'VBN')
ADJECTIVES = ('JJ', 'JJR', 'JJS')
ADVERBS = ('RB', 'RBR', 'RBS')


class POSIndexer:
    """Indexes PTB POS tags."""

    def __init__(self,
                 samples: Sequence[ptb.Sample],
                 distinguish: Optional[Sequence[str]] = None,
                 unk: str = UNK):
        """Maps each POS tag to an index.

        Args:
            samples (Sequence[ptb.PTBSample]): The samples from which to
                draw tags.
            distinguish (Optional[Sequence[str]]): The XPOS tags to
                distinguish. All tags not in this set will be collapsed to
                the unk tag. By default, all tags will be distinguished.
            unk (str): Tag to use when un-indexed XPOS is encountered.
                If distinguish is set, the tags not in that sequence will be
                set to this tag.

        """
        if distinguish is None:
            tags = {xpos for sample in samples for xpos in sample.xpos}
        else:
            tags = set(distinguish)

        self.indexer = {unk: 0}
        for xpos in sorted(tags):
            self.indexer[xpos] = len(self.indexer)
        self.unk = unk

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Index the part-of-speech tags for each sample.

        Args:
            samples (ptb.PTBSample): The sample to index XPOS tags for.

        Returns:
            torch.Tensor: Integer tags for each XPOS in the sample.

        """
        return torch.tensor([
            self.indexer.get(xpos, self.indexer[self.unk])
            for xpos in sample.xpos
        ])

    def __len__(self) -> int:
        """Returns the number of valid POS tags in this task."""
        return len(self.indexer)


class ControlPOSIndexer:
    """Maps words to arbitrary POS tags."""

    def __init__(self,
                 *groups: Sequence[ptb.Sample],
                 dist: Optional[Sequence[float]] = None):
        """Initialize the tagger.

        The tagger computes the empirical distribution of the samples, if not
        provided, and then uses it to generate arbitrary integer tags for each
        individual word type.

        Args:
            *groups (Sequence[ptb.PTBSample]): All samples, provided in one or
                more sequences, for which to generate tags.
            dist (Optional[Sequence[float]], optional): The empirical
                distribution to use when sampling tags for word type.
                By default, is computed from the list of samples.

        """
        samples = tuple(itertools.chain(*groups))
        if dist is None:
            counts: Dict[str, int] = collections.defaultdict(lambda: 0)
            for sample in samples:
                for pos in sample.xpos:
                    counts[pos] += 1
            dist = np.array([float(count) for count in counts.values()])
            dist /= np.sum(dist)
        assert dist is not None, 'uninitialized distribution?'
        self.dist = dist

        self.tags: Dict[str, int] = {}
        for sample in samples:
            for word in sample.sentence:
                if word not in self.tags:
                    self.tags[word] = np.random.choice(len(dist), p=dist) + 1

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Tag the given samples.

        Args:
            sample (ptb.PTBSample): The sample to tag.

        Returns:
            torch.Tensor: Integer tags for every word in the sentence.
                If the word type is unknown, it's tag will be 0.

        """
        return torch.tensor(
            [self.tags.get(word, 0) for word in sample.sentence])

    def __len__(self) -> int:
        """Returns the number of fake tags in this task."""
        return len(self.dist) + 1  # add 1 for unk tag


class POSTaskDataset(tasks.TaskDataset):
    """Iterates over (word representation, POS tag) pairs."""

    def __init__(
        self,
        representations: reps.RepresentationLayerDataset,
        annotations: Sequence[ptb.Sample],
        indexer: Union[POSIndexer, ControlPOSIndexer],
    ):
        """Maps each POS tag to an index.

        Args:
            representations (representations.RepresentationsLayerDataset): Word
                representations corresponding to the words to be tagged.
            annotations (Sequence[ptb.PTBSample]): The PTB annotations from
                which to pull POS tags.
            indexer (Union[POSIndexer, ControlPOSIndexer]): Callable mapping
                PTB annotations to integer tensors.

        Raises:
            ValueError: If number of representations/annotations do not match.

        """
        if len(representations) != len(annotations):
            raise ValueError(f'got {len(representations)} representations '
                             f'but {len(annotations)} annotations')

        self.representations = representations
        self.annotations = annotations
        self.indexer = indexer

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (representations, integral POS tags) for index'th sentence.

        Args:
            index (int): Index of the sentence in the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First tensor is shape
                (sentence_length, representation_dimension) containing word
                representations, and second is shape (sentence_length,)
                containing integral POS tags.

        """
        representations = self.representations[index]
        annotations = self.annotations[index]
        assert len(representations) == len(
            annotations.sentence), 'diff sentence lengths?'
        return representations, self.indexer(annotations)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields all (sentence representations, sentence POS tags) samples."""
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        """Returns the number of sentences (batches) in the dataset."""
        return len(self.annotations)

    @property
    def sample_representations_shape(self) -> Sequence[int]:
        """Returns the dimensionality of individual representations."""
        return (self.representations.dataset.dimension,)

    @property
    def sample_features_shape(self) -> Sequence[int]:
        """Returns the shape of each individual POS tag.

        Since POS tags are integral scalars, there is no such shape!
        """
        return ()

    def count_samples(self) -> int:
        """Returns the number of words in the dataset."""
        return sum(
            self.representations.dataset.length(index)
            for index in range(len(self.representations)))

    def count_unique_features(self) -> int:
        """Returns number of unique POS seen in data."""
        return len(self.indexer)
