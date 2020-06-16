"""Core experiments for the dependency label prediction task."""

import collections
import itertools
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

from lodimp.common import tasks
from lodimp.common.data import ptb
from lodimp.common.data import representations as reps

import numpy as np
import torch

UNK = 'unk'


class DLPIndexer:
    """Maps pairs of words to their syntactic relationship, if any."""

    def __init__(self, samples: Sequence[ptb.Sample], unk: str = UNK):
        """Map each relation label to an integer.

        Args:
            samples (Sequence[ptb.Sample]): The samples from which to determine
                possible relations.
            unk (str): Label to use when un-indexed dependency label is
                encountered.

        """
        labels = {rel for sample in samples for rel in sample.relations}
        self.indexer = {unk: 0}
        for label in sorted(labels):
            self.indexer[label] = len(self.indexer)
        self.unk = unk

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map all possible (word, word) pairs to labels.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, returns shape (W, W) matrix
                where element (v, w) is the index of the label describing
                the relationship between word v and w, if any. Defaults to
                the "unk" label, even if there is no relationship between
                v and w.

        """
        heads, relations = sample.heads, sample.relations
        labels = torch.empty(len(heads), len(heads), dtype=torch.long)
        labels.fill_(self.indexer[self.unk])
        for word, (head, rel) in enumerate(zip(heads, relations)):
            if head == -1:
                labels[word, word] = self.indexer[rel]
            else:
                label = self.indexer.get(rel, self.indexer[self.unk])
                labels[word, head] = label
        return labels

    def __len__(self) -> int:
        """Returns the number of unique labels for this task."""
        return len(self.indexer)


class ControlDLPIndexer:
    """Maps pairs of words to arbitrary syntactic relationships."""

    def __init__(self,
                 *groups: Sequence[ptb.Sample],
                 dist: Optional[Sequence[float]] = None):
        """Map each relation label to an arbitrary (integer) label.

        We only do this for pairs of words which have a head-dependent
        relationship in the original dataset.

        Args:
            *groups (Sequence[ptb.Samples]): The samples from which to pull
                possible word pairs.
            dist (Optional[Sequence[float]], optional): The empirical
                distribution to use when sampling tags per word type.
                By default, is computed from the list of samples.

        """
        samples = tuple(itertools.chain(*groups))

        if dist is None:
            counts: Dict[str, int] = collections.defaultdict(lambda: 0)
            for sample in samples:
                for relation in sample.relations:
                    counts[relation] += 1
            dist = np.array([float(count) for count in counts.values()])
            dist /= np.sum(dist)
        assert dist is not None, 'uninitialized distribution?'
        self.dist = dist

        self.rels: Dict[Tuple[str, str], int] = {}
        for sample in samples:
            sentence = sample.sentence
            heads = sample.heads
            for dep, head in enumerate(heads):
                if head == -1:
                    head = dep
                words = (sentence[dep], sentence[head])
                if words not in self.rels:
                    # Add one so that 0 is reserved for "no relationship" tag.
                    self.rels[words] = np.random.choice(len(dist), p=dist) + 1

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map all possible (word, word) pairs to labels.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, returns shape (W, W) matrix
                where element (v, w) is the index of the label describing
                the relationship between word v and w, if any. Defaults to
                the "unk" label, even if there is no relationship between
                v and w.

        """
        heads = sample.heads
        labels = torch.zeros(len(heads), len(heads), dtype=torch.long)
        for dep, head in enumerate(heads):
            if head == -1:
                head = dep
            words = (sample.sentence[dep], sample.sentence[head])
            labels[dep, head] = self.rels.get(words, 0)
        return labels

    def __len__(self) -> int:
        """Returns the number of relationships, including the null one."""
        return len(self.dist) + 1


class DLPTaskDataset(tasks.TaskDataset):
    """Iterates over (word representation pair, dependency label) pairs."""

    def __init__(
        self,
        representations: reps.RepresentationLayerDataset,
        annotations: Sequence[ptb.Sample],
        indexer: Union[DLPIndexer, ControlDLPIndexer],
    ):
        """Initializes dataset by mapping each dependency label to an index.

        Args:
            representations (representations.RepresentationsLayerDataset): Word
                representations corresponding to the words to be paired and
                labeled.
            annotations (Sequence[ptb.PTBSample]): The PTB annotations from
                which to pull dependency labels.
            indexer (Union[DLPIndexer, ControlDLPIndexer]): Callable mapping
                PTB dependency label annotations to integer tensors.

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
        rels = self.indexer(annotations)

        # Find all pairs of words sharing an edge.
        indexes = set(range(len(representations)))
        pairs = [(i, j) for i in indexes for j in indexes if rels[i, j]]
        assert pairs and len(pairs) == len(representations), 'missing edges?'

        # Stack everything before returning it.
        bigrams = torch.stack([
            torch.stack((representations[i], representations[j]))
            for i, j in pairs
        ])
        labels = torch.stack([rels[i, j] for i, j in pairs])

        return bigrams, labels

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields all (sentence representations, sentence POS tags) samples."""
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        """Returns the number of sentences (batches) in the dataset."""
        return len(self.annotations)

    @property
    def sample_representations_shape(self) -> Sequence[int]:
        """Returns the dimensionality of the representation pairs."""
        return (2, self.representations.dataset.dimension)

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
