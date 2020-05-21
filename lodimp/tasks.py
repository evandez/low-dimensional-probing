"""Defines real and control tasks."""

import collections
import itertools
from typing import Any, Dict, Optional, Sequence, Set

from lodimp.common.data import ptb

import numpy as np
import torch

POS_NOUNS = {'NN', 'NNS', 'NNP', 'NNPS'}
POS_NOUNS_PROPER = {'NNP', 'NNPS'}
POS_NOUNS_PLURAL = {'NNS', 'NNPS'}
POS_VERBS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
POS_VERBS_PRESENT = {'VBZ', 'VBP', 'VBG'}
POS_VERBS_PAST = {'VBD', 'VBN'}
POS_ADJECTIVES = {'JJ', 'JJR', 'JJS'}
POS_ADVERBS = {'RB', 'RBR', 'RBS'}


class Task:
    """Base class for all tasks."""

    def __init__(self, samples: Sequence[ptb.Sample], **kwargs: Any):
        """Preprocess the samples to construct the task.

        Args:
            samples (Sequence[ptb.Sample]): PTB samples on which to base
                the task.

        """
        pass

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Maps a sample to a tensor label.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: The tensor label.

        """
        raise NotImplementedError


class SizedTask(Task):
    """A task with a predefined label set."""

    def __len__(self) -> int:
        """Returns the number of valid labels in this task."""
        raise NotImplementedError


class POSTask(SizedTask):
    """Indexes PTB POS tags."""

    def __init__(self,
                 samples: Sequence[ptb.Sample],
                 tags: Optional[Set[str]] = None,
                 unk: str = 'UNK'):
        """Maps each POS tag to an index.

        Args:
            samples (Sequence[ptb.PTBSample]): The samples from which to
                draw tags.
            tags (Optional[Set[str]]): The XPOS tags to distinguish.
                All tags not in this set will be collapsed to the same tag.
                By default, all tags will be distinguished.
            unk (str): Tag to use when un-indexed XPOS is encountered.

        """
        if tags is None:
            tags = {xpos for sample in samples for xpos in sample.xpos}
        assert tags is not None, 'no tags to distinguish?'

        self.indexer = {xpos: index for index, xpos in enumerate(sorted(tags))}
        self.indexer[unk] = len(self.indexer)
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


# TODO(evandez): Allow restricting # of tags in control task.
class ControlPOSTask(SizedTask):
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
                more lists, for which to generate tags.
            dist (Optional[Sequence[float]], optional): The empirical
                distribution to use when sampling tags for word type.
                By default, is computed from the list of samples.

        """
        samples = list(itertools.chain(*groups))
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
                    self.tags[word] = np.random.choice(len(dist), p=dist)

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Tag the given samples.

        Args:
            sample (ptb.PTBSample): The sample to tag.

        Returns:
            torch.Tensor: Integer tags for every word in the sentence.

        """
        return torch.tensor([self.tags[word] for word in sample.sentence])

    def __len__(self) -> int:
        """Returns the number of fake tags in this task."""
        return len(self.dist)


class DependencyArcTask(Task):
    """Maps dependents to heads."""

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map dependents to heads.

        The label for word w is the index of word v in the sentence if word v
        is the head of word w. Note that if w is the root, it's head is w.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, this returns a length W
                tensor containing the index of its head.

        """
        return torch.tensor([
            head if head != -1 else word
            for word, head in enumerate(sample.heads)
        ])


class DependencyLabelTask(SizedTask):
    """Maps pairs of words to their syntactic relationship, if any."""

    def __init__(self,
                 samples: Sequence[ptb.Sample],
                 relations: Optional[Set[str]] = None,
                 unk: str = 'unk'):
        """Map each relation label to an integer.

        Args:
            samples: The samples from which to pull possible relations.
            relations: If set, only use these labels, and collapse the rest to
                the "unk" label.
            unk: Name for the unknown label.

        """
        if relations is None:
            relations = {rel for sample in samples for rel in sample.relations}
        assert relations is not None, 'unitialized relations?'

        self.indexer = {unk: 0}
        for rel in sorted(relations):
            self.indexer[rel] = len(self.indexer)
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
