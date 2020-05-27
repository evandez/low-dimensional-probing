"""Defines real and control tasks."""

import collections
import itertools
import random
from typing import Any, Dict, Optional, Sequence, Set, Tuple

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


POS_PUNCT = 'PUNCT'


class ControlDependencyArcTask(Task):
    """Constructs arbitrary parse "trees" for all samples."""

    def __init__(self, *groups: Sequence[ptb.Sample]):
        """Map each word type to a dependency arc behavior.

        We sample uniformly from three behaviors:
        - Always attach word to itself.
        - Always attach word to first word in sentence.
        - Always attach word to last word in sentence.

        """
        samples = tuple(itertools.chain(*groups))

        self.attach_to_self: Set[str] = set()
        self.attach_to_first: Set[str] = set()
        self.attach_to_last: Set[str] = set()
        behaviors = (self.attach_to_self, self.attach_to_first,
                     self.attach_to_last)

        for sample in samples:
            for word in sample.sentence:
                if not any(word in behavior for behavior in behaviors):
                    random.choice(behaviors).add(word)

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map dependents to (fake) heads.

        Same format as DependencyArcTask, but labels are assigned according
        to one of the three behaviors described in the constructor.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, this returns a length W
                tensor containing the index of its "head."

        Raises:
            ValueError: If word was not seen during initialization.

        """
        labels = []
        for index, word in enumerate(sample.sentence):
            if word in self.attach_to_first:
                labels.append(0)
            elif word in self.attach_to_self:
                labels.append(index)
            elif word in self.attach_to_last:
                length = len(sample.sentence)
                assert length > 1, 'sentence too short?'
                if sample.xpos[-1] == POS_PUNCT:
                    assert sample.xpos[-2] != POS_PUNCT, 'double punctuation?'
                    labels.append(length - 2)
                else:
                    labels.append(length - 1)
            else:
                raise ValueError(f'unknown word: {word}')
        return torch.tensor(labels, dtype=torch.long)


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


class ControlDependencyLabelTask(SizedTask):
    """Maps pairs of words to arbitrary syntactic relationships."""

    def __init__(self,
                 *groups: Sequence[ptb.Sample],
                 dist: Optional[Sequence[float]] = None):
        """Map each relation label to an arbitrary (integer) label.

        We only do this for pairs of words which have a head-dependent
        relationship in the original dataset.

        Args:
            groups: The samples from which to pull possible word pairs.
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

        Raises:
            ValueError: If word pair was not seen during initialization.

        """
        heads = sample.heads
        labels = torch.zeros(len(heads), len(heads), dtype=torch.long)
        for dep, head in enumerate(heads):
            if head == -1:
                head = dep
            words = (sample.sentence[dep], sample.sentence[head])
            if words not in self.rels:
                raise ValueError(f'unknown word pair: {words}')
            labels[dep, head] = self.rels[words]
        return labels

    def __len__(self) -> int:
        """Returns the number of relationships, including the null one."""
        return len(self.dist) + 1
