"""Defines real and control tasks."""

import collections
import itertools
from typing import Any, Dict, List, Optional, Sequence, Set

from lodimp import ptb

import numpy as np
import torch

PTB_POS_NOUNS = {'NN', 'NNS', 'NNP', 'NNPS'}
PTB_POS_VERBS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
PTB_POS_ADJECTIVES = {'JJ', 'JJR', 'JJS'}
PTB_POS_ADVERBS = {'RB', 'RBR', 'RBS'}


class Task:
    """Base class for all tasks."""

    def __init__(self, samples: Sequence[ptb.Sample], **kwargs: Any):
        """Preprocess the samples to construct the task.

        Args:
            samples (Sequence[ptb.Sample]): PTB samples on which to base
                the task.

        """
        raise NotImplementedError

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Maps a sample to a tensor label.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: The tensor label.

        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the number of valid labels in this task."""
        raise NotImplementedError


class RealPOSTask(Task):
    """Indexes PTB POS tags."""

    def __init__(self,
                 samples: List[ptb.Sample],
                 tags: Optional[Set[str]] = None,
                 unk: str = 'UNK'):
        """Maps each POS tag to an index.

        Args:
            samples (List[ptb.PTBSample]): The samples from which to draw tags.
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
class ControlPOSTask(Task):
    """Maps words to arbitrary POS tags."""

    def __init__(self,
                 *groups: List[ptb.Sample],
                 dist: Optional[List[float]] = None):
        """Initialize the tagger.

        The tagger computes the empirical distribution of the samples, if not
        provided, and then uses it to generate arbitrary integer tags for each
        individual word type.

        Args:
            *groups (List[ptb.PTBSample]): All samples, provided in one or more
                lists, for which to generate tags.
            dist (Optional[List[float]], optional): The empirical distribution
                to use when sampling tags for word type. By default, is
                computed from the list of samples.

        """
        samples = list(itertools.chain(*groups))
        if dist is None:
            counts: Dict[str, int] = collections.defaultdict(lambda: 0)
            for sample in samples:
                for pos in sample.xpos:
                    counts[pos] += 1
            dist = np.array([float(count) for count in counts.values()])
            dist /= np.sum(dist)
        assert dist is not None, 'uninitialized distribution'
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
        return len(self.tags)
