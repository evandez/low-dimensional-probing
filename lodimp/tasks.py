"""Defines real and control tasks."""

import collections
import itertools
from typing import Callable, Dict, List, Optional

from lodimp import ptb

import numpy as np
import torch

# For now, tasks are defined in terms of Penn Treebank samples.
Task = Callable[[ptb.Sample], torch.Tensor]


class PTBRealPOS:
    """Indexes PTB POS tags."""

    def __init__(self, samples: List[ptb.Sample]):
        """Maps each POS tag to an index.

        Args:
            samples (List[ptb.PTBSample]): The samples from which to draw tags.

        """
        self.indexer: Dict[str, int] = {}
        for sample in samples:
            for xpos in sample.xpos:
                if xpos not in self.indexer:
                    self.indexer[xpos] = len(self.indexer)

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Index the part-of-speech tags for each sample.

        Args:
            samples (ptb.PTBSample): The sample to index XPOS tags for.

        Returns:
            torch.Tensor: Integer tags for each XPOS in the sample.

        """
        return torch.tensor([self.indexer[xpos] for xpos in sample.xpos])


class PTBControlPOS:
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
