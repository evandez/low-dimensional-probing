"""Utilities for collating batches."""

import collections
from typing import Sequence

from lodimp import check

import torch
from torch.nn.utils import rnn


def pack(samples: Sequence[Sequence[torch.Tensor]]) -> Sequence[torch.Tensor]:
    """Collate samples of sequences into PackedSequences.

    Note that this process is destructive! The sequences may not be
    separated afterward because we toss the PackedSequence wrapper.

    Args:
        samples (Sequence[Sequence[torch.Tensor]]): The samples to collate.

    Raises:
        ValueError: If no samples given, if individual samples are empty, or
            if samples do not consist of equal numbers of elements.

    Returns:
        Sequence[torch.Tensor]: The packed sequences.

    """
    check.nonempty(samples)
    check.lengths(samples)
    for sample in samples:
        check.nonempty(sample)
        check.lengths(sample)

    separated = collections.defaultdict(list)
    for sample in samples:
        for index, item in enumerate(sample):
            separated[index].append(item)

    collated = []
    for index in sorted(separated.keys()):
        items = separated[index]
        collated.append(rnn.pack_sequence(items, enforce_sorted=False).data)

    return collated
