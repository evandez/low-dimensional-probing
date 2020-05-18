"""Utilities for interacting with the Ontonotes 5.0 dataset."""

import pathlib
from typing import List, NamedTuple, Sequence


class Sample(NamedTuple):
    """Defines an Ontonotes sample. Here, we need only SRL labels."""

    sentence: Sequence[str]
    roles: Sequence[Sequence[str]]


def load(path: pathlib.Path) -> Sequence[Sample]:
    """Load the given Ontonotes .conll file.

    Args:
        path (pathlib): Path to the .conll file.

    Returns:
        Sequence[Sample]: The parsed samples.

    """
    samples = []

    def add(sentence: Sequence[str], roles: Sequence[Sequence[str]]) -> None:
        """Add a sample to the running list of parsed samples.

        Args:
            sentence (Sequence[str]): Words in the sentence.
            roles (Sequence[Sequence[str]]): Roles corresponding to each
                word. Note each element in the inner list corresponds to a
                different role labeling, but in the output sample, each
                element corresponds to the entire role labeling.

        """
        assert len(sentence) == len(roles), 'more words than roles?'
        assert len({len(role) for role in roles}) == 1, 'missing roles?'
        samples.append(Sample(tuple(sentence), tuple(zip(*roles))))

    with path.open() as file:
        sentence: List[str] = []
        roles: List[List[str]] = []
        for line in file:
            line = line.strip()
            if not line and sentence:
                add(sentence, roles)
                sentence, roles = [], []
                continue
            elif not line or line.startswith('#'):
                continue

            components = line.split()
            assert len(components) > 12, 'no semantic roles?'
            sentence.append(components[3])
            roles.append(components[11:-1])

    if sentence:
        add(sentence, roles)

    return samples
