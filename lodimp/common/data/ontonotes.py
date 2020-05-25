"""Utilities for interacting with the Ontonotes 5.0 dataset."""

import pathlib
from typing import List, NamedTuple, Sequence


class Sample(NamedTuple):
    """Defines an Ontonotes sample. Here, we need only SRL labels."""

    sentence: Sequence[str]
    roles: Sequence[Sequence[str]]


def unparse(labeling: Sequence[str]) -> Sequence[str]:
    """Map role parse to a sequence of tags.

    Example input:  (*Arg1 *    *    *)   (V*) * (Arg2*   *)
    Example output: Arg1   Arg1 Arg1 Arg1 V    * Arg2     Arg2

    Args:
        labeling (Sequence[str]): The parsed role labeling.

    Returns:
        Sequence[str]: The unparsed labeling.

    """
    current = None
    unparsed = []
    for label in labeling:
        if label == '*':
            unparsed.append(current or '*')
        elif label.startswith('('):
            assert current is None, 'nested labeling?'
            if label.endswith('*)'):
                unparsed.append(label[1:-2])
            else:
                assert label.endswith('*'), 'irregular label start?'
                current = label[1:-1]
                assert current, 'no label name?'
                unparsed.append(current)
        elif label.endswith(')'):
            assert label == '*)', 'irregular label end?'
            assert current is not None, 'no label to end?'
            unparsed.append(current)
            current = None
    assert current is None, 'unfinished label?'
    assert len(unparsed) == len(labeling)
    return tuple(unparsed)


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
                element corresponds to the entire role labeling. Might be empty
                if the sentence is unlabeled.

        """
        assert not roles or len(sentence) == len(roles), 'nwords > nroles?'
        lengths = {len(labeling) for labeling in roles}
        assert not roles or len(lengths) == 1, 'incomplete labelings?'

        sentence = tuple(sentence)
        roles = tuple(zip(*roles))
        labelings = tuple(tuple(unparse(labeling)) for labeling in roles)
        samples.append(Sample(sentence, labelings))

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
            assert len(components) >= 12, 'incomplete column?'
            sentence.append(components[3])

            if len(components) > 12:
                roles.append(components[11:-1])
            else:
                assert not roles, 'unlabeled word?'

    if sentence:
        add(sentence, roles)

    return samples
