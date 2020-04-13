"""Utilities for interacting with the Penn Treebank."""

import pathlib
from typing import List, NamedTuple


class Sample(NamedTuple):
    """A sample in the PTB is a sentence and its annotation.

    In this project we only load the fields that we need.
    """
    sentence: List[str]
    xpos: List[str]


def load(path: pathlib.Path) -> List[Sample]:
    """Loads the given .conllx file.

    Args:
        path (pathlib.Path): The path to the .conllx file.

    Returns:
        List[Sample]: Parsed samples from the file, one per sentence.

    """
    samples = []
    with path.open() as file:
        sentence, xpos = [], []
        for line in file:
            if line.strip():
                components = line.strip().split()
                assert len(components) == 10, f'malformed line: {line}'
                sentence.append(components[1])
                xpos.append(components[4])
            elif sentence:
                assert len(sentence) == len(xpos)
                samples.append(Sample(sentence, xpos))
                sentence, xpos = [], []
        if sentence:
            samples.append(Sample(sentence, xpos))
    return samples
