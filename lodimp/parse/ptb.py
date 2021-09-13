"""Utilities for interacting with the Penn Treebank."""
import pathlib
from typing import NamedTuple, Sequence

from lodimp.utils.typing import PathLike


class Sample(NamedTuple):
    """A sample in the PTB is a sentence and its annotation.

    In this project we only load the fields that we need.
    """

    sentence: Sequence[str]
    xpos: Sequence[str]
    heads: Sequence[int]
    relations: Sequence[str]


def load(path: PathLike) -> Sequence[Sample]:
    """Load the given .conllx file.

    Args:
        path (PathLike): The path to the .conllx file.

    Returns:
        Sequence[Sample]: Parsed samples from the file, one per sentence.

    """
    samples = []
    with pathlib.Path(path).open() as file:
        sentence, xpos, heads, relations = [], [], [], []
        for line in file:
            if line.strip():
                components = line.strip().split()
                assert len(components) == 10, f'malformed line: {line}'
                sentence.append(components[1])
                xpos.append(components[4])
                head = components[6]
                assert head.isdigit(), f'bad head index: {head}'
                heads.append(int(head) - 1)
                relations.append(components[7])
            elif sentence:
                assert len(sentence) == len(xpos)
                samples.append(
                    Sample(
                        (*sentence,),
                        (*xpos,),
                        (*heads,),
                        (*relations,),
                    ))
                sentence, xpos, heads, relations = [], [], [], []
        if sentence:
            samples.append(
                Sample(
                    (*sentence,),
                    (*xpos,),
                    (*heads,),
                    (*relations,),
                ))
    return (*samples,)
