"""Unit tests for the ptb module."""

import pathlib
import tempfile

import pytest

from lodimp import ptb

CONLLX = '''\
1       The     _       DET     DT      _       2       det     _       _
2       company _       NOUN    NN      _       3       nsubj   _       _
3       expects _       VERB    VBZ     _       0       root    _       _
4       earnings _       NOUN    NNS     _       16      nsubj   _       _
5       .   _       PUNCT   .       _       2       punct   _       _

1       He      _       PRON    PRP     _       3       nsubjpass       _     _
2       was     _       AUX     VBD     _       3       auxpass _       _
3       named   _       VERB    VBN     _       0       root    _       _
4       chief   _       ADJ     JJ      _       6       amod    _       _

1       He      _       PRON    PRP     _       3       nsubjpass       _     _
2       took    _       VERB    VBD     _       0       root    _       _
3      .       _       PUNCT   .       _       6       punct   _       _
'''

SAMPLES = [
    ptb.Sample(
        ['The', 'company', 'expects', 'earnings', '.'],
        ['DT', 'NN', 'VBZ', 'NNS', '.'],
    ),
    ptb.Sample(
        ['He', 'was', 'named', 'chief'],
        ['PRP', 'VBD', 'VBN', 'JJ'],
    ),
    ptb.Sample(
        ['He', 'took', '.'],
        ['PRP', 'VBD', '.'],
    )
]


@pytest.yield_fixture
def path():
    """Yields a path to a fake .conllx file."""
    with tempfile.TemporaryDirectory() as tempdir:
        file = pathlib.Path(tempdir) / 'ptb.conllx'
        with file.open(mode='w') as handle:
            handle.write(CONLLX)
        yield file


def test_load(path):
    """Tests that ptb.load(...) parses .conllx file correctly."""
    actual = ptb.load(path)
    assert actual == SAMPLES
