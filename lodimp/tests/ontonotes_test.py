"""Unit tests for ontonotes module."""

import pathlib
import tempfile

from lodimp import ontonotes

import pytest

CONLL = '''\
#begin document (nw/wsj/02/wsj_0201); part 000
nw/wsj/02/wsj_0200   0   0        Ms.   NNP  (TOP(S(NP*      -    -   -   -          *    (ARG0*   -
nw/wsj/02/wsj_0200   0   1       Haag   NNP           *)     -    -   -   -    (PERSON)        *)  -
nw/wsj/02/wsj_0200   0   2      plays   VBZ        (VP*    play  02   -   -          *       (V*)  -
nw/wsj/02/wsj_0200   0   3    Elianti   NNP       (NP*))     -    -   -   -    (PERSON)   (ARG1*)  -
nw/wsj/02/wsj_0200   0   4          .     .          *))     -    -   -   -          *         *   -

nw/wsj/02/wsj_0221   0    0    Columbia    NNP       (TOP(S(NP*         -    -   -   -   (ORG*   (ARG1*            *   -
nw/wsj/02/wsj_0221   0    1    Pictures   NNPS                *)        -    -   -   -       *)       *)           *   -
nw/wsj/02/wsj_0221   0    2          is    VBZ             (VP*         -    -   -   -       *        *            *   -
nw/wsj/02/wsj_0221   0    3       being    VBG             (VP*         -    -   -   -       *        *            *   -
nw/wsj/02/wsj_0221   0    4    acquired    VBN             (VP*    acquire  01   -   -       *      (V*)           *   -
nw/wsj/02/wsj_0221   0    5          by     IN             (PP*         -    -   -   -       *   (ARG0*            *   -
nw/wsj/02/wsj_0221   0    6        Sony    NNP          (NP(NP*         -    -   -   -   (ORG*        *       (ARG1*   -
nw/wsj/02/wsj_0221   0    7       Corp.    NNP                *)        -    -   -   -       *)       *            *)  -
nw/wsj/02/wsj_0221   0    8           ,      ,                *         -    -   -   -       *        *            *   -
nw/wsj/02/wsj_0221   0    9       which    WDT      (SBAR(WHNP*)        -    -   -   -       *        *     (R-ARG1*)  -
nw/wsj/02/wsj_0221   0   10          is    VBZ           (S(VP*         -    -   -   -       *        *            *   -
nw/wsj/02/wsj_0221   0   11       based    VBN             (VP*       base  01   -   -       *        *          (V*)  -
nw/wsj/02/wsj_0221   0   12          in     IN             (PP*         -    -   -   -       *        *   (ARGM-LOC*   -
nw/wsj/02/wsj_0221   0   13       Japan    NNP   (NP*)))))))))))        -    -   -   -    (GPE)       *)           *)  -
nw/wsj/02/wsj_0221   0   14           .      .               *))        -    -   -   -       *        *            *   -

#end document
'''  # noqa: E501

SAMPLES = (
    ontonotes.Sample(
        ('Ms.', 'Haag', 'plays', 'Elianti', '.'),
        (('ARG0', 'ARG0', 'V', 'ARG1', '*'),),
    ),
    ontonotes.Sample(
        (
            'Columbia',
            'Pictures',
            'is',
            'being',
            'acquired',
            'by',
            'Sony',
            'Corp.',
            ',',
            'which',
            'is',
            'based',
            'in',
            'Japan',
            '.',
        ),
        (
            (
                'ARG1',
                'ARG1',
                '*',
                '*',
                'V',
                'ARG0',
                'ARG0',
                'ARG0',
                'ARG0',
                'ARG0',
                'ARG0',
                'ARG0',
                'ARG0',
                'ARG0',
                '*',
            ),
            (
                '*',
                '*',
                '*',
                '*',
                '*',
                '*',
                'ARG1',
                'ARG1',
                '*',
                'R-ARG1',
                '*',
                'V',
                'ARGM-LOC',
                'ARGM-LOC',
                '*',
            ),
        ),
    ),
)


@pytest.yield_fixture
def path():
    """Yields the path to a temporary, fake data file for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        file = pathlib.Path(tempdir) / 'data.conll'
        file.write_text(CONLL)
        yield file


def test_load(path):
    """Test load parses data correctly."""
    samples = ontonotes.load(path)
    assert len(samples) == len(SAMPLES)
    for actual, expected in zip(samples, SAMPLES):
        print(actual)
        assert actual == expected
