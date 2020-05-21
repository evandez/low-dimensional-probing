"""Embed an entire dataset with ELMo."""

import argparse
import pathlib
import subprocess
import sys

DATASETS = ('ptb', 'ontonotes')

PREFIXES = {
    'ptb': 'ptb3-wsj-',
    'ontonotes': 'ontonotes5-',
}
assert PREFIXES.keys() == set(DATASETS)

WORD_COLUMNS = {
    'ptb': 1,
    'ontonotes': 3,
}
assert WORD_COLUMNS.keys() == set(DATASETS)

parser = argparse.ArgumentParser(description='Embed dataset with ELMo.')
parser.add_argument('data', type=pathlib.Path, help='Path to data spltis.')
parser.add_argument('--dataset',
                    choices=DATASETS,
                    default='ptb',
                    help='Dataset to embed.')
options = parser.parse_args()

script = pathlib.Path(__file__).parent / 'conll_to_elmo.py'
for split in ('train', 'dev', 'test'):
    command = [
        'python3',
        str(script),
        str(options.data / f'{PREFIXES[options.dataset]}{split}.conllx'),
        str(options.data / f'raw.{split}.elmo-layers.hdf5'),
        '--word-column',
        str(WORD_COLUMNS[options.dataset]),
    ]
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
