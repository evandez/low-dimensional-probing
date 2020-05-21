"""Embed an entire dataset with ELMo."""

import argparse
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(description='Embed dataset with ELMo.')
parser.add_argument('data', type=pathlib.Path, help='Path to data spltis.')
parser.add_argument('--prefix', default='ptb3-wsj-', help='File prefixes.')
options = parser.parse_args()

script = pathlib.Path(__file__).parent / 'conll_to_elmo.py'
for split in ('train', 'dev', 'test'):
    command = [
        'python3',
        str(script),
        str(options.data / f'{options.prefix}{split}.conllx'),
        str(options.data / f'raw.{split}.{options.pretrained}-layers.hdf5'),
    ]
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
