"""Embed the Penn Treebank with BERT."""

import argparse
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(description='Embed PTB with BERT.')
parser.add_argument('data', type=pathlib.Path, help='Path to PTB data.')
parser.add_argument('--pretrained',
                    default='bert-base-uncased',
                    help='Forwarded to conll_to_bert.')
options = parser.parse_args()

script = pathlib.Path(__file__).parent / 'conll_to_bert.py'
for split in ('train', 'dev', 'test'):
    command = [
        'python3',
        str(script),
        str(options.data / f'ptb3-wsj-{split}.conllx'),
        str(options.data / f'raw.{options.pretrained}.{split}.hdf5'),
        '--pretrained', options.pretrained
    ]
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
