"""Embed an entire dataset with BERT."""

import argparse
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(description='Embed dataset with BERT.')
parser.add_argument('data', type=pathlib.Path, help='Path to data spltis.')
parser.add_argument('--prefix', default='ptb3-wsj-', help='File prefixes.')
parser.add_argument('--pretrained',
                    default='bert-base-uncased',
                    help='Forwarded to conll_to_bert.')
options = parser.parse_args()

script = pathlib.Path(__file__).parent / 'conll_to_bert.py'
for split in ('train', 'dev', 'test'):
    command = [
        'python3',
        str(script),
        str(options.data / f'{options.prefix}{split}.conllx'),
        str(options.data / f'raw.{split}.{options.pretrained}-layers.hdf5'),
        '--pretrained', options.pretrained
    ]
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
