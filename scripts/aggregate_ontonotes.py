"""Aggregate the WSJ portion of Ontonotes.

Annotations can be retrieved from, e.g., here:
https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/
"""

import argparse
import glob
import logging
import pathlib
import sys

parser = argparse.ArgumentParser(description='Aggregate Ontonotes WSJ data.')
parser.add_argument('data', type=pathlib.Path, help='Path to data root.')
parser.add_argument('--quiet',
                    action='store_const',
                    dest='log_level',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='Suppress logging output to warnings and above.')
options = parser.parse_args()

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

root = options.data / 'conll-formatted-ontonotes-5.0' / 'data'
for split in ('train', 'development', 'test'):
    glob_path = root / split / '**' / '*.gold_conll'
    split = 'dev' if split == 'development' else split
    out_path = options.data / f'ontonotes5-{split}.conllx'
    files = glob.glob(str(glob_path), recursive=True)
    with open(out_path, 'w') as out:
        for file in files:
            logging.info('%s set: appending %s', split, file)
            with open(file, 'r') as handle:
                out.writelines(handle.readlines()[1:-1])
