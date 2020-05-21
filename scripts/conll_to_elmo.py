"""Embed all sentences in the .conllx file with ELMo."""

import argparse
import pathlib
import subprocess
import sys
import tempfile
from typing import List

parser = argparse.ArgumentParser(description='Pre-embed sentences with ELMo.')
parser.add_argument('data', type=pathlib.Path, help='Path to .conll(x) file.')
parser.add_argument('out', type=pathlib.Path, help='Path to output file.')
parser.add_argument('--word-column', type=int, default=1, help='Column ')
options = parser.parse_args()

if not options.data.exists():
    raise FileNotFoundError(f'data file {options.data} not found')

with options.data.open() as data:
    lines = data.readlines()

buffer: List[str] = []
sentences: List[str] = []
for line in lines:
    line = line.strip()
    if not line and buffer:
        sentences.append(' '.join(buffer))
        continue
    elif not line or line.startswith('#'):
        continue
    buffer.append(line.split()[options.word_column])

with tempfile.TemporaryDirectory() as tempdir:
    raw = pathlib.Path(tempdir) / f'{options.data.name}.raw'
    with raw.open('w') as handle:
        handle.writelines(sentences)
    process = subprocess.run(
        ['allennlp', 'elmo', '--all',
         str(raw), str(options.out)])
    if process.returncode:
        sys.exit(process.returncode)
