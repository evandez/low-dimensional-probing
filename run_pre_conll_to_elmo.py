"""Embed all sentences in the .conllx file with ELMo."""
import argparse
import pathlib
import subprocess
import sys
import tempfile
from typing import List

from torch import cuda

ELMO_URL = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
ELMO_MODEL = '2x4096_512_2048cnn_2xhighway_5.5B'
WEIGHTS_FILE = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
WEIGHTS_URL = f'{ELMO_URL}/{ELMO_MODEL}/{WEIGHTS_FILE}'
OPTIONS_FILE = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_args.json'
OPTIONS_URL = f'{ELMO_URL}/{ELMO_MODEL}/{OPTIONS_FILE}'

parser = argparse.ArgumentParser(description='pre-embed sentences with elmo')
parser.add_argument('data_file',
                    type=pathlib.Path,
                    help='path to ptb .conll(x) file')
parser.add_argument('out_file', type=pathlib.Path, help='path to output file')
parser.add_argument('--word-column',
                    type=int,
                    default=1,
                    help='index of .conll column containing words')
parser.add_argument('--device', help='use this device (default: guessed)')
args = parser.parse_args()

if not args.data_file.exists():
    raise FileNotFoundError(f'data file {args.data_file} not found')

with args.data_file.open() as data:
    lines = data.readlines()

buffer: List[str] = []
sentences: List[str] = []
for line in lines:
    line = line.strip()
    if not line and buffer:
        sentences.append(' '.join(buffer))
        buffer = []
        continue
    elif not line or line.startswith('#'):
        continue
    buffer.append(line.split()[args.word_column])

if buffer:
    sentences.append(' '.join(buffer))

with tempfile.TemporaryDirectory() as tempdir:
    raw = pathlib.Path(tempdir) / f'{args.data_file.name}.raw'
    with raw.open('w') as handle:
        handle.write('\n'.join(sentences))
    command = ['allennlp', 'elmo', str(raw), str(args.out_file)]
    command += ['--weight-file', WEIGHTS_URL]
    command += ['--options-file', OPTIONS_URL]
    command += ['--all']

    device = args.device
    if device:
        if 'cuda' in args.device:
            device = device.replace('cuda:', '')
        command += ['--cuda-device', device]
    elif cuda.is_available():
        command += ['--cuda-device', '0']

    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
