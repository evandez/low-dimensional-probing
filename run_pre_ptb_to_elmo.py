"""Embed all sentences in the .conllx file with ELMo."""
import argparse
import pathlib
import subprocess
import sys
import tempfile
from typing import List

from ldp.parse import splits
from ldp.utils import logging

from torch import cuda

EPILOG = '''\
WARNING: To run this script, you need to install `allennlp==0.9.0` because
that is the last version that supports the `allennlp elmo` command.
We do not include this in `requirements.txt` because it shares too many
dependencies with the rest of this project, among other issues (e.g., it
cannot be run on Python >= 3.9). I suggest creating a separate venv for this.
'''

ELMO_URL = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
ELMO_MODEL = '2x4096_512_2048cnn_2xhighway_5.5B'
WEIGHTS_FILE = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
WEIGHTS_URL = f'{ELMO_URL}/{ELMO_MODEL}/{WEIGHTS_FILE}'
OPTIONS_FILE = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_args.json'
OPTIONS_URL = f'{ELMO_URL}/{ELMO_MODEL}/{OPTIONS_FILE}'

parser = argparse.ArgumentParser(description='pre-embed sentences with elmo',
                                 epilog=EPILOG)
parser.add_argument('data_dir',
                    type=pathlib.Path,
                    help='dir containing ptb .conll(x) files')
parser.add_argument('--out-dir',
                    type=pathlib.Path,
                    help='output dir (default: data_dir)')
parser.add_argument('--word-column',
                    type=int,
                    default=1,
                    help='index of .conll(x) column containing words')
parser.add_argument('--splits',
                    nargs='+',
                    default=splits.STANDARD_SPLITS,
                    help='splits to process (default: train, dev, test)')
parser.add_argument('--device', help='use this device (default: guessed)')
args = parser.parse_args()

logging.configure()
log = logging.getLogger(__name__)

data_dir = args.data_dir
if not data_dir.exists():
    raise FileNotFoundError(f'data dir {data_dir} not found')
out_dir = args.out_dir or data_dir

for split in args.splits:
    data_file = data_dir / f'ptb3-wsj-{split}.conllx'
    out_file = out_dir / f'elmo-{split}.h5'
    log.info('processing %s -> %s', data_file.name, out_file.name)

    # Parse the conll data.
    # TODO(evandez): Just use the parse lib?
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

    # Call the (very old) allennlp elmo command.
    with tempfile.TemporaryDirectory() as tempdir:
        raw = pathlib.Path(tempdir) / f'{data_file.name}.raw'
        with raw.open('w') as handle:
            handle.write('\n'.join(sentences))
        command = ['allennlp', 'elmo', str(raw), str(out_file)]
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
