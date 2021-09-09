"""Pre-compute representations for entire PTB."""
import argparse
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(description='compute bert embeddings for ptb')
parser.add_argument('data_dir',
                    type=pathlib.Path,
                    help='dir containing ptb conllx files')
parser.add_argument('model',
                    choices=('bert', 'elmo'),
                    help='model to compute reps with')
args = parser.parse_args()

# Basically just call this script for every split...
model = args.model
script = pathlib.Path(__file__).parent / f'run_pre_conll_to_{model}.py'
for split in ('train', 'dev', 'test'):
    command = [
        'python3',
        str(script),
        str(args.data / f'ptb3-wsj-{split}.conllx'),
        str(args.data / f'raw.{split}.{model}-layers.h5'),
    ]
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
