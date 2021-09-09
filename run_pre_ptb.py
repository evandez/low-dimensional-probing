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
                    choices=('bert', 'bert-random', 'elmo'),
                    help='model to compute reps with')
args = parser.parse_args()

# Basically just call this script for every split...
model_kind = args.model.split('-')[0]
script = pathlib.Path(__file__).parent / f'run_pre_conll_to_{model_kind}.py'
for split in ('train', 'dev', 'test'):
    command = [
        'python3',
        str(script),
        str(args.data / f'ptb3-wsj-{split}.conllx'),
        str(args.data / f'raw.{split}.{args.model}-layers.h5'),
    ]
    if args.model == 'bert-random':
        command += ['--random']
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
