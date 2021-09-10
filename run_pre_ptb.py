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
parser.add_argument('--out-dir',
                    type=pathlib.Path,
                    help='output directory (default: same as data_dir)')
args = parser.parse_args()

data_dir = args.data_dir
out_dir = args.out_dir
if out_dir is None:
    out_dir = args.data_dir

# Basically just call this script for every split...
model_kind = args.model.split('-')[0]
script = pathlib.Path(__file__).parent / f'run_pre_conll_to_{model_kind}.py'
for split in ('train', 'dev', 'test'):
    command = [
        'python3',
        str(script),
        str(data_dir / f'ptb3-wsj-{split}.conllx'),
        str(out_dir / f'raw.{split}.{args.model}-layers.h5'),
    ]
    if args.model == 'bert-random':
        command += ['--random']
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)
