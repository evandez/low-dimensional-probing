"""Learn a hierarchy of projections for detecting POS."""

import argparse
import pathlib
import shutil
import subprocess
import sys

from torch import cuda

parser = argparse.ArgumentParser(description='Learn projection hierarchy.')
parser.add_argument('data', type=pathlib.Path, help='Path to data directory.')
parser.add_argument('--projected-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/projected',
                    help='Path at which to store projected copies of dataset.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Save models here.')
parser.add_argument('--log-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/logs',
                    help='Save TB logs here.')
parser.add_argument('--dimension',
                    type=int,
                    default=10,
                    help='Dimension of full-task projection.')
parser.add_argument('--epochs',
                    type=int,
                    default=2500,
                    help='Total passes through training dataset.')
options = parser.parse_args()

root = subprocess.check_output(['git', 'rev-parse',
                                '--show-toplevel']).decode().strip()
assert root, 'no git root?'
module = pathlib.Path(root) / 'lodimp'
script = module / 'train.py'
base = [
    'python3',
    str(module),
    str(script), '--verbose', '--log-dir',
    str(options.log_dir), '--model-dir',
    str(options.model_dir), '--epochs',
    str(options.epochs)
]
if cuda.is_available():
    base.extend(['--cuda', '--no-batch'])

if options.model_dir.exists():
    shutil.rmtree(options.model_dir)
command = (*base, str(options.data), 'pos', str(options.dimension))
print(' '.join(command))
process = subprocess.run(command)
if process.returncode:
    sys.exit(process.returncode)

if options.projected_dir.exists():
    shutil.rmtree(options.projected_dir)
command = ('python3', str(module), str(module / 'project.py'),
           str(next(options.model_dir.iterdir())), str(options.data),
           str(options.projected_dir), '--verbose')
if cuda.is_available():
    command = (*command, '--cuda')
print(' '.join(command))
process = subprocess.run(command)
if process.returncode:
    sys.exit(process.returncode)

for dim in range(2, options.dimension):
    for task in ('pos-noun', 'pos-verb', 'pos-adj', 'pos-adv'):
        command = (*base, str(options.projected_dir), task, str(dim))
        print(' '.join(command))
        proc = subprocess.run(command)
        if proc.returncode:
            sys.exit(proc.returncode)
