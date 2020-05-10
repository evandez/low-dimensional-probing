"""Brute force: compute task accuracy for all projection ranks in a range."""

import argparse
import pathlib
import subprocess
import sys

from torch import cuda

parser = argparse.ArgumentParser(description='Run brute-force experiments.')
parser.add_argument('tasks', type=pathlib.Path, nargs='+', help='Task roots.')
parser.add_argument('--layers',
                    type=int,
                    nargs='+',
                    default=(0, 1, 2),
                    help='ELMo layers to run.')
parser.add_argument('--dims',
                    type=int,
                    nargs='+',
                    default=(2, 4, 8, 16, 32, 64),
                    help='Projection dimensionalities to run.')
parser.add_argument('--epochs',
                    type=int,
                    default=500,
                    help='Total passes through training dataset.')
parser.add_argument('--log-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/logs',
                    help='Save TB logs here.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Save models here.')
options = parser.parse_args()

root = subprocess.check_output(['git', 'rev-parse',
                                '--show-toplevel']).decode().strip()
assert root, 'no git root?'
module = pathlib.Path(root) / 'lodimp'
script = module / 'train.py'
command = ['python3', str(module.resolve()), str(script.resolve())]
command += ['--model-dir', str(options.model_dir.resolve())]
command += ['--log-dir', str(options.log_dir.resolve())]
command += ['--epochs', str(options.epochs)]
command += ['--verbose']
if cuda.is_available():
    command += ['--cuda', '--no-batch']

# Launch jobs, one task, layer, and dimension at a time.
for task in options.tasks:
    for layer in options.layers:
        path = task / f'elmo-{layer}'
        for dim in options.dims:
            args = command + [str(path.resolve()), str(dim)]
            print(' '.join(args))
            process = subprocess.run(args)
            if process.returncode:
                sys.exit(process.returncode)
