"""Regularization: vary regularization kind/weight and look at accuracy."""

import argparse
import pathlib
import subprocess
import sys

from torch import cuda

REGS = ('nuc', 'l1')

parser = argparse.ArgumentParser(description='Run brute-force experiments.')
parser.add_argument('tasks', type=pathlib.Path, nargs='+', help='Task roots.')
parser.add_argument('--regs',
                    choices=REGS,
                    default=REGS,
                    nargs='+',
                    help='Kinds of regularization to use.')
parser.add_argument('--weights',
                    type=float,
                    nargs='+',
                    default=(1e-4, 1e-3, 1e-2, 1e-1),
                    help='Regularization weights.')
parser.add_argument('--layers',
                    type=int,
                    nargs='+',
                    default=(0, 1, 2),
                    help='ELMo layers to run.')
parser.add_argument('--dimension',
                    type=int,
                    default=64,
                    help='Projection dimensionality.')
parser.add_argument('--epochs',
                    type=int,
                    default=2500,
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
        for reg in options.regs:
            for weight in options.weights:
                args = command + [
                    str(path.resolve()),
                    str(options.dimension), f'--{reg}',
                    str(weight)
                ]
                print(' '.join(args))
                process = subprocess.run(args)
                if process.returncode:
                    sys.exit(process.returncode)
