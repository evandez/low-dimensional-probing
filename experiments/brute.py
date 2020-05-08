"""Brute force: compute task accuracy for all projection ranks in a range."""

import argparse
import os
import subprocess
import sys

from torch import cuda

parser = argparse.ArgumentParser(description='Run brute-force experiments.')
parser.add_argument('data', help='Path to PTB data.')
parser.add_argument('--low', type=int, default=2, help='Low projection dim.')
parser.add_argument('--high', type=int, default=64, help='Top projection dim.')
parser.add_argument('--step', type=int, default=1, help='Dimenion step size.')
parser.add_argument('--epochs',
                    type=int,
                    default=500,
                    help='Total passes through training dataset.')
parser.add_argument('--tasks',
                    nargs='+',
                    default=('pos', 'pos-control'),
                    help='Tasks to run.')
parser.add_argument('--layers',
                    type=int,
                    default=(0, 1, 2),
                    nargs='+',
                    help='ELMo layers.')
parser.add_argument('--log-dir', default='/tmp/lodimp', help='TB log path.')
options = parser.parse_args()

root = subprocess.check_output(['git', 'rev-parse',
                                '--show-toplevel']).decode().strip()
assert root, 'no git root?'
module = os.path.join(root, 'lodimp')
script = os.path.join(module, 'train.py')
command = ['python3', module, script, os.path.abspath(options.data)]
command += ['--log-dir', os.path.abspath(options.log_dir)]
command += ['--epochs', str(options.epochs)]
command += ['--verbose']
if cuda.is_available():
    command += ['--cuda', '--no-batch']

low, high, step = options.low, options.high, options.step
if low < 1 or high < low or step < 1:
    raise ValueError(f'bad range: ({low}, {high}, {step})')

# Launch jobs, one task, layer, and dimension at a time.
for task in options.tasks:
    for layer in options.layers:
        for dim in range(low, high, step):
            args = command + [task, str(dim), '--elmo', str(layer)]
            print(' '.join(args))
            process = subprocess.run(args)
            if process.returncode:
                sys.exit(process.returncode)
