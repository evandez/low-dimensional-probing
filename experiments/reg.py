"""Regularization: vary regularization kind/weight and look at accuracy."""

import argparse
import os
import subprocess
import sys

import numpy as np
from torch import cuda

parser = argparse.ArgumentParser(description='Run regularizer experiments.')
parser.add_argument('data', help='Path to PTB data.')
parser.add_argument('--dim', type=int, default=64, help='Projection dim.')
parser.add_argument('--reg', default='l1', help='Regularization variety.')
parser.add_argument('--tasks',
                    nargs='+',
                    default=('real', 'control'),
                    help='Tasks to run.')
parser.add_argument('--layers',
                    type=int,
                    nargs='+',
                    default=(0, 1, 2),
                    help='ELMo layers.')
parser.add_argument('--low', type=float, default=0, help='Low L1 weight.')
parser.add_argument('--high', type=float, default=1., help='Top L1 weight.')
parser.add_argument('--step', type=float, default=.1, help='Weight step size.')
parser.add_argument('--epochs',
                    type=int,
                    default=500,
                    help='Total passes through training dataset.')
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
if cuda.is_available():
    command += ['--cuda', '--no-batch']

low, high, step = options.low, options.high, options.step
if low < 0 or high < low or step < 0:
    raise ValueError(f'bad range: ({low}, {high}, {step})')

# Launch jobs, one task at a time, one group at a time.
for task in options.tasks:
    for layer in options.layers:
        for lam in np.arange(low, high, step):
            args = command.copy()
            args += [task, str(options.dim)]
            args += [f'--{options.reg}', str(lam)]
            args += ['--elmo', str(layer)]
            print(' '.join(args))
            process = subprocess.run(args)
            if process.returncode:
                sys.exit(process.returncode)
