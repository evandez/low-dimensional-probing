"""Regularization: vary regularization kind/weight and look at accuracy."""

import argparse
import os
import subprocess
import sys

from torch import cuda

parser = argparse.ArgumentParser(description='Run regularizer experiments.')
parser.add_argument('data', help='Path to PTB data.')
parser.add_argument('--dim', type=int, default=64, help='Projection dim.')
parser.add_argument('--kind', default='l1', help='Regularization variety.')
parser.add_argument('--tasks',
                    nargs='+',
                    default=('real', 'control'),
                    help='Tasks to run.')
parser.add_argument('--low', type=float, default=0, help='Low L1 weight.')
parser.add_argument('--high', type=float, default=1., help='Top L1 weight.')
parser.add_argument('--step', type=float, default=.1, help='Weight step size.')
parser.add_argument('--pool', type=int, default=4, help='Max jobs at once.')
parser.add_argument('--log-dir', default='/tmp/lodimp', help='TB log path.')
options = parser.parse_args()

# TODO(evandez): De-dupe from other experiments...
root = subprocess.check_output(['git', 'rev-parse',
                                '--show-toplevel']).decode().strip()
assert root, 'no git root?'
module = os.path.join(root, 'lodimp')
script = os.path.join(module, 'train.py')
data = os.path.abspath(options.data)
logs = os.path.abspath(options.log_dir)
command = ['python3', module, script, data, '--log-dir', logs]
if cuda.is_available():
    command.append('--cuda')

low, high, step = options.low, options.high, options.step
if low < 1 or high < low or step < 0:
    raise ValueError(f'bad range: ({low}, {high}, {step})')

lams = list(range(low, high, step))
pool = options.pool
groups = [lams[start:start + pool] for start in range(0, len(lams), pool)]

# Launch jobs, one task at a time, one group at a time.
for task in options.tasks:
    for group in groups:
        processes = []
        for lam in group:
            args = command + [task, str(options.dim), f'--{options.reg}', lam]
            print(' '.join(args))
            process = subprocess.Popen(args)
            processes.append(process)

        # Wait for all processes to start before launching next. They should
        # all finish at similar times assuming --pool was chosen to maximize
        # resource use without clogging it.
        for process in processes:
            process.wait()
            if process.returncode:
                sys.exit(process.returncode)
