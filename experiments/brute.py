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
parser.add_argument('--tasks',
                    nargs='+',
                    default=('real', 'control'),
                    help='Tasks to run.')
parser.add_argument('--layers', default=(2,), nargs='+', help='ELMo layers.')
parser.add_argument('--step', type=int, default=1, help='Dimenion step size.')
parser.add_argument('--pool', type=int, default=2, help='Max jobs at once.')
parser.add_argument('--log-dir', default='/tmp/lodimp', help='TB log path.')
options = parser.parse_args()

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
if low < 1 or high < low or step < 1:
    raise ValueError(f'bad range: ({low}, {high}, {step})')

dims = list(range(low, high, step))
pool = options.pool
groups = [dims[start:start + pool] for start in range(0, len(dims), pool)]

# Launch jobs, one task at a time, one group at a time.
for task in options.tasks:
    for layer in options.layers:
        for group in groups:
            processes = []
            for dim in group:
                args = command + [task, str(dim), '--elmo', str(layer)]
                print(' '.join(args))
                process = subprocess.Popen(args)
                processes.append(process)

            # Wait for all processes to start before launching next. They
            # should all finish at similar times assuming --pool was chosen
            # to maximize resource use without clogging it.
            for process in processes:
                process.wait()
                if process.returncode:
                    sys.exit(process.returncode)
