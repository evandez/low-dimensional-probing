"""Brute force: compute task accuracy for all projection ranks in a range."""

import argparse
import pathlib
import subprocess
import sys
from typing import Sequence

from torch import cuda

TASKS = ('pos', 'dep_arc', 'dep_label')
NLAYERS = {'elmo': 3, 'bert-base-uncased': 12}

parser = argparse.ArgumentParser(description='Run brute-force experiments.')
parser.add_argument('task', choices=TASKS, help='Task to run.')
parser.add_argument('data', type=pathlib.Path, help='Path to task data.')
parser.add_argument('--probes', nargs='+', help='Probes to train.')
parser.add_argument('--model',
                    choices=NLAYERS.keys(),
                    default='elmo',
                    help='Representation model.')
parser.add_argument('--layers',
                    type=int,
                    nargs='+',
                    help='Representation layers to train against.')
parser.add_argument('--dims',
                    type=int,
                    nargs='+',
                    default=(2, 4, 8, 16, 32, 64),
                    help='Projection dimensionalities to run.')
parser.add_argument('--l1', type=float, nargs='+', help='L1 penalties.')
parser.add_argument('--nuc', type=float, nargs='+', help='Nuc norm penalties.')
parser.add_argument('--ablate',
                    action='store_true',
                    help='Ablate axes and retest after every run.')
parser.add_argument('--epochs',
                    type=int,
                    default=500,
                    help='Total passes through training dataset.')
parser.add_argument('--force-batch',
                    action='store_true',
                    help='Force batching, even if CUDA available.')
parser.add_argument('--wandb-group', help='Wandb group. Default is task name.')
parser.add_argument('--wandb-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/wandb',
                    help='Save wandb data here.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Save models here.')
options = parser.parse_args()

probes = options.probes
if not options.probes:
    if options.task == 'dep_arc':
        probes = ('bilinear', 'mlp')
    else:
        probes = ('linear', 'mlp')

root = pathlib.Path(__file__).resolve().parent.parent
module = root / 'lodimp'
script = module / f'{options.task}.py'
base = [
    'python3',
    str(module.resolve()),
    str(script.resolve()),
    str(options.data.resolve())
]
base += ['--model-dir', str(options.model_dir.resolve())]
base += ['--wandb-group', options.wandb_group or options.task]
base += ['--wandb-dir', str(options.wandb_dir.resolve())]
base += ['--epochs', str(options.epochs)]
if options.ablate:
    base += ['--ablate']
if cuda.is_available():
    base += ['--cuda']
    if not options.force_batch:
        base += ['--no-batch']


def run(command: Sequence[str]) -> None:
    """Run the given command, stopping execution if it fails."""
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process)


# Launch jobs, one task, layer, and dimension at a time.
for layer in options.layers or range(NLAYERS[options.model]):
    for dim in options.dims:
        for probe in probes:
            tag = f'{options.task}-{options.model}{layer}-{probe}-{dim}d'
            command = base.copy()
            command += ['--model', options.model]
            command += ['--layer', str(layer)]
            command += ['--dimension', str(dim)]
            command += ['--probe', probe]
            run(command + ['--wandb-name', tag])

            # TODO(evandez): Okay, this script needs to be broken up.
            if options.task in ('dep_arc', 'dep_label'):
                shared = command.copy()
                shared += ['--share-projection']
                shared += ['--wandb-name', tag + '-shared']
                run(shared)

            # Run regularized variants if requested.
            for reg, lams in (('l1', options.l1), ('nuc', options.nuc)):
                for lam in lams or []:
                    regularized = command.copy()
                    regularized += [f'--{reg}', str(lam)]
                    regularized += ['--wandb-name', f'{tag}-{lam}{reg}']
                    run(regularized)
