"""Learn a hierarchy of projections for detecting POS."""

import argparse
import pathlib
import subprocess
import sys
from typing import List

import wandb
from torch import cuda

NLAYERS = {'elmo': 3, 'bert-base-uncased': 13}

parser = argparse.ArgumentParser(description='Learn projection hierarchy.')
parser.add_argument('wandb_user', help='Weights and Biases user to use.')
parser.add_argument(
    'tasks',
    nargs='+',
    type=pathlib.Path,
    help='Data directories for tasks constructing the hierarchy, in order.')
parser.add_argument('--model',
                    choices=NLAYERS.keys(),
                    default='elmo',
                    help='Representation model.')
parser.add_argument('--layers',
                    type=int,
                    nargs='+',
                    help='Representation layers.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Save models here.')
parser.add_argument('--wandb-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/wandb',
                    help='Save wandb data here.')
parser.add_argument('--root-dimension',
                    type=int,
                    default=10,
                    help='Dimension of full-task projection.')
parser.add_argument(
    '--thresholds',
    nargs='+',
    type=float,
    help='Accuracy thresholds for each task. Must be one per task. '
    'Defaults to 0.9 for each.')
parser.add_argument('--epochs',
                    type=int,
                    default=500,
                    help='Total passes through training dataset.')
options = parser.parse_args()

if options.thresholds and len(options.thresholds) != len(options.tasks):
    raise ValueError('number of thresholds must match number of tasks')
thresholds = options.thresholds or (0.9,) * len(options.tasks)
assert len(thresholds) == len(options.tasks), 'bad number of thresholds?'

root = pathlib.Path(__file__).resolve().parent.parent
module = root / 'lodimp'
script = module / 'pos.py'
base = ['python3', str(script.resolve())]
base += ['--wandb-group', ':'.join([task.name for task in options.tasks])]
base += ['--wandb-dir', str(options.wandb_dir.resolve())]
base += ['--model-dir', str(options.model_dir.resolve())]
base += ['--epochs', str(options.epochs)]
base += ['--probe', 'mlp']
if cuda.is_available():
    base.extend(['--cuda', '--no-batch'])

api = wandb.Api()
for layer in options.layers or range(NLAYERS[options.model]):
    max_dimension = options.root_dimension
    compose_paths: List[pathlib.Path] = []
    for index, task in enumerate(options.tasks):
        for dimension in range(1, max_dimension + 1):
            command = base.copy()
            command += [str(task.resolve())]
            command += ['--model', options.model]
            command += ['--layer', str(layer)]
            command += ['--dimension', str(dimension)]

            wandb_id = wandb.util.generate_id()
            command += ['--wandb-id', wandb_id]

            tag = f'h{index}-{task.name}-{options.model}{layer}-d{dimension}'
            command += ['--wandb-name', tag]

            model_file = f'{tag}.pth'
            command += ['--model-file', f'{tag}.pth']

            if compose_paths:
                command += ['--compose', str(compose_paths[-1])]

            print(' '.join(command))
            proc = subprocess.run(command)
            if proc.returncode:
                sys.exit(proc.returncode)

            run = api.run(f'{options.wandb_user}/lodimp/{wandb_id}')
            if run.summary['accuracy'] > thresholds[index]:
                compose_paths.append(options.model_dir / model_file)
                max_dimension = dimension
                break
        if len(compose_paths) != index + 1:
            # We failed! Keep max_dimension the same and carry on with the
            # best we could do...
            compose_paths.append(options.model_dir / model_file)
