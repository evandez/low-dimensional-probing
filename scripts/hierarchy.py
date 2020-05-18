"""Learn a hierarchy of projections for detecting POS."""

import argparse
import pathlib
import shutil
import subprocess
import sys

from torch import cuda

parser = argparse.ArgumentParser(description='Learn projection hierarchy.')
parser.add_argument('root', type=pathlib.Path, help='Root task data.')
parser.add_argument('leaves',
                    type=pathlib.Path,
                    nargs='+',
                    help='Leaf task data.')
parser.add_argument('--layers',
                    type=int,
                    nargs='+',
                    default=(0, 1, 2),
                    help='ELMo layers to use.')
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
parser.add_argument('--epochs',
                    type=int,
                    default=2500,
                    help='Total passes through training dataset.')
options = parser.parse_args()

root = pathlib.Path(__file__).resolve().parent.parent
module = root / 'lodimp'
script = module / 'pos.py'
base = ['python3', str(module.resolve()), str(script.resolve())]
base += ['--wandb-group', 'hierarchy']
base += ['--wandb-dir', str(options.wandb_dir.resolve())]
base += ['--model-dir', str(options.model_dir.resolve())]
base += ['--epochs', str(options.epochs)]
base += ['--probe', 'linear']
if cuda.is_available():
    base.extend(['--cuda', '--no-batch'])

for layer in options.layers:
    if options.model_dir.exists():
        shutil.rmtree(options.model_dir)
    tag = f'root-{options.root.name}-elmo{layer}-d{options.root_dimension}'
    model_file = f'{tag}.pth'
    command = base.copy()
    command += [str(options.root.resolve())]
    command += ['--layer', str(layer)]
    command += ['--dimension', str(options.root_dimension)]
    command += ['--model-file', model_file]
    command += ['--wandb-name', tag]
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)

    compose_path = options.model_dir / model_file
    for dimension in range(2, options.root_dimension):
        for leaf in options.leaves:
            tag = f'leaf-{leaf.name}-elmo{layer}-d{dimension}'
            command = base.copy()
            command += [str(leaf.resolve())]
            command += ['--layer', str(layer)]
            command += ['--dimension', str(dimension)]
            command += ['--compose', str(compose_path)]
            command += ['--wandb-name', tag]
            command += ['--model-file', f'{tag}.pth']
            print(' '.join(command))
            proc = subprocess.run(command)
            if proc.returncode:
                sys.exit(proc.returncode)
