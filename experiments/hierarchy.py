"""Learn a hierarchy of projections for detecting POS."""

import argparse
import pathlib
import shutil
import subprocess
import sys

from torch import cuda

parser = argparse.ArgumentParser(description='Learn projection hierarchy.')
parser.add_argument('root', type=pathlib.Path, help='Root task to run.')
parser.add_argument('leaves',
                    type=pathlib.Path,
                    nargs='+',
                    help='Leaf tasks to run.')
parser.add_argument('--layers',
                    type=int,
                    nargs='+',
                    default=(0, 1, 2),
                    help='ELMo layers to use.')
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
    str(module.resolve()),
    str(script.resolve()), '--verbose', '--log-dir',
    str(options.log_dir.resolve()), '--model-dir',
    str(options.model_dir.resolve()), '--epochs',
    str(options.epochs)
]
if cuda.is_available():
    base.extend(['--cuda', '--no-batch'])

for layer in options.layers:
    if options.model_dir.exists():
        shutil.rmtree(options.model_dir)
    task = options.root / f'elmo-{layer}'
    model_file = f'root-{options.root.name}-l{layer}-d{options.dimension}.pth'
    command = (
        *base,
        str(task.resolve()),
        str(options.dimension),
        '--model-file',
        model_file,
    )
    print(' '.join(command))
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)

    for dimension in range(2, options.dimension):
        for leaf in options.leaves:
            task = leaf / f'elmo-{layer}'
            command = (
                *base,
                str(task.resolve()),
                str(dimension),
                '--compose',
                str(options.model_dir / model_file),
                '--model-file',
                f'leaf-{leaf.name}-l{layer}-d{dimension}.pth',
            )
            print(' '.join(command))
            proc = subprocess.run(command)
            if proc.returncode:
                sys.exit(proc.returncode)
