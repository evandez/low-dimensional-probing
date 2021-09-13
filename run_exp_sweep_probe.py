"""Forward to `run_exp_train_probe`, sweeping over projection ranks."""
import argparse
import os
import pathlib
import subprocess

from lodimp import datasets, tasks
from lodimp.parse import splits

parser = argparse.ArgumentParser(description='sweep over projection rank')
parser.add_argument('task',
                    choices=(tasks.PART_OF_SPEECH_TAGGING,
                             tasks.DEPENDENCY_LABEL_PREDICTION,
                             tasks.DEPENDENCY_EDGE_PREDICTION),
                    help='linguistic task')
parser.add_argument('data_dir', type=pathlib.Path, help='data directory')
parser.add_argument('--linear',
                    action='store_true',
                    help='use linear probe (default: mlp)')
parser.add_argument('--d-min',
                    type=int,
                    default=1,
                    help='min projection rank for sweep (default: 1)')
parser.add_argument('--d-max',
                    type=int,
                    help='max projection rank for sweep (default: rep dim)')
parser.add_argument('--d-step',
                    type=int,
                    default=1,
                    help='step size to take while sweeping (defualt: 1)')
parser.add_argument('--d-step-exp-after',
                    type=int,
                    default=32,
                    help='step exponentially after this rank (default: 32)')
parser.add_argument('--share-projection',
                    action='store_true',
                    help='when combining reps, project both with same matrix; '
                    'cannot be used if task is "pos"')
parser.add_argument('--representation-model',
                    choices=('elmo', 'bert', 'bert-random'),
                    default='elmo',
                    help='representations to probe (default: elmo)')
parser.add_argument('--representation-layers',
                    nargs='+',
                    type=int,
                    help='representation layers (default: all)')
parser.add_argument('--lr',
                    type=float,
                    help='learning rate (default: see run_exp_train_probe.py)')
parser.add_argument(
    '--epochs',
    type=int,
    help='training epochs (default: see run_exp_train_probe.py)')
parser.add_argument(
    '--patience',
    type=int,
    help='stop training if dev loss does not improve for this many epochs '
    '(default: see run_exp_train_probe.py)')
parser.add_argument(
    '--model-dir',
    type=pathlib.Path,
    default='results/sweep/probes',
    help='dir to write finished probes (default: results/sweep/probes)')
parser.add_argument('--representations-key',
                    help='key for representations dataset in h5 file '
                    '(default: see run_exp_train_probe.py)')
parser.add_argument('--features-key',
                    help='key for features dataset in h5 file '
                    '(default: tags)')
parser.add_argument('--wandb-group',
                    help='experiment group (default: generated)')
parser.add_argument('--no-batch',
                    action='store_true',
                    help='store entire dataset in RAM/GPU and do not batch it')
parser.add_argument('--cache',
                    action='store_true',
                    help='cache entire dataset in memory/GPU')
parser.add_argument(
    '--device', help='use this device (default: see run_exp_train_probe.py)')
args = parser.parse_args()

# Resolve layers for model.
model = args.representation_model
model_dir = args.data_dir / model
if not model_dir.exists():
    raise FileNotFoundError(f'no collated data for model: {model}')

layers = args.representation_layers
if layers is None:
    layers = [child.name for child in model_dir.iterdir() if child.is_dir()]
    if not layers:
        raise ValueError(f'no layer data for model: {model}')

# Load a small subset of the data to determine representation size.
data_dir = args.data_dir / model / str(layers[0])
dataset = datasets.CollatedTaskDataset(data_dir / f'{splits.DEV}.h5')

# Determine the ranks we will sweep over.
d_min = args.d_min
d_max = (args.d_max if args.d_max is not None else
         dataset.sample_representations_shape[-1])
d_step = args.d_step
d_step_exp_after = args.d_step_exp_after

ranks = list(range(d_min, min(d_max, d_step_exp_after) + 1, d_step))
if d_max > d_step_exp_after:
    current = 1 << d_step_exp_after.bit_length()
    while current < d_max:
        ranks.append(current)
        current *= 2
    ranks.append(min(current, d_max))

# Generate a wandb group if necessary.
wandb_group = args.wandb_group
if wandb_group is None:
    wandb_group = args.data_dir.name

# Start training!
for layer in layers:
    for rank in ranks:
        command = [
            'python3',
            'run_exp_train_probe.py',
            args.task,
            str(args.data_dir),
            '--representation-model',
            model,
            '--representation-layer',
            str(layer),
            '--project-to',
            str(rank),
            '--model-dir',
            str(args.model_dir / model / f'layer-{layer}/rank-{rank}'),
            '--wandb-group',
            wandb_group,
            '--wandb-name',
            f'{model}-l{layer}-r{rank}',
            '--quiet',
        ]

        # Boolean flags.
        for flag, value in (
            ('--linear', args.linear),
            ('--share-projection', args.share_projection),
            ('--cache', args.cache),
            ('--no-batch', args.no_batch),
        ):
            if value:
                command += [flag]

        # Non-boolean flags.
        for flag, value in (
            ('--lr', args.lr),
            ('--epochs', args.epochs),
            ('--patience', args.patience),
            ('--representations-key', args.representations_key),
            ('--features-key', args.features_key),
            ('--device', args.device),
        ):
            if value is not None:
                command += [flag, str(value)]

        # Go!
        print(' '.join(command))
        subprocess.call(command, env={'WANDB_SILENT': 'true', **os.environ})
