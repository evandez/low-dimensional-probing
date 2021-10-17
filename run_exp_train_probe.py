"""Train a single probe on a task.

This command trains a probe to predict linguistic features given only
representations. For performance reasons, it assumes the data has been
collated. Use the `run_pre_collate.py` script to do that.
"""
import argparse
import pathlib
from typing import Dict

from ldp import datasets, tasks
from ldp.models import probes, projections
from ldp.parse import splits
from ldp.tasks import dep, dlp, pos
from ldp.utils import env, logging

import torch
import wandb
from torch import cuda, nn

LINEAR_PROBE = 'linear'
MLP_PROBE = 'mlp'

PROBE_TYPES_BY_TASK = {
    tasks.PART_OF_SPEECH_TAGGING: {
        LINEAR_PROBE: probes.Linear,
        MLP_PROBE: probes.MLP,
    },
    tasks.DEPENDENCY_LABEL_PREDICTION: {
        LINEAR_PROBE: probes.Linear,
        MLP_PROBE: probes.MLP,
    },
    tasks.DEPENDENCY_EDGE_PREDICTION: {
        LINEAR_PROBE: probes.PairwiseBilinear,
        MLP_PROBE: probes.PairwiseMLP,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('task',
                    choices=(tasks.PART_OF_SPEECH_TAGGING,
                             tasks.DEPENDENCY_LABEL_PREDICTION,
                             tasks.DEPENDENCY_EDGE_PREDICTION),
                    help='linguistic task')
parser.add_argument('--linear',
                    action='store_const',
                    dest='probe_type',
                    const=LINEAR_PROBE,
                    default=MLP_PROBE,
                    help='use linear probe (default: mlp)')
parser.add_argument(
    '--project-to',
    type=int,
    help='project reps to this dimension (default: no projection)')
parser.add_argument('--share-projection',
                    action='store_true',
                    help='when combining reps, project both with same matrix; '
                    'cannot be used if task is "pos"')
parser.add_argument('--model',
                    choices=('elmo', 'bert', 'bert-random'),
                    default='elmo',
                    help='representations to probe (default: elmo)')
parser.add_argument('--layer',
                    type=int,
                    default=0,
                    help='representation layer (default: 0)')
parser.add_argument('--lr',
                    default=1e-3,
                    type=float,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs',
                    type=int,
                    default=25,
                    help='training epochs (default: 25)')
parser.add_argument(
    '--patience',
    type=int,
    default=4,
    help='stop training if dev loss does not improve for this many epochs '
    '(default: 4)')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='data directory (default: project data dir)')
parser.add_argument(
    '--results-dir',
    type=pathlib.Path,
    help='directory to write finished probe (default: project results dir)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: generated)')
parser.add_argument('--wandb-group',
                    help='experiment group (default: generated)')
parser.add_argument('--no-batch',
                    action='store_true',
                    help='store entire dataset in RAM/GPU and do not batch it')
parser.add_argument('--cache',
                    action='store_true',
                    help='cache entire dataset in memory/GPU')
parser.add_argument('--device', help='use this device (default: guessed)')
parser.add_argument('--quiet',
                    action='store_const',
                    dest='log_level',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='only log warnings and above')
args = parser.parse_args()

task = args.task
model = args.model
layer = args.layer
project_to = args.project_to

# Configure wandb immediately.
wandb.init(
    project='ldp',
    name=args.wandb_name or
    f'{model}-l{layer}-{"full" if project_to is None else f"r{project_to}"}',
    group=args.wandb_group or task,
    config={
        'task': task,
        'representations': {
            'model': model,
            'layer': layer,
        },
        'projection': {
            'dimension':
                project_to,
            'shared': (task != tasks.PART_OF_SPEECH_TAGGING and
                       args.share_projection),
        },
        'probe': {
            'model': args.probe_type,
        },
        'hyperparameters': {
            'epochs': args.epochs,
            'batched': not args.no_batch,
            'cached': args.cache,
            'lr': args.lr,
            'patience': args.patience,
        },
    })

if args.task == tasks.PART_OF_SPEECH_TAGGING and args.share_projection:
    raise ValueError('cannot set --share-projection when task is "pos"')

logging.configure(level=args.log_level)
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

# Prepare results/data directories.
data_root = args.data_dir or env.data_dir()
data_dir = data_root / 'ptb3/collated' / model / str(layer) / task

results_root = args.results_dir or env.results_dir()
results_dir = results_root / 'train-probe'
results_dir /= 'linear' if args.linear else 'mlp'
results_dir /= f'{task}/{model}/l{layer}'
results_dir /= f'r{project_to}' if project_to is not None else 'full'
results_dir.mkdir(parents=True, exist_ok=True)

# Load the datasets.
cache = device if args.cache else None
data: Dict[str, datasets.CollatedTaskDataset] = {}
for split in splits.STANDARD_SPLITS:
    split_file = data_dir / f'{split}.h5'
    if args.no_batch:
        data[split] = datasets.NonBatchingCollatedTaskDataset(split_file,
                                                              device=cache)
    else:
        data[split] = datasets.SentenceBatchingCollatedTaskDataset(
            split_file, device=cache)

# Start training!
probe: nn.Module
task = args.task
if task == tasks.PART_OF_SPEECH_TAGGING:
    probe, accuracy = pos.train(
        data[splits.TRAIN],
        data[splits.DEV],
        data[splits.TEST],
        probe_t=PROBE_TYPES_BY_TASK[args.task][args.probe_type],
        project_to=args.project_to,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=device,
        also_log_to_wandb=True)
elif task == tasks.DEPENDENCY_LABEL_PREDICTION:
    probe, accuracy = dlp.train(
        data[splits.TRAIN],
        data[splits.DEV],
        data[splits.TEST],
        probe_t=PROBE_TYPES_BY_TASK[args.task][args.probe_type],
        project_to=args.project_to,
        share_projection=args.share_projection,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=device,
        also_log_to_wandb=True)
elif task == tasks.DEPENDENCY_EDGE_PREDICTION:
    probe, accuracy = dep.train(
        data[splits.TRAIN],
        data[splits.DEV],
        data[splits.TEST],
        probe_t=PROBE_TYPES_BY_TASK[args.task][args.probe_type],
        project_to=args.project_to,
        share_projection=args.share_projection,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=device,
        also_log_to_wandb=True)
else:
    raise ValueError(f'unknown task: {task}')

probe_file = args.results_dir / 'probe.pth'
log.info('saving probe to %s', probe_file)
torch.save(probe, probe_file)
wandb.save(str(probe_file))

proj_file = args.results_dir / 'projection.pth'
log.info('saving projection to %s', proj_file)
torch.save(probe.project, proj_file)
wandb.save(str(proj_file))

# For convenience, compute POS nullspaces for downstream testing.
if task == tasks.PART_OF_SPEECH_TAGGING and probe.project is not None:
    log.info('task is pos, so computing projection nullspace')
    projection = probe.project
    assert isinstance(projection, projections.Projection)
    nullspace = projection.nullspace()

    nullspace_file = results_dir / 'nullspace.pth'
    log.info('saving nullspace to %s', nullspace_file)
    torch.save(nullspace, nullspace_file)
    wandb.save(str(nullspace_file))

log.info('test accuracy %f', accuracy)
wandb.summary['accuracy'] = accuracy
