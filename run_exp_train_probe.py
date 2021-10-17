"""Train a single probe on a task.

This command trains a probe to predict linguistic features given only
representations. For performance reasons, it assumes the data has been
collated. Use the `run_pre_collate.py` script to do that.
"""
import argparse
import pathlib
from typing import Dict

from lodimp import datasets, tasks
from lodimp.models import probes, projections
from lodimp.parse import splits
from lodimp.tasks import dep, dlp, pos
from lodimp.utils import linalg, logging

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
parser.add_argument('data_dir', type=pathlib.Path, help='data directory')
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
parser.add_argument('--representation-model',
                    choices=('elmo', 'bert', 'bert-random'),
                    default='elmo',
                    help='representations to probe (default: elmo)')
parser.add_argument('--representation-layer',
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
parser.add_argument(
    '--model-dir',
    type=pathlib.Path,
    default='results/probes',
    help='directory to write finished probe (default: results/probe)')
parser.add_argument(
    '--representations-key',
    default=datasets.DEFAULT_H5_REPRESENTATIONS_KEY,
    help='key for representations dataset in h5 file (default: reps)')
parser.add_argument('--features-key',
                    default=datasets.DEFAULT_H5_FEATURES_KEY,
                    help='key for features dataset in h5 file (default: tags)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: wandb default)')
parser.add_argument('--wandb-group',
                    help='experiment group (default: wandb default)')
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

if args.task == tasks.PART_OF_SPEECH_TAGGING and args.share_projection:
    raise ValueError('cannot set --share-projection when task is "pos"')

# Configure wandb immediately.
wandb.init(project='lodimp',
           name=args.wandb_name,
           group=args.wandb_group,
           config={
               'task': args.task,
               'representations': {
                   'model': args.representation_model,
                   'layer': args.representation_layer,
               },
               'projection': {
                   'dimension':
                       args.project_to,
                   'shared': (args.task != tasks.PART_OF_SPEECH_TAGGING and
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

args.model_dir.mkdir(parents=True, exist_ok=True)

logging.configure(level=args.log_level)
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

# Load the datasets.
model, layer = args.representation_model, args.representation_layer
data_dir = args.data_dir / model / str(layer)
cache = device if args.cache else None

data: Dict[str, datasets.CollatedTaskDataset] = {}
for split in splits.STANDARD_SPLITS:
    split_file = data_dir / f'{split}.h5'
    if args.no_batch:
        data[split] = datasets.NonBatchingCollatedTaskDataset(
            split_file,
            device=cache,
            representations_key=args.representations_key,
            features_key=args.features_key)
    else:
        data[split] = datasets.SentenceBatchingCollatedTaskDataset(
            split_file,
            device=cache,
            representations_key=args.representations_key,
            features_key=args.features_key)

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

probe_file = args.model_dir / 'probe.pth'
log.info('saving probe to %s', probe_file)
torch.save(probe, probe_file)
wandb.save(str(probe_file))

# For convenience, compute POS nullspaces for downstream testing.
if task == tasks.PART_OF_SPEECH_TAGGING and probe.project is not None:
    log.info('task is pos, so computing projection nullspace')
    projection = probe.project
    assert isinstance(projection, projections.Projection)
    rowspace = linalg.rowspace(projection.project.weight.data)
    nullspace = projections.Projection(*rowspace.shape)
    eye = torch.eye(len(rowspace), device=device)
    nullspace.project.weight.data[:] = eye - rowspace

    nullspace_file = args.model_dir / 'nullspace.pth'
    log.info('saving nullspace to %s', nullspace_file)
    torch.save(nullspace, nullspace_file)
    wandb.save(str(nullspace_file))

log.info('test accuracy %f', accuracy)
wandb.summary['accuracy'] = accuracy
