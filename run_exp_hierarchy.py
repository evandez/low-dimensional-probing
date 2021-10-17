"""Find a subspace hierarchy for a sequence of tasks."""
import argparse
import pathlib
from typing import Dict

from ldp import datasets, tasks
from ldp.models import probes, projections
from ldp.parse import splits
from ldp.tasks import pos
from ldp.utils import logging

import torch
import wandb
from torch import cuda

parser = argparse.ArgumentParser()
parser.add_argument('--linear',
                    action='store_true',
                    help='use linear probe (default: mlp)')
parser.add_argument('--max-rank',
                    type=int,
                    default=10,
                    help='max projection rank (default: 10)')
parser.add_argument(
    '--max-accuracy',
    type=float,
    default=.95,
    help='move to next task if accuracy exceeds this threshold '
    '(default: .95)')
parser.add_argument(
    '--accuracy-tolerance',
    type=float,
    default=.05,
    help='move to next task if accuracy falls within this tolerance '
    'of previous accuracy (default: .05)')
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
                    help='max training epochs (default: 25)')
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
    help='directory to write finished probes (default: results/probes)')
parser.add_argument(
    '--representations-key',
    default=datasets.DEFAULT_H5_REPRESENTATIONS_KEY,
    help='key for representations dataset in h5 file (default: reps)')
parser.add_argument('--features-key',
                    default=datasets.DEFAULT_H5_FEATURES_KEY,
                    help='key for features dataset in h5 file (default: tags)')
parser.add_argument('--wandb-group',
                    default='hierarchy',
                    help='experiment group (default: hierarchy)')
parser.add_argument('--wandb-name', help='Experiment name.')
parser.add_argument('--no-batch',
                    action='store_true',
                    help='store entire dataset in RAM/GPU and do not batch it')
parser.add_argument('--cache',
                    action='store_true',
                    help='cache entire dataset in memory/GPU')
parser.add_argument('--device', help='use this device (default: guessed)')
parser.add_argument('data_dir', type=pathlib.Path, help='data directory')
parser.add_argument('tasks', nargs='+', help='sequence of tasks in order')
args = parser.parse_args()

args.model_dir.mkdir(parents=True, exist_ok=True)
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

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

# Load the datasets.
model, layer = args.representation_model, args.representation_layer
data_root = args.data_dir / model / str(layer)
cache = device if args.cache else None

data: Dict[str, datasets.CollatedTaskDataset] = {}
for split in splits.STANDARD_SPLITS:
    split_path = data_root / f'{split}.h5'
    if args.no_batch:
        data[split] = datasets.NonBatchingCollatedTaskDataset(
            split_path,
            device=cache,
            representations_key=args.representations_key,
            features_key=args.features_key)
    else:
        data[split] = datasets.SentenceBatchingCollatedTaskDataset(
            split_path,
            device=cache,
            representations_key=args.representations_key,
            features_key=args.features_key)

current_rank = args.max_dimension
current_projection = None
current_accuracy = .95
for task in args.tasks:
    log.info('begin search for subspace encoding %s', task)
    for project_to in range(1, current_rank + 1):
        log.info('try task %s rank %d (<= %d) probe', task, project_to,
                 current_rank)
        probe, accuracy = pos.train(
            data[splits.TRAIN],
            data[splits.DEV],
            data[splits.TEST],
            probe_t=probes.Linear if args.linear else probes.MLP,
            project_to=project_to,
            project_from=current_projection,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            device=device)

        done = accuracy > current_accuracy - args.accuracy_tolerance
        done |= accuracy > args.max_accuracy
        done |= project_to == current_rank
        if done:
            log.info('best rank for %s is %d (<= %d)', task, project_to,
                     current_rank)
            current_projection = projections.Projection(
                current_rank, project_to, compose=current_projection)
            current_rank = project_to
            current_accuracy = min(accuracy, args.max_accuracy)

            model_file = args.model_dir / f'{task}.pth'
            log.info('writing probe to %s', model_file)
            torch.save(probe, model_file)
            wandb.save(model_file)

            break

    wandb.summary[task] = {
        'rank': current_rank,
        'accuracy': current_accuracy,
    }
