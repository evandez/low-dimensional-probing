"""Find a subspace hierarchy for a sequence of tasks."""
import argparse
import pathlib
from typing import Dict

from ldp import datasets, tasks
from ldp.models import probes, projections
from ldp.parse import splits
from ldp.tasks import pos
from ldp.utils import env, logging

import torch
import wandb
from torch import cuda

parser = argparse.ArgumentParser()
parser.add_argument('tasks', nargs='+', help='sequence of tasks in order')
parser.add_argument(
    '--data-dir',
    type=pathlib.Path,
    help='root dir containing data (default: project data dir)')
parser.add_argument(
    '--results-dir',
    type=pathlib.Path,
    help='dir to write trained probes (default: project results dir)')
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
                    help='max training epochs (default: 25)')
parser.add_argument(
    '--patience',
    type=int,
    default=4,
    help='stop training if dev loss does not improve for this many epochs '
    '(default: 4)')
parser.add_argument('--wandb-group',
                    default='hierarchy',
                    help='experiment group (default: hierarchy)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: generated)')
parser.add_argument('--no-batch',
                    action='store_true',
                    help='store entire dataset in RAM/GPU and do not batch it')
parser.add_argument('--cache',
                    action='store_true',
                    help='cache entire dataset in memory/GPU')
parser.add_argument('--device', help='use this device (default: guessed)')
args = parser.parse_args()

task = '_'.join(args.tasks)
model = args.model
layer = args.layer
project_to = args.project_to

wandb.init(project='ldp',
           name=args.wandb_name or task,
           group=args.wandb_group,
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

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

# Prepare data and results directories.
data_root = args.data_dir or env.data_dir()

results_root = args.results_dir or env.results_dir()
results_dir = results_root / 'hierarchy' / task
results_dir.mkdir(exist_ok=True, parents=True)

current_rank = args.max_dimension
current_projection = None
current_accuracy = .95
for task in args.tasks:
    # Load the datasets.
    data_dir = data_root / 'ptb3/collated' / task / model / str(layer)
    log.info('load data for from %s', data_dir)

    cache = device if args.cache else None
    data: Dict[str, datasets.CollatedTaskDataset] = {}
    for split in splits.STANDARD_SPLITS:
        split_path = data_dir / f'{split}.h5'
        if args.no_batch:
            data[split] = datasets.NonBatchingCollatedTaskDataset(split_path,
                                                                  device=cache)
        else:
            data[split] = datasets.SentenceBatchingCollatedTaskDataset(
                split_path, device=cache)

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
        wandb.log({'task': task, 'rank': project_to, 'accuracy': accuracy})

        done = accuracy > current_accuracy - args.accuracy_tolerance
        done |= accuracy > args.max_accuracy
        done |= project_to == current_rank
        if done:
            # Log results.
            log.info('best rank for %s is %d (<= %d)', task, project_to,
                     current_rank)
            wandb.summary[task] = {'rank': project_to, 'accuracy': accuracy}

            # Save models.
            model_file = results_dir / f'{task}-r{project_to}.pth'
            log.info('writing probe to %s', model_file)
            torch.save(probe, model_file)
            wandb.save(model_file)
            # TODO(evandez): Also save nullspace.

            # Update state.
            current_projection = projections.Projection(
                current_rank, project_to, compose=current_projection)
            current_rank = project_to
            current_accuracy = min(accuracy, args.max_accuracy)

            break
