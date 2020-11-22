"""Find a hierarchy of subspaces."""

import argparse
import logging
import pathlib
from typing import Dict

from lodimp import tasks
from lodimp.common import datasets
from lodimp.common.models import probes, projections
from lodimp.common.parse import splits
from lodimp.tasks import pos

import torch
import wandb


def parser() -> argparse.ArgumentParser:
    """Returns the argument parser for this command."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--linear',
                        action='store_true',
                        help='Use linear probe. Defaults to MLP.')
    parser.add_argument('--max-rank',
                        type=int,
                        default=10,
                        help='Maximum projection rank. Defaults to 10.')
    parser.add_argument(
        '--max-accuracy',
        type=float,
        default=.95,
        help='Move to next task if accuracy exceeds this threshold. '
        'Defaults to .95.')
    parser.add_argument(
        '--accuracy-tolerance',
        type=float,
        default=.05,
        help='Move to next task if accuracy falls within this tolerance '
        'of previous accuracy. Defaults to .05.')
    parser.add_argument('--representation-model',
                        choices=('elmo', 'bert-base-uncased'),
                        default='elmo',
                        help='Representations to probe. Default elmo.')
    parser.add_argument('--representation-layer',
                        type=int,
                        default=0,
                        help='Representation layer to probe. Default 0.')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='Learning rate. Default 1e-3.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Passes to make through dataset during training. Default 25.')
    parser.add_argument(
        '--patience',
        type=int,
        default=4,
        help='Epochs for dev loss to decrease to stop training. Default 4.')
    parser.add_argument('--model-dir',
                        type=pathlib.Path,
                        default='/tmp/lodimp/models',
                        help='Directory to write finished model.')
    parser.add_argument('--representations-key',
                        default=datasets.DEFAULT_H5_REPRESENTATIONS_KEY,
                        help='Key for representations dataset in h5 file.')
    parser.add_argument('--features-key',
                        default=datasets.DEFAULT_H5_FEATURES_KEY,
                        help='Key for features dataset in h5 file.')
    parser.add_argument('--wandb-id', help='Experiment ID. Use carefully!')
    parser.add_argument('--wandb-group',
                        default='hierarchy',
                        help='Experiment group.')
    parser.add_argument('--wandb-name', help='Experiment name.')
    parser.add_argument('--wandb-dir',
                        type=pathlib.Path,
                        default='/tmp/lodimp/wandb',
                        help='Directory to write Weights and Biases data.')
    parser.add_argument(
        '--no-batch',
        action='store_true',
        help='If set, training will use proper gradient descent, not SGD.'
        'This means the entire dataset will be treated as one giant batch. '
        'Be warned! This does not work with extremely large datasets because '
        'the entire dataset must fit into memory (or GPU memory, if --cuda '
        'is set. Use at your own risk.')
    parser.add_argument('--cache',
                        action='store_true',
                        help='Cache entire dataset in memory/GPU. '
                        'See the warning supplied for --no-batch.')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('data', type=pathlib.Path, help='Data path.')
    parser.add_argument('tasks', nargs='+', help='Sequence of tasks in order.')

    return parser


def run(options: argparse.Namespace) -> None:
    """Run the hierarchy experiment."""
    options.model_dir.mkdir(parents=True, exist_ok=True)
    options.wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project='lodimp',
               id=options.wandb_id,
               name=options.wandb_name,
               group=options.wandb_group,
               config={
                   'task': options.task,
                   'representations': {
                       'model': options.representation_model,
                       'layer': options.representation_layer,
                   },
                   'projection': {
                       'dimension':
                           options.project_to,
                       'shared':
                           (options.task != tasks.PART_OF_SPEECH_TAGGING and
                            options.share_projection),
                   },
                   'probe': {
                       'model': options.probe_type,
                   },
                   'hyperparameters': {
                       'epochs': options.epochs,
                       'batched': not options.no_batch,
                       'cached': options.cache,
                       'lr': options.lr,
                       'patience': options.patience,
                   },
               },
               dir=str(options.wandb_dir))
    assert wandb.run is not None, 'null run?'

    log = logging.getLogger(__name__)

    device = torch.device('cuda') if options.cuda else torch.device('cpu')
    log.info('using %s', device.type)

    # Load the datasets.
    model, layer = options.representation_model, options.representation_layer
    data_path = options.data / model / str(layer)
    cache = device if options.cache else None

    data: Dict[str, datasets.CollatedTaskDataset] = {}
    kwargs = dict(device=cache,
                  representations_key=options.representations_key,
                  features_key=options.features_key)
    for split in splits.STANDARD_SPLITS:
        split_path = data_path / f'{split}.hdf5'
        if options.no_batch:
            data[split] = datasets.NonBatchingCollatedTaskDataset(
                split_path, **kwargs)
        else:
            data[split] = datasets.SentenceBatchingCollatedTaskDataset(
                split_path, **kwargs)

    current_rank = options.max_dimension
    current_projection = None
    current_accuracy = .95
    for task in options.tasks:
        log.info('begin search for subspace encoding %s', task)
        for project_to in range(1, current_rank + 1):
            log.info('try task %s rank %d (<= %d) probe', task, project_to,
                     current_rank)
            probe, accuracy = pos.train(
                data[splits.TRAIN],
                data[splits.DEV],
                data[splits.TEST],
                probe_t=probes.Linear if options.linear else probes.MLP,
                project_to=project_to,
                project_from=current_projection,
                epochs=options.epochs,
                patience=options.patience,
                lr=options.lr,
                device=device)

            done = accuracy > current_accuracy - options.accuracy_tolerance
            done |= accuracy > options.max_accuracy
            done |= project_to == current_rank
            if done:
                log.info('best rank for %s is %d (<= %d)', task, project_to,
                         current_rank)
                current_projection = projections.Projection(
                    current_rank, project_to, compose=current_projection)
                current_rank = project_to
                current_accuracy = min(accuracy, options.max_accuracy)

                model_file = options.model_dir / f'{task}.pth'
                log.info('writing probe to %s', model_file)
                torch.save(probe, model_file)
                wandb.save(model_file)

                break

        wandb.run.summary[task] = {
            'rank': current_rank,
            'accuracy': current_accuracy,
        }
