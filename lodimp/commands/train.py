"""Defines the `lodimp train` command.

This command trains a probe to predict linguistic features given only
representations. For performance reasons, it assumes the data has been
collated. Use the `lodimp collate` command to do that.
"""

import argparse
import logging
import pathlib
from typing import Dict

from lodimp import tasks
from lodimp.common import datasets, linalg
from lodimp.common.models import probes, projections
from lodimp.common.parse import splits
from lodimp.tasks import dep, dlp, pos

import torch
import wandb
from torch import nn

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


def parser() -> argparse.ArgumentParser:
    """Returns the argument parser for this command."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--linear',
                        action='store_const',
                        dest='probe_type',
                        const=LINEAR_PROBE,
                        default=MLP_PROBE,
                        help='Use linear probe. Defaults to MLP.')
    parser.add_argument(
        '--project-to',
        type=int,
        help='Project reps to this dimension. Default is no projection.')
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
    parser.add_argument('--wandb-group', help='Experiment group.')
    parser.add_argument('--wandb-name', help='Experiment name.')
    parser.add_argument('--wandb-dir',
                        type=pathlib.Path,
                        default='/tmp/lodimp/wandb',
                        help='Path to write Weights and Biases data.')
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
    subparsers = parser.add_subparsers(dest='task')
    subparsers.add_parser(tasks.PART_OF_SPEECH_TAGGING)
    dlp_parser = subparsers.add_parser(tasks.DEPENDENCY_LABEL_PREDICTION)
    dep_parser = subparsers.add_parser(tasks.DEPENDENCY_EDGE_PREDICTION)
    parser.add_argument('data_dir', type=pathlib.Path, help='Data directory.')

    # Okay, we've defined the full command. Now define task-specific options.
    for subparser in (dlp_parser, dep_parser):
        subparser.add_argument(
            '--share-projection',
            action='store_true',
            help='When comparing reps, project both with same matrix.')

    return parser


def run(options: argparse.Namespace) -> None:
    """Run training with the given options.

    Args:
        options (argparse.Namespace): Parsed arguments. See parser() for list
            of flags.

    """
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
    data_dir = options.data_dir / model / str(layer)
    cache = device if options.cache else None

    data: Dict[str, datasets.CollatedTaskDataset] = {}
    kwargs = dict(device=cache,
                  representations_key=options.representations_key,
                  features_key=options.features_key)
    for split in splits.STANDARD_SPLITS:
        split_file = data_dir / f'{split}.hdf5'
        if options.no_batch:
            data[split] = datasets.NonBatchingCollatedTaskDataset(
                split_file, **kwargs)
        else:
            data[split] = datasets.SentenceBatchingCollatedTaskDataset(
                split_file, **kwargs)

    # Start training!
    probe: nn.Module
    task = options.task
    if task == tasks.PART_OF_SPEECH_TAGGING:
        probe, accuracy = pos.train(
            data[splits.TRAIN],
            data[splits.DEV],
            data[splits.TEST],
            probe_t=PROBE_TYPES_BY_TASK[options.task][options.probe_type],
            project_to=options.project_to,
            epochs=options.epochs,
            patience=options.patience,
            lr=options.lr,
            device=device,
            also_log_to_wandb=True)
    elif task == tasks.DEPENDENCY_LABEL_PREDICTION:
        probe, accuracy = dlp.train(
            data[splits.TRAIN],
            data[splits.DEV],
            data[splits.TEST],
            probe_t=PROBE_TYPES_BY_TASK[options.task][options.probe_type],
            project_to=options.project_to,
            share_projection=options.share_projection,
            epochs=options.epochs,
            patience=options.patience,
            lr=options.lr,
            device=device,
            also_log_to_wandb=True)
    elif task == tasks.DEPENDENCY_EDGE_PREDICTION:
        probe, accuracy = dep.train(
            data[splits.TRAIN],
            data[splits.DEV],
            data[splits.TEST],
            probe_t=PROBE_TYPES_BY_TASK[options.task][options.probe_type],
            project_to=options.project_to,
            share_projection=options.share_projection,
            epochs=options.epochs,
            patience=options.patience,
            lr=options.lr,
            device=device,
            also_log_to_wandb=True)
    else:
        raise ValueError(f'unknown task: {task}')

    probe_file = options.model_dir / 'probe.pth'
    log.info('saving probe to %s', probe_file)
    torch.save(probe, probe_file)
    wandb.save(str(probe_file))

    # For convenience, compute POS nullspaces for downstream testing.
    if task == tasks.PART_OF_SPEECH_TAGGING and probe.project is not None:
        log.info('task is pos, so computing projection nullspace')
        rowspace = linalg.rowspace(probe.project.weight.data)
        nullspace = projections.Projection(*rowspace.shape)
        eye = torch.eye(len(rowspace), device=device)
        nullspace.project.weight.data[:] = eye - rowspace

        nullspace_file = options.model_dir / 'nullspace.pth'
        log.info('saving nullspace to %s', nullspace_file)
        torch.save(nullspace, nullspace_file)
        wandb.save(str(nullspace_file))

    log.info('test accuracy %f', accuracy)
    wandb.run.summary['accuracy'] = accuracy
