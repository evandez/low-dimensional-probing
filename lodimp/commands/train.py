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
from lodimp.common import datasets
from lodimp.common.models import probes
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
    parser.add_argument('--project-to',
                        type=int,
                        default=64,
                        help='Dimensionality of projected space.')
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
    parser.add_argument('--model-path',
                        type=pathlib.Path,
                        default='/tmp/lodimp/models/probe.pth',
                        help='Directory to write finished model.')
    parser.add_argument('--wandb-id', help='Experiment ID. Use carefully!')
    parser.add_argument('--wandb-group', help='Experiment group.')
    parser.add_argument('--wandb-name', help='Experiment name.')
    parser.add_argument('--wandb-path',
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
    parser.add_argument('data', type=pathlib.Path, help='Data path.')

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
    options.model_path.parent.mkdir(parents=True, exist_ok=True)
    options.wandb_path.mkdir(parents=True, exist_ok=True)
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
               dir=str(options.wandb_path))
    assert wandb.run is not None, 'null run?'

    log = logging.getLogger(__name__)

    device = torch.device('cuda') if options.cuda else torch.device('cpu')
    log.info('using %s', device.type)

    # Load the datasets.
    model, layer = options.representation_model, options.representation_layer
    data_path = options.data / model / str(layer)
    cache = device if options.cache else None

    data: Dict[str, datasets.CollatedTaskDataset] = {}
    for split in splits.STANDARD_SPLITS:
        split_path = data_path / f'{split}.hdf5'
        if options.no_batch:
            data[split] = datasets.NonBatchingCollatedTaskDataset(split_path,
                                                                  device=cache)
        else:
            data[split] = datasets.SentenceBatchingCollatedTaskDataset(
                split_path, device=cache)

    # Start training!
    probe: nn.Module
    if options.task == tasks.PART_OF_SPEECH_TAGGING:
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
    elif options.task == tasks.DEPENDENCY_LABEL_PREDICTION:
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
    elif options.task == tasks.DEPENDENCY_EDGE_PREDICTION:
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
        raise ValueError(f'unknown task: {options.task}')

    logging.info('saving model to %s', options.model_path)
    torch.save(probe, options.model_path)
    wandb.save(str(options.model_path))

    logging.info('test accuracy %f', accuracy)
    wandb.run.summary['accuracy'] = accuracy
