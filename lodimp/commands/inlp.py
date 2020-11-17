"""Defines the `lodimp inlp` command.

INLP is a method for ablating from a representation all information is linearly
predictive of some pre-defined attributes. In our case, the attributes are
parts of speech, and the representations are contextual word representations.
"""

import argparse
import logging
import pathlib
from typing import Dict

from lodimp.common import datasets
from lodimp.common.parse import splits
from lodimp.tasks import pos

import torch
import wandb


def parser() -> argparse.ArgumentParser:
    """Returns the argument parser for this command."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('data', type=pathlib.Path, help='Data path.')
    parser.add_argument('--representation-model',
                        choices=('elmo', 'bert-base-uncased'),
                        default='bert-base-uncased',
                        help='Representations to probe. Default BERT.')
    parser.add_argument('--representation-layer',
                        type=int,
                        default=0,
                        help='Representation layer to probe. Default 0.')
    parser.add_argument('--project-to',
                        type=int,
                        default=64,
                        help='Dimensionality of projected space.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Maximum number of passes to make through dataset when training '
        'each linear probe. Default 25.')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='Learning rate. Default 1e-3.')
    parser.add_argument('--attempts',
                        type=int,
                        default=100,
                        help='Maximum projections to compose.')
    parser.add_argument(
        '--tolerance',
        type=int,
        default=5e-2,
        help='Tolerance for determining when probe accuracy is at chance.')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--no-batch',
                        action='store_true',
                        help='Do not batch data.')
    parser.add_argument('--cache',
                        action='store_true',
                        help='Cache entire dataset in memory/GPU.')
    parser.add_argument('--wandb-id', help='Experiment ID. Use carefully!')
    parser.add_argument('--wandb-group', help='Experiment group.')
    parser.add_argument('--wandb-name', help='Experiment name.')
    parser.add_argument('--wandb-path',
                        type=pathlib.Path,
                        default='/tmp/lodimp/wandb',
                        help='Path to write Weights and Biases data.')
    parser.add_argument('--model-path',
                        type=pathlib.Path,
                        default='/tmp/lodimp/models/probe.pth',
                        help='Directory to write nullspace projection.')
    return parser


def run(options: argparse.Namespace) -> None:
    """Run INLP with the given options.

    Args:
        options (argparse.Namespace): Parsed arguments. See parser() for list
            of flags.

    """
    options.model_path.mkdir(parents=True, exist_ok=True)
    options.wandb_path.mkdir(parents=True, exist_ok=True)
    wandb.init(project='lodimp',
               id=options.wandb_id,
               name=options.wandb_name,
               group=options.wandb_group,
               reinit=True,
               config={
                   'task': 'pos',
                   'representations': {
                       'model': options.representation_model,
                       'layer': options.representation_layer,
                   },
                   'projection': {
                       'dimension': options.project_to,
                   },
                   'probe': {
                       'model': 'linear',
                   },
                   'hyperparameters': {
                       'epochs': options.epochs,
                       'batched': not options.no_batch,
                       'cached': options.cache,
                       'lr': options.lr,
                   },
               },
               dir=str(options.wandb_path))
    log = logging.getLogger(__name__)

    device = torch.device('cuda') if options.cuda else torch.device('cpu')
    log.info('using %s', device.type)

    representaiton_model = options.representaiton_model
    representation_layer = options.representation_layer
    data_path = options.data / representaiton_model / str(representation_layer)
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

    nullspace = pos.inlp(
        data[splits.TRAIN],
        data[splits.DEV],
        data[splits.TEST],
        rank=options.project_to,
        attempts=options.attempts,
        tolerance=options.tolerance,
        lr=options.lr,
        epochs=options.epochs,
        device=device,
        also_log_to_wandb=True,
    )

    log.info('saving projection to %s', options.model_path)
    torch.save(nullspace, options.model_path)
    wandb.save(str(options.model_path))
