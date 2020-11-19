"""Defines the `lodimp axis-alignment` command.

This command measures how axis-aligned a projection is. Given a probe and
the projection that was learned with it, we zero each row of the projection
one at a time and measure the probe's dev accuracy post-ablation. We determine
which row hurts dev accuracy the least when ablated, and then we zero it
permanently and measure test accuracy. We repeat until no row remain.
"""

import argparse
import logging
import pathlib
from typing import Dict

from lodimp import tasks
from lodimp.common import datasets
from lodimp.common.models import probes, projections
from lodimp.common.parse import splits
from lodimp.tasks import dlp, pos

import torch
import wandb
from torch import nn


def parser() -> argparse.ArgumentParser:
    """Returns the argument parser for this command."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--representation-model',
                        choices=('elmo', 'bert-base-uncased'),
                        default='elmo',
                        help='Representations to probe. Default elmo.')
    parser.add_argument('--representation-layer',
                        type=int,
                        default=0,
                        help='Representation layer to probe. Default 0.')
    parser.add_argument(
        '--no-batch',
        action='store_true',
        help='Store entire dataset in RAM/GPU and do not batch it.')
    parser.add_argument('--wandb-id', help='Experiment ID. Use carefully!')
    parser.add_argument('--wandb-group', help='Experiment group.')
    parser.add_argument('--wandb-name', help='Experiment name.')
    parser.add_argument('--wandb-path',
                        type=pathlib.Path,
                        default='/tmp/lodimp/wandb',
                        help='Path to write Weights and Biases data.')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--cache',
                        action='store_true',
                        help='Cache entire dataset in memory/GPU.')
    parser.add_argument('--download-probe-to',
                        type=pathlib.Path,
                        help='If set, treat probe argument as WandB run path '
                        'and restore probe from that run to this path.')
    parser.add_argument('--representations-key',
                        default=datasets.DEFAULT_H5_REPRESENTATIONS_KEY,
                        help='Key for representations dataset in h5 file.')
    parser.add_argument('--features-key',
                        default=datasets.DEFAULT_H5_FEATURES_KEY,
                        help='Key for features dataset in h5 file.')
    parser.add_argument('--breaks-key',
                        default=datasets.DEFAULT_H5_BREAKS_KEY,
                        help='Key for breaks dataset in h5 file.')
    parser.add_argument('task',
                        choices=(
                            tasks.PART_OF_SPEECH_TAGGING,
                            tasks.DEPENDENCY_LABEL_PREDICTION,
                        ),
                        help='Task on which probe was trained.')
    parser.add_argument('data', type=pathlib.Path, help='Task data path.')
    parser.add_argument('probe',
                        help='Path to probe, or WandB run path if '
                        '--download-probe-to is set.')
    return parser


def run(options: argparse.Namespace) -> None:
    """Run axis-alignment experiment with the given options.

    Args:
        options (argparse.Namespace): Parsed arguments. See parser() for list
            of flags.

    """
    if options.task not in (tasks.PART_OF_SPEECH_TAGGING,
                            tasks.DEPENDENCY_LABEL_PREDICTION):
        raise ValueError(f'unsupported task: {options.task}')

    # Set up environment. Do not set config yet, as we have to read the probe.
    options.wandb_path.mkdir(parents=True, exist_ok=True)
    wandb.init(project='lodimp',
               id=options.wandb_id,
               name=options.wandb_name,
               group=options.wandb_group,
               dir=str(options.wandb_path))
    assert wandb.run is not None, 'null run?'
    log = logging.getLogger(__name__)

    device = torch.device('cuda') if options.cuda else torch.device('cpu')
    log.info('using %s', device.type)

    probe: nn.Module
    if options.download_probe_to:
        probe_path = options.download_probe_to
        log.info('downloading probe from %s to %s', options.probe, probe_path)
        wandb.restore(probe_path, run_path=options.probe)
        assert probe_path.exists()
    else:
        probe_path = pathlib.Path(options.probe)

    log.info('reading probe from %s', probe_path)
    probe = torch.load(options.probe_path, map_location=device)

    if not (isinstance(probe, probes.Linear) or isinstance(probe, probes.MLP)):
        raise ValueError(f'bad probe type: {probe.__class__}')
    projection = probe.project
    assert projection is not None, 'no projection?'
    assert isinstance(projection, projections.Projection)

    # Now we can set the full run config...
    wandb.run.config['task'] = options.task
    wandb.run.config['representations'] = {
        'model': options.representation_model,
        'layer': options.representation_layer,
    }
    wandb.run.config['projection'] = {
        'dimension': projection.out_features,
    }
    wandb.run.config['probe'] = {
        # TODO(evandez): Share this with train command.
        'model': {
            probes.Linear: 'linear',
            probes.MLP: 'mlp',
        }[probe.__class__]
    }

    # TODO(evandez): Factor out and share this code as well.
    representation_model = options.representation_model
    representation_layer = options.representation_layer
    data_path = options.data / representation_model / str(representation_layer)
    cache = device if options.cache else None

    data: Dict[str, datasets.CollatedTaskDataset] = {}
    kwargs = dict(device=cache,
                  representations_key=options.representations_key,
                  features_key=options.features_key,
                  breaks_key=options.breaks_key)
    for split in splits.STANDARD_SPLITS:
        split_path = data_path / f'{split}.hdf5'
        if options.no_batch:
            data[split] = datasets.NonBatchingCollatedTaskDataset(
                split_path, **kwargs)
        else:
            data[split] = datasets.SentenceBatchingCollatedTaskDataset(
                split_path, **kwargs)

    if options.task == tasks.PART_OF_SPEECH_TAGGING:
        pos.axis_alignment(probe,
                           datasets[splits.DEV],
                           datasets[splits.TEST],
                           device=device,
                           also_log_to_wandb=True)
    else:
        dlp.axis_alignment(probe,
                           datasets[splits.DEV],
                           datasets[splits.TEST],
                           device=device,
                           also_log_to_wandb=True)
