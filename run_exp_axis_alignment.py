"""Measure how axis-aligned a representation is for a linguistic task.

This command measures how axis-aligned a projection is. Given a probe and
the projection that was learned with it, we zero each row of the projection
one at a time and measure the probe's dev accuracy post-ablation. We determine
which row hurts dev accuracy the least when ablated, and then we zero it
permanently and measure test accuracy. We repeat until no row remain.
"""
import argparse
import pathlib
from typing import Dict

from lodimp import datasets, tasks
from lodimp.models import probes, projections
from lodimp.parse import splits
from lodimp.tasks import dlp, pos
from lodimp.utils import logging

import torch
import wandb
from torch import cuda, nn

parser = argparse.ArgumentParser()
parser.add_argument('--representation-model',
                    choices=('elmo', 'bert-base-uncased'),
                    default='elmo',
                    help='representations to probe (default: elmo)')
parser.add_argument('--representation-layer',
                    type=int,
                    default=0,
                    help='layer to probe (default: 0)')
parser.add_argument('--no-batch',
                    action='store_true',
                    help='store entire dataset in RAM/GPU and do not batch it')
parser.add_argument('--cache',
                    action='store_true',
                    help='cache entire dataset in memory/GPU')
parser.add_argument('--device', help='use this device (default: guessed)')
parser.add_argument('--wandb-id', help='experiment ID, use carefully!')
parser.add_argument('--wandb-group',
                    default='axis-alignment',
                    help='experiment group (default: axis-alignment)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: generated)')
parser.add_argument('--wandb-path',
                    type=pathlib.Path,
                    help='path to write wandb data (default: wandb default)')
parser.add_argument('--download-probe-to',
                    type=pathlib.Path,
                    help='if set, treat probe argument as wandb run path '
                    'and restore probe from that run to this path')
parser.add_argument(
    '--representations-key',
    default=datasets.DEFAULT_H5_REPRESENTATIONS_KEY,
    help='key for representations dataset in h5 file (default: reps)')
parser.add_argument('--features-key',
                    default=datasets.DEFAULT_H5_FEATURES_KEY,
                    help='key for features dataset in h5 file (default: tags)')
parser.add_argument('task',
                    choices=(
                        tasks.PART_OF_SPEECH_TAGGING,
                        tasks.DEPENDENCY_LABEL_PREDICTION,
                    ),
                    help='task on which probe was trained')
parser.add_argument('data_dir', type=pathlib.Path, help='task data dir')
parser.add_argument('probe_file',
                    help='path to probe weights, or wandb run path if '
                    '--download-probe-to is set')
args = parser.parse_args()

if args.task not in (tasks.PART_OF_SPEECH_TAGGING,
                     tasks.DEPENDENCY_LABEL_PREDICTION):
    raise ValueError(f'unsupported task: {args.task}')

# Set up environment. Do not set config yet, as we have to read the probe.
args.wandb_path.mkdir(parents=True, exist_ok=True)
wandb.init(project='lodimp',
           id=args.wandb_id,
           name=args.wandb_name,
           group=args.wandb_group,
           dir=str(args.wandb_path))
assert wandb.run is not None, 'null run?'

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

probe: nn.Module
probe_file: pathlib.Path
if args.download_probe_to:
    probe_file = args.download_probe_to
    log.info('downloading probe from %s to %s', args.probe, probe_file)
    wandb.restore(probe_file, run_path=args.probe_file)
    assert probe_file.exists()
else:
    probe_file = args.probe_file

log.info('reading probe from %s', probe_file)
probe = torch.load(probe_file, map_location=device)

if not (isinstance(probe, probes.Linear) or isinstance(probe, probes.MLP)):
    raise ValueError(f'bad probe type: {probe.__class__}')
projection = probe.project
assert projection is not None, 'no projection?'
assert isinstance(projection, projections.Projection)

# Now we can set the full run config...
wandb.config.update({
    'task': args.task,
    'representations': {
        'model': args.representation_model,
        'layer': args.representation_layer,
    },
    'projection': {
        'dimension': projection.out_features,
    },
    'probe': {
        'model': {
            probes.Linear: 'linear',
            probes.MLP: 'mlp',
        }[probe.__class__],
    },
})

# TODO(evandez): Factor out and share this code as well.
representation_model = args.representation_model
representation_layer = args.representation_layer
data_root = args.data_dir / representation_model / str(representation_layer)
cache = device if args.cache else None

data: Dict[str, datasets.CollatedTaskDataset] = {}
for split in splits.STANDARD_SPLITS:
    split_path = data_root / f'{split}.hdf5'
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

if args.task == tasks.PART_OF_SPEECH_TAGGING:
    pos.axis_alignment(probe,
                       data[splits.DEV],
                       data[splits.TEST],
                       device=device,
                       also_log_to_wandb=True)
else:
    assert args.task == tasks.DEPENDENCY_LABEL_PREDICTION
    dlp.axis_alignment(probe,
                       data[splits.DEV],
                       data[splits.TEST],
                       device=device,
                       also_log_to_wandb=True)
