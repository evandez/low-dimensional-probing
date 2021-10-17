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

from ldp import datasets, tasks
from ldp.models import probes, projections
from ldp.parse import splits
from ldp.tasks import dlp, pos
from ldp.utils import env, logging

import torch
import wandb
from torch import cuda

parser = argparse.ArgumentParser(description='run axis alignment experiments')
parser.add_argument('task',
                    choices=(tasks.PART_OF_SPEECH_TAGGING,
                             tasks.DEPENDENCY_LABEL_PREDICTION),
                    help='task on which probe was trained')
parser.add_argument(
    '--data-dir',
    type=pathlib.Path,
    help='root dir containing data (default: project data dir)')
parser.add_argument(
    '--results-dir',
    help='root dir containing trained probes (default: project results dir)')
parser.add_argument('--linear',
                    action='store_const',
                    help='use linear probe (default: mlp)')
parser.add_argument('--model',
                    choices=('elmo', 'bert', 'bert-random'),
                    default='elmo',
                    help='representations to probe (default: elmo)')
parser.add_argument('--layer',
                    type=int,
                    default=0,
                    help='layer to probe (default: 0)')
parser.add_argument('--rank',
                    type=int,
                    default=8,
                    help='rank of probe to use (default: 8)')
parser.add_argument('--no-batch',
                    action='store_true',
                    help='store entire dataset in RAM/GPU and do not batch it')
parser.add_argument('--cache',
                    action='store_true',
                    help='cache entire dataset in memory/GPU')
parser.add_argument('--device', help='use this device (default: guessed)')
parser.add_argument('--wandb-group',
                    default='axis-alignment',
                    help='experiment group (default: axis-alignment)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: generated)')
parser.add_argument(
    '--representations-key',
    default=datasets.DEFAULT_H5_REPRESENTATIONS_KEY,
    help='key for representations dataset in h5 file (default: reps)')
parser.add_argument('--features-key',
                    default=datasets.DEFAULT_H5_FEATURES_KEY,
                    help='key for features dataset in h5 file (default: tags)')
args = parser.parse_args()

task = args.task
model = args.model
layer = args.layer
rank = args.rank

# Set up environment. Do not set config yet, as we have to read the probe.
wandb.init(project='ldp',
           name=args.wandb_name or f'{task}-{model}-l{layer}-r{rank}',
           group=args.wandb_group)

if args.task not in (tasks.PART_OF_SPEECH_TAGGING,
                     tasks.DEPENDENCY_LABEL_PREDICTION):
    raise ValueError(f'unsupported task: {args.task}')

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

results_root = args.results_dir or env.results_dir()
results_dir = results_root / 'train-probe'
results_dir /= 'linear' if args.linear else 'mlp'
results_dir /= f'{task}/{model}/l{layer}'
results_dir /= f'r{rank}' if rank is not None else 'full'

probe_file = results_dir / 'probe.pth'
log.info('reading probe from %s', probe_file)
probe = torch.load(probe_file, map_location=device)

# Now we can set the full run config...
wandb.config.update({
    'task': task,
    'representations': {
        'model': args.model,
        'layer': args.layer,
    },
    'projection': {
        'rank': rank,
    },
    'probe': {
        'model': {
            probes.Linear: 'linear',
            probes.MLP: 'mlp',
        }[probe.__class__],
    },
})

data_root = args.data_dir or env.data_dir()
data_dir = data_root / 'ptb3/collated' / task / model / str(layer)
cache = device if args.cache else None

data: Dict[str, datasets.CollatedTaskDataset] = {}
for split in splits.STANDARD_SPLITS:
    split_path = data_dir / f'{split}.h5'
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
