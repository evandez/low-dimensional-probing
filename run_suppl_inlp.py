"""Run INLP on the ordinary POS task to obtain a nullspace projection.

INLP is a method for ablating from a representation all information is linearly
predictive of some pre-defined attributes. In our case, the attributes are
parts of speech, and the representations are contextual word representations.
"""
import argparse
import pathlib
from typing import Dict

from ldp import datasets
from ldp.parse import splits
from ldp.tasks import pos
from ldp.utils import env, logging

import torch
import wandb
from torch import cuda

parser = argparse.ArgumentParser(description='run inlp')
parser.add_argument('task',
                    choices=('pos', 'pos-verb', 'pos-noun'),
                    help='task to run inlp on')
parser.add_argument(
    '--data-dir',
    type=pathlib.Path,
    help='root dir containing data (default: project data dir)')
parser.add_argument(
    '--results-dir',
    type=pathlib.Path,
    help='root dir to write inlp results (default: project results dir)')
parser.add_argument('--model',
                    choices=('elmo', 'bert', 'bert-random'),
                    default='bert',
                    help='representations to probe (default: bert)')
parser.add_argument('--layer',
                    type=int,
                    default=0,
                    help='representation layer (default: 0)')
parser.add_argument('--project-to',
                    type=int,
                    default=64,
                    help='dimensionality of projected space (default: 64)')
parser.add_argument('--epochs',
                    type=int,
                    default=25,
                    help='training epochs (default: 25)')
parser.add_argument('--lr',
                    default=1e-3,
                    type=float,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--attempts',
                    type=int,
                    default=100,
                    help='max projections to compose (default: 100)')
parser.add_argument(
    '--tolerance',
    type=int,
    default=5e-2,
    help='tolerance for determining when probe accuracy is at chance '
    '(default: 5e-2)')
parser.add_argument('--device', help='use this device (default: guessed)')
parser.add_argument('--no-batch',
                    action='store_true',
                    help='do not batch data')
parser.add_argument('--cache',
                    action='store_true',
                    help='cache entire dataset in memory/GPU')
parser.add_argument('--wandb-group',
                    default='inlp',
                    help='experiment group (default: inlp)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: generated)')
args = parser.parse_args()

task = args.task
model = args.model
layer = args.layer

wandb.init(project='ldp',
           name=args.wandb_name or f'{task}-{model}-l{layer}',
           group=args.wandb_group,
           reinit=True,
           config={
               'task': task,
               'representations': {
                   'model': args.model,
                   'layer': args.layer,
               },
               'projection': {
                   'dimension': args.project_to,
               },
               'probe': {
                   'model': 'linear',
               },
               'hyperparameters': {
                   'epochs': args.epochs,
                   'batched': not args.no_batch,
                   'cached': args.cache,
                   'lr': args.lr,
               },
           })

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

data_root = args.data_dir or env.data_dir()
data_dir = data_root / 'ptb3/collated' / task / model / str(layer)

results_root = args.results_dir or env.results_dir()
results_dir = results_root / 'inlp' / task
results_dir.mkdir(exist_ok=True, parents=True)

cache = device if args.cache else None
data: Dict[str, datasets.CollatedTaskDataset] = {}
for split in splits.STANDARD_SPLITS:
    split_path = data_root / f'{split}.h5'
    if args.no_batch:
        data[split] = datasets.NonBatchingCollatedTaskDataset(split_path,
                                                              device=cache)
    else:
        data[split] = datasets.SentenceBatchingCollatedTaskDataset(
            split_path, device=cache)

nullspace = pos.inlp(
    data[splits.TRAIN],
    data[splits.DEV],
    data[splits.TEST],
    rank=args.project_to,
    attempts=args.attempts,
    tolerance=args.tolerance,
    lr=args.lr,
    epochs=args.epochs,
    device=device,
    also_log_to_wandb=True,
)

nullspace_file = results_dir / 'nullspace.pth'
log.info('saving projection to %s', nullspace_file)
torch.save(nullspace, nullspace_file)
wandb.save(str(nullspace_file))
