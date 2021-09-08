"""Run INLP on the ordinary POS task to obtain a nullspace projection.

INLP is a method for ablating from a representation all information is linearly
predictive of some pre-defined attributes. In our case, the attributes are
parts of speech, and the representations are contextual word representations.
"""
import argparse
import pathlib
from typing import Dict

from lodimp.common import datasets, logging
from lodimp.common.parse import splits
from lodimp.tasks import pos

import torch
import wandb
from torch import cuda

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=pathlib.Path, help='data directory')
parser.add_argument('--representation-model',
                    choices=('elmo', 'bert-base-uncased'),
                    default='bert-base-uncased',
                    help='representations to probe (default: bert)')
parser.add_argument('--representation-layer',
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
parser.add_argument('--wandb-id', help='experiment ID, use carefully')
parser.add_argument('--wandb-group',
                    default='inlp',
                    help='experiment group (default: inlp)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: generated)')
parser.add_argument('--wandb-path',
                    type=pathlib.Path,
                    help='path to write wandb data (default: wandb default)')
parser.add_argument('--model-path',
                    type=pathlib.Path,
                    default='results/probes/inlp',
                    help='directory to write nullspace projection.')
parser.add_argument(
    '--representations-key',
    default=datasets.DEFAULT_H5_REPRESENTATIONS_KEY,
    help='key for representations dataset in h5 file (default: reps)')
parser.add_argument('--features-key',
                    default=datasets.DEFAULT_H5_FEATURES_KEY,
                    help='key for features dataset in h5 file (default: tags)')
args = parser.parse_args()

args.model_path.parent.mkdir(parents=True, exist_ok=True)
args.wandb_path.mkdir(parents=True, exist_ok=True)
wandb.init(project='lodimp',
           id=args.wandb_id,
           name=args.wandb_name,
           group=args.wandb_group,
           reinit=True,
           config={
               'task': 'pos',
               'representations': {
                   'model': args.representation_model,
                   'layer': args.representation_layer,
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
           },
           dir=str(args.wandb_path))

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using %s', device)

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

log.info('saving projection to %s', args.model_path)
torch.save(nullspace, args.model_path)
wandb.save(str(args.model_path))
