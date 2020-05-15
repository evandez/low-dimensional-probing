"""Entry point for all experiments."""

import argparse
import copy
import itertools
import logging
import math
import pathlib
import sys
from typing import Dict, Union

from lodimp import datasets, linalg, models

import torch
import torch.utils.data
import wandb
from torch import nn, optim

parser = argparse.ArgumentParser(description='Train a POS tagger.')
parser.add_argument('task', type=pathlib.Path, help='Task directory.')
parser.add_argument('layer', type=int, help='ELMo layer.')
parser.add_argument('dim', type=int, help='Projection dimensionality.')
parser.add_argument('probe',
                    choices=(
                        'linear',
                        'mlp',
                        'pairwise-bilinear',
                        'pairwise-mlp',
                    ),
                    help='Use MLP probe.')
parser.add_argument('--no-batch', action='store_true', help='Do not batch.')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='Passes to make through dataset during training.')
parser.add_argument('--l1', type=float, help='Add L1 norm penalty.')
parser.add_argument('--nuc', type=float, help='Add nuclear norm penalty')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('--patience',
                    type=int,
                    default=4,
                    help='Epochs for dev loss to decrease to stop training.')
parser.add_argument('--compose',
                    nargs='+',
                    type=pathlib.Path,
                    help='Compose these projections with learned projection.')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device.')
parser.add_argument('--wandb-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp',
                    help='Path to write Weights and Biases data.')
parser.add_argument('--wandb-group', help='Experiment group.')
parser.add_argument('--wandb-name', help='Experiment name.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Directory to write finished model.')
parser.add_argument('--projection-file',
                    help='Projection save file. Generated by default.')
parser.add_argument('--probe-file',
                    help='Probe model save file. Generated by default.')
parser.add_argument('--quiet',
                    dest='log_level',
                    action='store_const',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='Only show warning or error messages.')
options = parser.parse_args()

# Initialize wandb so we can track any failures right away.
options.wandb_dir.mkdir(parents=True, exist_ok=True)
wandb.init(project='lodimp',
           name=options.wandb_name,
           group=options.wandb_group,
           config={
               'task': options.task.name,
               'representations': {
                   'model': 'elmo',
                   'layer': options.layer,
               },
               'projection': {
                   'dimension': options.dim,
                   'compositions': len(options.compose or []),
               },
               'probe': {
                   'model': options.probe,
               },
               'hyperparameters': {
                   'epochs': options.epochs,
                   'batched': not options.no_batch,
                   'lr': options.lr,
                   'patience': options.patience,
                   'regularization': {
                       'l1': options.l1,
                       'nuc': options.nuc,
                   },
               },
           },
           dir=str(options.wandb_dir))

# Quick validations.
if not options.task.exists():
    raise FileNotFoundError(f'task directory does not exist: {options.task}')

root = options.task / f'elmo-{options.layer}'
if not root.exists():
    raise FileNotFoundError(f'layer {options.layer} partition not found')

for path in options.compose or []:
    if not path.exists():
        raise FileNotFoundError(f'model does not exist: {path}')

# Set up.
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

device = torch.device('cuda') if options.cuda else torch.device('cpu')
logging.info('using %s', device.type)

options.model_dir.mkdir(parents=True, exist_ok=True)
logging.info('model(s) will be written to %s', options.model_dir)

# Load data.
data = {}
for split in ('train', 'dev', 'test'):
    file = root / f'{split}.h5'
    if not file.exists():
        raise FileNotFoundError(f'{split} partition not found')
    data[split] = datasets.TaskDataset(root / f'{split}.h5')
train, dev, test = data['train'], data['dev'], data['test']

ndims, ngrams = train.ndims, train.ngrams
logging.info('samples consist of %d-grams of %d dimensions', ngrams, ndims)

nlabels = train.nlabels
logging.info('task has %s labels',
             'variable number of' if nlabels is None else str(nlabels))

# Batch data.
loaders: Dict[str, Union[datasets.SentenceTaskDataset,
                         datasets.ChunkedTaskDataset]] = {}
for split, dataset in data.items():
    if options.no_batch:
        logging.info('batching disabled, collating %s set', split)
        # You might be wondering: why not use a DataLoader here, like a normal
        # person? It's because DataLoaders unexpectedly copy the data in some
        # cases. This is problematic if, for example, your data is already
        # stored on the GPU and copying it would result in an OOM error.
        # We make our lives easier by simply iterating over datasets.
        loaders[split] = datasets.ChunkedTaskDataset(
            dataset,
            chunks=dataset.breaks if nlabels is None else 1,
            device=device)
    else:
        logging.info('batching %s set by sentence', split)
        loaders[split] = datasets.SentenceTaskDataset(dataset)

# Initialize the projection.
if options.compose:
    proj = torch.load(options.compose, map_location=device).extend(options.dim)
else:
    proj = models.Projection(ndims, options.dim).to(device)

probe: nn.Module
input_dimension = ngrams * options.dim
if options.probe == 'linear':
    assert nlabels is not None
    probe = nn.Linear(input_dimension, nlabels)
elif options.probe == 'mlp':
    assert nlabels is not None
    probe = models.MLP(input_dimension, nlabels)
elif options.probe == 'pairwise-bilinear':
    probe = models.PairwiseBilinear(input_dimension)
else:
    assert options.probe == 'pairwise-mlp'
    probe = models.PairwiseMLP(input_dimension)
probe = probe.to(device)

parameters = itertools.chain(proj.parameters(), probe.parameters())
optimizer = optim.Adam(parameters, lr=options.lr)
ce = nn.CrossEntropyLoss().to(device)


def criterion(*args: torch.Tensor) -> torch.Tensor:
    """Returns CE loss with regularizers."""
    loss = ce(*args)
    if options.l1:
        return loss + options.l1 * proj.weight.norm(p=1)
    if options.nuc:
        return loss + options.nuc * proj.weight.norm(p='nuc')
    return loss


# Train the model.
best_dev_loss, bad_epochs = float('inf'), 0
for epoch in range(options.epochs):
    proj.train()
    probe.train()
    for batch, (reps, tags) in enumerate(loaders['train']):
        reps, tags = reps.to(device), tags.to(device)
        projected = proj(reps).view(-1, input_dimension)
        preds = probe(projected)
        loss = criterion(preds, tags)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({'train loss': loss})
        logging.info('epoch %d batch %d loss %f', epoch, batch, loss.item())

    proj.eval()
    probe.eval()
    total, count = 0., 0
    for reps, tags in loaders['dev']:
        reps, tags = reps.to(device), tags.to(device)
        projected = proj(reps).view(-1, input_dimension)
        preds = probe(projected)
        total += criterion(preds, tags).item() * len(reps)  # Undo mean.
        count += len(reps)
    dev_loss = total / count
    erank = linalg.effective_rank(proj.project.weight)
    logging.info('epoch %d dev loss %f erank %f', epoch, dev_loss, erank)
    wandb.log({'dev loss': dev_loss, 'erank': erank})

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs > options.patience:
        logging.info('patience exceeded, training is now over')
        break

# Write finished models.
for name, model_file, model in (('proj', options.projection_file, proj),
                                ('probe', options.probe_file, probe)):
    model_file = model_file or f'{name}.pth'
    model_path = options.model_dir / model_file
    torch.save(model, model_path)
    wandb.save(str(model_path))
    logging.info('%s saved to %s', name, model_path)

# Evaluate the models.
erank = linalg.effective_rank(proj.project.weight)
truncated = copy.deepcopy(proj)
weights = linalg.truncate(truncated.project.weight.data, math.ceil(erank))
truncated.project.weight.data = weights

for name, model in (('full', proj), ('truncated', truncated)):
    logging.info('testing on %s model', name)
    correct, count = 0., 0
    for reps, tags in loaders['test']:
        reps, tags = reps.to(device), tags.to(device)
        projected = model(reps).view(-1, input_dimension)
        preds = probe(projected).argmax(dim=1)
        correct += preds.eq(tags).sum().item()
        count += len(reps)
    accuracy = correct / count
    wandb.log({f'{name} accuracy': accuracy})
    logging.info('%s accuracy %.3f', name, accuracy)
