"""Entry point for all experiments."""

import argparse
import copy
import logging
import pathlib
import sys
from typing import Dict, Set, Union

from lodimp import datasets, linalg, models

import torch
import torch.utils.data
import wandb
from torch import nn, optim

parser = argparse.ArgumentParser(description='Train a POS tagger.')
parser.add_argument('data', type=pathlib.Path, help='Data directory.')
parser.add_argument('layer', type=int, help='ELMo layer.')
parser.add_argument('dim', type=int, help='Projection dimensionality.')
parser.add_argument('probe', choices=('linear', 'mlp'), help='Probe model.')
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
parser.add_argument('--model-file',
                    default='probe.pth',
                    help='Model file name.')
parser.add_argument('--ablate',
                    action='store_true',
                    help='Also test ablated model.')
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
               'task': 'pos',
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
if not options.data.exists():
    raise FileNotFoundError(f'task directory does not exist: {options.data}')

root = options.data / f'elmo-{options.layer}'
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

ndims = data['train'].ndims
logging.info('reps have %d dimensions', ndims)

nlabels = data['train'].nlabels
assert nlabels is not None, 'no label count?'
logging.info('task has %d labels', nlabels)

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
        loaders[split] = datasets.ChunkedTaskDataset(dataset, device=device)
    else:
        logging.info('batching %s set by sentence', split)
        loaders[split] = datasets.SentenceTaskDataset(dataset)

# Initialize the projection.
if options.compose:
    compose = torch.load(options.compose, map_location=device)
    projection = models.Projection(ndims, options.dim, compose=compose)
else:
    projection = models.Projection(ndims, options.dim)

probe: Union[models.Linear, models.MLP]
if options.probe == 'linear':
    assert nlabels is not None
    probe = models.Linear(options.dim, nlabels, project=projection)
else:
    assert options.probe == 'mlp', 'unknown model?'
    probe = models.MLP(options.dim, nlabels, project=projection)
probe = probe.to(device)

optimizer = optim.Adam(probe.parameters(), lr=options.lr)
ce = nn.CrossEntropyLoss().to(device)


def criterion(*args: torch.Tensor) -> torch.Tensor:
    """Returns CE loss with regularizers."""
    loss = ce(*args)
    if options.l1:
        return loss + options.l1 * projection.project.weight.norm(p=1)
    if options.nuc:
        return loss + options.nuc * projection.project.weight.norm(p='nuc')
    return loss


# Train the model.
best_dev_loss, bad_epochs = float('inf'), 0
for epoch in range(options.epochs):
    probe.train()
    for batch, (reps, tags) in enumerate(loaders['train']):
        reps, tags = reps.to(device), tags.to(device)
        preds = probe(reps)
        loss = criterion(preds, tags)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({'train loss': loss})
        logging.info('epoch %d batch %d loss %f', epoch, batch, loss.item())

    probe.eval()
    total, count = 0., 0
    for reps, tags in loaders['dev']:
        reps, tags = reps.to(device), tags.to(device)
        preds = probe(reps)
        total += criterion(preds, tags).item() * len(reps)  # Undo mean.
        count += len(reps)
    dev_loss = total / count
    erank = linalg.effective_rank(projection.project.weight)
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
model_path = options.model_dir / options.model_file
torch.save(probe, model_path)
wandb.save(str(model_path))
logging.info('model saved to %s', model_path)


# Evaluate the model.
def test(model: nn.Module) -> float:
    """Evaluate model accuracy on the test set.

    Args:
        model (nn.Module): The module to evaluate.

    Returns:
        float: Fraction of test set correctly classified.

    """
    correct, count = 0., 0
    for reps, tags in loaders['test']:
        reps, tags = reps.to(device), tags.to(device)
        preds = model(reps).argmax(dim=1)
        correct += preds.eq(tags).sum().item()
        count += len(reps)
    return correct / count


accuracy = test(probe)
wandb.run.summary['accuracy'] = accuracy
logging.info('test accuracy %.3f', accuracy)

# Measure whether or not the projection is axis aligned.
if options.ablate:
    logging.info('will ablate axes one by one and retest')
    axes = set(range(projection.project.in_features))
    ablated: Set[int] = set()
    accuracies = []
    while axes:
        best_axis, best_accuracy = 0, 0.
        for axis in axes:
            model = copy.deepcopy(probe)
            assert model.project is not None, 'no projection?'
            model.project.project.weight.data[:, sorted(ablated | {axis})] = 0
            accuracy = test(model)
            if accuracy > best_accuracy:
                best_axis = axis
                best_accuracy = accuracy
        logging.info('ablating axis %d, accuracy %f', best_axis, best_accuracy)
        axes.remove(best_axis)
        ablated.add(best_axis)
        accuracies.append(best_accuracy)
    wandb.run.summary['ablated accuracies'] = torch.tensor(accuracies)
