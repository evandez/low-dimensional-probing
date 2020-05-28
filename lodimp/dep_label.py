"""Train and evaluate a probe on the dependency label task."""

# flake8: noqa
import argparse
import copy
import logging
import pathlib
import sys
from typing import Dict, Set, Union
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from lodimp import datasets
from lodimp.common.models import probes, projections

import torch
import torch.utils.data
import wandb
from torch import nn, optim

parser = argparse.ArgumentParser(description='Train on dependency label task.')
parser.add_argument('data', type=pathlib.Path, help='Data directory.')
parser.add_argument(
    '--model',
    choices=('bert-base-uncased', 'bert-large-uncased', 'elmo'),
    default='elmo',
    help='Representation model to use.',
)
parser.add_argument('--layer', type=int, default=0, help='Model layer.')
parser.add_argument('--dimension',
                    type=int,
                    default=64,
                    help='Projection dimensionality.')
parser.add_argument('--probe',
                    choices=('linear', 'mlp'),
                    default='linear',
                    help='Probe architecture.')
parser.add_argument('--share-projection',
                    action='store_true',
                    help='When comparing reps, project both with same matrix.')
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
parser.add_argument('--ablate',
                    action='store_true',
                    help='Also test ablated model.')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device.')
parser.add_argument('--wandb-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp',
                    help='Path to write Weights and Biases data.')
parser.add_argument('--wandb-group', help='Experiment group.')
parser.add_argument('--wandb-name', help='Experiment name.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/model',
                    help='Directory to write finished model.')
parser.add_argument('--model-file',
                    default='probe.pth',
                    help='Model file name.')
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
               'task': 'dep-label',
               'representations': {
                   'model': options.model,
                   'layer': options.layer,
               },
               'projection': {
                   'dimension': options.dimension,
                   'shared': options.share_projection,
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
    raise FileNotFoundError(f'data directory does not exist: {options.data}')

root = options.data / options.model / str(options.layer)
if not root.exists():
    raise FileNotFoundError(f'layer {options.layer} partition not found')

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

ntags = data['train'].ntags
assert ntags is not None, 'no label count?'
logging.info('task has %d labels', ntags)

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

# Initialize the projection(s).
if options.share_projection:
    projection = projections.Projection(ndims, options.dimension)
else:
    projection = projections.Projection(2 * ndims, 2 * options.dimension)

probe: Union[probes.Linear, probes.MLP]
if options.probe == 'linear':
    probe = probes.Linear(2 * options.dimension, ntags, project=projection)
else:
    assert options.probe == 'mlp'
    probe = probes.MLP(2 * options.dimension, ntags, project=projection)
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
    logging.info('epoch %d dev loss %f', epoch, dev_loss)
    wandb.log({'dev loss': dev_loss})

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


# Evaluate the models.
def test(model: nn.Module, loader: str = 'test') -> float:
    """Evaluate model accuracy on the test set.

    Args:
        model (nn.Module): The module to evaluate.
        loader (str): Key for the dataset to evaluate on.

    Returns:
        float: Fraction of test set correctly classified.

    """
    correct, count = 0., 0
    for reps, tags in loaders[loader]:
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
    # logging.info('will ablate axes one by one and retest')
    # axes = set(range(projection.project.in_features))
    # ablated: Set[int] = set()
    # accuracies = []
    # while axes:
    #     best_model, best_axis, best_accuracy = probe, -1, -1.
    #     for axis in axes:
    #         model = copy.deepcopy(best_model)
    #         assert model.project is not None, 'no projection?'
    #         model.project.project.weight.data[:, sorted(ablated | {axis})] = 0
    #         accuracy = test(model, loader='dev')
    #         if accuracy > best_accuracy:
    #             best_model = model
    #             best_axis = axis
    #             best_accuracy = accuracy
    #     accuracy = test(best_model)
    #     logging.info('ablating axis %d, accuracy %f', best_axis, accuracy)
    #     axes.remove(best_axis)
    #     ablated.add(best_axis)
    #     accuracies.append(accuracy)
    # wandb.run.summary['ablated accuracies'] = torch.tensor(accuracies)

    def ablate(axes: Set[int]) -> nn.Module:
        """Abalate the given axes from the probe.

        Args:
            axes (Set[int]): The axes to ablate.

        Returns:
            nn.Module: The probe with the axes ablated from the projection.

        """
        model = copy.deepcopy(probe)
        assert model.project is not None, 'no projection?'
        model.project.project.weight.data[:, sorted(axes)] = 0
        return model.eval()

    axes = set(range(projection.project.in_features))
    dev_accuracies = []
    for axis in axes:
        model = ablate({axis})
        dev_accuracy = test(model, loader='dev')
        logging.info('axis %d/dev accuracy %f', axis, dev_accuracy)
        dev_accuracies.append(dev_accuracy)
    wandb.run.summary['ablated dev accuracy'] = torch.tensor(dev_accuracies)

    ablated = set()
    test_accuracies = []
    to_ablate = sorted(enumerate(dev_accuracies),
                       key=lambda x: x[1],
                       reverse=True)
    for axis, _ in to_ablate:
        ablated.add(axis)
        model = ablate(ablated)
        test_accuracy = test(model)
        logging.info('%d axes/test accuracy %f', len(ablated), test_accuracy)
        test_accuracies.append(test_accuracy)
    wandb.run.summary['ablated test accuracy'] = torch.tensor(test_accuracies)
