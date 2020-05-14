"""Entry point for all experiments."""

import argparse
import collections
import logging
import math
import pathlib
import sys
from typing import Dict, List, Union

from lodimp import datasets, linalg, probes

import torch
import torch.utils.data
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import tensorboard as tb

parser = argparse.ArgumentParser(description='Train a POS tagger.')
parser.add_argument('task', type=pathlib.Path, help='Task directory.')
parser.add_argument('layer', type=int, help='ELMo layer.')
parser.add_argument('dim', type=int, help='Projection dimensionality.')
parser.add_argument('--no-batch', action='store_true', help='Do not batch.')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='Passes to make through dataset during training.')
parser.add_argument('--l1', type=float, help='Add L1 norm penalty.')
parser.add_argument('--nuc', type=float, help='Add nuclear norm penalty')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('--lr-reduce',
                    type=float,
                    default=0.5,
                    help='Shrink LR at this rate when --lr-patience exceeded.')
parser.add_argument('--lr-patience',
                    type=int,
                    default=1,
                    help='Shrink LR after this many epochs dev loss decrease.')
parser.add_argument('--patience',
                    type=int,
                    default=4,
                    help='Epochs for dev loss to decrease to stop training.')
parser.add_argument('--compose',
                    nargs='+',
                    type=pathlib.Path,
                    help='Compose these projections with learned projection.')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device.')
parser.add_argument('--log-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/logs',
                    help='Path to write TensorBoard logs.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Directory to write finished model.')
parser.add_argument('--model-file', help='Save file name. Defaults to TB tag.')
parser.add_argument('--verbose',
                    dest='log_level',
                    action='store_const',
                    const=logging.INFO,
                    default=logging.WARNING,
                    help='Print lots of logs to stdout.')
options = parser.parse_args()

# Quick validations.
if not options.task.exists():
    raise FileNotFoundError(f'data directory does not exist: {options.data}')
root = options.task / f'elmo-{options.layer}'
if not root.exists():
    raise FileNotFoundError(f'layer {options.layer} partition not found')
for path in options.compose or []:
    if not path.exists():
        raise FileNotFoundError(f'model does not exist: {options.compose}')

# Set up.
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

options.log_dir.mkdir(parents=True, exist_ok=True)
logging.info('tensorboard will write to %s', options.log_dir)

options.model_dir.mkdir(parents=True, exist_ok=True)
logging.info('model(s) will be written to %s', options.model_dir)

device = torch.device('cuda') if options.cuda else torch.device('cpu')
logging.info('using %s', device.type)

# Identify this run.
hparams = collections.OrderedDict()
hparams['proj'] = options.dim
hparams['task'] = options.task.name
hparams['layer'] = options.layer
if options.l1:
    hparams['l1'] = options.l1
if options.nuc:
    hparams['nuc'] = options.nuc
tag = '-'.join(f'{key}_{value}' for key, value in hparams.items())
logging.info('job tag is %s', tag)

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

# Initialize compositions, if any.
compose: nn.Module = nn.Identity()
if options.compose:
    projections: List[nn.Linear] = []
    dim = ndims
    for path in options.compose:
        logging.info('composing with projection at %s', path)
        model = torch.load(path, map_location=device)
        if not isinstance(model, nn.Linear):
            raise ValueError(f'bad projection type: {type(model)}')
        if model.in_features != dim:
            raise ValueError(f'cannot compose {dim}d and {model.in_features}d')
        projections.append(model)
        dim = model.out_features
    compose = nn.Sequential(*projections)
    ndims = dim
compose = compose.to(device)

# Initialize model, optimizer, loss, etc.
projection = nn.Linear(ndims, options.dim)
probe: nn.Module
if nlabels is not None:
    probe = nn.Linear(ngrams * options.dim, nlabels)
else:
    probe = probes.Bilinear(ngrams * options.dim)
probe = probe.to(device)

optimizer = optim.Adam(probe.parameters(), lr=options.lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                           factor=options.lr_reduce,
                                           patience=options.lr_patience,
                                           threshold=1e-6)
ce = nn.CrossEntropyLoss().to(device)


def criterion(*args: torch.Tensor) -> torch.Tensor:
    """Returns CE loss with regularizers."""
    loss = ce(*args)
    if options.l1:
        return loss + options.l1 * projection.weight.norm(p=1)
    if options.nuc:
        return loss + options.nuc * projection.weight.norm(p='nuc')
    return loss


# Train the model.
with tb.SummaryWriter(log_dir=options.log_dir, filename_suffix=tag) as writer:
    for epoch in range(options.epochs):
        projection.train()
        probe.train()
        for batch, (reps, tags) in enumerate(loaders['train']):
            reps, tags = reps.to(device), tags.to(device)
            with torch.no_grad():
                reps = compose(reps)
            projected = projection(reps).view(-1, ngrams * options.dim)
            preds = probe(projected)
            loss = criterion(preds, tags)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iteration = epoch * len(loaders['train']) + batch
            writer.add_scalar(f'{tag}/train-loss', loss.item(), iteration)
            logging.info('iteration %d loss %f', iteration, loss.item())

        projection.eval()
        probe.eval()
        total, count = 0., 0
        for reps, tags in loaders['dev']:
            reps, tags = reps.to(device), tags.to(device)
            with torch.no_grad():
                reps = compose(reps)
            projected = projection(reps).view(-1, ngrams * options.dim)
            preds = probe(projected)
            total += criterion(preds, tags).item() * len(reps)  # Undo mean.
            count += len(reps)
        dev_loss = total / count
        erank = linalg.effective_rank(projection.weight)
        logging.info('epoch %d dev loss %f erank %f', epoch, dev_loss, erank)

        writer.add_scalar(f'{tag}/dev-loss', dev_loss, epoch)
        writer.add_scalar(f'{tag}/erank', erank, epoch)

        scheduler.step(dev_loss)
        if scheduler.num_bad_epochs > options.patience:  # type: ignore
            logging.info('patience exceeded, training complete')
            break

    # Write finished model.
    model_file = options.model_file or f'{tag}.pth'
    model_path = options.model_dir / model_file
    torch.save(projection, model_path)
    logging.info('model saved to %s', model_path)

    # Test the model with and without truncated rank.
    erank = linalg.effective_rank(projection.weight)
    logging.info('effective rank %.3f', erank)
    results = {'erank': erank}

    truncated = nn.Linear(ndims, options.dim).to(device)
    truncated.load_state_dict(projection.state_dict())
    weights = linalg.truncate(truncated.weight.data, math.ceil(erank))
    truncated.weight.data = weights

    for name, proj in (('full', projection), ('truncated', truncated)):
        logging.info('testing on %s model', name)
        correct, count = 0., 0
        for reps, tags in loaders['test']:
            reps, tags = reps.to(device), tags.to(device)
            with torch.no_grad():
                reps = compose(reps)
            projected = proj(reps).view(-1, ngrams * options.dim)
            preds = probe(projected).argmax(dim=1)
            correct += preds.eq(tags).sum().item()
            count += len(reps)
        accuracy = correct / count
        results[f'{name}-accuracy'] = accuracy
        writer.add_scalar(f'{tag}/{name}-accuracy', accuracy)
        logging.info('%s accuracy %.3f', name, accuracy)

    # Write metrics.
    writer.add_hparams(dict(hparams), results)
