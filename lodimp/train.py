"""Entry point for all experiments."""

import argparse
import collections
import logging
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
parser_ex = parser.add_mutually_exclusive_group()
parser_ex.add_argument('--batch-size',
                       type=int,
                       default=128,
                       help='Sentences per minibatch.')
parser_ex.add_argument('--no-batch', action='store_true', help='Do not batch.')
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

loaders: Dict[str, Union[torch.utils.data.DataLoader,
                         datasets.ChunkedTaskDataset]] = {}
for split, dataset in data.items():
    if options.no_batch:
        logging.info(f'batching disabled, collating {split} set')
        # You might be wondering: why not use a DataLoader here, like a normal
        # person? It's because DataLoaders unexpectedly copy the data in some
        # cases. This is problematic if, for example, your data is already
        # stored on the GPU and copying it would result in an OOM error.
        loaders[split] = datasets.ChunkedTaskDataset(dataset, device=device)
    else:
        loaders[split] = torch.utils.data.DataLoader(
            dataset, batch_size=options.batch_size, shuffle=True)

features = data['train'].nfeatures
logging.info('samples have %d features', features)

classes = data['train'].nlabels
logging.info('task has %d distinct tag(s)', classes)

# Initialize compositions, if any.
compose: nn.Module = nn.Identity()
if options.compose:
    projections: List[probes.Projection] = []
    dim = features
    for path in options.compose:
        # TODO(evandez): Validate model type.
        model = torch.load(path, map_location=device).project
        if model.in_features != dim:
            raise ValueError(
                f'cannot compose out dim {dim}, in dim {model.in_features}')
        logging.info('composing with projection at %s', path)
        projections.append(model)
        dim = model.out_features
    compose = nn.Sequential(*projections)
    features = dim
compose.to(device)

# Initialize model, optimizer, loss, etc.
probe = probes.ProjectedLinear(features, options.dim, classes).to(device)
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
        return loss + options.l1 * probe.project.weight.norm(p=1)
    if options.nuc:
        return loss + options.nuc * probe.project.weight.norm(p='nuc')
    return loss


# Train the model.
with tb.SummaryWriter(log_dir=options.log_dir, filename_suffix=tag) as writer:
    for epoch in range(options.epochs):
        probe.train()
        for batch, (reps, tags) in enumerate(loaders['train']):
            reps, tags = reps.to(device), tags.to(device)
            with torch.no_grad():
                reps = compose(reps)
            preds = probe(reps)
            loss = criterion(preds, tags)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iteration = epoch * len(loaders['train']) + batch
            writer.add_scalar(f'{tag}/train-loss', loss.item(), iteration)
            logging.info('iteration %d loss %f', iteration, loss.item())

        probe.eval()
        total, count = 0., 0
        for reps, tags in loaders['dev']:
            reps, tags = reps.to(device), tags.to(device)
            with torch.no_grad():
                reps = compose(reps)
            preds = probe(reps)
            total += criterion(preds, tags).item() * len(reps)  # Undo mean.
            count += len(reps)
        dev_loss = total / count
        erank = linalg.effective_rank(probe.project.weight)
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
    torch.save(probe, model_path)
    logging.info('model saved to %s', model_path)

    # Test the model with and without truncated rank.
    erank = linalg.effective_rank(probe.project.weight)
    logging.info('effective rank %.3f', erank)
    results = {'erank': erank}

    truncated = probes.ProjectedLinear(features, options.dim,
                                       classes).to(device)
    truncated.load_state_dict(probe.state_dict())
    weights = linalg.truncate(truncated.project.weight.data, int(erank) + 1)
    truncated.project.weight.data = weights

    for name, model in (('full', probe), ('truncated', truncated)):
        logging.info('testing on %s model', name)
        correct, count = 0., 0
        for reps, tags in loaders['test']:
            reps, tags = reps.to(device), tags.to(device)
            with torch.no_grad():
                reps = compose(reps)
            preds = model(reps).argmax(dim=1)
            correct += preds.eq(tags).sum().item()
            count += len(reps)
        accuracy = correct / count
        results[f'{name}-accuracy'] = accuracy
        writer.add_scalar(f'{tag}/{name}-accuracy', accuracy)
        logging.info('%s accuracy %.3f', name, accuracy)

    # Write metrics.
    writer.add_hparams(dict(hparams), results)
