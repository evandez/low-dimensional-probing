"""Entry point for all experiments."""
import argparse
import collections
import logging
import pathlib
import sys
from typing import List, Optional

from lodimp import datasets, probes, ptb, tasks

import torch
from torch import distributions, nn, optim
from torch.nn.utils import rnn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils import tensorboard as tb


# TODO(evandez): Move this to its own module.
def pack(samples: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Collate samples of sequences into PackedSequences.

    This is meant to be used as the collate_fn for a torch.utils.DataLoader
    when the elements of the dataset are sequences of arbitrary length.
    Note that this process is destructive! The sequences may not be
    separated afterward because we toss the PackedSequence wrapper.

    Args:
        samples (List[List[torch.Tensor]]): The samples to collate.

    Raises:
        ValueError: If the samples do not consist of equal numbers of elements.

    Returns:
        List[torch.Tensor]: The packed sequences.

    """
    if not samples:
        return []

    length = len(samples[-1])
    for sample in samples:
        if len(sample) != length:
            raise ValueError(f'bad sample lengths: {len(sample)} vs. {length}')
        seq_len = len(sample[-1])
        for item in sample:
            if len(item) != seq_len:
                raise ValueError(f'bad seq lengths: {len(item)} vs. {seq_len}')

    separated: List[List[torch.Tensor]] = [[] for _ in range(length)]
    for sample in samples:
        for index, item in enumerate(sample):
            separated[index].append(item)

    collated = []
    for items in separated:
        collated.append(rnn.pack_sequence(items, enforce_sorted=False).data)

    return collated


parser = argparse.ArgumentParser(description='Train a POS tagger.')
parser.add_argument('data', type=pathlib.Path, help='Data directory.')
parser.add_argument('task', choices=['real', 'control'], help='Task variant.')
parser.add_argument('dim', type=int, help='Projection dimensionality.')
parser.add_argument('--elmo',
                    choices=(0, 1, 2),
                    type=int,
                    default=2,
                    help='ELMo layer to use.')
parser.add_argument('--batch-size',
                    type=int,
                    default=64,
                    help='Sentences per minibatch.')
parser.add_argument('--epochs',
                    type=int,
                    default=25,
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
parser.add_argument('--cuda', action='store_true', help='Use CUDA device.')
parser.add_argument('--log-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/logs',
                    help='Path to write TensorBoard logs.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Path to write finished model(s).')
parser.add_argument('--verbose',
                    dest='log_level',
                    action='store_const',
                    const=logging.INFO,
                    default=logging.WARNING,
                    help='Print lots of logs to stdout.')
options = parser.parse_args()

# Set up.
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

options.log_dir.mkdir(parents=True, exist_ok=True)
logging.info('tensorboard will write to %s', options.log_dir)

options.model_dir.mkdir(parents=True, exist_ok=True)
logging.info('model(s) will be written to %s', options.model_dir)

# Identify this run.
hparams = collections.OrderedDict()
hparams['proj'] = options.dim
hparams['task'] = options.task
if options.l1:
    hparams['l1'] = options.l1
if options.nuc:
    hparams['nuc'] = options.nuc
tag = '-'.join(f'{key}_{value}' for key, value in hparams.items())
logging.info('job tag is %s', tag)

# Load data.
ptbs, elmos = {}, {}
for split in ('train', 'dev', 'test'):
    path = options.data / f'ptb3-wsj-{split}.conllx'
    ptbs[split] = ptb.load(path)
    logging.info('loaded ptb %s set from %s', split, path)

    path = options.data / f'raw.{split}.elmo-layers.hdf5'
    elmos[split] = datasets.ELMoRepresentationsDataset(path, options.elmo)
    logging.info('loaded elmo %s reps from %s', split, path)
    assert len(ptbs[split]) == len(elmos[split]), 'mismatched datasets?'
elmo_dim = elmos['train'].dimension
logging.info('elmo layer %d has dim %d', options.elmo, elmo_dim)

task: Optional[tasks.Task] = None
if options.task == 'real':
    task = tasks.PTBRealPOS(ptbs['train'])
    classes = len(task.indexer)
else:
    task = tasks.PTBControlPOS(*ptbs.values())
    classes = len(task.tags)
assert task is not None, 'unitialized task?'
logging.info('will train on %s task', options.task)

loaders = {}
for split in elmos.keys():
    dataset = datasets.ZippedDatasets(elmos[split],
                                      datasets.PTBDataset(ptbs[split], task))
    loaders[split] = data.DataLoader(dataset,
                                     batch_size=options.batch_size,
                                     collate_fn=pack,
                                     shuffle=True)

# Initialize model, optimizer, loss, etc.
device = torch.device('cuda') if options.cuda else torch.device('cpu')
probe = probes.Projection(elmo_dim, options.dim, classes).to(device)
optimizer = optim.Adam(probe.parameters(), lr=options.lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                           factor=options.lr_reduce,
                                           patience=options.lr_patience)
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
            preds = probe(reps)
            loss = criterion(preds, tags)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iteration = epoch * len(loaders['train']) + batch
            writer.add_scalar(f'{tag}/train-loss', loss.item(), iteration)
            logging.info('iteration %d loss %.3f', iteration, loss.item())

        probe.eval()
        total, count = 0., 0
        for reps, tags in loaders['dev']:
            reps, tags = reps.to(device), tags.to(device)
            preds = probe(reps)
            total += criterion(preds, tags).item() * len(reps)  # Undo mean.
            count += len(reps)
        dev_loss = total / count
        writer.add_scalar(f'{tag}/dev-loss', dev_loss, epoch)
        logging.info('epoch %d dev loss %.3f', epoch, dev_loss)
        scheduler.step(dev_loss)
        if scheduler.num_bad_epochs > options.patience:  # type: ignore
            logging.info('patience exceeded, training complete')
            break

    # Write finished model.
    model_file = tag + '.pth'
    model_path = options.model_dir / model_file
    torch.save(probe.state_dict(), model_path)
    logging.info('model saved to %s', model_path)

    # Test the model.
    correct, count = 0., 0
    for reps, tags in loaders['test']:
        reps, tags = reps.to(device), tags.to(device)
        preds = probe(reps).argmax(dim=1)
        correct += preds.eq(tags).sum().item()
        count += len(reps)
    accuracy = correct / count

    # Also compute effective rank.
    _, s, _ = torch.svd(probe.project.weight, compute_uv=False)
    rank = distributions.Categorical(logits=s).entropy()

    # Write metrics.
    writer.add_hparams(hparams, {'accuracy': accuracy, 'erank': rank})
    logging.info('test accuracy %.3f', accuracy)
