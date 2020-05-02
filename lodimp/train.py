"""Entry point for all experiments."""
import argparse
import collections
import logging
import pathlib
import sys
from typing import Dict, List, Optional

from lodimp import datasets, metrics, probes, ptb, tasks

import torch
from torch import nn, optim
from torch.nn.utils import rnn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils import tensorboard as tb


# TODO(evandez): Move the functions below to separate modules.
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
hparams['layer'] = options.elmo
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
logging.info('using elmo layer %d which has dim %d', options.elmo, elmo_dim)

task: Optional[tasks.Task] = None
if options.task == 'real':
    task = tasks.PTBRealPOS(ptbs['train'])
    classes = len(task.indexer)
else:
    task = tasks.PTBControlPOS(*ptbs.values())
    classes = len(task.tags)
assert task is not None, 'unitialized task?'
logging.info('will train on %s task', options.task)

loaders: Dict[str, data.DataLoader] = {}
for split in elmos.keys():
    dataset: data.Dataset = datasets.ZippedDatasets(
        elmos[split], datasets.PTBDataset(ptbs[split], task))
    if options.no_batch:
        logging.info(f'batching disabled, collating {split} set')
        dataset = datasets.CollatedDataset(dataset, collate_fn=pack)
        loaders[split] = data.DataLoader(dataset)
    else:
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
            reps, tags = reps.squeeze().to(device), tags.squeeze().to(device)
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
            reps, tags = reps.squeeze().to(device), tags.squeeze().to(device)
            preds = probe(reps)
            total += criterion(preds, tags).item() * len(reps)  # Undo mean.
            count += len(reps)
        dev_loss = total / count
        erank = metrics.effective_rank(probe.project.weight)
        logging.info('epoch %d dev loss %f erank %f', epoch, dev_loss, erank)

        writer.add_scalar(f'{tag}/dev-loss', dev_loss, epoch)
        writer.add_scalar(f'{tag}/erank', erank, epoch)

        scheduler.step(dev_loss)
        if scheduler.num_bad_epochs > options.patience:  # type: ignore
            logging.info('patience exceeded, training complete')
            break

    # Write finished model.
    model_file = f'{tag}.pth'
    model_path = options.model_dir / model_file
    torch.save(probe.state_dict(), model_path)
    logging.info('model saved to %s', model_path)

    # Test the model with and without truncated rank.
    erank = metrics.effective_rank(probe.project.weight)
    logging.info('effective rank %.3f', erank)
    results = {'erank': erank}

    truncated = probes.Projection(elmo_dim, options.dim, classes).to(device)
    truncated.load_state_dict(probe.state_dict())
    u, s, v = torch.svd(truncated.project.weight)
    s[int(erank):] = 0
    truncated.project.weight.data = u.mm(torch.diag(s)).mm(v.t())

    for name, model in (('full', probe), ('truncated', truncated)):
        correct, count = 0., 0
        for reps, tags in loaders['test']:
            reps, tags = reps.squeeze().to(device), tags.squeeze().to(device)
            preds = probe(reps).argmax(dim=1)
            correct += preds.eq(tags).sum().item()
            count += len(reps)
        accuracy = correct / count
        results[f'{name}-accuracy'] = accuracy
        logging.info('%s accuracy %.3f', name, accuracy)

    # Write metrics.
    writer.add_hparams(dict(hparams), results)
