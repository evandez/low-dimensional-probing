"""Entry point for all experiments."""
import argparse
import pathlib
from typing import List, Optional

from lodimp import datasets, probes, ptb, tasks

import torch
from torch import nn, optim
from torch.nn.utils import rnn
from torch.utils import data, tensorboard


# TODO(evandez): Move this to its own module.
def collate_seq(samples: List[List[torch.Tensor]]) -> List[torch.Tensor]:
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
                    default=10,
                    help='Passes to make through dataset during training.')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device.')
parser.add_argument('--log-dir',
                    default='/tmp/lodimp/train',
                    help='Path to write TensorBoard logs.')

subparsers = parser.add_subparsers(dest='probe')
mlp_parser = subparsers.add_parser('mlp', help='Use an MLP probe.')
mlp_parser.add_argument('--layers', type=int, default=2, help='MLP layers.')
mlp_parser.add_argument('--hidden', type=int, default=1024, help='Hidden dim.')
proj_parser = subparsers.add_parser('proj', help='Use a projection probe.')
proj_parser.add_argument('dim', type=int, help='Projection dimensionality.')
options = parser.parse_args()

ptbs, elmos = {}, {}
for split in ('train', 'dev', 'test'):
    ptbs[split] = ptb.load(options.data / f'ptb3-wsj-{split}.conllx')
    elmos[split] = datasets.ELMoRepresentationsDataset(
        options.data / f'raw.{split}.elmo-layers.hdf5', options.elmo)
    assert len(ptbs[split]) == len(elmos[split]), 'mismatched datasets?'

task: Optional[tasks.Task] = None
if options.task == 'real':
    task = tasks.PTBRealPOS(ptbs['train'])
    classes = len(task.indexer)
else:
    task = tasks.PTBControlPOS(*ptbs.values())
    classes = len(task.tags)
assert task is not None, 'unitialized task?'

loaders = {}
for split in elmos.keys():
    dataset = datasets.ZippedDatasets(elmos[split],
                                      datasets.PTBDataset(ptbs[split], task))
    loaders[split] = data.DataLoader(dataset,
                                     batch_size=options.batch_size,
                                     collate_fn=collate_seq,
                                     shuffle=True)

probe: Optional[nn.Module] = None
if options.probe == 'mlp':
    probe = probes.MLP(elmos['train'].dimension,
                       classes,
                       hidden_dimension=options.hidden,
                       hidden_layers=options.layers)
    label = f'MLP-{options.layers}-{options.task}'
else:
    probe = probes.Projection(elmos['train'].dimension, options.dim, classes)
    label = f'Proj-{options.task}-{options.dim}'
assert probe is not None, 'unitialized probe?'

device = torch.device('cuda') if options.cuda else torch.device('cpu')
probe.to(device)
criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
optimizer = optim.Adam(probe.parameters(), lr=options.lr)
writer = tensorboard.SummaryWriter(log_dir=options.log_dir)

for epoch in range(options.epochs):
    probe.train()
    for batch, (reps, tags) in enumerate(loaders['train']):
        reps, tags = reps.to(device), tags.to(device)
        predictions = probe(reps)
        loss = criterion(predictions, tags) / len(reps)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar(f'{label}/Train-Loss', loss.item(),
                          epoch * len(loaders['train']) + batch)

    probe.eval()
    dev_loss, dev_size = 0., 0
    for reps, tags in loaders['dev']:
        reps, tags = reps.to(device), tags.to(device)
        predictions = probe(reps)
        dev_loss += criterion(predictions, tags).item()
        dev_size += len(reps)
    writer.add_scalar(f'{label}/Dev-Loss', dev_loss / dev_size, epoch)

correct, total = 0., 0
for reps, tags in loaders['test']:
    reps, tags = reps.to(device), tags.to(device)
    predictions = probe(reps).argmax(dim=1)
    correct += predictions.eq(tags).sum().item()
    total += len(reps)

hparams = {'probe': label, 'task': options.task}
metrics = {'hparam/accuracy': correct / total}
writer.add_hparams(hparams, metrics)
writer.close()
