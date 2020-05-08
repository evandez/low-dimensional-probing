"""Preprocess PTB data for some task.

This script computes the mapping from representation (or pairs of
representations) to labels, and then saves the mapping to an h5 file
to be used by the training script.

While the mappings themselves are not expensive to construct, doing
so during training can be costly. It often results in many, many random
accesses to the data files. In fact, most the computation time is spent
on reading and collating the data.

We cut out that complexity entirely by preparing the task ahead of time
and keeping the training script (mostly) agnostic to the task at hand.
"""

import argparse
import functools as ft
import logging
import pathlib
import sys
from typing import Callable, Dict, Sequence, Type

from lodimp import datasets, ptb, tasks

import h5py
import torch

TaskFactory = Callable[[Sequence[ptb.Sample]], tasks.Task]
UNIGRAM_TASKS: Dict[str, TaskFactory] = {
    'pos': tasks.POSTask,
    'pos-verb': ft.partial(tasks.POSTask, tags=tasks.POS_VERBS),
    'pos-noun': ft.partial(tasks.POSTask, tags=tasks.POS_NOUNS),
    'pos-adj': ft.partial(tasks.POSTask, tags=tasks.POS_ADJECTIVES),
    'pos-adv': ft.partial(tasks.POSTask, tags=tasks.POS_ADVERBS),
    'pos-control': tasks.ControlPOSTask,
}
BIGRAM_TASKS: Dict[str, Type[tasks.Task]] = {
    'dep-arc': tasks.DependencyArcTask,
    'dep-label': tasks.DependencyLabelTask,
}

parser = argparse.ArgumentParser(description='Preprocess PTB for some task.')
parser.add_argument('data', type=pathlib.Path, help='Path to PTB directory.')
parser.add_argument('task',
                    choices=(*UNIGRAM_TASKS.keys(), *BIGRAM_TASKS.keys()),
                    help='Task to preproces.')
parser.add_argument('--out', type=pathlib.Path, help='Path to output h5 file.')
parser.add_argument('--splits',
                    nargs='+',
                    default=('train', 'dev', 'test'),
                    help='Dataset splits to preprocess.')
parser.add_argument('--ptb-prefix',
                    default='ptb3-wsj-',
                    help='Prefix of PTB files.')
parser.add_argument('--ptb-suffix',
                    default='.conllx',
                    help='Suffix of PTB files.')
parser.add_argument('--elmo-prefix',
                    default='raw.',
                    help='Prefix of ELMo files.')
parser.add_argument('--elmo-suffix',
                    default='.elmo-layers.hdf5',
                    help='Suffix of ELMo files.')
parser.add_argument(
    '--elmo-layers',
    type=int,
    nargs='+',
    choices=(0, 1, 2),
    default=(0, 1, 2),
    help='ELMo layers to use. Separate files generated for each.')
parser.add_argument('--verbose',
                    dest='log_level',
                    action='store_const',
                    const=logging.INFO,
                    default=logging.WARNING,
                    help='Print lots of logs to stdout.')
options = parser.parse_args()

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

ptbs, elmos = {}, {}
for split in options.splits:
    conllx = options.data / f'{options.ptb_prefix}{split}{options.ptb_suffix}'
    logging.info('reading ptb %s set from %s', split, conllx)
    ptbs[split] = ptb.load(conllx)

    elmo = options.data / f'{options.elmo_prefix}{split}{options.elmo_suffix}'
    logging.info('reading elmo %s set from %s', split, elmo)
    elmos[split] = [
        datasets.ELMoRepresentationsDataset(elmo, layer)
        for layer in options.elmo_layers
    ]

samples = ptbs['train']  # TODO(evandez): Make this an option.
if options.task in UNIGRAM_TASKS:
    task = UNIGRAM_TASKS[options.task](samples)
    logging.info('will prepare for unigram task "%s"', options.task)
else:
    task = BIGRAM_TASKS[options.task](samples)
    logging.info('will prepare for bigram task "%s"', options.task)

for split in options.splits:
    for layer in options.elmo_layers:
        directory = options.out or options.data
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f'{options.task}-{split}-l{layer}.h5'
        logging.info('will write %s layer %d task to %s', split, layer, file)

        with h5py.File(file, 'w') as h5f:
            reps = elmos[split][layer]
            labels = [task(sample) for sample in samples]

            # Determine important dimensions.
            nfeatures, nlabels = reps.dimension, len(task)
            if options.task in BIGRAM_TASKS:
                nfeatures *= 2
                nsamples = sum([len(label)**2 for label in labels])
            else:
                nsamples = sum([len(label) for label in labels])
            logging.info(
                'found %d samples, %d features per sample, and %d classes',
                nsamples, nfeatures, nlabels)

            logging.info('writing features to %s', file)
            dataset = h5f.create_dataset('features',
                                         shape=(nsamples, nfeatures),
                                         dtype='f')
            if options.task in UNIGRAM_TASKS:
                start = 0
                for index in range(len(reps)):
                    current = reps[index]
                    dataset[start:start + len(current)] = current.numpy()
                    start += len(current)
                assert start == len(dataset)
            else:
                start = 0
                for index in range(len(reps)):
                    current = reps[index]
                    pairs = torch.stack([
                        torch.cat((reps[i], reps[j]))
                        for i in range(len(reps))
                        for j in range(len(reps))
                    ])
                    dataset[start:start + len(pairs)] = pairs
                    start += len(pairs)
                assert start == len(dataset)

            logging.info('writing labels to %s', file)
            dataset = h5f.create_dataset(
                'labels',
                data=torch.cat([label.flatten() for label in labels]).numpy())
            assert len(dataset) == nsamples
