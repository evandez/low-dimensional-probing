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
import random
import sys
from typing import Callable, Dict, List, Sequence, Type

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

samples: List[ptb.Sample] = []
for split in options.splits:
    samples.extend(ptbs[split])

if options.task in UNIGRAM_TASKS:
    task = UNIGRAM_TASKS[options.task](samples)
    logging.info('will prepare for unigram task "%s"', options.task)
else:
    task = BIGRAM_TASKS[options.task](samples)
    logging.info('will prepare for bigram task "%s"', options.task)

for layer in options.elmo_layers:
    tag = f'{options.task}-elmo-l{layer}'
    directory = (options.out or options.data) / tag
    directory.mkdir(parents=True, exist_ok=True)
    logging.info('writing splits for task %s to %s', tag, directory)

    for split in options.splits:
        file = directory / f'{split}.h5'
        logging.info('will write split %s to %s', split, file)
        with h5py.File(file, 'w') as h5f:
            reps = elmos[split][layer]
            labels = [task(sample) for sample in ptbs[split]]

            # Determine important dimensions.
            nfeatures, nlabels = reps.dimension, len(task)
            if options.task in BIGRAM_TASKS:
                # For bigram tasks, the features correspond to two stacked
                # representations, and the labels correspond to a relationship
                # between them, if any. We assume each representation connects
                # with exactly one other representation (in a directed sense).
                # We sample a linear number of negative samples from each
                # sentence to balance positive with negative examples.
                nfeatures *= 2
                nsamples = sum([
                    len(label) * 2 if len(label) > 1 else 1 for label in labels
                ])
            else:
                nsamples = sum([len(label) for label in labels])
            logging.info(
                'found %d samples, %d features per sample, and %d classes',
                nsamples, nfeatures, nlabels)

            features_out = h5f.create_dataset('features',
                                              shape=(nsamples, nfeatures),
                                              dtype='f')
            labels_out = h5f.create_dataset('labels',
                                            shape=(nsamples,),
                                            dtype='u8')
            if options.task in UNIGRAM_TASKS:
                logging.info('writing features to %s', file)
                start = 0
                for index in range(len(reps)):
                    logging.info('processing %d of %d', index + 1, len(reps))
                    current = reps[index]
                    features_out[start:start + len(current)] = current.numpy()
                    start += len(current)
                assert start == len(features_out)

                logging.info('writing labels to %s', file)
                labels_out[:] = torch.cat(labels).numpy()
            else:
                logging.info('writing features and labels to %s', file)
                start = 0
                for index in range(len(reps)):
                    logging.info('processing %d of %d', index + 1, len(reps))
                    rep, label = reps[index], labels[index]
                    assert label.shape == (len(rep), len(rep))

                    # Take all positive examples, and O(n) negative examples.
                    # We assume negative examples have label 0.
                    idxs = set(range(len(rep)))
                    pairs = [(i, j) for i in idxs for j in idxs if label[i, j]]
                    assert pairs and len(pairs) == len(rep)
                    if len(rep) > 1:
                        negatives = []
                        for i, j in pairs:
                            choices = tuple(idxs - {j})
                            negative = (i, random.choice(choices))
                            negatives.append(negative)
                        pairs += negatives
                        assert len(pairs) == 2 * len(rep)

                    # Write the results to the file.
                    end = start + len(pairs)
                    features = [torch.cat((rep[i], rep[j])) for i, j in pairs]
                    features_out[start:end] = torch.stack(features).numpy()
                    labels_out[start:end] = torch.stack(
                        [label[i, j] for i, j in pairs]).numpy()
                    start = end
                assert start == nsamples
