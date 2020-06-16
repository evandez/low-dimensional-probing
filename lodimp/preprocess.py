"""Preprocess PTB data for some task.

This script computes the mapping from representation (or pairs of
representations) to tags, and then saves the mapping to an h5 file
to be used by the training script.

While the mappings themselves are not expensive to construct, doing
so during training can be costly. It often results in many, many random
accesses to the data files. In fact, most the computation time is spent
on reading and collating the data.

We cut out that complexity entirely by preparing the task ahead of time
and keeping the training script (mostly) agnostic to the task at hand.
"""
# flake8: noqa
import argparse
import functools as ft
import logging
import pathlib
import sys
from typing import Callable, Dict, List, Sequence

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from lodimp import maps
from lodimp.common.data import ptb, representations

import h5py
import torch

TaskFactory = Callable[[Sequence[ptb.Sample]], maps.Task]

# Unigram tasks map single words to tags.
UNIGRAM_TASKS: Dict[str, TaskFactory] = {
    'pos': maps.POSTask,
    'pos-verb': ft.partial(maps.POSTask, tags=maps.POS_VERBS),
    'pos-verb-pres': ft.partial(maps.POSTask, tags=maps.POS_VERBS_PRESENT),
    'pos-verb-past': ft.partial(maps.POSTask, tags=maps.POS_VERBS_PAST),
    'pos-noun': ft.partial(maps.POSTask, tags=maps.POS_NOUNS),
    'pos-noun-proper': ft.partial(maps.POSTask, tags=maps.POS_NOUNS_PROPER),
    'pos-noun-plural': ft.partial(maps.POSTask, tags=maps.POS_NOUNS_PLURAL),
    'pos-adj': ft.partial(maps.POSTask, tags=maps.POS_ADJECTIVES),
    'pos-adv': ft.partial(maps.POSTask, tags=maps.POS_ADVERBS),
    'pos-control': maps.ControlPOSTask,
    'dep-arc': maps.DependencyArcTask,
    'dep-arc-control': maps.ControlDependencyArcTask,
}

# Bigram tasks map pairs of words to tags.
BIGRAM_TASKS: Dict[str, TaskFactory] = {
    'dep-label': maps.DependencyLabelTask,
    'dep-label-control': maps.ControlDependencyLabelTask,
}

NLAYERS = {
    'elmo': 3,
    'bert-base-uncased': 13,
}

SPLITS = ('train', 'dev', 'test')

parser = argparse.ArgumentParser(description='Preprocess PTB for some task.')
parser.add_argument('data', type=pathlib.Path, help='Path to PTB directory.')
parser.add_argument('task',
                    choices=(*UNIGRAM_TASKS.keys(), *BIGRAM_TASKS.keys()),
                    help='Task to preproces.')
parser.add_argument(
    '--model',
    choices=NLAYERS.keys(),
    default='elmo',
    help='Representation model to use.',
)
parser.add_argument('--layers',
                    nargs='+',
                    type=int,
                    help='Layers to preprocess. Defaults to all.')
parser.add_argument('--out', type=pathlib.Path, help='Path to output h5 file.')
parser.add_argument('--quiet',
                    dest='log_level',
                    action='store_const',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='Only print warning and errors to stdout.')
options = parser.parse_args()

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

ptbs, reps_by_split = {}, {}
for split in SPLITS:
    conllx = options.data / f'ptb3-wsj-{split}.conllx'
    logging.info('reading ptb %s set from %s', split, conllx)
    ptbs[split] = ptb.load(conllx)

    h5 = options.data / f'raw.{split}.{options.model}-layers.hdf5'
    logging.info('reading %s %s set from %s', options.model, split, h5)
    reps_by_split[split] = [
        representations.RepresentationDataset(h5).layer(layer)
        for layer in range(NLAYERS[options.model])
    ]

samples: List[ptb.Sample] = []
for split in SPLITS:
    samples.extend(ptbs[split])

if options.task in UNIGRAM_TASKS:
    task = UNIGRAM_TASKS[options.task](samples)
    logging.info('will prepare for unigram task "%s"', options.task)
else:
    task = BIGRAM_TASKS[options.task](samples)
    logging.info('will prepare for bigram task "%s"', options.task)

root = (options.out or options.data) / options.task
for layer in options.layers or range(NLAYERS[options.model]):
    directory = root / options.model / str(layer)
    directory.mkdir(parents=True, exist_ok=True)
    logging.info('writing splits for layer %d to %s', layer, directory)

    for split in SPLITS:
        file = directory / f'{split}.h5'
        logging.info('will write split %s to %s', split, file)
        with h5py.File(file, 'w') as h5f:
            reps = reps_by_split[split][layer]
            tags = [task(sample) for sample in ptbs[split]]

            # Determine important dimensions.
            nsents = len(reps)
            ndims = reps.dataset.dimension
            nsamples = sum(len(tag) for tag in tags)

            logging.info('found %d sentences, %d samples for task, %dd reps',
                         nsents, nsamples, ndims)
            breaks_out = h5f.create_dataset('breaks',
                                            shape=(nsents,),
                                            dtype='i')
            reps_out = h5f.create_dataset(
                'reps',
                shape=(nsamples, 2, ndims) if options.task in BIGRAM_TASKS else
                (nsamples, ndims),
                dtype='f')
            tags_out = h5f.create_dataset('tags', shape=(nsamples,), dtype='i')
            if isinstance(task, maps.SizedTask):
                # Write out how many valid tags there are, if that quantity
                # is defined for the task.
                tags_out.attrs['ntags'] = len(task)

            if options.task in BIGRAM_TASKS:
                logging.info('writing reps and tags to %s', file)
                start = 0
                for index in range(nsents):
                    logging.info('processing %d of %d', index + 1, nsents)
                    rep, tag = reps[index], tags[index]
                    assert tag.shape == (len(rep), len(rep))

                    # Take only positive examples.
                    idxs = set(range(len(rep)))
                    pairs = [(i, j) for i in idxs for j in idxs if tag[i, j]]
                    assert pairs and len(pairs) == len(rep)

                    # Write the results to the file.
                    breaks_out[index] = start
                    end = start + len(pairs)
                    bigrams = [torch.stack((rep[i], rep[j])) for i, j in pairs]
                    reps_out[start:end] = torch.stack(bigrams).numpy()
                    tags_out[start:end] = torch.stack(
                        [tag[i, j] for i, j in pairs]).numpy()
                    start = end
                assert start == nsamples
            else:
                logging.info('writing reps to %s', file)
                start = 0
                for index in range(nsents):
                    logging.info('processing %d of %d', index + 1, nsents)
                    breaks_out[index] = start
                    current = reps[index]
                    reps_out[start:start + len(current)] = current.numpy()
                    start += len(current)
                assert start == len(reps_out)

                logging.info('writing tags to %s', file)
                tags_out[:] = torch.cat(tags).numpy()
