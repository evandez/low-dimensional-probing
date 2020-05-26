"""Preprocess OntoNotes for SRL.

TODO(evandez): Fold this hot garbage into the updated codebase...
"""
# flake8: noqa
import argparse
import functools as ft
import logging
import pathlib
import sys
from typing import Callable, Dict, List, Sequence

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from lodimp import tasks
from lodimp.common.data import ontonotes, representations

import h5py
import torch

NLAYERS = {
    'elmo': 3,
    'bert-base-uncased': 12,
}

SPLITS = ('train', 'dev', 'test')

parser = argparse.ArgumentParser(description='Preprocess OntoNotes for SRL.')
parser.add_argument('data', type=pathlib.Path, help='Path to OntoNotes.')
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
parser.add_argument('--cache',
                    action='store_true',
                    help='Cache input and output files in memory.')
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

driver = 'H5FD_CORE' if options.cache else None

annotations, reps_by_split = {}, {}
for split in SPLITS:
    conllx = options.data / f'ontonotes5-{split}.conllx'
    logging.info('reading ontonotes %s set from %s', split, conllx)
    annotations[split] = ontonotes.load(conllx)

    h5 = options.data / f'raw.{split}.{options.model}-layers.hdf5'
    logging.info('reading %s %s set from %s', options.model, split, h5)
    reps_by_split[split] = [
        representations.RepresentationDataset(h5, driver=driver).layer(layer)
        for layer in range(NLAYERS[options.model])
    ]

root = (options.out or options.data) / 'srl'
for layer in options.layers or range(NLAYERS[options.model]):
    directory = root / options.model / str(layer)
    directory.mkdir(parents=True, exist_ok=True)
    logging.info('writing splits for layer %d to %s', layer, directory)

    for split in SPLITS:
        file = directory / f'{split}.h5'
        logging.info('will write split %s to %s', split, file)
        with h5py.File(file, 'w', driver=driver) as h5f:
            reps = reps_by_split[split][layer]
            samples = annotations[split]
            assert len(reps) == len(samples)

            # Determine important dimensions.
            nsents = len(reps)
            nsamples = sum(len(sample.sentence) for sample in samples)
            ndims = reps.dataset.dimension

            logging.info('found %d sentences, %d samples for task, %dd reps',
                         nsents, nsamples, ndims)
            breaks_out = h5f.create_dataset('breaks',
                                            shape=(nsents,),
                                            dtype='i')
            reps_out = h5f.create_dataset('reps',
                                          shape=(nsamples, ndims),
                                          dtype='f')

            logging.info('writing reps to %s', file)
            start = 0
            for index in range(nsents):
                logging.info('processing %d of %d', index + 1, nsents)
                breaks_out[index] = start
                current = reps[index]
                reps_out[start:start + len(current)] = current.numpy()
                start += len(current)
            assert start == len(reps_out)

            # Just create tags so it looks like the real deal...
            logging.info('writing fake tags dataset...')
            tags_out = h5f.create_dataset('tags', shape=(nsamples,), dtype='i')
            tags_out[:] = -1
