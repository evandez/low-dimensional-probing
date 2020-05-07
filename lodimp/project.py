"""Linearly project an entire dataset of representation."""

import argparse
import logging
import pathlib
import shutil
import sys

import h5py
import torch

parser = argparse.ArgumentParser(description='Project all representations.')
parser.add_argument('model', type=pathlib.Path, help='Projection model path.')
parser.add_argument('data', type=pathlib.Path, help='Input data directory.')
parser.add_argument('out', type=pathlib.Path, help='Output data directory.')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device.')
parser.add_argument('--verbose',
                    dest='log_level',
                    action='store_const',
                    const=logging.INFO,
                    default=logging.WARNING,
                    help='Print lots of logs to stdout.')
options = parser.parse_args()

if not options.model.exists():
    raise FileNotFoundError(f'model not found: {options.model}')
if not options.data.exists():
    raise FileNotFoundError(f'data directory not found: {options.data}')

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

device = torch.device('cuda') if options.cuda else torch.device('cpu')
logging.info('using %s', device.type)

logging.info('reading model from %s', options.model)
projection = torch.load(options.model, map_location=device).project

if options.out.exists():
    logging.info('clearing output directory %s', options.out)
    shutil.rmtree(options.out)

logging.info('copying dataset from %s to %s', options.data, options.out)
shutil.copytree(options.data, options.out)
logging.info('will write all data to %s', options.out)

for split in ('train', 'dev', 'test'):
    path = options.out / f'raw.{split}.elmo-layers.hdf5'
    logging.info('reading %s reps from %s', split, path)
    with h5py.File(path, 'a') as file:
        for key in file.keys() - {'sentence_to_index'}:
            original = torch.tensor(file[key][:])
            with torch.no_grad():
                projected = projection(original)
            del file[key]
            file.create_dataset(key, data=projected)
            logging.info('projected %s sample %s', split, key)

logging.info('all projections complete')
