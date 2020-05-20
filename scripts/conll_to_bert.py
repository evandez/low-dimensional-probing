"""Embed all sentences in the .conllx file with BERT."""

import argparse
import logging
import pathlib
import sys
from typing import List

import h5py
import torch
import transformers
from torch import cuda

parser = argparse.ArgumentParser(description='Pre-embed sentences with BERT.')
parser.add_argument('data', type=pathlib.Path, help='Path to .conll(x) file.')
parser.add_argument('out', type=pathlib.Path, help='Path to output file.')
parser.add_argument('--word-column',
                    type=int,
                    default=1,
                    help='Index of .conll column containing words.')
parser.add_argument('--pretrained',
                    default='bert-base-uncased',
                    help='Pre-trained weights to use.')
parser.add_argument('--quiet',
                    dest='log_level',
                    action='store_const',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='Only print warning and errors to stdout.')
options = parser.parse_args()

if not options.data.exists():
    raise FileNotFoundError(f'data file {options.data} not found')

# Set up.
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
logging.info('using %s', device.type)

sentences = []
partial: List[str] = []
with open(options.data, 'r') as conll:
    for line in conll:
        line = line.strip()
        if not line and partial:
            sentences.append(tuple(partial))
            partial = []
        if not line or line.startswith('#'):
            continue
        components = line.split()
        partial.append(components[options.word_column])
    if partial:
        sentences.append(tuple(partial))

logging.info('loading bert model: %s', options.pretrained)
tokenizer = transformers.BertTokenizer.from_pretrained(options.pretrained)
model = transformers.BertModel.from_pretrained(options.pretrained).to(device)
model.eval()

with h5py.File(options.out, 'w') as out:
    logging.info('will write %d embeddings to %s', len(sentences), options.out)
    for index, sentence in enumerate(sentences):
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        dataset = out.create_dataset(
            str(index),
            shape=(model.config.num_hidden_layers, len(tokens),
                   model.config.hidden_size),
            dtype='f',
        )

        inputs = torch.tensor([tokens], device=device)
        with torch.no_grad():
            embedded, *_ = model(inputs)
        dataset[:] = embedded
        logging.info('encoded sentence %d', index)
