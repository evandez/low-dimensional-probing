"""Embed all sentences in the .conllx file with BERT."""
import argparse
import json
import pathlib
from typing import List

from lodimp.common import logging

import h5py
import numpy as np
import torch
import transformers
from torch import cuda

parser = argparse.ArgumentParser(description='pre-embed sentences with bert')
parser.add_argument('data_file',
                    type=pathlib.Path,
                    help='path to ptb .conll(x) file')
parser.add_argument('out_file', type=pathlib.Path, help='path to output file')
parser.add_argument('--word-column',
                    type=int,
                    default=1,
                    help='index of .conll column containing words')
parser.add_argument('--device', help='use this device (default: guessed)')
args = parser.parse_args()

if not args.data_file.exists():
    raise FileNotFoundError(f'data file {args.data_file} not found')

# Set up.
logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
logging.info('using %s', device)

sentences = []
partial: List[str] = []
with args.data_file.open('r') as conll:
    for line in conll:
        line = line.strip()
        if not line and partial:
            sentences.append(tuple(partial))
            partial = []
        if not line or line.startswith('#'):
            continue
        components = line.split()
        partial.append(components[args.word_column])
    if partial:
        sentences.append(tuple(partial))

logging.info('loading model...')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

bert = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)
bert.eval()
# We want the hidden states, so we have to hack the config a little bit...
bert.encoder.output_hidden_states = True

with h5py.File(str(args.out_file), 'w') as out_file:
    logging.info('will write %d embeddings to %s', len(sentences),
                 args.out_file)
    for index, sentence in enumerate(sentences):
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        dataset = out_file.create_dataset(
            str(index),
            shape=(bert.config.num_hidden_layers + 1, len(tokens),
                   bert.config.hidden_size),
            dtype='f',
        )

        inputs = torch.tensor([tokens], device=device)
        with torch.no_grad():
            _, _, hiddens = bert(inputs)
            assert len(hiddens) == 13
            embeddings = torch.cat(hiddens)
            assert embeddings.shape == (13, len(tokens), 768)

        dataset[:] = embeddings.cpu().numpy()
        logging.info('encoded sentence %d', index)

    # Create a sentence_to_index item so bert hdf5 files look like ELMo ones.
    joined = [' '.join(sentence) for sentence in sentences]
    sentence_to_index = {sentence: i for i, sentence in enumerate(joined)}
    sentence_to_index_str = json.dumps(sentence_to_index).encode()
    out_file.create_dataset('sentence_to_index',
                            data=np.array([sentence_to_index_str]))
