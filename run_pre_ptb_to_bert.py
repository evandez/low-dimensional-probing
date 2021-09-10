"""Embed all sentences in the .conllx file with BERT."""
import argparse
import json
import pathlib
from typing import List

from lodimp.parse import splits
from lodimp.utils import logging

import h5py
import numpy as np
import torch
import transformers
from torch import cuda
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='pre-embed sentences with bert')
parser.add_argument('data_dir',
                    type=pathlib.Path,
                    help='dir containing .conll(x) files')
parser.add_argument('--out-dir',
                    type=pathlib.Path,
                    help='output dir (default: data_dir)')
parser.add_argument('--word-column',
                    type=int,
                    default=1,
                    help='index of .conll column containing words')
parser.add_argument('--random',
                    action='store_true',
                    help='randomly initialize bert')
parser.add_argument('--bert-config',
                    default='bert-base-uncased',
                    help='pretrained bert config (default: bert-base-uncased)')
parser.add_argument('--splits',
                    nargs='+',
                    default=splits.STANDARD_SPLITS,
                    help='splits to process (default: train, dev, test)')
parser.add_argument('--device', help='use this device (default: guessed)')
args = parser.parse_args()

data_dir = args.data_dir
if not data_dir.exists():
    raise FileNotFoundError(f'data dir {data_dir} not found')
out_dir = args.out_dir or data_dir

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Load the model.
tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_config)
if args.random:
    random_weights_dir = out_dir / 'bert-base-uncased-random'
    if random_weights_dir.exists():
        bert = transformers.BertModel.from_pretrained(str(random_weights_dir))
        config = bert.config
    else:
        config = transformers.BertConfig.from_pretrained(args.bert_config)
        bert = transformers.BertModel(config)
else:
    bert = transformers.BertModel.from_pretrained(args.bert_config)
    config = bert.config
assert isinstance(config, transformers.BertConfig)
bert.to(device).eval()

# Process each split.
for split in ('train', 'dev', 'test'):
    data_file = data_dir / f'ptb3-wsj-{split}.conllx'
    out_file = out_dir / f'bert{"-random" if args.random else ""}-{split}.h5'

    # TODO(evandez): Use parse lib?
    sentences = []
    partial: List[str] = []
    with data_file.open('r') as conll:
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

    with h5py.File(str(out_file), 'w') as handle:
        for index, sentence in tqdm(tuple(enumerate(sentences)),
                                    desc='embed sentences'):
            tokens = tokenizer.encode(sentence, add_special_tokens=False)
            dataset = handle.create_dataset(
                str(index),
                shape=(config.num_hidden_layers + 1, len(tokens),
                       config.hidden_size),
                dtype='f',
            )

            inputs = torch.tensor([tokens], device=device)
            with torch.no_grad():
                outputs = bert(inputs, output_hidden_states=True)
                hiddens = outputs.hidden_states
                assert len(hiddens) == 13
                embeddings = torch.cat(hiddens)
                assert embeddings.shape == (13, len(tokens), 768)

            dataset[:] = embeddings.cpu().numpy()

        # Create a sentence_to_index item so bert h5 files look like ELMo ones.
        joined = [' '.join(sentence) for sentence in sentences]
        sentence_to_index = {sentence: i for i, sentence in enumerate(joined)}
        sentence_to_index_str = json.dumps(sentence_to_index).encode()
        out_file.create_dataset('sentence_to_index',
                                data=np.array([sentence_to_index_str]))
