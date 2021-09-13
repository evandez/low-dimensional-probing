"""Collate PTB data into compact h5 files."""
import argparse
import itertools
import pathlib

from lodimp import datasets, tasks
from lodimp.parse import ptb, representations, splits
from lodimp.tasks import dep, dlp, pos
from lodimp.utils import logging

EPILOG = '''\
This command reads representations and linguistic annotations from disk,
converts the string annotations to integer annotations, and writes the
result to a contiguous array. This reduces the IO bottleneck during training.
Indeed, for sufficiently small datasets like PTB, we can load the entire
collated dataset into GPU memory and skip batching for an absurd performance
increase.

All downstream commands, like `lodimp train`, assume the data is provided
in the collated format output by this command.
'''

PART_OF_SPEECH_SUBTASKS = {
    'nouns': pos.NOUNS,
    'proper-nouns': pos.NOUNS_PROPER,
    'common-nouns': pos.NOUNS_COMMON,
    'verbs': pos.VERBS,
    'verbs-present': pos.VERBS_PRESENT,
    'verbs-past': pos.VERBS_PAST,
    'adverbs': pos.ADVERBS,
    'adjectives': pos.ADJECTIVES,
}

ELMO = 'elmo'
BERT = 'bert'
BERT_RANDOM = 'bert-random'
REPRESENTATION_MODELS = (ELMO, BERT, BERT_RANDOM)
REPRESENTATION_FILES_BY_MODEL = {
    ELMO: splits.ELMO_REPRESENTATIONS,
    BERT: splits.BERT_REPRESENTATIONS,
    BERT_RANDOM: splits.BERT_RANDOM_REPRESENTATIONS,
}

ELMO_LAYERS = (0, 1, 2)
BERT_LAYERS = (0, 1, 4, 8, 12)
BERT_RANDOM_LAYERS = BERT_LAYERS
REPRESENTATION_LAYERS_BY_MODEL = {
    ELMO: ELMO_LAYERS,
    BERT: BERT_LAYERS,
    BERT_RANDOM: BERT_RANDOM_LAYERS,
}

parser = argparse.ArgumentParser(epilog=EPILOG)
parser.add_argument('task',
                    choices=(tasks.PART_OF_SPEECH_TAGGING,
                             tasks.DEPENDENCY_LABEL_PREDICTION,
                             tasks.DEPENDENCY_EDGE_PREDICTION),
                    help='task to collate reps for')
parser.add_argument('data_dir', type=pathlib.Path, help='data dir')
parser.add_argument('--out-dir',
                    type=pathlib.Path,
                    help='output dir (default: data_dir / "collated" / task)')
parser.add_argument('--representation-model',
                    choices=REPRESENTATION_MODELS,
                    default=ELMO,
                    help='representation model to use (default: elmo)')
parser.add_argument('--representation-layers',
                    metavar='LAYER',
                    nargs='+',
                    type=int,
                    help='layers to collate (default: same as paper)')
parser.add_argument('--control',
                    action='store_true',
                    help='collate control version of this task')
parser.add_argument('--subtask',
                    choices=PART_OF_SPEECH_SUBTASKS.keys(),
                    help='when collating pos, only use a subset of tags, '
                    'collapsing the rest; cannot use if --control set '
                    'or if task is not pos')
parser.add_argument('--force',
                    action='store_true',
                    help='overwrite any existing files')
args = parser.parse_args()

logging.configure()
log = logging.getLogger(__name__)

if args.task == tasks.PART_OF_SPEECH_TAGGING:
    if args.control and args.subtask:
        raise ValueError('cannot set both --control and --subtask')
elif args.subtask:
    raise ValueError('cannot set --subtask when task != "pos"')

data_dir = args.data_dir
if not data_dir.exists():
    raise ValueError(f'data dir not found: {data_dir}')

out_dir = args.out_dir
if out_dir is None:
    out_name = f'{args.task}{"-control" if args.control else ""}'
    out_dir = data_dir / 'collated' / out_name
out_dir.mkdir(parents=True, exist_ok=True)
log.info('will write collated data to directory %s', out_dir)

files = splits.join(REPRESENTATION_FILES_BY_MODEL[args.representation_model],
                    splits.PTB_ANNOTATIONS,
                    root=data_dir)

reps, annos = {}, {}
for split in splits.STANDARD_SPLITS:
    reps_path = files[split].representations
    log.info('reading %s set reps from %s', split, reps_path)
    reps[split] = representations.RepresentationDataset(reps_path)

    annos_path = files[split].annotations
    log.info('reading %s set annotations from %s', split, annos_path)
    annos[split] = ptb.load(files[split].annotations)

dataset: datasets.TaskDataset
layers = args.representation_layers
if layers is None:
    layers = REPRESENTATION_LAYERS_BY_MODEL[args.representation_model]
for layer in layers:
    layer_path = out_dir / args.representation_model / str(layer)
    log.info('collating data for %s layer %d in directory %s',
             args.representation_model, layer, layer_path)
    layer_path.mkdir(parents=True, exist_ok=True)

    for split in splits.STANDARD_SPLITS:
        split_reps = reps[split].layer(layer)
        split_annos = annos[split]

        if args.task == tasks.PART_OF_SPEECH_TAGGING:
            if args.subtask:
                tags = PART_OF_SPEECH_SUBTASKS[args.subtask]
                log.info('will only distinguish these POS tags: %r', tags)
                dataset = pos.POSTaskDataset(split_reps,
                                             split_annos,
                                             distinguish=tags)
            elif args.control:
                log.info('will use control POS task')
                dataset = pos.POSTaskDataset(
                    split_reps,
                    split_annos,
                    indexer=pos.ControlPOSIndexer,
                    samples=tuple(itertools.chain(*annos.values())))
            else:
                log.info('will use ordinary POS task')
                dataset = pos.POSTaskDataset(split_reps, split_annos)

        elif args.task == tasks.DEPENDENCY_LABEL_PREDICTION:
            if args.control:
                log.info('will use control DLP task')
                dataset = dlp.DLPTaskDataset(
                    split_reps,
                    split_annos,
                    indexer=dlp.ControlDLPIndexer,
                    samples=tuple(itertools.chain(*annos.values())))
            else:
                log.info('will use ordinary DLP task')
                dataset = dlp.DLPTaskDataset(split_reps, split_annos)

        elif args.task == tasks.DEPENDENCY_EDGE_PREDICTION:
            if args.control:
                log.info('will use control DLP task')
                dataset = dep.DEPTaskDataset(
                    split_reps,
                    split_annos,
                    indexer=dep.ControlDEPIndexer,
                    samples=tuple(itertools.chain(*annos.values())))
            else:
                log.info('will use ordinary DEP task')
                dataset = dep.DEPTaskDataset(split_reps, split_annos)

        else:
            raise ValueError(f'unknown task: {args.task}')

        split_path = layer_path / f'{split}.h5'
        log.info('collating %s set to %s', split, split_path)
        dataset.collate(split_path, force=args.force)
