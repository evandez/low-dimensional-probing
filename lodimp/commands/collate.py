"""Defines the `lodimp collate` command."""

import argparse
import itertools
import logging
import pathlib

from lodimp import tasks
from lodimp.common import datasets
from lodimp.common.parse import ptb, representations, splits
from lodimp.tasks import dep, dlp, pos

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
BERT = 'bert-base-uncased'
REPRESENTATION_MODELS = (ELMO, BERT)
REPRESENTATION_FILES_BY_MODEL = {
    ELMO: splits.ELMO_REPRESENTATIONS,
    BERT: splits.BERT_REPRESENTATIONS
}


def parser() -> argparse.ArgumentParser:
    """Returns the argument parser for this command."""
    parser = argparse.ArgumentParser(add_help=False, epilog=EPILOG)
    parser.add_argument('--representation-model',
                        choices=REPRESENTATION_MODELS,
                        default=ELMO,
                        help='Representation model to use.')
    parser.add_argument(
        '--representation-layers',
        metavar='LAYER',
        nargs='+',
        type=int,
        help='Layers to collate. Each layer will be collated to a separate '
        'file. Defaults to all layers.')
    parser.add_argument('--control',
                        action='store_true',
                        help='Collate the control version of this task.')
    parser.add_argument('--force',
                        action='store_true',
                        help='Overwrite any existing files.')

    subparsers = parser.add_subparsers(dest='task')
    pos_parser = subparsers.add_parser(tasks.PART_OF_SPEECH_TAGGING)
    pos_parser.add_argument('--subtask',
                            choices=PART_OF_SPEECH_SUBTASKS.keys(),
                            help='Only use a subset of part of speech tags, '
                            'collapse the rest. Cannot use with --control.')
    subparsers.add_parser(tasks.DEPENDENCY_LABEL_PREDICTION)
    subparsers.add_parser(tasks.DEPENDENCY_EDGE_PREDICTION)

    parser.add_argument('data', type=pathlib.Path, help='Path to data.')
    parser.add_argument('out', type=pathlib.Path, help='Output path.')
    return parser


def run(options: argparse.Namespace) -> None:
    """Run training with the given options.

    Args:
        options (argparse.Namespace): Parsed arguments. See parser() for list
            of flags.

    """
    if options.task == tasks.PART_OF_SPEECH_TAGGING:
        if options.control and options.subtask:
            raise ValueError('cannot set both --control and --subtask')
    if not options.data.exists():
        raise ValueError(f'data path not found: {options.data}')

    log = logging.getLogger(__name__)

    log.info('will write collated data to directory %s', options.out)
    options.out.mkdir(parents=True, exist_ok=True)

    files = splits.join(
        REPRESENTATION_FILES_BY_MODEL[options.representation_model],
        splits.PTB_ANNOTATIONS,
        root=options.data)

    reps, annos = {}, {}
    for split in splits.STANDARD_SPLITS:
        reps_path = files[split].representations
        log.info('reading %s set reps from %s', split, reps_path)
        reps[split] = representations.RepresentationDataset(reps_path)

        annos_path = files[split].annotations
        log.info('reading %s set annotations from %s', split, annos_path)
        annos[split] = ptb.load(files[split].annotations)

    dataset: datasets.TaskDataset
    layers = options.representation_layers or range(reps[splits.TRAIN].layers)
    for layer in layers:
        layer_path = options.out / options.representation_model / str(layer)
        log.info('collating data for %s layer %d in directory %s',
                 options.representation_model, layer, layer_path)
        layer_path.mkdir(parents=True, exist_ok=True)

        for split in splits.STANDARD_SPLITS:
            split_reps = reps[split].layer(layer)
            split_annos = annos[split]

            if options.task == tasks.PART_OF_SPEECH_TAGGING:
                if options.subtask:
                    tags = PART_OF_SPEECH_SUBTASKS[options.subtask]
                    log.info('will only distinguish these POS tags: %r', tags)
                    dataset = pos.POSTaskDataset(split_reps,
                                                 split_annos,
                                                 distinguish=tags)
                elif options.control:
                    log.info('will use control POS task')
                    dataset = pos.POSTaskDataset(
                        split_reps,
                        split_annos,
                        indexer=pos.ControlPOSIndexer,
                        samples=itertools.chain(*annos.values()))
                else:
                    log.info('will use ordinary POS task')
                    dataset = pos.POSTaskDataset(split_reps, split_annos)

            elif options.task == tasks.DEPENDENCY_LABEL_PREDICTION:
                if options.control:
                    log.info('will use control DLP task')
                    dataset = dlp.DLPTaskDataset(
                        split_reps,
                        split_annos,
                        indexer=dlp.ControlDLPIndexer,
                        samples=itertools.chain(*annos.values()))
                else:
                    log.info('will use ordinary DLP task')
                    dataset = dlp.DLPTaskDataset(split_reps, split_annos)

            elif options.task == tasks.DEPENDENCY_EDGE_PREDICTION:
                if options.control:
                    log.info('will use control DLP task')
                    dataset = dep.DEPTaskDataset(
                        split_reps,
                        split_annos,
                        indexer=dep.ControlDEPIndexer,
                        samples=itertools.chain(*annos.values()))
                else:
                    log.info('will use ordinary DEP task')
                    dataset = dep.DEPTaskDataset(split_reps, split_annos)

            else:
                raise ValueError(f'unknown task: {options.task}')

            split_path = layer_path / f'{split}.hdf5'
            log.info('collating %s set to %s', split, split_path)
            dataset.collate(split_path, force=options.force)
