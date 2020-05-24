"""Entrypoint for all LoDimP scripts."""

import argparse
import logging
import pathlib
import sys

import torch
import wandb

# Unfortunately we have to muck with sys.path to avoid a wrapper script.
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

from lodimp.common.data import splits  # noqa: E402
from lodimp.common.models import probes  # noqa: E402
from lodimp import pos  # noqa: E402
import lodimp.pos.preprocess  # noqa: E402

POS = 'pos'
DEP_LABEL = 'dep_label'
DEP_ARC = 'dep_arc'
SRL = 'srl'
TASKS = (POS, DEP_LABEL, DEP_ARC, SRL)
POS_SUBTASKS = {
    'nouns': lodimp.pos.preprocess.NOUNS,
    'nouns-proper': lodimp.pos.preprocess.NOUNS_PROPER,
    'nouns-plural': lodimp.pos.preprocess.NOUNS_PLURAL,
    'verbs': lodimp.pos.preprocess.VERBS,
    'verbs-past': lodimp.pos.preprocess.VERBS_PAST,
    'verbs-pres': lodimp.pos.preprocess.VERBS_PRESENT,
    'adjectives': lodimp.pos.preprocess.ADJECTIVES,
    'adverbs': lodimp.pos.preprocess.ADVERBS,
}

ANNOTATIONS = {
    POS: splits.PTB_ANNOTATIONS,
    DEP_LABEL: splits.PTB_ANNOTATIONS,
    DEP_ARC: splits.PTB_ANNOTATIONS,
    SRL: splits.ONTONOTES_ANNOTATIONS,
}

ELMO = 'elmo'
BERT = 'bert'
MODELS = (ELMO, BERT)
REPRESENTATIONS = {
    ELMO: splits.ELMO_REPRESENTATIONS,
    BERT: splits.BERT_REPRESENTATIONS
}

# Define all commands! I know, it's a lot. But you'll survive.
parser = argparse.ArgumentParser(description='Run a LoDimP script.')
parser.add_argument('--quiet',
                    dest='log_level',
                    action='store_const',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='Only show warning or error messages.')
subparsers = parser.add_subparsers(dest='subcommand')

preprocess_parser = subparsers.add_parser('preprocess')
preprocess_parser.add_argument('--model',
                               choices=MODELS,
                               default='elmo',
                               help='Representation model to use.')
preprocess_parser.add_argument('--layers',
                               metavar='LAYER',
                               nargs='+',
                               type=int,
                               help='Layers to preprocess. Defaults to all.')

preprocess_subparsers = preprocess_parser.add_subparsers(dest='task')
preprocess_pos_parser = preprocess_subparsers.add_parser(POS)
preprocess_pos_parser.add_argument(
    '--subtask',
    choices=POS_SUBTASKS.keys(),
    help='Only use a subset of part of speech tags, collapse the rest.')
preprocess_subparsers.add_parser(DEP_LABEL)
preprocess_subparsers.add_parser(DEP_ARC)
preprocess_subparsers.add_parser(SRL)

preprocess_parser.add_argument('data', type=pathlib.Path, help='Path to data.')
preprocess_parser.add_argument('out', type=pathlib.Path, help='Output path.')

train_parser = subparsers.add_parser('train')
train_parser.add_argument('--mlp', action='store_true', help='Use MLP probe.')
train_parser.add_argument('--project-to',
                          type=int,
                          default=64,
                          help='Dimensionality of projected space.')
train_parser.add_argument('--representation-model',
                          choices=('elmo', 'bert-base-uncased'),
                          default='elmo',
                          help='Representations to probe. Default elmo.')
train_parser.add_argument('--representation-layer',
                          type=int,
                          default=0,
                          help='Representation layer to probe. Default 0.')
train_parser.add_argument('--lr',
                          default=1e-3,
                          type=float,
                          help='Learning rate. Default 1e-3.')
train_parser.add_argument(
    '--epochs',
    type=int,
    default=25,
    help='Passes to make through dataset during training. Default 25.')
train_parser.add_argument(
    '--patience',
    type=int,
    default=4,
    help='Epochs for dev loss to decrease to stop training. Default 4.')
train_parser.add_argument(
    '--no-batch',
    action='store_true',
    help='Store entire dataset in RAM/GPU and do not batch it.')
train_parser.add_argument('--model-path',
                          type=pathlib.Path,
                          default='/tmp/lodimp/models/probe.pth',
                          help='Directory to write finished model.')
train_parser.add_argument('--wandb-id', help='Experiment ID. Use carefully!')
train_parser.add_argument('--wandb-group', help='Experiment group.')
train_parser.add_argument('--wandb-name', help='Experiment name.')
train_parser.add_argument('--wandb-path',
                          type=pathlib.Path,
                          default='/tmp/lodimp/wandb',
                          help='Path to write Weights and Biases data.')
train_parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
train_subparsers = train_parser.add_subparsers(dest='task')
train_parser.add_argument('data', type=pathlib.Path, help='Data path.')

train_pos_parser = train_subparsers.add_parser('pos')
train_pos_parser.add_argument(
    '--project-from',
    type=pathlib.Path,
    help='Compose projection stored here with learned projection.')

train_dep_arc_parser = train_subparsers.add_parser('dep-arc')
train_dep_label_parser = train_subparsers.add_parser('dep-label')
train_srl_parser = train_subparsers.add_parser('srl')
for sp in (train_dep_arc_parser, train_dep_label_parser, train_srl_parser):
    sp.add_argument('--share-projection',
                    action='store_true',
                    help='When comparing reps, project both with same matrix.')

axis_alignment_parser = subparsers.add_parser('axis-alignment')
axis_alignment_parser.add_argument(
    '--representation-model',
    choices=('elmo', 'bert-base-uncased'),
    default='elmo',
    help='Representations to probe. Default elmo.')
axis_alignment_parser.add_argument(
    '--representation-layer',
    type=int,
    default=0,
    help='Representation layer to probe. Default 0.')
axis_alignment_parser.add_argument(
    '--no-batch',
    action='store_true',
    help='Store entire dataset in RAM/GPU and do not batch it.')
axis_alignment_parser.add_argument('--wandb-id',
                                   help='Experiment ID. Use carefully!')
axis_alignment_parser.add_argument('--wandb-group', help='Experiment group.')
axis_alignment_parser.add_argument('--wandb-name', help='Experiment name.')
axis_alignment_parser.add_argument(
    '--wandb-path',
    type=pathlib.Path,
    default='/tmp/lodimp/wandb',
    help='Path to write Weights and Biases data.')
axis_alignment_parser.add_argument('--cuda',
                                   action='store_true',
                                   help='Use CUDA.')
axis_alignment_subparsers = axis_alignment_parser.add_subparsers(dest='task')
axis_alignment_subparsers.add_parser(POS)
axis_alignment_parser.add_argument('data',
                                   type=pathlib.Path,
                                   help='Task data path.')
axis_alignment_parser.add_argument('probe',
                                   type=pathlib.Path,
                                   help='Probe path.')
options = parser.parse_args()

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)
print(options.subcommand)
if options.subcommand == 'preprocess':
    if options.task != 'pos':
        raise NotImplementedError('only pos supported right now')

    data = splits.join(REPRESENTATIONS[options.model],
                       ANNOTATIONS[options.task],
                       root=options.data)
    pos.preprocess.run(data,
                       options.out / options.model,
                       tags=POS_SUBTASKS.get(options.subtask),
                       layers=options.layers)

elif options.subcommand == 'train':
    if options.task != 'pos':
        raise NotImplementedError('only pos supported right now')
    wandb.init(project='lodimp',
               id=options.wandb_id,
               name=options.wandb_name,
               group=options.wandb_group,
               config={
                   'task': options.task,
                   'representations': {
                       'model': options.representation_model,
                       'layer': options.representation_layer,
                   },
                   'projection': {
                       'dimension': options.project_to,
                       'composed': bool(options.project_from),
                   },
                   'probe': {
                       'model': options.probe,
                   },
                   'hyperparameters': {
                       'epochs': options.epochs,
                       'batched': not options.no_batch,
                       'lr': options.lr,
                       'patience': options.patience,
                       'regularization': {
                           'l1': options.l1,
                           'nuc': options.nuc,
                       },
                   },
               },
               dir=str(options.wandb_path))

    device = torch.device('cuda') if options.cuda else torch.device('cpu')
    logging.info('using %s', device.type)

    model, layer = options.representation_model, options.representation_layer
    data_path = options.data / model / str(layer)

    probe, accuracy = pos.train(
        data_path,
        probe_t=probes.MLP if options.mlp else probes.Linear,
        project_to=options.project_to,
        project_from=torch.load(options.project_from, map_location=device)
        if options.project_from else None,
        epochs=options.epochs,
        patience=options.patience,
        lr=options.lr,
        batch=not options.no_batch,
        device=device,
        also_log_to_wandb=True)
    torch.save(probe, options.model_path)
    wandb.save(options.model_path)
    wandb.run.summary['accuracy'] = accuracy

elif options.subcommand == 'axis-alignment':
    if options.task != 'pos':
        raise NotImplementedError('only pos supported right now')

    device = torch.device('cuda') if options.cuda else torch.device('cpu')
    logging.info('using %s', device.type)

    model, layer = options.representation_model, options.representation_layer
    data_path = options.data / model / str(layer)

    probe = torch.load(options.probe, map_location=device)

    pos.axis_alignment(probe,
                       data_path,
                       batch=not options.no_batch,
                       device=device,
                       also_log_to_wandb=True)
