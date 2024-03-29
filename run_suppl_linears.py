"""Trains a bunch of linear probes."""
import argparse
import pathlib
import subprocess
from typing import NamedTuple


class Config(NamedTuple):
    """A run config."""

    model: str
    layer: int
    task: str
    dimension: int


ELMO = 'elmo'
BERT = 'bert'

POS = 'pos'
POSC = 'pos-control'
DLP = 'dep-label'
DLPC = 'dep-label-control'
DEP = 'dep-arc'
DEPC = 'dep-arc-control'
TASKS = {
    POS: 'pos',
    POSC: 'pos',
    DLP: 'dlp',
    DLPC: 'dlp',
    DEP: 'dep',
    DEPC: 'dep',
}

CONFIGS = (
    Config(model=ELMO, layer=0, task=POS, dimension=6),
    Config(model=ELMO, layer=0, task=POSC, dimension=256),
    Config(model=ELMO, layer=0, task=DLP, dimension=5),
    Config(model=ELMO, layer=0, task=DLPC, dimension=512),
    Config(model=ELMO, layer=0, task=DEP, dimension=11),
    Config(model=ELMO, layer=0, task=DEPC, dimension=13),
    Config(model=ELMO, layer=1, task=POS, dimension=5),
    Config(model=ELMO, layer=1, task=POSC, dimension=256),
    Config(model=ELMO, layer=1, task=DLP, dimension=3),
    Config(model=ELMO, layer=1, task=DLPC, dimension=512),
    Config(model=ELMO, layer=1, task=DEP, dimension=13),
    Config(model=ELMO, layer=1, task=DEPC, dimension=17),
    Config(model=ELMO, layer=2, task=POS, dimension=6),
    Config(model=ELMO, layer=2, task=POSC, dimension=256),
    Config(model=ELMO, layer=2, task=DLP, dimension=4),
    Config(model=ELMO, layer=2, task=DLPC, dimension=256),
    Config(model=ELMO, layer=2, task=DEP, dimension=21),
    Config(model=ELMO, layer=2, task=DEPC, dimension=23),
    Config(model=BERT, layer=0, task=POS, dimension=4),
    Config(model=BERT, layer=0, task=POSC, dimension=26),
    Config(model=BERT, layer=0, task=DLP, dimension=6),
    Config(model=BERT, layer=0, task=DLPC, dimension=256),
    Config(model=BERT, layer=0, task=DEP, dimension=13),
    Config(model=BERT, layer=0, task=DEPC, dimension=7),
    Config(model=BERT, layer=1, task=POS, dimension=5),
    Config(model=BERT, layer=1, task=POSC, dimension=128),
    Config(model=BERT, layer=1, task=DLP, dimension=6),
    Config(model=BERT, layer=1, task=DLPC, dimension=256),
    Config(model=BERT, layer=1, task=DEP, dimension=17),
    Config(model=BERT, layer=1, task=DEPC, dimension=9),
    Config(model=BERT, layer=4, task=POS, dimension=6),
    Config(model=BERT, layer=4, task=POSC, dimension=128),
    Config(model=BERT, layer=4, task=DLP, dimension=5),
    Config(model=BERT, layer=4, task=DLPC, dimension=256),
    Config(model=BERT, layer=4, task=DEP, dimension=13),
    Config(model=BERT, layer=4, task=DEPC, dimension=10),
    Config(model=BERT, layer=8, task=POS, dimension=7),
    Config(model=BERT, layer=8, task=POSC, dimension=256),
    Config(model=BERT, layer=8, task=DLP, dimension=5),
    Config(model=BERT, layer=8, task=DLPC, dimension=64),
    Config(model=BERT, layer=8, task=DEP, dimension=13),
    Config(model=BERT, layer=8, task=DEPC, dimension=12),
    Config(model=BERT, layer=12, task=POS, dimension=11),
    Config(model=BERT, layer=12, task=POSC, dimension=256),
    Config(model=BERT, layer=12, task=DLP, dimension=6),
    Config(model=BERT, layer=12, task=DLPC, dimension=24),
    Config(model=BERT, layer=12, task=DEP, dimension=20),
    Config(model=BERT, layer=12, task=DEPC, dimension=11),
)

parser = argparse.ArgumentParser(description='train linear probes')
parser.add_argument('tasks_dir', type=pathlib.Path, help='tasks directory')
parser.add_argument('--cuda', action='store_true', help='use cuda')
args = parser.parse_args()

for config in CONFIGS:
    command = ['python3', 'run_exp_train_probe.py']
    command += ['--linear']
    command += ['--cache']
    command += ['--model', config.model]
    command += ['--layer', str(config.layer)]
    command += ['--project-to', str(config.dimension)]
    command += ['--wandb-group', 'linear']

    # Task-dependent runtime settings.
    if config.task not in (DEP, DEPC):
        command += ['--no-batch']
        command += ['--epochs', '1000']
    else:
        command += ['--epochs', '4']

    # Args-dependent runtime settings.
    if args.device:
        command += ['--device', args.device]

    command += [TASKS[config.task], str(args.tasks_dir / config.task)]
    print(' '.join(command))
    subprocess.call(command)
