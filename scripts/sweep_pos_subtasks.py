"""Trains a bunch of linear probes."""
import argparse
import pathlib
import subprocess

parser = argparse.ArgumentParser(description='train linear probes')
parser.add_argument('tasks_dir', type=pathlib.Path, help='tasks directory')
parser.add_argument('--ranks',
                    type=int,
                    default=10,
                    help='number of ranks to sweep')
parser.add_argument('--cuda', action='store_true', help='use cuda')
options = parser.parse_args()

TASKS = ('pos-noun', 'pos-verb', 'pos-verb-pres', 'pos-noun-common')

for task in TASKS:
    for rank in range(1, options.ranks + 1):
        command = ['python3', 'lodimp', 'train']
        command += ['--no-batch', '--cache']
        command += ['--epochs', '1000']
        command += ['--representation-model', 'bert-base-uncased']
        command += ['--representation-layer', '12']
        command += ['--project-to', str(rank)]
        command += ['--wandb-group', task]
        command += ['--wandb-name', f'{task}-bert-base-uncased12-{rank}d']

        # Args-dependent runtime settings.
        if options.cuda:
            command += ['--cuda']

        command += ['pos', str(options.tasks_dir / task)]
        subprocess.call(command)
