"""Wrapper for executing LoDimP python scripts.

In particular, adjusts PYTHONPATH accordingly to ensure
scripts in subdirectories know how to import first-party packages.
Also saves run configuration so that experiments are repeatable.
"""

import argparse
import logging
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description='Run an ITIS script.')
parser.add_argument('--debug',
                    action='store_true',
                    help='Log debug statements.')
parser.add_argument('--detach',
                    action='store_true',
                    help='Start a screen session and detach.')
parser.add_argument('--also-log-to',
                    metavar='FILE',
                    help='Path to a file in where logs should be dumped.')
parser.add_argument(
    'command',
    nargs=argparse.REMAINDER,
    help='Everything after flags is taken to be the full command to run.')
options = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if options.debug else logging.INFO)

# Validate the command
if not options.command:
    raise ValueError('Command is empty (probably failed to parse).')
script = os.path.abspath(options.command[0])
if not script.endswith('.py'):
    raise ValueError('Script must be a python script.')
if not os.path.isfile(script):
    raise ValueError('No such script: %s' % script)

# Fill in some missing details.
args = ['python3']
args.extend(options.command)
if options.detach:
    # If we want to use screen, we have to handle logging slightly
    # differently.
    screen = ['screen', '-S', os.path.basename(args[0])]
    if options.also_log_to:
        screen.extend(['-L', '-Logfile', options.also_log_to])
    screen.append('-dm')
    screen.extend(args)
    args = screen
elif options.also_log_to:
    args.extend(['2>&1', '|', 'tee', options.also_log_to])
command = ' '.join(args)
logging.debug(command)

# Find the project root.
root = subprocess.check_output(['git', 'rev-parse',
                                '--show-toplevel']).decode().strip()
if not root:
    raise EnvironmentError('Could not determine current git project root.')
logging.debug('Project root is %s', root)

# Set environment variables.
python_path = os.environ.get('PYTHONPATH')
python_path = '%s:%s' % (python_path, root) if python_path else root
os.environ['PYTHONPATH'] = python_path
hdf5_use_file_locking = os.environ.get('HDF5_USE_FILE_LOCKING')
if not hdf5_use_file_locking:
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
logging.debug(os.environ)

# Go! Use shell for command so we can tee as necessary.
sys.exit(subprocess.call(command, env=os.environ, shell=True))
