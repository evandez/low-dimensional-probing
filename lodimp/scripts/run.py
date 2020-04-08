"""Wrapper for executing python scripts.

In particular, adjusts PYTHONPATH accordingly to ensure
scripts in subdirectories know how to import first-party packages.
Also saves run configuration so that experiments are repeatable.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from typing import List


def execute(command: str, root: str) -> int:
    """Execute the given command on a shell.

    Sets extra environment variables as necessary.

    Args:
        command: The command to execute as a string.
        root: Project root used to set PYTHONPATH.

    Returns:
        The subprocess return code.

    """
    python_path = os.environ.get('PYTHONPATH')
    python_path = '%s:%s' % (python_path, root) if python_path else root
    os.environ['PYTHONPATH'] = python_path

    hdf5_use_file_locking = os.environ.get('HDF5_USE_FILE_LOCKING')
    if not hdf5_use_file_locking:
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    logging.debug(command)

    # Use shell for command so we can tee as necessary.
    return subprocess.call(command, env=os.environ, shell=True)


def complete_command(command: List[str], log: str = None,
                     detach: bool = False) -> str:
    """Complete the given command, adding environment.

    Args:
        command: The original command.
        log: If set, also log to this path. By default everything is
            logged in the normal way to stdout/stderr.
        detach: If set, create and detach screen session for the command.
            By default, screen is not invoked.

    Returns:
        Completed command as a string to execute on a shell.

    """
    result = ['python3']
    result.extend(command)

    # If we want to use screen, we have to handle logging slightly
    # differently.
    if detach:
        screen = ['screen', '-S', os.path.basename(command[0])]
        if log:
            screen.extend(['-L', '-Logfile', log])
        screen.append('-dm')
        screen.extend(result)
        result = screen
    elif log:
        result.extend(['2>&1', '|', 'tee', log])

    return ' '.join(result)


def validate_command(command: List[str]) -> None:
    """Validate the given command is a python script.

    Args:
        command: The command to verify.

    """
    if not command:
        raise ValueError('Command is empty (probably failed to parse).')

    script = os.path.abspath(command[0])
    if not script.endswith('.py'):
        raise ValueError('Script must be a python script.')
    if not os.path.isfile(script):
        raise ValueError('No such script: %s' % script)


def ensure_dependencies(root: str) -> None:
    """Verify all pip dependencies are installed.

    Args:
        root: The project root in which to search for requirements.

    """
    pattern = os.path.join(root, '**/requirements.txt')
    paths = glob.glob(pattern, recursive=True)
    for path in paths:
        command = ['pip3', 'install', '-r', path]

        # Print command for trace.
        print(' '.join(command))

        # Exit this script cleanly if there is a problem.
        # This prevents obnoxious stack traces.
        error = subprocess.call(['pip3', 'install', '-r', path],
                                stdout=subprocess.DEVNULL)
        if error:
            sys.exit(error)


def project_root() -> str:
    """Returns the current git project root."""
    root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
    if not root:
        raise EnvironmentError('Could not determine current git project root.')
    return root.decode().strip()


def parse_args() -> argparse.Namespace:
    """Return parsed command-line options for this script."""
    parser = argparse.ArgumentParser(description='Run an ITIS script.')
    parser.add_argument(
        '--debug', action='store_true', help='If set, logs debug statements.')
    parser.add_argument(
        '--detach',
        action='store_true',
        help='If set, start a screen session and detach.')
    parser.add_argument(
        '--also-log-to',
        metavar='FILE',
        help='Path to a file in where logs should be dumped.')
    parser.add_argument(
        '--check-requirements',
        action='store_true',
        help='If set, try to search for and sync with requirements.txt.')
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='Everything after flags is taken to be the full command to run.')
    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    """Run the script.

    Args:
        options: Command-line options.

    """
    logging.basicConfig(level=logging.DEBUG if options.debug else logging.INFO)

    logging.debug('Determining project root...')
    root = project_root()
    logging.debug('Project root is %s', root)

    if options.check_requirements:
        logging.debug('Ensuring python dependencies installed...')
        ensure_dependencies(root)

    validate_command(options.command)
    command = complete_command(
        options.command, detach=options.detach, log=options.also_log_to)
    sys.exit(execute(command, root))


if __name__ == '__main__':
    main(parse_args())
