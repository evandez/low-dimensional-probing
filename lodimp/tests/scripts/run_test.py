"""Unit tests for run wrapper script."""

import os
import subprocess
import tempfile
from unittest import mock

from lodimp.scripts import run

import pytest


def test_project_root(mocker):
    """Test project_root calls correct git command."""
    check_output = mocker.patch.object(subprocess, 'check_output')
    check_output.return_value = b'root   '
    actual = run.project_root()
    assert actual == 'root'
    assert check_output.call_args_list == [
        mock.call(['git', 'rev-parse', '--show-toplevel']),
    ]


def test_ensure_dependencies(mocker):
    """Test ensure_dependencies finds all requirements.txt files."""
    call = mocker.patch.object(subprocess, 'call', return_value=0)
    with tempfile.TemporaryDirectory() as tempdir:
        path_a = os.path.join(tempdir, 'requirements.txt')
        path_b = os.path.join(tempdir, 'subdir/requirements.txt')
        os.makedirs(os.path.dirname(path_b))
        with open(path_a, 'w') as handle_a, open(path_b, 'w') as handle_b:
            handle_a.write('dep_a==1')
            handle_b.write('dep_b==2')

        run.ensure_dependencies(tempdir)
        assert call.call_count == 2
        for path in (path_a, path_b):
            assert mock.call(['pip3', 'install', '-r', path],
                             stdout=subprocess.DEVNULL) in call.call_args_list


def test_validate_command_valid():
    """Test validate_command allows good commands."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'foo.py')
        with open(path, 'w') as handle:
            handle.write('print("hello, world!")')
        run.validate_command([path])


@pytest.mark.parametrize('command,expected', [
    ([], '.* empty .*'),
    (['ls', '-al'], '.* python .*'),
    (['foo.py'], 'No such script: .*'),
])
def test_validate_command_invalid(command, expected):
    """Test validate_command dies with correct error messages."""
    with pytest.raises(ValueError, match=expected):
        run.validate_command(command)


@pytest.mark.parametrize('kwargs,expected', [
    (dict(command=['foo.py']), 'python3 foo.py'),
    (
        dict(command=['scripts/foo.py'], detach=True),
        'screen -S foo.py -dm python3 scripts/foo.py',
    ),
    (
        dict(command=['foo.py'], log='out.log'),
        'python3 foo.py 2>&1 | tee out.log',
    ),
    (
        dict(command=['scripts/foo.py'], detach=True, log='out.log'),
        'screen -S foo.py -L -Logfile out.log -dm python3 scripts/foo.py',
    ),
])
def test_complete_command(kwargs, expected):
    """Test complete_command produces expected command strings."""
    actual = run.complete_command(**kwargs)
    assert actual == expected


@pytest.mark.parametrize('command,root', (('python3 foo.py', 'root'),))
def test_execute(mocker, command, root):
    """Test execute augments PYTHONPATH and redirects correctly."""
    # Clear existing env so that it does not pollute the test.
    os.environ.clear()
    call = mocker.patch.object(subprocess, 'call', return_value=0)

    actual = run.execute(command, root)
    assert call.call_count == 1
    assert actual == 0

    [(args, kwargs)] = call.call_args_list
    assert args == (command,)
    assert kwargs == {'shell': True, 'env': os.environ}
    assert os.environ['PYTHONPATH'] == root
    assert os.environ['HDF5_USE_FILE_LOCKING'] == 'FALSE'
