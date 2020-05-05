"""Unit tests for the check module."""

from lodimp import check

import pytest

NAME = 'name'


def test_nonempty_pass():
    """Test nonempty passes when it should."""
    check.nonempty([1])


def test_nonempty_fail():
    """Test nonempty fails when it should."""
    with pytest.raises(ValueError, match=f'.*{NAME}.*'):
        check.nonempty([], name=NAME)


def test_lengths_pass():
    """Test lengths passes when it should."""
    check.lengths([])
    check.lengths([[1, 2], [3, 4]])
    check.lengths(['foo', 'bar'])


def test_lengths_fail():
    """Test lengths fails when it should."""
    with pytest.raises(ValueError, match=f'.*{NAME}.*'):
        check.lengths([[1], [2, 3]], name=NAME)
