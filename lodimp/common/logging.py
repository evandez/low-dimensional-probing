"""Wraps built-in logging module."""
import logging
import sys
from typing import Any


def __getattr__(name: str) -> Any:
    """Forward to built-in `logging`."""
    return getattr(logging, name)


def configure(**kwargs: Any) -> None:
    """Configure logging."""
    kwargs.setdefault('stream', sys.stdout)
    kwargs.setdefault('format', '%(asctime)s %(levelname)-8s %(message)s')
    kwargs.setdefault('datefmt', '%Y-%m-%d %H:%M:%S')
    kwargs.setdefault('level', logging.INFO)
    logging.basicConfig(**kwargs)
