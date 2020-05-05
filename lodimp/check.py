"""Some higher-order assertions."""

from typing import Optional, Sequence, Sized


def nonempty(sized: Sized, name: Optional[str] = None) -> None:
    """Check that the given sequence is nonempty.

    Args:
        sized (Sized): The sequence to validate.
        name (Optional[str]): If set, display this name in the error message.

    Raises:
        ValueError: If the sequence is empty.

    """
    if not len(sized):
        name = name or 'sequence'
        raise ValueError(f'{name} is empty')


def lengths(items: Sequence[Sized], name: Optional[str] = None) -> None:
    """Check that all items have the same size.

    Args:
        items (Sequence[Sized]): The items to validate.
        name (Optional[str]): If set, display this name in the error message.

    Raises:
        ValueError: If any two of the items have a different length.

    """
    lengths = {len(item) for item in items}
    if len(lengths) > 1:
        name = name or 'sequence'
        raise ValueError(f'{name} has mismatched lengths: {lengths}')
