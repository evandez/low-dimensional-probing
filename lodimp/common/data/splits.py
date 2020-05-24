"""Constants and utilities for interacting with dataset splits."""

import pathlib
from typing import Dict, Mapping, NamedTuple, Optional, Union

# These constants define standard dataset splits for both PTB/OntoNotes.
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
STANDARD_SPLITS = (TRAIN, DEV, TEST)

# These constants define conventional file names for all kinds of
# representations we will use throughout the dataset.
ELMO_REPRESENTATIONS = {
    key: f'raw.{key}.elmo-layers.hdf5' for key in STANDARD_SPLITS
}
BERT_REPRESENTATIONS = {
    key: f'raw.{key}.bert-base-uncased-layers.hdf5' for key in STANDARD_SPLITS
}

# These constants define conventional file names for all kinds of
# annotations we will use throughout the dataset.
PTB_ANNOTATIONS = {key: f'ptb3-wsj-{key}.conllx' for key in STANDARD_SPLITS}
ONTONOTES_ANNOTATIONS = {
    key: f'ontonotes5-{key}.conll' for key in STANDARD_SPLITS
}


class Split(NamedTuple):
    """Defines a dataset split as it exists on disk.

    Note that this class verifies that the files do indeed exist.
    """
    # Representations are typically an h5 file storing pre-computed word
    # embeddings for every sample in the split.
    representations: pathlib.Path

    # Annotations are typically some variant of a .conll file containing
    # both the original words and whatever labels we need for probing tasks.
    annotations: pathlib.Path


PathLike = Union[str, pathlib.Path]


def ensure(representations: PathLike, annotations: PathLike) -> Split:
    """Construct a Split, ensuring the files exist.

    Args:
        representations (PathLike): The word representations for the split.
        annotations (PathLike): The annotations for the split.

    Raises:
        FileNotFoundError: If representations/annotations files not found.

    Returns:
        Split: The valid, constructed split.

    """
    representations = pathlib.Path(representations)
    if not representations.exists():
        raise FileNotFoundError(f'reps not found: {representations}')

    annotations = pathlib.Path(annotations)
    if not annotations.exists():
        raise FileNotFoundError(f'annotations not found: {annotations}')

    return Split(representations=representations, annotations=annotations)


def join(representations: Mapping[str, PathLike],
         annotations: Mapping[str, PathLike],
         root: Optional[PathLike] = None,
         validate: bool = True) -> Dict[str, Split]:
    """Join representations and annotations into split.

    This function is best used in conjunction with the constants at the
    top of this file, for example:

        data = splits.join(splits.ELMO_REPRESENTATIONS,
                           splits.PTB_ANNOTAITONS,
                           root=path_to_my_data)

    Args:
        representations (Mapping[str, PathLike]): Map from split key
            to representations path.
        annotations (Mapping[str, PathLike]): Map from split key to
            annotations path.
        root (PathLike): Prepend this path to the representations
            and annotations paths. By default nothing is prepended.
        validate (bool, optional): Ensure the root and final split paths
            exist. Defaults to True.

    Raises:
        FileNotFoundError: If `validate=True` and root/split paths do
            not exist.
        ValueError: If `representations` and `annotations` dicts have different
            split keys.

    Returns:
        Dict[str, Split]: Mapping from split key to (potentially validated)
            Split(...) tuple.

    """
    if root is not None:
        root = pathlib.Path(root)
        if validate and not root.exists():
            raise FileNotFoundError(f'root {root} does not exist')

    if representations.keys() != annotations.keys():
        raise ValueError(f'reps have splits {representations.keys()} '
                         f'but annotations have splits {annotations.keys()}')

    splits = {}
    for key in representations.keys():
        split_reps_path = pathlib.Path(representations[key])
        split_annotations_path = pathlib.Path(annotations[key])
        if root is not None:
            split_reps_path = root / split_reps_path
            split_annotations_path = root / split_annotations_path

        if validate:
            splits[key] = ensure(representations=split_reps_path,
                                 annotations=split_annotations_path)
        else:
            splits[key] = Split(representations=split_reps_path,
                                annotations=split_annotations_path)

    return splits
