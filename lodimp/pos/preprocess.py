"""Preprocess PTB for part of speech tagging.

The functions here map part of speech tags to numerical indices and pre-collate
them alongside pre-computed word representations. This allows us to avoid
batching during training.
"""

import itertools
import logging
import pathlib
from typing import Dict, Optional, Sequence

from lodimp.common.data import ptb, representations, splits

import h5py
import torch

NOUNS = ('NN', 'NNS', 'NNP', 'NNPS')
NOUNS_PROPER = ('NNP', 'NNPS')
NOUNS_PLURAL = ('NNS', 'NNPS')
VERBS = ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
VERBS_PRESENT = ('VBZ', 'VBP', 'VBG')
VERBS_PAST = ('VBD', 'VBN')
ADJECTIVES = ('JJ', 'JJR', 'JJS')
ADVERBS = ('RB', 'RBR', 'RBS')


class POSTask:
    """Indexes PTB POS tags."""

    def __init__(self,
                 samples: Sequence[ptb.Sample],
                 tags: Optional[Sequence[str]] = None,
                 unk: str = 'UNK'):
        """Maps each POS tag to an index.

        Args:
            samples (Sequence[ptb.PTBSample]): The samples from which to
                draw tags.
            tags (Optional[Set[str]]): The XPOS tags to distinguish.
                All tags not in this set will be collapsed to the same tag.
                By default, all tags will be distinguished.
            unk (str): Tag to use when un-indexed XPOS is encountered.

        """
        if tags is None:
            tags = tuple(xpos for sample in samples for xpos in sample.xpos)
        assert tags is not None, 'no tags to distinguish?'
        assert unk not in tags, 'unk found in dataset?'

        self.unk = unk
        self.indexer = {unk: 0}
        for xpos in sorted(set(tags)):
            self.indexer[xpos] = len(self.indexer)

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Index the part-of-speech tags for each sample.

        Args:
            samples (ptb.PTBSample): The sample to index XPOS tags for.

        Returns:
            torch.Tensor: Integer tags for each XPOS in the sample.

        """
        return torch.tensor([
            self.indexer.get(xpos, self.indexer[self.unk])
            for xpos in sample.xpos
        ])

    def __len__(self) -> int:
        """Returns the number of valid POS tags in this task."""
        return len(self.indexer)


class POSTagsDataset(torch.utils.data.Dataset):
    """Tags PTB samples and allows for iteration over them."""

    def __init__(self, samples: Sequence[ptb.Sample], task: POSTask):
        """Initialize the dataset.

        Args:
            samples (Sequence[ptb.Sample]): The samples to tag.
            task (POSTask): Mapping from part of speech to integer tag.

        """
        self.samples = samples
        self.task = task

    def __getitem__(self, index: int) -> torch.Tensor:
        """Compute the tags for the index'th sample.

        Args:
            index (int): Index of the sample (sentence) to tag.

        Raises:
            IndexError: If the index is out of bounds.

        Returns:
            torch.Tensor: Shape (L,) integer tensor, where L is the length
                of the index'th sample's sentence.

        """
        if index < 0 or index >= len(self):
            raise IndexError(f'index out of bounds: {index}')
        return self.task(self.samples[index])

    def __len__(self) -> int:
        """Returns the number of samples (sentences) in the dataset."""
        return len(self.samples)


def collate(
    reps: representations.RepresentationLayerDataset,
    tags: POSTagsDataset,
    out: pathlib.Path,
    breaks_key: str = 'breaks',
    reps_key: str = 'reps',
    tags_key: str = 'tags',
    force: bool = False,
) -> None:
    """Collate the representations and their tags into an h5 file.

    The h5 file will have two datasets: one for representations,
    which will be a contiguous array of all representations in order,
    and one for POS tags represented as integers. The datasets will
    align, in that the i-th representation in the representations dataset
    should be tagged with the i-th tag in the tags dataset.

    Args:
        reps (representations.RepresentationLayerDataset): Representations to
            collate.
        tags (POSTagsDataset): Tags corresponding to each representation.
            Dataset length should match the length of `reps`, and the length of
            each tag sequence should match the length of the corresponding
            representation sequence in `reps`.
        out (pathlib.Path): Path at which to write output file. Must not exist,
            unless force is set to True.
        breaks_key (str, optional): Key to use for the breaks dataset
            in the output h5 file. Defaults to 'breaks'.
        reps_key (str, optional): Key to use for the representations dataset
            in the output h5 file. Defaults to 'reps'.
        tags_key (str, optional): Same as above, for the dataset of tags.
            Defaults to 'tags'.
        force (bool, optional): Overwrite the output file if it exists.
            Defaults to False.

    Raises:
        FileExistsError: If output file exists and `force=False`.
        ValueError: If length of `reps` does not equal length of `tags`,
            or if any samples therein have different sequence lengths.

    """
    if out.exists() and not force:
        raise FileExistsError(f'{out} exists, set force=True to overwrite')

    if len(reps) != len(tags):
        raise ValueError(f'got {len(tags)} samples in tags '
                         f'but {len(reps)} in reps')

    nsamples = 0
    for index in range(len(reps)):
        reps_length = reps.dataset.length(index)
        tags_length = len(tags[index])
        if reps_length != tags_length:
            raise ValueError(f'got {reps_length} reps but {tags_length} tags '
                             f'for sample {index}')
        nsamples += reps_length

    log = logging.getLogger(__name__)
    log.info('%d samples to collate, %d reps/tags total', len(reps), nsamples)

    with h5py.File(out, 'w') as handle:
        breaks_out = handle.create_dataset(breaks_key,
                                           shape=(len(reps),),
                                           dtype='i')
        reps_out = handle.create_dataset(
            reps_key,
            shape=(nsamples, reps.dataset.dimension),
            dtype='f',
        )
        tags_out = handle.create_dataset(
            tags_key,
            shape=(nsamples,),
            dtype='i',
        )
        start = 0
        for index in range(len(reps)):
            log.info('processing %d of %d', index + 1, len(reps))
            reps_curr = reps[index]
            tags_curr = tags[index]
            end = start + len(reps_curr)
            breaks_out[index] = start
            reps_out[start:end] = reps_curr
            tags_out[start:end] = tags_curr
            start = end
        assert end == nsamples, 'did not finish writing?'


def run(data: Dict[str, splits.Split],
        out: pathlib.Path,
        layers: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
        breaks_key: str = 'breaks',
        reps_key: str = 'reps',
        tags_key: str = 'tags',
        force: bool = False) -> None:
    """Preprocess a POS task from the given splits.

    Preprocessing involves collating a special h5 file for every split
    and every representation layer. See collate(...) above for how that
    works.

    Args:
        data (Dict[str, split.Split]): The dataset splits to process.
            Key is an identifier for the split, e.g. "train," and values
            are paths to the data itself. Mappings from POS to integer will
            be determined using all splits, but ultimately each split will be
            collated to a separate file.
        out (pathlib.Path): Path at which to write collated files.
        layers (Optional[Sequence[int]], optional): Only preprocess these
            representation layers. Defaults to all layers.
        tags (Optional[Sequence[str]]): Only distinguish these POS tags, and
            collapse the rest to a single UNK tag. See `POSTask.__init__`.
            Defaults to distinguishing all tags.
        breaks_key (str, optional): Key to use for the breaks dataset
            in the output h5 file. Defaults to 'breaks'.
        reps_key (str, optional): Key to use for the representations dataset
            in the output h5 file. Defaults to 'reps'.
        tags_key (str, optional): Same as above, for the dataset of tags.
            Defaults to 'tags'.
        force (bool): Overwrite existing collated h5 files. Otherwise die
            if existing files are found. Defaults to False.

    Raises:
        FileExistsError: If output directory exists and `force=False`.

    """
    if not force and out.exists():
        raise FileExistsError(f'{out} exists, set force=True to overwrite')

    log = logging.getLogger(__name__)

    samples, reps = {}, {}
    for key, split in data.items():
        annotations_path = split.annotations
        log.info('reading ptb %s set: %s', key, annotations_path)
        samples[key] = ptb.load(annotations_path)

        reps_path = split.representations
        log.info('reading reps %s set: %s', key, reps_path)
        reps[key] = representations.RepresentationDataset(reps_path)
    assert len({r.layers for r in reps.values()}) == 1, 'mismatched layers?'

    task = POSTask(tuple(itertools.chain(*samples.values())), tags=tags)
    assert len(task) > 1, 'degenerate task?'

    for key in data:
        split_reps = reps[key]
        split_tags = POSTagsDataset(samples[key], task)
        for layer in layers or range(split_reps.layers):
            out_h5 = out / str(layer) / f'{key}.h5'
            out_h5.parent.mkdir(parents=True, exist_ok=force)
            log.info('collating %s set, layer %d to %s', key, layer, out_h5)
            collate(split_reps.layer(layer),
                    split_tags,
                    out_h5,
                    breaks_key=breaks_key,
                    reps_key=reps_key,
                    tags_key=tags_key,
                    force=force)
