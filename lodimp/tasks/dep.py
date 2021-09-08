"""Core experiments for dependency edge prediction task."""

import logging
import random
from typing import Any, Iterator, Optional, Sequence, Set, Tuple, Type, Union

from lodimp.common import datasets, learning
from lodimp.common.models import probes, projections
from lodimp.common.parse import ptb
from lodimp.common.parse import representations as reps
from lodimp.common.typing import Device

import torch


class DEPIndexer:
    """Maps dependents to heads."""

    def __init__(self, **kwargs: Any):
        """Does nothing, only exists for sake of type checking."""

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map dependents to heads.

        The label for word w is the index of word v in the sentence if word v
        is the head of word w. Note that if w is the root, it's head is w.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, this returns a length W
                tensor containing the index of its head.

        """
        return torch.tensor([
            head if head != -1 else word
            for word, head in enumerate(sample.heads)
        ])


class ControlDEPIndexer:
    """Constructs arbitrary parse "trees" for all samples."""

    def __init__(self, samples: Sequence[ptb.Sample]):
        """Map each word type to a dependency arc behavior.

        We sample uniformly from three behaviors:
        - Always attach word to itself.
        - Always attach word to first word in sentence.
        - Always attach word to last word in sentence.

        Args:
            samples (Sequence[ptb.PTBSample]): All samples for which to
                generate tags.

        """
        self.attach_to_self: Set[str] = set()
        self.attach_to_first: Set[str] = set()
        self.attach_to_last: Set[str] = set()
        behaviors = (self.attach_to_self, self.attach_to_first,
                     self.attach_to_last)

        for sample in samples:
            for word in sample.sentence:
                if not any(word in behavior for behavior in behaviors):
                    random.choice(behaviors).add(word)

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map dependents to (fake) heads.

        Same format as DependencyArcTask, but labels are assigned according
        to one of the three behaviors described in the constructor.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, this returns a length W
                tensor containing the index of its "head."

        Raises:
            ValueError: If word was not seen during initialization.

        """
        labels = []
        for index, word in enumerate(sample.sentence):
            if word in self.attach_to_first:
                labels.append(0)
            elif word in self.attach_to_self:
                labels.append(index)
            elif word in self.attach_to_last:
                length = len(sample.sentence)
                if length == 1:
                    labels.append(0)
                elif sample.xpos[-1] == 'PUNCT':
                    assert sample.xpos[-2] != 'PUNCT', 'double punctuation?'
                    labels.append(length - 2)
                else:
                    labels.append(length - 1)
            else:
                raise ValueError(f'unknown word: {word}')
        return torch.tensor(labels, dtype=torch.long)


class DEPTaskDataset(datasets.TaskDataset):
    """Iterates over (word representation, index of head) pairs."""

    def __init__(
        self,
        representations: reps.RepresentationLayerDataset,
        annotations: Sequence[ptb.Sample],
        indexer: Type[Union[DEPIndexer, ControlDEPIndexer]] = DEPIndexer,
        **kwargs: Any,
    ):
        """Initialize the task dataset.

        All kwargs are forwarded to the indexer upon instantiation.

        Args:
            representations (representations.RepresentationsLayerDataset): Word
                representations corresponding to the words to be tagged.
            annotations (Sequence[ptb.PTBSample]): The PTB annotations from
                which to pull head indices.
            indexer (Type[Union[DEPIndexer, ControlDEPIndexer]]): Type of the
                indexer to use for mapping PTB annotations to integer tensors.
                If set to ControlDEPIndexer, annotations will be passed as the
                `samples` argument unless that parameter is set in kwargs.

        Raises:
            ValueError: If number of representations/annotations do not match.

        """
        if len(representations) != len(annotations):
            raise ValueError(f'got {len(representations)} representations '
                             f'but {len(annotations)} annotations')

        self.representations = representations
        self.annotations = annotations

        kwargs = kwargs.copy()
        if indexer is ControlDEPIndexer:
            kwargs.setdefault('samples', annotations)
        self.indexer = indexer(**kwargs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (representations, head indices) for index'th sentence.

        Args:
            index (int): Index of the sentence in the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First tensor is shape
                (sentence_length, representation_dimension) containing word
                representations, and second is shape (sentence_length,)
                containing integral head indices. The indices will range
                between (-1, sentence_length - 1), and an index of -1 means
                the word is attached to itself.

        """
        representations = self.representations[index]
        annotations = self.annotations[index]
        assert len(representations) == len(
            annotations.sentence), 'diff sentence lengths?'
        return representations, self.indexer(annotations)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields all (sentence representations, sentence POS tags) samples."""
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        """Returns the number of sentences (batches) in the dataset."""
        return len(self.annotations)

    @property
    def sample_representations_shape(self) -> Sequence[int]:
        """Returns the dimensionality of individual representations."""
        return (self.representations.dataset.dimension,)

    @property
    def sample_features_shape(self) -> Sequence[int]:
        """Returns the shape of each individual POS tag.

        Since POS tags are integral scalars, there is no such shape!
        """
        return ()

    def count_samples(self) -> int:
        """Returns the number of words in the dataset."""
        return sum(
            self.representations.dataset.length(index)
            for index in range(len(self.representations)))

    def count_unique_features(self) -> Optional[int]:
        """Returns number of unique POS seen in data."""
        return None


Probe = Union[probes.PairwiseBilinear, probes.PairwiseMLP]


def train(train_dataset: datasets.TaskDataset,
          dev_dataset: datasets.TaskDataset,
          test_dataset: datasets.TaskDataset,
          probe_t: Type[Probe] = probes.PairwiseBilinear,
          project_to: Optional[int] = None,
          share_projection: bool = False,
          epochs: int = 25,
          patience: int = 4,
          lr: float = 1e-3,
          device: Optional[Device] = None,
          also_log_to_wandb: bool = False) -> Tuple[Probe, float]:
    """Train a probe on dependency edge prediction.

    Args:
        train_dataset (datasets.TaskDataset): Training data for probe.
        dev_dataset (datasets.TaskDataset): Validation data for probe, used for
            early stopping.
        test_dataset (datasets.TaskDataset): Test data for probe, used to
            compute final accuracy after training.
        probe_t (Type[Probe], optional): Probe type to train.
            Defaults to probes.Linear.
        project_to (Optional[int], optional): Project representations to this
            dimensionality. Defaults to no projection.
        share_projection (bool): If set, project the left and right components
            of pairwise probes with the same projection. E.g. if the probe is
            bilinear of the form xAy, we will always compute (Px)A(Py) as
            opposed to (Px)A(Qy) for distinct projections P, Q. Defaults to NOT
            shared.
        epochs (int, optional): Maximum passes through the training dataset.
            Defaults to 25.
        patience (int, optional): Allow dev loss to not improve for this many
            epochs, then stop training. Defaults to 4.
        lr (float, optional): Learning rate for optimizer. Defaults to 1e-3.
        device (Optional[Device], optional): Torch device on which to
            train probe. Defaults to CPU.
        also_log_to_wandb (Optional[pathlib.Path], optional): If set, log
            training data to wandb. By default, wandb is not used.

    Returns:
        Tuple[Probe, float]: The trained probe and its test accuracy.

    """
    log = logging.getLogger(__name__)

    device = device or 'cpu'

    ndims = train_dataset.sample_representations_shape[-1]
    log.info('representations have dimension %d', ndims)

    if project_to is None or project_to == ndims:
        logging.info('projection dim = reps dim, not projecting')
        projection = None
    elif share_projection:
        projection = projections.PairwiseProjection(
            projections.Projection(ndims, project_to))
    else:
        projection = projections.PairwiseProjection(
            projections.Projection(ndims, project_to),
            right=projections.Projection(ndims, project_to))

    probe = probe_t(project_to or ndims, project=projection)
    learning.train(probe,
                   train_dataset,
                   dev_dataset=dev_dataset,
                   stopper=learning.EarlyStopping(patience=patience),
                   epochs=epochs,
                   lr=lr,
                   device=device,
                   also_log_to_wandb=also_log_to_wandb)
    accuracy = learning.test(probe, test_dataset, device=device)
    return probe, accuracy
