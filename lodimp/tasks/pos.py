"""Defines core experiments for part of speech tagging task."""

import collections
import copy
import logging
from typing import (Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple,
                    Type, Union)

from lodimp.common import datasets, learning, linalg
from lodimp.common.models import probes, projections
from lodimp.common.parse import ptb
from lodimp.common.parse import representations as reps

import numpy as np
import torch
import wandb

# Standard unknown symbol.
UNK = 'unk'

# Frequently used POS tags.
NOUNS = ('NN', 'NNS', 'NNP', 'NNPS')
NOUNS_PROPER = ('NNP', 'NNPS')
NOUNS_PLURAL = ('NNS', 'NNPS')
VERBS = ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
VERBS_PRESENT = ('VBZ', 'VBP', 'VBG')
VERBS_PAST = ('VBD', 'VBN')
ADJECTIVES = ('JJ', 'JJR', 'JJS')
ADVERBS = ('RB', 'RBR', 'RBS')


class POSIndexer:
    """Indexes PTB POS tags."""

    def __init__(self,
                 samples: Sequence[ptb.Sample],
                 distinguish: Optional[Sequence[str]] = None,
                 unk: str = UNK):
        """Maps each POS tag to an index.

        Args:
            samples (Sequence[ptb.PTBSample]): The samples from which to
                draw tags.
            distinguish (Optional[Sequence[str]]): The XPOS tags to
                distinguish. All tags not in this set will be collapsed to
                the unk tag. By default, all tags will be distinguished.
            unk (str): Tag to use when un-indexed XPOS is encountered.
                If distinguish is set, the tags not in that sequence will be
                set to this tag.

        """
        if distinguish is None:
            tags = {xpos for sample in samples for xpos in sample.xpos}
        else:
            tags = set(distinguish)

        self.indexer = {unk: 0}
        for xpos in sorted(tags):
            self.indexer[xpos] = len(self.indexer)
        self.unk = unk

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


class ControlPOSIndexer:
    """Maps words to arbitrary POS tags."""

    def __init__(self,
                 samples: Sequence[ptb.Sample],
                 dist: Optional[Sequence[float]] = None):
        """Initialize the tagger.

        The tagger computes the empirical distribution of the samples, if not
        provided, and then uses it to generate arbitrary integer tags for each
        individual word type.

        Args:
            samples (Sequence[ptb.PTBSample]): All samples for which to
                generate tags.
            dist (Optional[Sequence[float]], optional): The empirical
                distribution to use when sampling tags for word type.
                By default, is computed from the list of samples.

        """
        if dist is None:
            counts: Dict[str, int] = collections.defaultdict(lambda: 0)
            for sample in samples:
                for pos in sample.xpos:
                    counts[pos] += 1
            dist = np.array([float(count) for count in counts.values()])
            dist /= np.sum(dist)
        assert dist is not None, 'uninitialized distribution?'
        self.dist = dist

        self.tags: Dict[str, int] = {}
        for sample in samples:
            for word in sample.sentence:
                if word not in self.tags:
                    self.tags[word] = np.random.choice(len(dist), p=dist) + 1

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Tag the given samples.

        Args:
            sample (ptb.PTBSample): The sample to tag.

        Returns:
            torch.Tensor: Integer tags for every word in the sentence.
                If the word type is unknown, it's tag will be 0.

        """
        return torch.tensor(
            [self.tags.get(word, 0) for word in sample.sentence])

    def __len__(self) -> int:
        """Returns the number of fake tags in this task."""
        return len(self.dist) + 1  # add 1 for unk tag


class POSTaskDataset(datasets.TaskDataset):
    """Iterates over (word representation, POS tag) pairs."""

    def __init__(
        self,
        representations: reps.RepresentationLayerDataset,
        annotations: Sequence[ptb.Sample],
        indexer: Type[Union[POSIndexer, ControlPOSIndexer]] = POSIndexer,
        **kwargs: Any,
    ):
        """Maps each POS tag to an index.

        Keyword arguments are forwarded to indexer when instantiated.

        Args:
            representations (representations.RepresentationsLayerDataset): Word
                representations corresponding to the words to be tagged.
            annotations (Sequence[ptb.PTBSample]): The PTB annotations from
                which to pull POS tags.
            indexer (Type[Union[POSIndexer, ControlPOSIndexer]]): Type of
                indexer for mapping PTB annotations to integer tensors. Will
                be instantiated with given annotations unless `samples` is
                set in kwargs.

        Raises:
            ValueError: If number of representations/annotations do not match.

        """
        if len(representations) != len(annotations):
            raise ValueError(f'got {len(representations)} representations '
                             f'but {len(annotations)} annotations')

        self.representations = representations
        self.annotations = annotations

        kwargs = kwargs.copy()
        kwargs.setdefault('samples', annotations)
        self.indexer = indexer(**kwargs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (representations, integral POS tags) for index'th sentence.

        Args:
            index (int): Index of the sentence in the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First tensor is shape
                (sentence_length, representation_dimension) containing word
                representations, and second is shape (sentence_length,)
                containing integral POS tags.

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

    def count_unique_features(self) -> int:
        """Returns number of unique POS seen in data."""
        return len(self.indexer)


# Define the valid probe types for this task.
Probe = Union[probes.Linear, probes.MLP]


def train(train_dataset: datasets.TaskDataset,
          dev_dataset: datasets.TaskDataset,
          test_dataset: datasets.TaskDataset,
          probe_t: Type[Probe] = probes.Linear,
          project_to: int = 10,
          project_from: Optional[projections.Projection] = None,
          epochs: int = 25,
          patience: int = 4,
          lr: float = 1e-3,
          device: Optional[torch.device] = None,
          also_log_to_wandb: bool = False) -> Tuple[Probe, float]:
    """Train a probe on part of speech tagging.

    Args:
        train_dataset (datasets.TaskDataset): Training data.
        dev_dataset (datasets.TaskDataset): Validation data, used for early
            stopping.
        test_dataset (datasets.TaskDataset): Test data, used to compute final
            accuracy after training.
        probe_t (Type[Probe], optional): Probe type to train.
            Defaults to probes.Linear.
        project_to (int, optional): Project representations to this
            dimensionality. Defaults to 10.
        project_from (Optional[projections.Projection], optional): Project
            representations with this projection before applying the final
            projection, which will be the only one with learnable parameters.
            Defaults to None.
        epochs (int, optional): Maximum passes through the training dataset.
            Defaults to 25.
        patience (int, optional): Allow dev loss to not improve for this many
            epochs, then stop training. Defaults to 4.
        lr (float, optional): Learning rate for optimizer. Defaults to 1e-3.
        device (Optional[torch.device], optional): Torch device on which to
            train probe. Defaults to CPU.
        also_log_to_wandb (Optional[pathlib.Path], optional): If set, log
            training data to wandb. By default, wandb is not used.

    Returns:
        Tuple[Probe, float]: The trained probe and its test accuracy.

    """
    log = logging.getLogger(__name__)

    ndims = train_dataset.sample_representations_shape[-1]
    log.info('representations have dimension %d')

    ntags = train_dataset.count_unique_features()
    assert ntags is not None, 'no tag count, maybe dataset is for other task?'
    log.info('part of speech task has %d tags', ntags)

    proj = projections.Projection(ndims, project_to, compose=project_from)
    probe = probe_t(project_to, ntags, project=proj)
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


def axis_alignment(probe: Probe,
                   dev_dataset: datasets.TaskDataset,
                   test_dataset: datasets.TaskDataset,
                   device: Optional[torch.device] = None,
                   also_log_to_wandb: bool = False) -> Sequence[float]:
    """Measure whether the given probe is axis aligned.

    Args:
        probe (Probe): The probe to evaluate.
        dev_dataset (datasets.TaskDataset): Data used to determine which axes
            to cut.
        test_dataset (datasets.TaskDataset): Data used to determine the effect
            of cutting an axis.
        device (Optional[torch.device], optional): Torch device on which to
            train probe. Defaults to CPU.
        also_log_to_wandb (bool, optional): If set, log results to wandb.

    Returns:
        The sequence of accuracies obtained by ablating the least harmful
        axes, in order.

    """
    log = logging.getLogger(__name__)

    projection = probe.project
    assert projection is not None, 'no projection?'

    axes = set(range(projection.project.in_features))
    ablated: Set[int] = set()
    accuracies = []
    while axes:
        best_model, best_axis, best_accuracy = probe, -1, -1.
        for axis in axes:
            model = copy.deepcopy(best_model).eval()
            assert model.project is not None, 'no projection?'
            model.project.project.weight.data[:, sorted(ablated | {axis})] = 0
            accuracy = learning.test(model, dev_dataset, device=device)
            if accuracy > best_accuracy:
                best_model = model
                best_axis = axis
                best_accuracy = accuracy
        accuracy = learning.test(best_model, test_dataset, device=device)

        log.info('ablating axis %d, test accuracy %f', best_axis, accuracy)
        if also_log_to_wandb:
            wandb.log({
                'axis': best_axis,
                'dev accuracy': best_accuracy,
                'test accuracy': accuracy,
            })

        axes.remove(best_axis)
        ablated.add(best_axis)
        accuracies.append(accuracy)

    return accuracies


def inlp(train_dataset: datasets.TaskDataset,
         dev_dataset: datasets.TaskDataset,
         test_dataset: datasets.TaskDataset,
         rank: int = 10,
         attempts: int = 100,
         tolerance: float = 5e-2,
         epochs: int = 25,
         lr: float = 1e-3,
         device: Optional[torch.device] = None,
         also_log_to_wandb: bool = False) -> projections.Projection:
    """Compute the nullspace for all linear part of speech information.

    Applies the method from this paper: https://arxiv.org/abs/2004.07667

    Args:
        train_dataset (datasets.TaskDataset): Training data for the probe.
        dev_dataset (datasets.TaskDataset): Validation data for the probe.
        test_dataset (datasets.TaskDataset): Test data for evaluating probe
            accuracy after nullspace projection.
        rank (int, optional): Maximum rank of linear classifier.
            Achieved via LR factorization. Defaults to 10.
        attempts (int, optional): Maximum number of nullspace projections
            to compose before giving up. Defaults to 100.
        tolerance (float, optional): How close to chance accuracy we can
            get before giving up.
        epochs (int, optional): Number of passes through the training set
            for training each classifier. Defaults to 25.
        lr (float, optional): Learning rate for training each probe.
            Defaults to 1e-3.
        device (Optional[torch.device], optional): Torch device on which to
            train linear models. Defaults to CPU.
        also_log_to_wandb (bool, optional): If set, log results to wandb.

    Returns:
        projections.Projection: Projection onto the "part of speech" nullspace.

    """
    log = logging.getLogger(__name__)

    device = device or torch.device('cpu')

    ndims = train_dataset.sample_representations_shape[-1]
    log.info('representations have dimension %d')

    ntags = train_dataset.count_unique_features()
    assert ntags is not None, 'no tag count, maybe h5 file stores other task?'
    log.info('part of speech task has %d tags', ntags)

    # Cache some useful values.
    eye = torch.eye(ndims, device=device)
    zero = torch.zeros_like(eye)

    rowspaces: List[torch.Tensor] = []

    def get_nullspace_projection() -> torch.Tensor:
        """Returns the current nullspace projection."""
        return eye - linalg.rowspace(sum(rowspaces, zero))

    for attempt in range(attempts):
        nullspace = None
        if rowspaces:
            nullspace = projections.Projection(ndims, ndims)
            matrix = nullspace.project.weight.data
            matrix[:] = get_nullspace_projection()

        projection = projections.Projection(ndims, rank, compose=nullspace)
        classifier = probes.Linear(rank, ntags, project=projection)
        learning.train(classifier,
                       train_dataset,
                       dev_dataset=dev_dataset,
                       epochs=epochs,
                       lr=lr,
                       device=device)

        projection_matrix = projection.project.weight.data
        classifier_matrix = classifier.classify.weight.data
        rowspace = linalg.rowspace(classifier_matrix.mm(projection_matrix))
        rowspaces.append(rowspace)

        accuracy = learning.test(classifier, test_dataset, device=device)

        logging.info('attempt %d accuracy %f', attempt, accuracy)
        if also_log_to_wandb:
            wandb.log({'accuracy': accuracy})

        if accuracy < 1 / ntags + tolerance:
            break

    nullspace = projections.Projection(ndims, ndims)
    nullspace.project.weight.data[:] = get_nullspace_projection()
    return nullspace
