"""Core experiments for the dependency label prediction task."""
import collections
import copy
import logging
from typing import (Any, Dict, Iterator, Optional, Sequence, Set, Tuple, Type,
                    Union)

from lodimp import datasets, learning
from lodimp.models import probes, projections
from lodimp.parse import ptb
from lodimp.parse import representations as reps
from lodimp.utils.typing import Device

import numpy
import torch
import wandb

UNK = 'unk'


class DLPIndexer:
    """Map pairs of words to their syntactic relationship, if any."""

    def __init__(self, samples: Sequence[ptb.Sample], unk: str = UNK):
        """Map each relation label to an integer.

        Args:
            samples (Sequence[ptb.Sample]): The samples from which to determine
                possible relations.
            unk (str): Label to use when un-indexed dependency label is
                encountered.

        """
        labels = {rel for sample in samples for rel in sample.relations}
        self.indexer = {unk: 0}
        for label in sorted(labels):
            self.indexer[label] = len(self.indexer)
        self.unk = unk

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map all possible (word, word) pairs to labels.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, returns shape (W, W) matrix
                where element (v, w) is the index of the label describing
                the relationship between word v and w, if any. Defaults to
                the "unk" label, even if there is no relationship between
                v and w.

        """
        heads, relations = sample.heads, sample.relations
        labels = torch.empty(len(heads), len(heads), dtype=torch.long)
        labels.fill_(self.indexer[self.unk])
        for word, (head, rel) in enumerate(zip(heads, relations)):
            if head == -1:
                labels[word, word] = self.indexer[rel]
            else:
                label = self.indexer.get(rel, self.indexer[self.unk])
                labels[word, head] = label
        return labels

    def __len__(self) -> int:
        """Return the number of unique labels for this task."""
        return len(self.indexer)


class ControlDLPIndexer:
    """Map pairs of words to arbitrary syntactic relationships."""

    def __init__(self,
                 samples: Sequence[ptb.Sample],
                 dist: Optional[Union[numpy.ndarray, Sequence[float]]] = None):
        """Map each relation label to an arbitrary (integer) label.

        We only do this for pairs of words which have a head-dependent
        relationship in the original dataset.

        Args:
            samples (Sequence[ptb.Samples]): The samples from which to pull
                possible word pairs.
            dist (Optional[Union[numpy.ndarray, Sequence[float]]], optional): A
                distribution to use when sampling tags per word type.
                By default, is computed from the list of samples.

        """
        if dist is None:
            counts: Dict[str, int] = collections.defaultdict(lambda: 0)
            for sample in samples:
                for relation in sample.relations:
                    counts[relation] += 1
            dist = numpy.array([float(count) for count in counts.values()])
            dist /= numpy.sum(dist)
        assert dist is not None, 'uninitialized distribution?'
        self.dist = dist

        self.rels: Dict[Tuple[str, str], int] = {}
        for sample in samples:
            sentence = sample.sentence
            heads = sample.heads
            for dep, head in enumerate(heads):
                if head == -1:
                    head = dep
                words = (sentence[dep], sentence[head])
                if words not in self.rels:
                    # Add one so that 0 is reserved for "no relationship" tag.
                    rel = numpy.random.choice(len(dist), p=dist) + 1
                    self.rels[words] = rel

    def __call__(self, sample: ptb.Sample) -> torch.Tensor:
        """Map all possible (word, word) pairs to labels.

        Args:
            sample (ptb.Sample): The sample to label.

        Returns:
            torch.Tensor: For length W sentence, returns shape (W, W) matrix
                where element (v, w) is the index of the label describing
                the relationship between word v and w, if any. Defaults to
                the "unk" label, even if there is no relationship between
                v and w.

        """
        heads = sample.heads
        labels = torch.zeros(len(heads), len(heads), dtype=torch.long)
        for dep, head in enumerate(heads):
            if head == -1:
                head = dep
            words = (sample.sentence[dep], sample.sentence[head])
            labels[dep, head] = self.rels.get(words, 0)
        return labels

    def __len__(self) -> int:
        """Return the number of relationships, including the null one."""
        return len(self.dist) + 1


class DLPTaskDataset(datasets.TaskDataset):
    """Iterate over (word representation pair, dependency label) pairs."""

    def __init__(
        self,
        representations: reps.RepresentationLayerDataset,
        annotations: Sequence[ptb.Sample],
        indexer: Type[Union[DLPIndexer, ControlDLPIndexer]] = DLPIndexer,
        **kwargs: Any,
    ):
        """Initialize dataset by mapping each dependency label to an index.

        The kwargs are forwarded to indexer when it is instantiated.

        Args:
            representations (representations.RepresentationsLayerDataset): Word
                representations corresponding to the words to be paired and
                labeled.
            annotations (Sequence[ptb.PTBSample]): The PTB annotations from
                which to pull dependency labels.
            indexer (Union[DLPIndexer, ControlDLPIndexer]): Type of the indexer
                to use for mapping PTB dependency label annotations to integer
                tensors. Instantiated with given annotations unless the
                samples keyword is set in kwargs.

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
        rels = self.indexer(annotations)

        # Find all pairs of words sharing an edge.
        indexes = set(range(len(representations)))
        pairs = [(i, j) for i in indexes for j in indexes if rels[i, j]]
        assert pairs and len(pairs) == len(representations), 'missing edges?'

        # Stack everything before returning it.
        bigrams = torch.stack([
            torch.stack((representations[i], representations[j]))
            for i, j in pairs
        ])
        labels = torch.stack([rels[i, j] for i, j in pairs])

        return bigrams, labels

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield all (sentence representations, sentence POS tags) samples."""
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        """Return the number of sentences (batches) in the dataset."""
        return len(self.annotations)

    @property
    def sample_representations_shape(self) -> Sequence[int]:
        """Return the dimensionality of the representation pairs."""
        return (2, self.representations.dataset.dimension)

    @property
    def sample_features_shape(self) -> Sequence[int]:
        """Return the shape of each individual POS tag.

        Since POS tags are integral scalars, there is no such shape!
        """
        return ()

    def count_samples(self) -> int:
        """Return the number of words in the dataset."""
        return sum(
            self.representations.dataset.length(index)
            for index in range(len(self.representations)))

    def count_unique_features(self) -> int:
        """Return number of unique POS seen in data."""
        return len(self.indexer)


# Define the valid probe types for this task.
Probe = Union[probes.Linear, probes.MLP]


def train(train_dataset: datasets.TaskDataset,
          dev_dataset: datasets.TaskDataset,
          test_dataset: datasets.TaskDataset,
          probe_t: Type[Probe] = probes.Linear,
          project_to: Optional[int] = None,
          share_projection: bool = False,
          epochs: int = 25,
          patience: int = 4,
          lr: float = 1e-3,
          device: Optional[Device] = None,
          also_log_to_wandb: bool = False) -> Tuple[Probe, float]:
    """Train a probe on dependency label prediction.

    Args:
        train_dataset (TaskDataset): Training data for probe.
        dev_dataset (TaskDataset): Validation data for probe, used for early
            stopping.
        test_dataset (TaskDataset): Test data for probe, used to compute
            final accuracy after training.
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

    ntags = train_dataset.count_unique_features()
    assert ntags is not None, 'no label count, is dataset for different task?'
    log.info('dependency labeling task has %d tags', ntags)

    if project_to is None or ndims == project_to:
        logging.info('projection dim = reps dim, not projecting')
        projection = None
    elif share_projection:
        projection = projections.Projection(ndims, project_to)
    else:
        projection = projections.Projection(2 * ndims, 2 * project_to)

    probe = probe_t(2 * (project_to or ndims), ntags, project=projection)
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


# TODO(evandez): May as well commonize this, since it's shared with POS.
def axis_alignment(probe: Probe,
                   dev_dataset: datasets.TaskDataset,
                   test_dataset: datasets.TaskDataset,
                   device: Optional[Device] = None,
                   also_log_to_wandb: bool = False) -> Sequence[float]:
    """Measure whether the given probe is axis aligned.

    Args:
        probe (Probe): The probe to evaluate.
        dev_dataset (datasets.TaskDataset): Data used to determine which axes
            to cut.
        test_dataset (datasets.TaskDataset): Data used to determine the effect
            of cutting an axis.
        device (Optional[Device], optional): Torch device on which to
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
