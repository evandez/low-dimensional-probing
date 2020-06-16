"""Core experiments for part of speech tagger probes."""

import copy
import logging
import pathlib
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, Union

from lodimp.common import learning, linalg, tasks
from lodimp.common.data import splits
from lodimp.common.models import probes, projections

import torch
import wandb

# Define the valid probe types for this task.
Probe = Union[probes.Linear, probes.MLP]


def load(
    data_path: pathlib.Path,
    data_splits: Sequence[str] = splits.STANDARD_SPLITS,
    batch: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, tasks.TaskDataset]:
    """Load task data.

    Args:
        data_path (pathlib.Path): Path to preprocessed task data.
        data_splits (Sequence[str], optional): Splits to load.
            Defaults to all standard splits (i.e. train, dev, test).
        batch (bool, optional): If true, batch the dataset by sentence.
            Otherwise, the data will be loaded into memory/GPU all at once.
            Defaults to True.
        device (Optional[torch.device], optional): When `batch=False`, send
            full dataset to this device. Defaults to CPU.

    Returns:
        Dict[str, tasks.TaskDataset]: Mapping from split name to dataset.

    """
    log = logging.getLogger(__name__)
    datasets: Dict[str, tasks.TaskDataset] = {}
    for split in splits.STANDARD_SPLITS:
        path = data_path / f'{split}.h5'
        if batch:
            log.info('loading task %s set from %s', split, path)
            datasets[split] = tasks.SentenceBatchingCollatedTaskDataset(
                path, device=device)
        else:
            log.info('loading and collating task %s set from %s', split, path)
            datasets[split] = tasks.NonBatchingCollatedTaskDataset(
                path, device=device)
    return datasets


def train(data_path: pathlib.Path,
          probe_t: Type[Probe] = probes.Linear,
          project_to: int = 10,
          project_from: Optional[projections.Projection] = None,
          epochs: int = 25,
          batch: bool = True,
          cache: bool = False,
          patience: int = 4,
          lr: float = 1e-3,
          device: Optional[torch.device] = None,
          also_log_to_wandb: bool = False) -> Tuple[Probe, float]:
    """Train a probe on part of speech tagging.

    Args:
        data_path (pathlib.Path): Path to preprocessed task data.
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
        batch (bool, optional): If true, batch the dataset by sentence.
            Otherwise, the data will be loaded into memory/GPU all at once.
            Defaults to True.
        cache (bool, optional): If true, load entire dataset onto memory/GPU
            before training. Defaults to False.
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

    datasets = load(data_path, batch=batch, device=device if cache else None)

    ndims = datasets[splits.TRAIN].sample_representations_shape[-1]
    log.info('representations have dimension %d')

    ntags = datasets[splits.TRAIN].count_unique_features()
    assert ntags is not None, 'no tag count, maybe h5 file stores other task?'
    log.info('part of speech task has %d tags', ntags)

    proj = projections.Projection(ndims, project_to, compose=project_from)
    probe = probe_t(project_to, ntags, project=proj)
    learning.train(probe,
                   datasets[splits.TRAIN],
                   dev_dataset=datasets[splits.DEV],
                   stopper=learning.EarlyStopping(patience=patience),
                   epochs=epochs,
                   lr=lr,
                   device=device,
                   also_log_to_wandb=also_log_to_wandb)
    accuracy = learning.test(probe, datasets[splits.TEST], device=device)
    return probe, accuracy


def axis_alignment(probe: Probe,
                   data_path: pathlib.Path,
                   batch: bool = True,
                   cache: bool = False,
                   device: Optional[torch.device] = None,
                   also_log_to_wandb: bool = False) -> Sequence[float]:
    """Measure whether the given probe is axis aligned.

    Args:
        probe (Probe): The probe to evaluate.
        data_path (pathlib.Path): Path to preprocessed task data.
        batch (bool, optional): If true, batch the dataset by sentence.
            Otherwise, the data will be loaded into memory/GPU all at once.
            Defaults to True.
        cache (bool, optional): If true, load entire dataset onto memory/GPU
            before training. Defaults to False.
        device (Optional[torch.device], optional): Torch device on which to
            train probe. Defaults to CPU.
        also_log_to_wandb (bool, optional): If set, log results to wandb.

    Returns:
        The sequence of accuracies obtained by ablating the least harmful
        axes, in order.

    """
    log = logging.getLogger(__name__)

    datasets = load(data_path,
                    data_splits=(splits.DEV, splits.TEST),
                    batch=batch,
                    device=device if cache else None)

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
            accuracy = learning.test(model, datasets[splits.DEV])
            if accuracy > best_accuracy:
                best_model = model
                best_axis = axis
                best_accuracy = accuracy
        accuracy = learning.test(best_model, datasets[splits.TEST])

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


def nullify(data_path: pathlib.Path,
            rank: int = 10,
            attempts: int = 100,
            tolerance: float = 5e-2,
            epochs: int = 25,
            batch: bool = True,
            cache: bool = False,
            lr: float = 1e-3,
            device: Optional[torch.device] = None,
            also_log_to_wandb: bool = False) -> projections.Projection:
    """Compute the nullspace for all linear part of speech information.

    This function iteratively computes the nullspace of part of speech

    Args:
        data_path (pathlib.Path): Path to preprocessed task data.
        rank (int, optional): Maximum rank of linear classifier.
            Achieved via LR factorization. Defaults to 10.
        attempts (int, optional): Maximum number of nullspace projections
            to compose before giving up. Defaults to 100.
        tolerance (float, optional): How close to chance accuracy we can
            get before giving up.
        epochs (int, optional): Number of passes through the training set
            for training each classifier. Defaults to 25.
        batch (bool, optional): If true, batch the dataset by sentence.
            Otherwise, the data will be loaded into memory/GPU all at once.
            Defaults to True.
        cache (bool, optional): If true, load entire dataset onto memory/GPU
            before training. Defaults to False.
        lr (float, optional): [description]. Defaults to 1e-3.
        device (Optional[torch.device], optional): Torch device on which to
            train linear models. Defaults to CPU.
        also_log_to_wandb (bool, optional): If set, log results to wandb.

    Returns:
        projections.Projection: Projection onto the "part of speech" nullspace.

    """
    log = logging.getLogger(__name__)

    device = device or torch.device('cpu')
    datasets = load(data_path,
                    data_splits=(splits.TRAIN, splits.TEST),
                    batch=batch,
                    device=device if cache else None)

    ndims = datasets[splits.TRAIN].sample_representations_shape[-1]
    log.info('representations have dimension %d')

    ntags = datasets[splits.TRAIN].count_unique_features()
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
                       datasets[splits.TRAIN],
                       epochs=epochs,
                       lr=lr,
                       device=device)

        projection_matrix = projection.project.weight.data
        classifier_matrix = classifier.classify.weight.data
        rowspace = linalg.rowspace(classifier_matrix.mm(projection_matrix))
        rowspaces.append(rowspace)

        accuracy = learning.test(classifier,
                                 datasets[splits.TEST],
                                 device=device)

        logging.info('attempt %d accuracy %f', attempt, accuracy)
        if also_log_to_wandb:
            wandb.log({'accuracy': accuracy})

        if accuracy < 1 / ntags + tolerance:
            break

    nullspace = projections.Projection(ndims, ndims)
    nullspace.project.weight.data[:] = get_nullspace_projection()
    return nullspace
