"""Core experiments for dependency arc prediction probes."""

import copy
import logging
import pathlib
from typing import Dict, Optional, Sequence, Set, Tuple, Type, Union

from lodimp.common import learning
from lodimp.common.data import splits
from lodimp.common.models import probes, projections

import torch
import wandb

Probe = Union[probes.PairwiseBilinear, probes.PairwiseMLP]


def load(
    data_path: pathlib.Path,
    data_splits: Sequence[str] = splits.STANDARD_SPLITS,
    device: Optional[torch.device] = None,
) -> Dict[str, learning.TaskDataset]:
    """Load task data.

    Args:
        data_path (pathlib.Path): Path to preprocessed task data.
        data_splits (Sequence[str], optional): Splits to load.
            Defaults to all standard splits (i.e. train, dev, test).
        device (Optional[torch.device], optional): Send full dataset
            to this device. Defaults to CPU.

    Returns:
        Dict[str, learning.TaskDataset]: Mapping from split name to dataset.

    """
    log = logging.getLogger(__name__)
    datasets: Dict[str, learning.TaskDataset] = {}
    for split in splits.STANDARD_SPLITS:
        path = data_path / f'{split}.h5'
        log.info('loading task %s set from %s', split, path)
        datasets[split] = learning.SentenceBatchingTaskDataset(path,
                                                               device=device)
    return datasets


def train(data_path: pathlib.Path,
          probe_t: Type[Probe] = probes.PairwiseBilinear,
          project_to: int = 10,
          share_projection: bool = False,
          epochs: int = 25,
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
        share_projection (bool): If set, project the left and right components
            of pairwise probes with the same projection. E.g. if the probe is
            bilinear of the form xAy, we will always compute (Px)A(Py) as
            opposed to (Px)A(Qy) for distinct projections P, Q. Defaults to NOT
            shared.
        epochs (int, optional): Maximum passes through the training dataset.
            Defaults to 25.
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

    device = device or torch.device('cpu')
    datasets = load(data_path, device=device if cache else None)

    ndims = datasets[splits.TRAIN].dimension
    log.info('representations have dimension %d')

    if share_projection:
        projection = projections.PairwiseProjection(
            projections.Projection(ndims, project_to))
    else:
        projection = projections.PairwiseProjection(
            projections.Projection(ndims, project_to),
            right=projections.Projection(ndims, project_to))

    probe = probe_t(project_to, project=projection)
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
                   cache: bool = False,
                   device: Optional[torch.device] = None,
                   also_log_to_wandb: bool = False) -> Sequence[float]:
    """Measure whether the given probe is axis aligned.

    Args:
        probe (Probe): The probe to evaluate.
        data_path (pathlib.Path): Path to preprocessed task data.
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

    device = device or torch.device('cpu')
    datasets = load(data_path,
                    data_splits=(splits.DEV, splits.TEST),
                    device=device if cache else None)

    projection = probe.project
    assert projection is not None, 'no projection?'
    shared = projection.right is None

    multiplier = 1 if shared else 2
    axes = set(range(multiplier * projection.in_features))
    ablated: Set[int] = set()
    accuracies = []
    while axes:
        best_model, best_axis, best_accuracy = probe, -1, -1.
        for axis in axes:
            model = copy.deepcopy(best_model)
            assert model.project is not None, 'no projection?'

            indices = ablated | {axis}
            if shared:
                # If we are sharing projections, then ablating the left also
                # ablates the right. Easy!
                model.project.left.project.weight.data[:, sorted(indices)] = 0
            else:
                # If we are not sharing projections, then the "left" and
                # "right" projections contain disjoint axes. we have to
                # manually determine which axis belongs to which projection.
                coordinates = {(i // projection.in_features,
                                i % projection.in_features) for i in indices}
                lefts = {ax for (proj, ax) in coordinates if not proj}
                rights = {ax for (proj, ax) in coordinates if proj}
                assert len(lefts) + len(rights) == len(indices), 'bad mapping?'
                model.project.left.project.weight.data[:, sorted(lefts)] = 0
                assert model.project.right is not None, 'null right proj?'
                model.project.right.project.weight.data[:, sorted(rights)] = 0

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
