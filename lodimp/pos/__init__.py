"""Core experiments for part of speech tagger probes."""

import logging
import pathlib
from typing import Dict, Optional, Type, Union

from lodimp.common import learning
from lodimp.common.data import splits
from lodimp.common.models import probes, projections

import torch

# Define the valid probe types for this task.
Probe = Union[probes.Linear, probes.MLP]


def train(data: pathlib.Path,
          probe_t: Type[Probe] = probes.Linear,
          project_to: int = 10,
          project_from: Optional[projections.Projection] = None,
          epochs: int = 25,
          batch: bool = True,
          patience: int = 4,
          lr: float = 1e-3,
          device: Optional[torch.device] = None,
          also_log_to_wandb: bool = False) -> Probe:
    """Train a probe on part of speech tagging.

    Args:
        data (pathlib.Path): Path to preprocessed task data.
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
        patience (int, optional): Allow dev loss to not improve for this many
            epochs, then stop training. Defaults to 4.
        lr (float, optional): Learning rate for optimizer. Defaults to 1e-3.
        device (Optional[torch.device], optional): Torch device on which to
            train probe. Defaults to CPU.
        also_log_to_wandb (bool, optional): If set, log training metrics
            to wandb. Defaults to False.

    Returns:
        Probe: The trained probe.

    """
    log = logging.getLogger(__name__)

    datasets: Dict[str, learning.TaskDataset] = {}
    for split in splits.STANDARD_SPLITS:
        path = data / f'{split}.h5'
        if batch:
            log.info('loading task %s set from %s', split, path)
            datasets[split] = learning.SentenceIterableTaskDataset(path)
        else:
            log.info('loading and collating task %s set from %s', split, path)
            datasets[split] = learning.InMemoryTaskDataset(path, device=device)

    ndims = datasets[splits.TRAIN].dimension
    log.info('representations have dimension %d')

    ntags = datasets[splits.TRAIN].tags.attrs.get('ntags')
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
    return probe
