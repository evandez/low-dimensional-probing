"""Defines functions for training models."""
import logging
from typing import Optional

from lodimp.common import datasets
from lodimp.common.typing import Device

import torch
import wandb
from torch import nn, optim


class EarlyStopping:
    """Observes a numerical value and determines when it has not improved."""

    def __init__(self, patience: int = 3, decreasing: bool = True):
        """Initialize the early stopping tracker.

        Args:
            patience (int): Allow tracked value to not improve over its
                best value this many times. Defaults to 3.
            decreasing (bool, optional): If True, the tracked value "improves"
                if it decreases. If False, it "improves" if it increases.
                Defaults to True.

        """
        self.patience = patience
        self.decreasing = decreasing
        self.best = float('inf') if decreasing else float('-inf')
        self.num_bad = 0

    def __call__(self, value: float) -> bool:
        """Considers the new tracked value and decides whether to stop.

        Args:
            value (float): The new tracked value.

        Returns:
            bool: True if patience has been exceeded.

        """
        improved = self.decreasing and value < self.best
        improved |= not self.decreasing and value > self.best
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1

        return self.num_bad > self.patience


def train(probe: nn.Module,
          train_dataset: datasets.TaskDataset,
          dev_dataset: Optional[datasets.TaskDataset] = None,
          stopper: Optional[EarlyStopping] = None,
          device: Optional[Device] = None,
          lr: float = 1e-3,
          epochs: int = 25,
          also_log_to_wandb: bool = False) -> None:
    """Train a probe on the given data.

    The probe is always trained against cross entropy and optimized with Adam
    using the default hyperparameters, bar learning rate, which can be set by
    the caller.

    This function makes only three assumptions:
    (1) Probe takes one tensor as input and produces one tensor as output.
    (2) Probe's outputs are logits that can be passed as the first argument
        to `torch.nn.CrossEntropyLoss`.
    (3) Inputs are paired with integral tensors that can be directly passed as
        the second argument to the loss above.

    Args:
        probe (nn.Module): The model to train.
        train_dataset (datasets.TaskDataset): The data on which to train.
            Iterates are (tensor, tensor) pairs, the former being model
            inputs that will be fed directly to the probe, and the latter
            integral tags that will be fed directly to the cross entropy loss.
            These tensors will not be reshaped in any way; the probe will have
            to do that internally.
        dev_dataset (Optional[datasets.TaskDataset], optional): Same format as
            `train_dataset`. If set, probe will be evaluated on this dataset
            after every epoch. Defaults to None.
        stopper (Optional[EarlyStopping], optional): If set, track the loss
            value and end training when patience is exceeded. Dev loss is used
            if `dev_data` is set, otherwise uses training loss.
            Defaults to None.
        device (Optional[Device], optional): Send probe, loss function,
            and all tensors in dataset to this device throughout training.
            By default, device is not changed on any module or tensor.
        lr (float, optional): Learning rate for Adam optimization.
            Defaults to 1e-3.
        epochs (int, optional): Maximum number of passes to make through
            `train_data`. Defaults to 25.
        also_log_to_wandb (bool, optional): Log train and, if applicable, dev
            losses to wandb. Defaults to False.

    """
    log = logging.getLogger(__name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    if device:
        probe = probe.to(device)
        criterion = criterion.to(device)

    for epoch in range(epochs):
        probe.train()
        for iteration, (inputs, tags) in enumerate(train_dataset):
            if device:
                inputs, tags = inputs.to(device), tags.to(device)

            optimizer.zero_grad()
            preds = probe(inputs)
            loss = criterion(preds, tags)
            loss.backward()
            optimizer.step()

            log.info('epoch %d batch %d train loss %f', epoch + 1,
                     iteration + 1, loss.item())

            if also_log_to_wandb:
                wandb.log({'train accuracy': loss})

            if not dev_dataset and stopper and stopper(loss.item()):
                log.info('patience on train loss exceed, training is now over')
                return

        if dev_dataset is not None:
            probe.eval()
            dev_loss, count = 0., 0
            for inputs, tags in dev_dataset:
                if device is not None:
                    inputs, tags = inputs.to(device), tags.to(device)
                preds = probe(inputs)
                dev_loss += criterion(preds, tags).item() * len(inputs)
                count += len(inputs)
            dev_loss /= count

            log.info('epoch %d dev loss %f', epoch + 1, dev_loss)

            if also_log_to_wandb:
                # Technically this creates a new step, when it should use the
                # same step as the last train loss log event. It irks me, but
                # oh well.
                wandb.log({'dev accuracy': dev_loss})

            if stopper and stopper(dev_loss):
                log.info('patience on dev loss exceed, training is now over')
                return


def test(probe: nn.Module,
         dataset: datasets.TaskDataset,
         device: Optional[Device] = None) -> float:
    """Compute classification accuracy of a probe on the given data.

    This function makes the same assumptions as `train` above. Importantly,
    it assumes the last dimension of the probes outputs are logits, and will
    take the argmax across that dimension to determine class predictions.

    Args:
        probe (nn.Module): The classifier probe.
        dataset (datasets.TaskDataset): Inputs to classify paired with labels.
            See `train_dataset` parameter in `train`.
        device (Optional[Device], optional): Probe, inputs, and tags
            will be sent to this device. By default, device is not changed
            on any module or tensor.

    Raises:
        ValueError: If `dataset` is empty.

    Returns:
        float: Fraction of correctly classified data points.

    """
    if device:
        probe.to(device)

    total, count = 0, 0
    for inputs, tags in dataset:
        if device:
            inputs, tags = inputs.to(device), tags.to(device)
        count += len(inputs)
        with torch.no_grad():
            total += probe(inputs).argmax(dim=-1).eq(tags).sum().item()
    assert total <= count, 'more correct than counted?'

    # There should never be a case where we hand this function an empty
    # dataset, so explode if that is the case.
    if not count:
        raise ValueError('no data in dataset')

    return total / count
