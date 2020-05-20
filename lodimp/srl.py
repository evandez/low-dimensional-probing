"""Preprocess Ontonotes for SRL.

TODO(evandez): This script is a hastily-made disaster.
Need to fold this into the rest of the codebase. Right now, everything in
this script is a special case, so pay attention, dear reader.
"""

import argparse
import copy
import itertools
import logging
import pathlib
import sys
from typing import Dict, Iterator, Optional, Sequence, Set, Tuple, Union

from lodimp import datasets, models, ontonotes

import torch
import wandb
from torch import nn, optim
from torch.utils import data

parser = argparse.ArgumentParser(description='Preprocess Ontonotes for SRL.')
parser.add_argument('data', type=pathlib.Path, help='Path to data.')
parser.add_argument('--layer', type=int, default=0, help='ELMo layer.')
parser.add_argument('--dimension',
                    type=int,
                    default=64,
                    help='Projection dimensionality.')
parser.add_argument('--probe',
                    choices=('bilinear', 'mlp'),
                    default='bilinear',
                    help='Probe architecture.')
parser.add_argument('--share-projection',
                    action='store_true',
                    help='When comparing reps, project both with same matrix.')
parser.add_argument('--no-batch', action='store_true', help='Do not batch.')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='Passes to make through dataset during training.')
parser.add_argument('--l1', type=float, help='Add L1 norm penalty.')
parser.add_argument('--nuc', type=float, help='Add nuclear norm penalty')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('--patience',
                    type=int,
                    default=4,
                    help='Epochs for dev loss to decrease to stop training.')
parser.add_argument('--ablate',
                    action='store_true',
                    help='Also test axis-ablated projection.')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device.')
parser.add_argument('--wandb-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp',
                    help='Path to write Weights and Biases data.')
parser.add_argument('--wandb-group', help='Experiment group.')
parser.add_argument('--wandb-name', help='Experiment name.')
parser.add_argument('--model-dir',
                    type=pathlib.Path,
                    default='/tmp/lodimp/models',
                    help='Directory to write finished model.')
parser.add_argument('--model-file',
                    default='probe.pth',
                    help='Model file name.')
parser.add_argument('--quiet',
                    dest='log_level',
                    action='store_const',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='Only show warning or error messages.')
options = parser.parse_args()

# Initialize wandb so we can track any failures right away.
options.wandb_dir.mkdir(parents=True, exist_ok=True)
wandb.init(project='lodimp',
           name=options.wandb_name,
           group=options.wandb_group,
           config={
               'task': 'srl',
               'representations': {
                   'model': 'elmo',
                   'layer': options.layer,
               },
               'projection': {
                   'dimension': options.dimension,
               },
               'probe': {
                   'model': options.probe,
               },
               'hyperparameters': {
                   'epochs': options.epochs,
                   'batched': not options.no_batch,
                   'lr': options.lr,
                   'patience': options.patience,
                   'regularization': {
                       'l1': options.l1,
                       'nuc': options.nuc,
                   },
               },
           },
           dir=str(options.wandb_dir))

if not options.data.exists():
    raise FileNotFoundError(f'task directory does not exist: {options.data}')

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

device = torch.device('cuda') if options.cuda else torch.device('cpu')
logging.info('using %s', device.type)

options.model_dir.mkdir(parents=True, exist_ok=True)
logging.info('model(s) will be written to %s', options.model_dir)

annotations, elmos = {}, {}
for split in ('train', 'dev', 'test'):
    conll = options.data / f'ontonotes5-wsj-{split}.conll'
    logging.info('reading ontonotes %s set from %s', split, conll)
    annotations[split] = ontonotes.load(conll)

    elmo = options.data / f'raw.{split}.elmo-layers.hdf5'
    logging.info('reading elmo %s set from %s', split, elmo)
    elmos[split] = datasets.ELMoRepresentationsDataset(elmo, options.layer)


class SemanticRoleLabelingTask:
    """Maps words to labels in one or more semantic role parses.

    Typically there is one semantic role parse per verb in the sentence.
    """

    def __init__(self, *samples: Sequence[ontonotes.Sample]):
        """Construct the semantic role labeling task.

        Args:
            samples (Sequence[ontonotes.Sample]): The samples from which
                to determine the list of all tags.

        """
        self.indexer: Dict[str, int] = {}
        for sample in itertools.chain(*samples):
            for role in sample.roles:
                for label in role:
                    if label not in self.indexer:
                        self.indexer[label] = len(self.indexer)

    def __call__(
        self,
        sample: ontonotes.Sample,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert role labels to integer tensor.

        Args:
            sample (ontonotes.Sample): The sample to label.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: If sample has N words
                and R role labelings, first tensor is shape (R,)
                tensor containing indices of theme words, and second is a
                shape (N, R) integer tensor where element (n, r)
                is the label for the r'th role parse of the n'th word.

        """
        themes, labels = [], []
        for labeling in sample.roles:
            labels.append([self.indexer[label] for label in labeling])
            if 'V' in labeling:
                themes.append(labeling.index('V'))
            else:
                # If there is no verb, this must be a WSJ sample that has no
                # SRLs. (There are several in the dataset.) We assert this
                # is the case for sanity and then set the theme to the first
                # word, because the label will be ignored by the loss anyway.
                assert len(sample.roles) == 1, 'not only one labeling?'
                assert set(labeling) == {ontonotes.IGNORE}, 'no ignored label?'
                themes.append(0)
        return torch.tensor(themes), torch.tensor(labels)

    def __len__(self) -> int:
        """Returns the number of unique tags for this SRL task."""
        return len(self.indexer)


task = SemanticRoleLabelingTask(*tuple(annotations.values()))


class SemanticRoleLabelingDataset(data.IterableDataset):
    """Simple wrapper around representations and annotations data."""

    def __init__(self, reps: datasets.ELMoRepresentationsDataset,
                 samples: Sequence[ontonotes.Sample]):
        """Initialize and preprocess the data.

        Args:
            reps (datasets.ELMoRepresentationsDataset): Dataset of reps.
            samples (Sequence[ontonotes.Sample]): Annotations for reps.

        """
        self.reps = reps
        self.themes = []
        self.labels = []
        for sample in samples:
            themes, labels = task(sample)
            self.themes.append(themes)
            self.labels.append(labels)

        self.cache = None
        if options.no_batch:
            self.cache = [reps[index].to(device) for index in range(len(reps))]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Yields a (reps, themes, roles) tuple for each sentence."""
        for index in range(len(self)):
            if self.cache is not None:
                rep = self.cache[index]
            else:
                rep = self.reps[index]
            yield rep, self.themes[index], self.labels[index]

    def __len__(self) -> int:
        """Returns the number of samples (sentences) in the dataset."""
        return len(self.reps)


loaders = {}
for split in ('train', 'dev', 'test'):
    loaders[split] = SemanticRoleLabelingDataset(elmos[split],
                                                 annotations[split])


class Bilinear(nn.Module):
    """A bilinear probe."""

    def __init__(self,
                 left_features: int,
                 right_features: int,
                 out_features: int,
                 project: Optional[models.PairwiseProjection] = None):
        """Initialize the architecture.

        Args:
            left_features (int): Number of features in left components.
            right_features (int): Number of features in right components.
            out_features (int): Number of output features.
            project (Optional[PairwiseProjection], optional): Apply this
                transformation to inputs first. By default, no transformation.

        """
        super().__init__()

        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.project = project
        self.compat = nn.Bilinear(left_features, right_features, out_features)

    def forward(self, lefts: torch.Tensor,
                rights: torch.Tensor) -> torch.Tensor:
        """Compute compatibility between lefts and rights.

        Args:
            lefts (torch.Tensor): Left items to compute compatibility for.
                Should have shape (*, left_features).
            rights (torch.Tensor): Right items to compute compatibility for.
                Should have shape (*, right_features).

        Returns:
            torch.Tensor: Compatibilities of shape (*, out_features).

        """
        if self.project is not None:
            lefts, rights = self.project(lefts, rights)
        return self.compat(lefts, rights)


class MLP(nn.Module):
    """MLP that computes pairwise compatibilities between representations."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: Optional[int] = None,
                 project: Optional[models.PairwiseProjection] = None):
        """Initialize the network.

        Args:
            in_features (int): Number of features in inputs.
            out_features (int): Number of features in outputs.
            hidden_features (Optional[int], optional): Number of features in
                MLP hidden layer. Defaults to the same as MLP's input features.
            project (Optional[PairwiseProjection], optional): See
                PairwiseBilinear.__init__ docstring.

        """
        super().__init__()

        hidden_features = hidden_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.project = project
        self.mlp = nn.Sequential(nn.Linear(in_features * 2, hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(hidden_features, out_features))

    def forward(self, lefts: torch.Tensor,
                rights: torch.Tensor) -> torch.Tensor:
        """Compute pairwise compatibilities for given matrix.

        Args:
            inputs (torch.Tensor): Shape (N, in_features) matrix,
                where N is the number of representations to compare.

        Raises:
            ValueError: If input is misshapen.

        Returns:
            torch.Tensor: Size (N, N) matrix representing pairwise
                compatibilities.

        """
        if self.project is not None:
            lefts, rights = self.project(lefts, rights)
        pairs = torch.cat((lefts, rights), dim=-1)
        return self.mlp(pairs)


ndims = elmos['train'].dimension
if options.share_projection:
    projection = models.PairwiseProjection(
        models.Projection(ndims, options.dimension),)
else:
    projection = models.PairwiseProjection(
        models.Projection(ndims, options.dimension),
        models.Projection(ndims, options.dimension))

probe: Union[Bilinear, MLP]
if options.probe == 'bilinear':
    probe = Bilinear(options.dimension,
                     options.dimension,
                     len(task),
                     project=projection)
else:
    assert options.probe == 'mlp'
    probe = MLP(options.dimension, len(task), project=projection)
probe = probe.to(device)

optimizer = optim.Adam(probe.parameters(), lr=options.lr)
ignore_index = task.indexer[ontonotes.IGNORE]
ce = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)


def criterion(*args: torch.Tensor) -> torch.Tensor:
    """Returns CE loss with regularizers."""
    loss = ce(*args)
    for subproj in (projection.left, projection.right):
        if options.l1:
            loss += options.l1 * subproj.project.weight.data.norm(p=1)
        if options.nuc:
            loss += options.l1 * subproj.project.weight.data.norm(p='nuc')
    return loss


best_dev_loss, bad_epochs = float('inf'), 0
for epoch in range(options.epochs):
    probe.train()
    for batch, (reps, themes, labels) in enumerate(loaders['train']):
        lefts = reps[themes].unsqueeze(1).expand(-1, len(reps), -1)
        rights = reps.unsqueeze(0).expand(len(themes), -1, -1)
        preds = probe(lefts, rights)
        loss = criterion(preds.view(-1, probe.out_features), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({'train loss': loss})
        logging.info('epoch %d batch %d loss %f', epoch, batch, loss.item())

    probe.eval()
    total, count = 0., 0
    for reps, themes, labels in loaders['dev']:
        lefts = reps[themes].unsqueeze(1).expand(-1, len(reps), -1)
        rights = reps.unsqueeze(0).expand(len(themes), -1, -1)
        preds = probe(lefts, rights).view(-1, probe.out_features)
        total += criterion(preds, labels.view(-1)).item() * len(reps)
        count += len(reps)
    dev_loss = total / count
    logging.info('epoch %d dev loss %f', epoch, dev_loss)
    wandb.log({'dev loss': dev_loss})

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs > options.patience:
        logging.info('patience exceeded, training is now over')
        break

# Write finished models.
model_path = options.model_dir / options.model_file
torch.save(probe, model_path)
wandb.save(str(model_path))
logging.info('model saved to %s', model_path)


# Evaluate the models.
def test(model: nn.Module) -> float:
    """Evaluate model accuracy on the test set.

    Args:
        model (nn.Module): The module to evaluate.

    Returns:
        float: Fraction of test set correctly classified.

    """
    correct, count = 0., 0
    for reps, themes, labels in loaders['test']:
        lefts = reps[themes].unsqueeze(1).expand(-1, len(reps), -1)
        rights = reps.unsqueeze(0).expand(len(themes), -1, -1)
        preds = model(lefts, rights).argmax(dim=-1)
        correct += preds.eq(labels).sum().item()
        count += len(reps)
    return correct / count


accuracy = test(probe)
wandb.run.summary['accuracy'] = accuracy
logging.info('test accuracy %.3f', accuracy)

# Measure whether or not the projection is axis aligned.
if options.ablate:
    logging.info('will ablate axes one by one and retest')
    multiplier = 1 if options.share_projection else 2
    axes = set(range(multiplier * projection.in_features))
    ablated: Set[int] = set()
    accuracies = []
    while axes:
        best_axis, best_accuracy = 0, 0.
        for axis in axes:
            model = copy.deepcopy(probe)
            assert model.project is not None, 'no projection?'

            indices = ablated | {axis}
            if options.share_projection:
                # If we are sharing projections, then ablating the left also
                # ablates the right. Easy!
                model.project.left.project.weight.data[:, sorted(indices)] = 0
            else:
                # If we are not sharing projections, then the "left" and
                # "right" projections contain disjoint axes. we have to
                # manually determine which axis belongs to which projection.
                coordinates = {(i // projection.in_features,
                                i % projection.in_features) for i in indices}
                ls = {ax for (proj, ax) in coordinates if not proj}
                rs = {ax for (proj, ax) in coordinates if proj}
                assert len(ls) + len(rs) == len(indices), 'bad mapping?'
                model.project.left.project.weight.data[:, sorted(ls)] = 0
                model.project.right.project.weight.data[:, sorted(rs)] = 0

            accuracy = test(model)
            if accuracy > best_accuracy:
                best_axis = axis
                best_accuracy = accuracy

        logging.info('ablating axis %d, accuracy %f', best_axis, best_accuracy)
        axes.remove(best_axis)
        ablated.add(best_axis)
        accuracies.append(best_accuracy)

    wandb.run.summary['ablated accuracies'] = torch.tensor(accuracies)
