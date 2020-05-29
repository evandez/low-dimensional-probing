"""Preprocess Ontonotes for SRL.

TODO(evandez): This script is a hastily-made disaster.
Need to fold this into the rest of the codebase. Right now, everything in
this script is a special case, so pay attention, dear reader.
"""

# flake8: noqa
import argparse
import collections
import copy
import itertools
import logging
import pathlib
import random
import sys
from typing import Dict, Iterator, Sequence, Set, Tuple, Union
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from lodimp import datasets
from lodimp.common import learning
from lodimp.common.data import ontonotes
from lodimp.common.models import probes, projections

import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.utils import data

parser = argparse.ArgumentParser(description='Train an SRL probe.')
parser.add_argument('data', type=pathlib.Path, help='Path to data.')
parser.add_argument(
    '--model',
    choices=('bert-base-uncased', 'elmo'),
    default='elmo',
    help='Representation model to use.',
)
parser.add_argument('--layer',
                    type=int,
                    default=0,
                    help='Representation layer.')
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
                    default=3,
                    help='Passes to make through dataset during training.')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
# See comment down below. No early stopping for now.
# parser.add_argument('--patience',
#                     type=int,
#                     default=4,
#                     help='Epochs for dev loss to decrease to stop training.')
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
wandb.init(
    project='lodimp',
    name=options.wandb_name,
    group=options.wandb_group,
    config={
        'task': 'srl-control',
        'representations': {
            'model': options.model,
            'layer': options.layer,
        },
        'projection': {
            'dimension': options.dimension,
            'shared': options.share_projection,
        },
        'probe': {
            'model': options.probe,
        },
        'hyperparameters': {
            'epochs': options.epochs,
            'batched': not options.no_batch,
            'lr': options.lr,
            #    'patience': options.patience,
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

annotations, reps_by_split = {}, {}
for split in ('train', 'dev', 'test'):
    # Load annotations.
    conll = options.data / f'ontonotes5-{split}.conllx'
    logging.info('reading ontonotes %s set from %s', split, conll)
    annotations[split] = ontonotes.load(conll)

    # Load preprocessed representations.
    h5 = (options.data / 'srl' / options.model / str(options.layer) /
          f'{split}.h5')
    logging.info('reading %s %s set from %s', options.model, split, h5)
    reps_by_split[split] = learning.SentenceBatchingTaskDataset(
        h5, device=device if options.no_batch else None)


class SemanticRoleLabelingTask:
    """Maps words to labels in one or more semantic role parses.

    Typically there is one semantic role parse per verb in the sentence.
    """

    def __init__(self, *groups: Sequence[ontonotes.Sample]):
        """Construct the semantic role labeling task.

        Args:
            groups (Sequence[ontonotes.Sample]): The samples from which
                to determine the list of all tags.

        """
        samples = tuple(itertools.chain(*groups))
        counts: Dict[str, int] = collections.defaultdict(lambda: 0)
        for sample in samples:
            for labeling in sample.roles:
                for label in labeling:
                    if label != '*' and label != 'V':
                        counts[label] += 1
        self.dist = np.array([float(count) for count in counts.values()])
        self.dist /= np.sum(self.dist)

        self.ignore_index = 0
        self.v_index = len(self.dist)
        self.star_index = self.v_index + 1

        def random_tag() -> int:
            """Returns a random tag according to the empirical distribution."""
            # Choose a random tag from the distribution of tags, and add one to
            # it because 0 is reserved for the "ignore" tag.
            return np.random.choice(len(self.dist), p=self.dist) + 1

        self.label_no_span: Set[str] = set()
        self.label_left_span: Dict[str, int] = {}
        self.label_right_span: Dict[str, int] = {}
        self.label_both_spans: Dict[str, Tuple[int, int]] = {}
        behaviors = (
            self.label_no_span,
            self.label_left_span,
            self.label_right_span,
            self.label_both_spans,
        )
        for sample in samples:
            for labeling in sample.roles:
                for index, role in enumerate(labeling):
                    if role == 'V':
                        word = sample.sentence[index]
                        behavior = random.choice(behaviors)
                        if behavior is self.label_no_span:
                            assert isinstance(behavior, set), 'appeasing mypy'
                            behavior.add(word)
                        elif behavior is self.label_both_spans:
                            assert isinstance(behavior, dict), 'appeasing mypy'
                            behavior[word] = (random_tag(), random_tag())
                        else:
                            assert isinstance(behavior, dict), 'appeasing mypy'
                            behavior[word] = random_tag()

    def __call__(
        self,
        sample: ontonotes.Sample,
        size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert role labels to integer tensor.

        Args:
            sample (ontonotes.Sample): The sample to label.
            size (int): Assume this many themes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: If sample has N words
                and R role labelings, first tensor is shape (R,)
                tensor containing indices of theme words, and second is a
                shape (N, R) integer tensor where element (n, r)
                is the label for the r'th role parse of the n'th word.

        """
        themes = torch.zeros(size, dtype=torch.long)
        labels = torch.zeros(size, len(sample.sentence), dtype=torch.long)
        for index, labeling in enumerate(sample.roles):
            candidates = [(index, sample.sentence[index])
                          for index, label in enumerate(labeling)
                          if label == 'V']
            assert len(candidates) == 1, 'double theme?'
            (theme_index, theme_word), = candidates

            themes[index] = theme_index

            if theme_word in self.label_no_span:
                left_label = right_label = self.star_index
            elif theme_word in self.label_left_span:
                left_label = self.label_left_span[theme_word]
                right_label = self.star_index
            elif theme_word in self.label_right_span:
                left_label = self.star_index
                right_label = self.label_right_span[theme_word]
            else:
                assert theme_word in self.label_both_spans, 'unseen theme?'
                left_index, right_index = self.label_both_spans[theme_word]
            labels[index, :theme_index] = left_index
            labels[index, theme_index] = self.v_index
            labels[index, theme_index + 1:] = right_index

        return themes, labels

    def __len__(self) -> int:
        """Returns the number of unique tags for this SRL task."""
        return len(self.dist) + 2


task = SemanticRoleLabelingTask(*tuple(annotations.values()))


class SemanticRoleLabelingDataset(data.IterableDataset):
    """Simple wrapper around representations and annotations data."""

    def __init__(self, reps: learning.SentenceBatchingTaskDataset,
                 samples: Sequence[ontonotes.Sample]):
        """Initialize and preprocess the data.

        Args:
            reps (datasets.SentenceBatchingTaskDataset): Dataset of reps.
            samples (Sequence[ontonotes.Sample]): Annotations for reps.

        """
        self.reps = reps

        self.indices = []
        for index, sample in enumerate(samples):
            if not sample.roles:
                continue
            self.indices.append(index)

        max_themes = max(len(sample.roles) for sample in samples)
        themes = []
        labels = []
        for sample in samples:
            sample_themes, sample_labels = task(sample, max_themes)
            themes.append(sample_themes)
            labels.append(sample_labels)
        self.themes = torch.stack(themes).to(device)
        self.labels = torch.cat(labels, dim=-1).to(device)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Return the sample at the given index.

        Args:
            index (int): Sample to retrieve.

        Returns:
            Tuple[torch.Tensor, ...]: Sample as (reps, themes, roles)

        """
        reps, _ = self.reps[self.indices[index]]
        themes = self.themes[self.indices[index]]
        start = self.reps.breaks[self.indices[index]]
        end = start + len(reps)
        labels = self.labels[:, start:end]
        return reps, themes, labels

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Yields a (reps, themes, roles) tuple for each sentence."""
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        """Returns the number of samples (sentences) in the dataset."""
        return len(self.indices)


loaders = {}
for split in ('train', 'dev', 'test'):
    logging.info('precomputing %s labels', split)
    loaders[split] = SemanticRoleLabelingDataset(reps_by_split[split],
                                                 annotations[split])

ndims = reps_by_split['train'].dimension
if options.share_projection:
    projection = projections.PairwiseProjection(
        projections.Projection(ndims, options.dimension),)
else:
    projection = projections.PairwiseProjection(
        projections.Projection(ndims, options.dimension),
        projections.Projection(ndims, options.dimension))

probe: Union[probes.Bilinear, probes.BiMLP]
if options.probe == 'bilinear':
    probe = probes.Bilinear(options.dimension, len(task), project=projection)
else:
    assert options.probe == 'mlp'
    probe = probes.BiMLP(options.dimension, len(task), project=projection)
probe = probe.to(device)

optimizer = optim.Adam(probe.parameters(), lr=options.lr)
criterion = nn.CrossEntropyLoss(ignore_index=task.ignore_index).to(device)

best_dev_loss, bad_epochs = float('inf'), 0
for epoch in range(options.epochs):
    probe.train()
    for batch, (reps, themes, labels) in enumerate(loaders['train']):
        reps = reps.to(device)
        themes = themes.to(device)
        labels = labels.to(device)
        lefts = reps[themes].unsqueeze(1).expand(-1, len(reps), -1)
        rights = reps.unsqueeze(0).expand(len(themes), -1, -1)
        preds = probe(lefts, rights)
        loss = criterion(
            preds.view(-1, probe.out_features),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({'train loss': loss})
        logging.info('epoch %d batch %d loss %f', epoch, batch, loss.item())

    # NOTE: We do not bother with early stopping. The dataset is so large,
    # and we have so many of these probes to train, that we can only afford
    # a few epochs of training. And this is fine because each epochs contains
    # many gradient steps.
    # probe.eval()
    # total, count = 0., 0
    # for reps, themes, labels in loaders['dev']:
    #     reps = reps.to(device)
    #     themes = themes.to(device)
    #     labels = labels.to(device)
    #     lefts = reps[themes].unsqueeze(1).expand(-1, len(reps), -1)
    #     rights = reps.unsqueeze(0).expand(len(themes), -1, -1)
    #     preds = probe(lefts, rights).view(-1, probe.out_features)
    #     total += criterion(preds, labels.view(-1)).item() * len(reps)
    #     count += len(reps) * len(themes)
    # dev_loss = total / count
    # logging.info('epoch %d dev loss %f', epoch, dev_loss)
    # wandb.log({'dev loss': dev_loss})

    # if dev_loss < best_dev_loss:
    #     best_dev_loss = dev_loss
    #     bad_epochs = 0
    # else:
    #     bad_epochs += 1

    # if bad_epochs > options.patience:
    #     logging.info('patience exceeded, training is now over')
    #     break

# Write finished models.
model_path = options.model_dir / options.model_file
torch.save(probe, model_path)
wandb.save(str(model_path))
logging.info('model saved to %s', model_path)


# Evaluate the models.
def test(model: nn.Module, loader: str = 'test') -> float:
    """Evaluate model accuracy on the test set.

    Args:
        model (nn.Module): The module to evaluate.
        loader (str): Dataset loader to use.

    Returns:
        float: Fraction of test set correctly classified.

    """
    correct, count = 0., 0
    for index, (reps, themes, labels) in enumerate(loaders[loader]):
        reps = reps.to(device)
        themes = themes.to(device)
        labels = labels.to(device)

        nthemes = int((labels.sum(dim=-1) != 0).sum().item())
        lefts = reps[themes[:nthemes]].unsqueeze(1).expand(-1, len(reps), -1)
        rights = reps.unsqueeze(0).expand(nthemes, -1, -1)
        preds = model(lefts, rights).argmax(dim=-1)
        correct += preds.eq(labels[:nthemes]).sum().item()
        count += len(reps) * nthemes
        logging.info('tested sample %d of %d', index + 1, len(loaders['test']))
    return correct / count


accuracy = test(probe)
wandb.run.summary['accuracy'] = accuracy
logging.info('test accuracy %.3f', accuracy)

# Measure whether or not the projection is axis aligned.
if options.ablate:

    def ablate(axes: Set[int]) -> nn.Module:
        """Abalate the given axes from the probe.

        Args:
            axes (Set[int]): The axes to ablate.

        Returns:
            nn.Module: The probe with the axes ablated from the projection.

        """
        model = copy.deepcopy(probe)
        assert model.project is not None, 'no projection?'
        if options.share_projection:
            # If we are sharing projections, then ablating the left also
            # ablates the right. Easy!
            model.project.left.project.weight.data[:, sorted(axes)] = 0
        else:
            # If we are not sharing projections, then the "left" and
            # "right" projections contain disjoint axes. we have to
            # manually determine which axis belongs to which projection.
            coordinates = {(i // projection.in_features,
                            i % projection.in_features) for i in axes}
            lefts = {ax for (proj, ax) in coordinates if not proj}
            rights = {ax for (proj, ax) in coordinates if proj}
            assert len(lefts) + len(rights) == len(axes), 'bad mapping?'
            model.project.left.project.weight.data[:, sorted(lefts)] = 0
            assert model.project.right is not None, 'null right proj?'
            model.project.right.project.weight.data[:, sorted(rights)] = 0
        return model.eval()

    multiplier = 1 if options.share_projection else 2
    axes = tuple(range(multiplier * projection.in_features))
    logging.info('will ablate %d axes and determine importance', len(axes))

    dev_accuracies = []
    for axis in axes:
        model = ablate({axis})
        dev_accuracy = test(model, loader='dev')
        logging.info('axis %d/dev accuracy %f', axis, dev_accuracy)
        dev_accuracies.append(dev_accuracy)
    wandb.run.summary['ablated dev accuracy'] = torch.tensor(dev_accuracies)

    ablated = set()
    test_accuracies = []
    to_ablate = sorted(enumerate(dev_accuracies),
                       key=lambda x: x[1],
                       reverse=True)
    for axis, _ in to_ablate:
        ablated.add(axis)
        model = ablate(ablated)
        test_accuracy = test(model)
        logging.info('%d axes/test accuracy %f', len(ablated), test_accuracy)
        test_accuracies.append(test_accuracy)
    wandb.run.summary['ablated test accuracy'] = torch.tensor(test_accuracies)
