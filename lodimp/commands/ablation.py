"""Ablate subspaces from representations and measure LM performance.

This command ablates a subspace from last-layer BERT representations and then
feeds the representations to the BERT LM and evaluates it on a subject-verb
agreement benchmark.

Subspaces are ablated by applying a low-dimensional (but shape-preserving)
projection to the representations. One way to create this projection is to
use the `python lodimp inlp` command.

Benchmarks are taken from SyntaxGym at https://syntaxgym.org. This script
only supports the subject-verb agreement suites.
"""

import argparse
import dataclasses
import logging
import pathlib
from typing import Any, Iterator, Mapping, Sequence

from lodimp.common.models import projections
from lodimp.common.parse import syntax_gym

import spacy
import spacy.lang.en
import torch
import transformers
import wandb


def parser() -> argparse.ArgumentParser:
    """Returns the argument parser for this command."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('projection_file',
                        type=pathlib.Path,
                        help='projection file')
    parser.add_argument('sg_files',
                        type=pathlib.Path,
                        nargs='+',
                        help='syntax gym json files.')
    parser.add_argument('--wandb-group',
                        default='ablation',
                        help='experiment group')
    parser.add_argument('--wandb-name', help='experiment name')
    parser.add_argument('--wandb-path',
                        type=pathlib.Path,
                        default='/tmp/lodimp/wandb',
                        help='path to write wandb data')
    parser.add_argument('--bert-config',
                        default='bert-base-uncased',
                        help='bert config to evaluate on')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    return parser


@dataclasses.dataclass(frozen=True)
class WordRepresentationResults:
    """Predictions and targets for a given word and rep (real vs. nulled)."""

    word_prob: float
    top5_words: Sequence[str]
    top5_probs: Sequence[float]
    top5_nouns: int
    top5_verbs: int
    decoded: str

    def dump(self) -> Mapping[str, Any]:
        """Dump results to a dictionary."""
        return {
            'word_prob': self.word_prob,
            'top5_words': self.top5_words,
            'top5_probs': self.top5_probs,
            'top5_nouns': self.top5_nouns,
            'top5_verbs': self.top5_verbs,
            'decoded': self.decoded,
        }


@dataclasses.dataclass(frozen=True)
class WordResults:
    """Predictions and targets for a given word."""

    word: str
    real: WordRepresentationResults
    nulled: WordRepresentationResults

    def dump(self) -> Mapping[str, Any]:
        """Dump results to a dictionary."""
        return {
            'word': self.word,
            'real': self.real.dump(),
            'nulled': self.nulled.dump(),
        }


@dataclasses.dataclass(frozen=True)
class ConditionResults:
    """Predictions for noun and verb words in a condition."""

    condition: syntax_gym.Condition
    noun: WordResults
    verb: WordResults

    def dump(self) -> Mapping[str, Any]:
        """Dump the results to a dictionary."""
        return {
            'item': self.condition.item,
            'condition': self.condition.name,
            'sentence': self.condition.sentence,
            'noun': self.noun.dump(),
            'verb': self.verb.dump(),
        }


@dataclasses.dataclass(frozen=True)
class SyntaxGymEvaluator:
    """Evaluates ablated representations on Syntax Gym suites."""

    projection: projections.Projection
    tokenizer: transformers.BertTokenizer
    bert: transformers.BertForMaskedLM

    # Optional configuration, usually does not need to be changed.
    mask: str = '[MASK]'
    device: torch.device = torch.device('cpu')
    nlp: spacy.lang.en.English = dataclasses.field(
        default_factory=lambda: spacy.load('en_core_web_sm'))

    @property
    def mask_vocabulary_index(self) -> int:
        """Return the vocabulary index on the mask token."""
        mask_token, = self.tokenizer.encode(self.mask,
                                            add_special_tokens=False)
        return mask_token

    def embed(self, tokens: Sequence[int]) -> torch.Tensor:
        """Feed the tokens to BERT and return the last hidden layer.

        Args:
            tokens (Sequence[int]): Vocabulary indices of tokens.

        Returns:
            torch.Tensor: Shape (1, len(sequences), representation_size)
                tensor containing outputs of last layer of BERT.

        """
        inputs = torch.tensor([tokens], device=self.device)
        representations, *_ = self.bert.bert(inputs)
        return representations

    def lm(self, representations: torch.Tensor) -> torch.Tensor:
        """Feed reps to the LM and return a (log) distribution over words.

        Args:
            representations (torch.Tensor): Last layer of hidden reps
                of shape (1, sequence_length, representation_size).

        Returns:
            torch.Tensor: Shape (1, sequence_length, vocabulary_size) tensor
                containing log probabilities for each word in the sequence.

        """
        logits = self.bert.cls(representations)
        return torch.log_softmax(logits, dim=-1)

    def word_representations(
            self, representations: torch.Tensor, word_index_in_sentence: int,
            word_index_in_vocabulary: int) -> WordRepresentationResults:
        """Process the given representations.

        Args:
            representations (torch.Tensor): Representations from the final
                layer of BERT. Should have shape
                (1, sequence_length, vocabulary_size).
            word_index_in_sentence (int): Index of word in the sentence.
            word_index_in_vocabulary (int): Index of word in vocabulary.

        Returns:
            WordRepresentationResults: The results.

        """
        log_probs = self.lm(representations).squeeze()

        word_prob = log_probs[word_index_in_sentence][word_index_in_vocabulary]

        top5_probs, top5_indices = log_probs[word_index_in_sentence].topk(
            k=5, dim=-1)
        top5_words = self.tokenizer.decode(top5_indices,
                                           clean_up_tokenization_spaces=False)

        top5_pos = [token.pos_ for token in self.nlp(top5_words)]
        assert len(top5_pos) == 5, 'should only be 5 tokens?'
        top5_nouns = top5_pos.count('NOUN')
        top5_verbs = top5_pos.count('VERB')

        decoded_indices = log_probs.argmax(dim=-1).tolist()
        decoded = self.tokenizer.decode(decoded_indices)

        return WordRepresentationResults(word_prob=word_prob.item(),
                                         top5_words=top5_words,
                                         top5_probs=top5_probs.tolist(),
                                         top5_nouns=top5_nouns,
                                         top5_verbs=top5_verbs,
                                         decoded=decoded)

    def word(self, condition: syntax_gym.Condition,
             region_name: str) -> WordResults:
        """Get the (single) word from region of a condition.

        Args:
            condition (syntax_gym.Condition): The condition.
            region_name (str): The region name.

        Returns:
            Word: The parsed word.

        """
        word = condition.get_region_by_name(region_name).content

        # Handle annoying special case where word is compound noun.
        parts = word.split()
        if len(parts) > 1:
            assert word.startswith('driver'), 'must be "taxi driver(s)"'
            word = parts[-1]

        # Compute important meatadata.
        word_index_in_vocabulary, = self.tokenizer.encode(
            word, add_special_tokens=False)

        tokens = self.tokenizer.encode(condition.sentence)
        word_index_in_sentence = tokens.index(word_index_in_vocabulary)

        # Compute representations
        tokens[word_index_in_sentence] = self.mask_vocabulary_index
        real_reps = self.embed(tokens)
        nulled_reps = real_reps.clone()
        nulled_reps[0][word_index_in_sentence] = self.projection(
            real_reps[0][word_index_in_sentence])

        # Compute probabilities/top-5 words/etc.
        real = self.word_representations(real_reps, word_index_in_sentence,
                                         word_index_in_vocabulary)
        nulled = self.word_representations(nulled_reps, word_index_in_sentence,
                                           word_index_in_vocabulary)

        return WordResults(word=word, real=real, nulled=nulled)

    def condition(self,
                  condition: syntax_gym.Condition,
                  noun_region_name: str = 'np_subject',
                  verb_region_name: str = 'matrix_v') -> ConditionResults:
        """Process the noun and verb in the condition.

        Args:
            condition (syntax_gym.Condition): The condition to process.
            noun_region_name (str, optional): Name of noun region. Defaults to
                "np_subject".
            verb_region_name (str, optional): Name of verb region. Defaults to
                "matrix_v".

        Returns:
            ConditionResults: The condition results.

        """
        noun = self.word(condition, noun_region_name)
        verb = self.word(condition, verb_region_name)
        return ConditionResults(condition=condition, noun=noun, verb=verb)

    def item(self, item: syntax_gym.Item) -> Iterator[ConditionResults]:
        """Process all conditions in the item.

        Args:
            item (syntax_gym.Item): The Syntax Gym item.

        Returns:
            ItemResults: Results from processing all conditions.

        """
        for condition in item.conditions:
            yield self.condition(condition)

    def suite(self, suite: syntax_gym.Suite) -> Iterator[ConditionResults]:
        """Evaluate the suite.

        Args:
            suite (syntax_gym.Suite): The SyntaxGym suite.

        """
        for item in suite.items:
            for condition_result in self.item(item):
                yield condition_result


def run(options: argparse.Namespace) -> None:
    """Run the ablation experiment with the given options."""
    options.wandb_path.mkdir(parents=True, exist_ok=True)
    wandb.init(project='lodimp',
               id=options.wandb_id,
               name=options.wandb_name,
               group=options.wandb_group,
               dir=str(options.wandb_path))
    assert wandb.run is not None, 'null run?'

    log = logging.getLogger(__name__)

    device = torch.device('cuda' if options.cuda else 'cpu')
    log.info('using device: %s', device.type)

    projection_file = options.projection_file
    log.info('loading projection from: %s', projection_file.resolve())
    projection = torch.load(projection_file, map_location=device)

    bert_config = options.bert_config

    log.info('loading bert tokenizer with config "%s"', bert_config)
    tokenizer = transformers.BertTokenizer.from_pretrained(bert_config)

    log.info('loading bert lm with config "%s"', bert_config)
    bert = transformers.BertForMaskedLM.from_pretrained(bert_config).eval()

    evaluate = SyntaxGymEvaluator(projection, tokenizer, bert, device=device)
    for sg_file in options.sg_files:
        logging.info('evaluating suite %s', sg_file)
        suite = syntax_gym.load_suite_json(sg_file)
        result = evaluate.suite(suite, projection, tokenizer, bert)
        wandb.log(result.dump())
