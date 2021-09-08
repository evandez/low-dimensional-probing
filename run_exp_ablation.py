"""Ablate subspaces from representations and measure LM performance.

This command ablates a subspace from last-layer BERT representations and then
feeds the representations to the BERT LM and evaluates it on a subject-verb
agreement benchmark.

Subspaces are ablated by applying a low-dimensional (but shape-preserving)
projection to the representations. One way to create this projection is to
use the `run_inlp.py` script in this directory.

Benchmarks are taken from SyntaxGym at https://syntaxgym.org. This script
only supports the subject-verb agreement suites.
"""
import argparse
import dataclasses
import pathlib
from typing import AbstractSet, Any, Iterator, Mapping, Sequence, cast

from lodimp.models import projections
from lodimp.parse import syntax_gym
from lodimp.utils import logging

import spacy
import spacy.lang.en
import torch
import transformers
import wandb
from torch import cuda

parser = argparse.ArgumentParser()
parser.add_argument('projection_file',
                    type=pathlib.Path,
                    help='projection file')
parser.add_argument('sg_files',
                    type=pathlib.Path,
                    nargs='+',
                    help='syntax gym json files')
parser.add_argument('--wandb-group',
                    default='ablation',
                    help='experiment group (default: ablation)')
parser.add_argument('--wandb-name',
                    help='experiment name (default: generated)')
parser.add_argument('--wandb-path',
                    type=pathlib.Path,
                    help='path to write wandb data (default: wandb default)')
parser.add_argument(
    '--bert-config',
    default='bert-base-uncased',
    help='bert config to evaluate on (default: bert-base-uncased)')
parser.add_argument('--device', help='use this device (default: guessed)')
args = parser.parse_args()


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


ItemResults = Mapping[str, ConditionResults]


@dataclasses.dataclass(frozen=True)
class SyntaxGymEvaluator:
    """Evaluates ablated representations on Syntax Gym suites."""

    projection: projections.Projection
    tokenizer: transformers.BertTokenizer
    bert: transformers.BertForMaskedLM

    # Optional configuration, usually does not need to be changed.
    mask: str = '[MASK]'
    device: str = 'cpu'
    nlp: spacy.lang.en.English = dataclasses.field(
        default_factory=lambda: cast(spacy.lang.en.English,
                                     spacy.load('en_core_web_sm')))
    verb_upos: AbstractSet[str] = frozenset({'AUX', 'VERB'})
    noun_upos: AbstractSet[str] = frozenset({'NOUN', 'PRON', 'PROPN'})

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
        with torch.no_grad():
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
        with torch.no_grad():
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
        if len(top5_pos) != 5:
            logging.warning(
                'top5 words computation returned != 5 words: '
                '"%s"; this can happen when bert predicts a '
                'special token as one of its top-5 words and it '
                'should not affect the results', top5_words)

        top5_nouns = 0
        for tag in self.noun_upos:
            top5_nouns += top5_pos.count(tag)
        assert top5_nouns <= 5, 'should only be 5 tokens?'

        top5_verbs = 0
        for tag in self.verb_upos:
            top5_verbs += top5_pos.count(tag)
        assert top5_verbs <= 5, 'should only be 5 tokens?'

        decoded_indices = log_probs.argmax(dim=-1).tolist()
        decoded = self.tokenizer.decode(decoded_indices)

        return WordRepresentationResults(word_prob=word_prob.item(),
                                         top5_words=top5_words.split(),
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
            word = parts[-1]
            assert word.startswith('driver'), 'must be "taxi driver(s)"'

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

    def item(self, item: syntax_gym.Item) -> ItemResults:
        """Process all conditions in the item.

        Args:
            item (syntax_gym.Item): The Syntax Gym item.

        Returns:
            ItemResults: Results from processing all conditions.

        """
        results = {}
        for condition in item.conditions:
            results[condition.name] = self.condition(condition)
        return results

    def suite(self, suite: syntax_gym.Suite) -> Iterator[ItemResults]:
        """Evaluate the suite.

        Args:
            suite (syntax_gym.Suite): The SyntaxGym suite.

        Yields:
            ItemResults: Results from processing each item.

        """
        for item in suite.items:
            yield self.item(item)


args.wandb_path.mkdir(parents=True, exist_ok=True)
wandb.init(project='lodimp',
           name=args.wandb_name,
           group=args.wandb_group,
           dir=str(args.wandb_path))
assert wandb.run is not None, 'null run?'

logging.configure()
log = logging.getLogger(__name__)

device = args.device or 'cuda' if cuda.is_available() else 'cpu'
log.info('using device: %s', device)

projection_file = args.projection_file
log.info('loading projection from: %s', projection_file.resolve())
projection = torch.load(projection_file, map_location=device).eval()

bert_config = args.bert_config

log.info('loading bert tokenizer with config "%s"', bert_config)
tokenizer = transformers.BertTokenizer.from_pretrained(bert_config)

log.info('loading bert lm with config "%s"', bert_config)
bert = transformers.BertForMaskedLM.from_pretrained(bert_config).eval()

evaluate = SyntaxGymEvaluator(projection, tokenizer, bert, device=device)

noun_prob_diff_before, noun_prob_diff_after = 0., 0.
verb_prob_diff_before, verb_prob_diff_after = 0., 0.
nouns_before, nouns_after = 0, 0
verbs_before, verbs_after = 0, 0
count = 0

for sg_file in args.sg_files:
    logging.info('evaluating suite %s', sg_file)
    suite = syntax_gym.load_suite_json(sg_file)
    for result in evaluate.suite(suite):
        # Log each condition as a separate item.
        for subresult in result.values():
            wandb.log({'suite': sg_file.name, **subresult.dump()})

        # Record verb aggregates.
        for match_key, mismatch_key in (
            ('match_sing', 'mismatch_sing'),
            ('match_plural', 'mismatch_plural'),
        ):
            match = result[match_key]
            mismatch = result[mismatch_key]

            verb_prob_diff_before += (match.verb.real.word_prob -
                                      mismatch.verb.real.word_prob)
            verb_prob_diff_after += (match.verb.nulled.word_prob -
                                     mismatch.verb.nulled.word_prob)

            verbs_before += match.verb.real.top5_verbs
            verbs_after += match.verb.nulled.top5_verbs

        # Record noun aggregates. Notice the change in keys here; the
        # match vs. mismatch tags in Syntax Gym refer to *verbs* matching,
        # so we have to flip it around when we work with *nouns* matching.
        for match_key, mismatch_key in (
            ('match_plural', 'mismatch_sing'),
            ('match_sing', 'mismatch_plural'),
        ):
            match = result[match_key]
            mismatch = result[mismatch_key]

            noun_prob_diff_before += (match.noun.real.word_prob -
                                      mismatch.noun.real.word_prob)
            noun_prob_diff_after += (match.noun.nulled.word_prob -
                                     mismatch.noun.nulled.word_prob)

            nouns_before += match.noun.real.top5_nouns
            nouns_after += match.noun.nulled.top5_nouns

        count += 2

    wandb.run.summary['avg_nouns_before'] = nouns_before / count
    wandb.run.summary['avg_nouns_after'] = nouns_after / count

    wandb.run.summary['avg_verbs_before'] = verbs_before / count
    wandb.run.summary['avg_verbs_after'] = verbs_after / count

    wandb.run.summary['avg_noun_prob_diff_before'] =\
        noun_prob_diff_before / count
    wandb.run.summary['avg_noun_prob_diff_after'] =\
        noun_prob_diff_after / count

    wandb.run.summary['avg_verb_prob_diff_before'] =\
        verb_prob_diff_before / count
    wandb.run.summary['avg_verb_prob_diff_after'] =\
        verb_prob_diff_after / count
