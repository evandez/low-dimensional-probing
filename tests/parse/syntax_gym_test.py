"""Unit tests for the syntax_gym module."""
import pathlib
import tempfile

from lodimp.parse import syntax_gym

import pytest

JSON = '''\
{
    "items": [
        {
            "conditions": [
                {
                    "condition_name": "condition-1",
                    "regions": [
                        {
                            "content": "item-1-condition-1-region-1",
                            "region_number": 1
                        },
                        {
                            "content": "item-1-condition-1-region-2",
                            "region_number": 2
                        }
                    ],
                    "content": "condition-2-content"
                },
                {
                    "condition_name": "condition-2",
                    "regions": [
                        {
                            "content": "item-1-condition-2-region-1",
                            "region_number": 1
                        },
                        {
                            "content": "item-1-condition-2-region-2",
                            "region_number": 2
                        }
                    ],
                    "content": "condition-2-content"
                }
            ],
            "item_number": 1
        },
        {
            "conditions": [
                {
                    "condition_name": "condition-1",
                    "regions": [
                        {
                            "content": "item-2-condition-1-region-1",
                            "region_number": 1
                        },
                        {
                            "content": "item-2-condition-1-region-2",
                            "region_number": 2
                        }
                    ],
                    "content": "condition-2-content"
                },
                {
                    "condition_name": "condition-2",
                    "regions": [
                        {
                            "content": "item-2-condition-2-region-1",
                            "region_number": 1
                        },
                        {
                            "content": "item-2-condition-2-region-2",
                            "region_number": 2
                        }
                    ],
                    "content": "condition-2-content"
                }
            ],
            "item_number": 2
        }
    ],
    "region_meta": {
        "1": "region-1",
        "2": "region-2"
    }
}
'''

EXPECTED_REGION_META = {'1': 'region-1', '2': 'region-2'}
EXPECTED_SUITE = syntax_gym.Suite(
    items=(
        syntax_gym.Item(
            number=1,
            region_meta=EXPECTED_REGION_META,
            conditions=(
                syntax_gym.Condition(
                    item=1,
                    name='condition-1',
                    regions=(
                        syntax_gym.Region(
                            number=1, content='item-1-condition-1-region-1'),
                        syntax_gym.Region(
                            number=2, content='item-1-condition-1-region-2'),
                    ),
                    region_meta=EXPECTED_REGION_META,
                ),
                syntax_gym.Condition(
                    item=1,
                    name='condition-2',
                    regions=(
                        syntax_gym.Region(
                            number=1, content='item-1-condition-2-region-1'),
                        syntax_gym.Region(
                            number=2, content='item-1-condition-2-region-2'),
                    ),
                    region_meta=EXPECTED_REGION_META,
                ),
            ),
        ),
        syntax_gym.Item(
            number=2,
            region_meta=EXPECTED_REGION_META,
            conditions=(
                syntax_gym.Condition(
                    item=2,
                    name='condition-1',
                    regions=(
                        syntax_gym.Region(
                            number=1, content='item-2-condition-1-region-1'),
                        syntax_gym.Region(
                            number=2, content='item-2-condition-1-region-2'),
                    ),
                    region_meta=EXPECTED_REGION_META,
                ),
                syntax_gym.Condition(
                    item=2,
                    name='condition-2',
                    regions=(
                        syntax_gym.Region(
                            number=1,
                            content='item-2-condition-2-region-1',
                        ),
                        syntax_gym.Region(
                            number=2,
                            content='item-2-condition-2-region-2',
                        ),
                    ),
                    region_meta=EXPECTED_REGION_META,
                ),
            ),
        ),
    ),
    region_meta=EXPECTED_REGION_META,
)


@pytest.yield_fixture
def json_file():
    """Yield a suite JSON file for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'suite.json'
        with path.open('w') as handle:
            handle.write(JSON)
        yield path


def test_load_suite_json(json_file):
    """Test load_suite_json on a basic file."""
    actual = syntax_gym.load_suite_json(json_file)
    assert actual == EXPECTED_SUITE


def test_load_suite_bad_file():
    """Test load_suite_json dies on bad file."""
    bad = pathlib.Path('bad.json')
    with pytest.raises(FileNotFoundError, match=f'.*{bad}.*'):
        syntax_gym.load_suite_json(bad)
