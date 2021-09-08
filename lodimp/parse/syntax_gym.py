"""Parse Syntax Gym suites."""
import dataclasses
import json
import pathlib
from typing import Mapping, Sequence

from lodimp.utils.typing import PathLike


@dataclasses.dataclass(frozen=True)
class Region:
    """A single region within a condition."""

    number: int
    content: str


@dataclasses.dataclass(frozen=True)
class Condition:
    """A single condition within an item."""

    item: int
    name: str
    regions: Sequence[Region]
    region_meta: Mapping[str, str]

    def get_region_by_number(self, number: int) -> Region:
        """Get a region by its number.

        Args:
            number (int): The region number.

        Raises:
            KeyError: If no region with the given number exists
                or if muliple regions with the same number exist.

        Returns:
            Region: The region.

        """
        matches = [reg for reg in self.regions if reg.number == number]
        if not matches:
            raise KeyError(f'no region with number: {number}')
        elif len(matches) > 1:
            raise KeyError(f'multiple regions with same number: {matches}')
        region, = matches
        return region

    def get_region_by_name(self, name: str) -> Region:
        """Get the given region by name.

        Args:
            name (str): Name of the region.

        Raises:
            KeyError: If no region with the given name exists
                or if muliple regions with the same name exist.
            ValueError: If number corresponding to region is not an integer.

        Returns:
            Region: The region.

        """
        matches = [num for num, key in self.region_meta.items() if key == name]
        if not matches:
            raise KeyError(f'no region with name: "{name}"')
        elif len(matches) > 1:
            raise KeyError(f'multiple regions with same name: {matches}')

        number, = matches
        if not number.isdigit():
            raise ValueError(f'region number is not int: "{number}"')

        return self.get_region_by_number(int(number))

    @property
    def sentence(self) -> str:
        """Return the full sentence represented by this condition."""
        return ' '.join(region.content for region in self.regions)


@dataclasses.dataclass(frozen=True)
class Item:
    """A single item in a suite."""

    number: int
    conditions: Sequence[Condition]
    region_meta: Mapping[str, str]

    def get_condition_by_name(self, name: str) -> Condition:
        """Get a condition by name.

        Args:
            name (str): Name of the condition.

        Raises:
            KeyError: If no condition with the given name exists or if multiple
                conditions have the same name.

        Returns:
            Condition: The condition.

        """
        matches = [cond for cond in self.conditions if cond.name == name]
        if not matches:
            raise KeyError(f'no region with name: "{name}"')
        elif len(matches) > 1:
            raise KeyError(f'multiple regions with same name: {matches}')
        condition, = matches
        return condition


@dataclasses.dataclass(frozen=True)
class Suite:
    """Represents a Syntax Gym suite."""

    items: Sequence[Item]
    region_meta: Mapping[str, str]


def load_suite_json(json_file: PathLike,
                    region_meta_key: str = 'region_meta',
                    items_key: str = 'items',
                    item_number_key: str = 'item_number',
                    conditions_key: str = 'conditions',
                    condition_name_key: str = 'condition_name',
                    regions_key: str = 'regions',
                    region_number_key: str = 'region_number',
                    region_content_key: str = 'content') -> Suite:
    """Load the given Syntax Gym suite.

    Args:
        path (pathlib.Path): Path to the suite JSON file.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON is malformed.

    Returns:
        Suite: The loaded suite.

    """
    json_file = pathlib.Path(json_file)
    if not json_file.is_file():
        raise FileNotFoundError(f'json file not found: {json_file}')

    content = json.load(json_file.open('r'))
    for key in (region_meta_key, items_key):
        if key not in content:
            raise ValueError(f'suite missing key: {key}')

    region_meta = content[region_meta_key]

    # Parse items...
    items = []
    for item_index, item_content in enumerate(content[items_key]):
        item_number = item_content.get(item_number_key)
        if item_number is None:
            raise ValueError(f'item {item_index} missing item_number')

        # For every item, parse conditions...
        item_conditions = []
        for condition_index, condition_content in enumerate(
                item_content[conditions_key]):
            condition_name = condition_content.get(condition_name_key)
            if condition_name is None:
                raise ValueError(
                    f'item {item_index} condition {condition_index} '
                    'missing name')
            if not isinstance(condition_name, str):
                raise ValueError(
                    f'item {item_index} condition {condition_index} '
                    f'name not str: {condition_name}')

            # For every condition, parse regions...
            condition_regions = []
            for region_index, region_content in enumerate(
                    condition_content[regions_key]):
                region_number = region_content.get(region_number_key)
                region_content = region_content.get(region_content_key)
                for name, value in (('region number', region_number),
                                    ('region content', region_content)):
                    if value is None:
                        raise ValueError(
                            f'item {item_index} condition {condition_index} '
                            f'region {region_index} missing {name}')

                region = Region(region_number, region_content)
                condition_regions.append(region)

            condition = Condition(item_number, condition_name,
                                  tuple(condition_regions), region_meta)
            item_conditions.append(condition)

        item = Item(item_number, tuple(item_conditions), region_meta)
        items.append(item)

    return Suite(tuple(items), region_meta)
