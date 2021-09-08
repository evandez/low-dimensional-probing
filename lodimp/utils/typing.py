"""Some common type aliases."""
import pathlib
from typing import Union

import torch

Device = Union[str, torch.device]
PathLike = Union[str, pathlib.Path]
