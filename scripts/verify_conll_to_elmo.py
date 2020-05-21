"""Sanity check that we are using same ELMo as Hewitt paper."""

import argparse
import pathlib

import h5py
import torch

parser = argparse.ArgumentParser(description='Test conll_to_elmo.py script.')
parser.add_argument('actual', type=pathlib.Path, help='Generated hdf5.')
parser.add_argument('expected', type=pathlib.Path, help='Expected hdf5.')
options = parser.parse_args()

actuals = h5py.File(options.actual, 'r')
expecteds = h5py.File(options.expected, 'r')

assert len(actuals.keys()) == len(expecteds.keys())
for index in range(len(expecteds) - 1):
    actual = torch.tensor(actuals[str(index)][:])
    expected = torch.tensor(expecteds[str(index)][:])
    assert torch.allclose(actual, expected, atol=2e-4)
