"""This module contains all task-specific code.

We work with four linguistically motivated tasks: part of speech tagging (POS),
dependency edge prediction (DEP), dependency label prediction (DLP), and
semantic role labeling (SRL).

Task-specific code is meant to be minimal. Most core logic can be found in
common modules.
"""

# Mapping from task to abbreviation. Useful when defining CLIs...
PART_OF_SPEECH_TAGGING = 'pos'
DEPENDENCY_LABEL_PREDICTION = 'dlp'
DEPENDENCY_EDGE_PREDICTION = 'dep'
