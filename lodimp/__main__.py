"""Entrypoint for all LoDimP scripts."""

import argparse
import logging
import pathlib
import sys

# Unfortunately we have to muck with sys.path to avoid a wrapper script.
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

from lodimp.commands import axis_alignment, collate, inlp, train  # noqa: E402

parser = argparse.ArgumentParser(description='Run a LoDimP script.')
parser.add_argument('--quiet',
                    dest='log_level',
                    action='store_const',
                    const=logging.WARNING,
                    default=logging.INFO,
                    help='Only show warning or error messages.')
subparsers = parser.add_subparsers(dest='command')
subparsers.add_parser('collate', parents=[collate.parser()])
subparsers.add_parser('train', parents=[train.parser()])
subparsers.add_parser('inlp', parents=[inlp.parser()])
subparsers.add_parser('axis-alignment', parents=[axis_alignment.parser()])
options = parser.parse_args()

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=options.log_level)

if options.command == 'collate':
    collate.run(options)
elif options.command == 'train':
    train.run(options)
elif options.command == 'inlp':
    inlp.run(options)
elif options.command == 'axis-alignment':
    axis_alignment.run(options)
