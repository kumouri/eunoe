"""The `eunoe` command line."""
from __future__ import annotations

import argparse

from eunoe import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="eunoe", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    parser.print_help()
    return 0
