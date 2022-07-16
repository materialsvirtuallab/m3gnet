"""
CLI for m3gnet
"""

import argparse
import sys
import logging

from pymatgen.core.structure import Structure
import tensorflow as tf
from m3gnet.models import Relaxer

logging.captureWarnings(True)
tf.get_logger().setLevel(logging.ERROR)


def relax_structure(args):
    """
    Handle view commands.

    :param args: Args from command.
    """

    s = Structure.from_file(args.infile)

    if args.verbose:
        print("Starting structure")
        print(s)
        print("Relaxing...")
    relaxer = Relaxer()
    relax_results = relaxer.relax(s)
    final_structure = relax_results["final_structure"]

    if args.outfile is not None:
        final_structure.to(filename=args.outfile)
        print(f"Structure written to {args.outfile}!")
    else:
        print("Final structure")
        print(final_structure)

    return 0


def main():
    """
    Handle main.
    """
    parser = argparse.ArgumentParser(
        description="""
    This script works based on several sub-commands with their own options. To see the options for the
    sub-commands, type "m3g sub-command -h".""",
        epilog="""Author: M3Gnet""",
    )

    subparsers = parser.add_subparsers()

    p_relax = subparsers.add_parser("relax", help="Relax crystal structures.")

    p_relax.add_argument(
        "-i",
        "--infile",
        dest="infile",
        required=True,
        help="Input file containing structure. Common structures support by pmg.Structure.from_file method.",
    )

    p_relax.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output.",
    )

    p_relax.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        default=None,
        help="Output file. If None is given, the structure is output to stdout.",
    )

    p_relax.set_defaults(func=relax_structure)

    args = parser.parse_args()

    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(-1)
    return args.func(args)


if __name__ == "__main__":
    main()
