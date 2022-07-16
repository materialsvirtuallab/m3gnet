"""
CLI for m3gnet
"""

import argparse
import sys

from pymatgen.core.structure import Structure
from m3gnet.models import Relaxer


def relax_structure(args):
    """
    Handle view commands.

    :param args: Args from command.
    """

    s = Structure.from_file(args.infile)

    relaxer = Relaxer()  # This loads the default pre-trained model

    relax_results = relaxer.relax(s)

    final_structure = relax_results["final_structure"]
    final_energy = relax_results["trajectory"].energies[-1] / 2

    if args.outfile is not None:
        final_structure.to(filename=args.outfile)
    else:
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
        epilog="""Author: Pymatgen Development Team""",
    )

    subparsers = parser.add_subparsers()

    p = subparsers.add_parser(
        "relax",
        help="Relax crystal structures.",
    )

    p_relax = subparsers.add_parser("relax", help="Relax crystal structures.")

    p_relax.add_argument(
        "-i",
        "--infile",
        dest="infile",
        required=True,
        help="Input file containg structure. Common structures support by pmg.Structure.from_file method.",
    )

    p_relax.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        default=None,
        help="Output structure",
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
