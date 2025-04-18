from glob import glob
import os
import argparse
from sourmash.logging import notify
from sourmash.sourmash_args import check_scaled_bounds, check_num_bounds


def add_moltype_args(parser):
    parser.add_argument(
        "--protein",
        dest="protein",
        action="store_true",
        help="choose a protein signature; by default, a nucleotide signature is used",
    )
    parser.add_argument(
        "--no-protein",
        dest="protein",
        action="store_false",
        help="do not choose a protein signature",
    )
    parser.set_defaults(protein=False)

    parser.add_argument(
        "--dayhoff",
        dest="dayhoff",
        action="store_true",
        help="choose Dayhoff-encoded amino acid signatures",
    )
    parser.add_argument(
        "--no-dayhoff",
        dest="dayhoff",
        action="store_false",
        help="do not choose Dayhoff-encoded amino acid signatures",
    )
    parser.set_defaults(dayhoff=False)

    parser.add_argument(
        "--hp",
        "--hydrophobic-polar",
        dest="hp",
        action="store_true",
        help="choose hydrophobic-polar-encoded amino acid signatures",
    )
    parser.add_argument(
        "--no-hp",
        "--no-hydrophobic-polar",
        dest="hp",
        action="store_false",
        help="do not choose hydrophobic-polar-encoded amino acid signatures",
    )
    parser.set_defaults(hp=False)

    parser.add_argument(
        "--skipm1n3",
        "--skipmer-m1n3",
        dest="skipm1n3",
        action="store_true",
        help="choose skipmer (m1n3) signatures",
    )

    parser.add_argument(
        "--no-skipm1n3",
        "--no-skipmer-m1n3",
        dest="skipm1n3",
        action="store_false",
        help="do not choose skipmer (m1n3) signatures",
    )
    parser.set_defaults(skipm1n3=False)

    parser.add_argument(
        "--skipm2n3",
        "--skipmer-m2n3",
        dest="skipm2n3",
        action="store_true",
        help="choose skipmer (m2n3) signatures",
    )
    parser.add_argument(
        "--no-skipm2n3",
        "--no-skipmer-m2n3",
        dest="skipm2n3",
        action="store_false",
        help="do not choose skipmer (m2n3) signatures",
    )
    parser.set_defaults(skipm2n3=False)

    parser.add_argument(
        "--dna",
        "--rna",
        "--nucleotide",
        dest="dna",
        default=None,
        action="store_true",
        help="choose a nucleotide signature (default: True)",
    )
    parser.add_argument(
        "--no-dna",
        "--no-rna",
        "--no-nucleotide",
        dest="dna",
        action="store_false",
        help="do not choose a nucleotide signature",
    )
    parser.set_defaults(dna=None)


def add_construct_moltype_args(parser):
    add_moltype_args(parser)
    parser.set_defaults(dna=True)


def add_ksize_arg(parser, *, default=None):
    "Add -k/--ksize to argparse parsers, with specified default."
    if default:
        message = f"k-mer size to select; default={default}"
    else:
        message = "k-mer size to select; no default."

    parser.add_argument(
        "-k",
        "--ksize",
        metavar="K",
        default=default,
        type=int,
        help=message,
    )


# https://stackoverflow.com/questions/55324449/how-to-specify-a-minimum-or-maximum-float-value-with-argparse#55410582
def range_limited_float_type(arg):
    """Type function for argparse - a float within some predefined bounds"""
    min_val = 0
    max_val = 1
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("\n\tERROR: Must be a floating point number.")
    if f < min_val or f > max_val:
        raise argparse.ArgumentTypeError(
            f"\n\tERROR: Argument must be >{str(min_val)} and <{str(max_val)}."
        )
    return f


def add_tax_threshold_arg(parser, containment_default=0.1, ani_default=None):
    parser.add_argument(
        "--containment-threshold",
        default=containment_default,
        type=range_limited_float_type,
        help=f"minimum containment threshold for classification; default={containment_default}",
    )
    parser.add_argument(
        "--ani-threshold",
        "--aai-threshold",
        default=ani_default,
        type=range_limited_float_type,
        help=f"minimum ANI threshold (nucleotide gather) or AAI threshold (protein gather) for classification; default={ani_default}",
    )


def add_picklist_args(parser):
    parser.add_argument(
        "--picklist",
        default=None,
        help="select signatures based on a picklist, i.e. 'file.csv:colname:coltype'",
    )
    parser.add_argument(
        "--picklist-require-all",
        default=False,
        action="store_true",
        help="require that all picklist values be found or else fail",
    )


def add_pattern_args(parser):
    parser.add_argument(
        "--include-db-pattern",
        default=None,
        help="search only signatures that match this pattern in name, filename, or md5",
    )
    parser.add_argument(
        "--exclude-db-pattern",
        default=None,
        help="search only signatures that do not match this pattern in name, filename, or md5",
    )


def opfilter(path):
    return not path.startswith("__") and path not in ["utils"]


def command_list(dirpath):
    paths = glob(os.path.join(dirpath, "*.py"))
    filenames = [os.path.basename(path) for path in paths]
    basenames = [
        os.path.splitext(path)[0] for path in filenames if not path.startswith("__")
    ]
    basenames = filter(opfilter, basenames)
    return sorted(basenames)


def add_scaled_arg(parser, default=None):
    parser.add_argument(
        "--scaled",
        metavar="FLOAT",
        type=check_scaled_bounds,
        default=default,
        help="downsample to this scaled; value should be between 100 and 1e6",
    )


def add_num_arg(parser, default=0):
    parser.add_argument(
        "-n",
        "--num-hashes",
        "--num",
        metavar="N",
        type=check_num_bounds,
        default=default,
        help="num value should be between 50 and 50000",
    )


def check_rank(args):
    """Check '--rank'/'--position'/'--lin-position' argument matches selected taxonomy."""
    standard_ranks = [
        "strain",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "superkingdom",
    ]
    if args.lins:
        if args.rank.isdigit():
            return str(args.rank)
        raise argparse.ArgumentTypeError(
            f"Invalid '--rank'/'--position' input: '{args.rank}'. '--lins' is specified. Rank must be an integer corresponding to a LIN position."
        )
    elif args.rank in standard_ranks:
        return args.rank
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid '--rank'/'--position' input: '{args.rank}'. Please choose: 'strain', 'species', 'genus', 'family', 'order', 'class', 'phylum', 'superkingdom'"
        )


def add_rank_arg(parser):
    parser.add_argument(
        "-r",
        "--rank",
        "--position",
        "--lin-position",
        help="For non-default output formats. Classify to this rank (tax genome) or summarize taxonomy at this rank and above (tax metagenome). \
              Note that the taxonomy CSV must contain lineage information at this rank, and that LIN positions start at 0. \
              Choices: 'strain', 'species', 'genus', 'family', 'order', 'class', 'phylum', 'superkingdom' or an integer LIN position",
    )


def check_tax_outputs(
    args,
    rank_required=["krona"],
    incompatible_with_lins=None,
    use_lingroup_format=False,
):
    "Handle ouput format combinations"
    # check that rank is passed for formats requiring rank.
    if not args.rank:
        if any(x in rank_required for x in args.output_format):
            raise ValueError(
                f"Rank (--rank) is required for {', '.join(rank_required)} output formats."
            )

    if args.lins:
        # check for outputs incompatible with lins
        if incompatible_with_lins:
            if any(x in args.output_format for x in incompatible_with_lins):
                raise ValueError(
                    f"The following outputs are incompatible with '--lins': : {', '.join(incompatible_with_lins)}"
                )
        # check that lingroup file exists if needed
        if args.lingroup:
            if use_lingroup_format and "lingroup" not in args.output_format:
                args.output_format.append("lingroup")
        elif "lingroup" in args.output_format:
            raise ValueError(
                "Must provide lingroup csv via '--lingroup' in order to output a lingroup report."
            )
    elif args.lingroup or "lingroup" in args.output_format:
        raise ValueError(
            "Must enable LIN taxonomy via '--lins' in order to use lingroups."
        )

    # check that only one output format is specified if writing to stdout
    if len(args.output_format) > 1:
        if args.output_base == "-":
            raise ValueError(
                f"Writing to stdout is incompatible with multiple output formats {args.output_format}"
            )
    elif not args.output_format:
        # change to "human" for 5.0
        args.output_format = ["csv_summary"]

    return args.output_format
