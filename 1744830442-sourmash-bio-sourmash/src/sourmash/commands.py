"""
Functions implementing the main command-line subcommands.
"""

import csv
import os
import os.path
import sys
import shutil
import io
import enum

import screed
from .compare import (
    compare_all_pairs,
    compare_serial_containment,
    compare_serial_max_containment,
    compare_serial_avg_containment,
)
from . import MinHash
from .sbtmh import load_sbt_index, create_sbt_index
from . import signature as sig
from . import sourmash_args
from .logging import notify, error, print_results, set_quiet
from .sourmash_args import FileOutput, FileOutputCSV, SaveSignaturesToLocation
from .search import prefetch_database, PrefetchResult
from .index import LazyLinearIndex
from sourmash.index.revindex import DiskRevIndex


WATERMARK_SIZE = 10000


class EnumIndexType(enum.StrEnum):  # used in 'index'
    SBT = "SBT"
    ROCKSDB = "rocksdb"
    ZIP = "zip"


def _get_screen_width():
    # default fallback is 80x24
    (col, rows) = shutil.get_terminal_size()

    return col


def compare(args):
    "Compare multiple signature files and create a distance matrix."
    import numpy

    set_quiet(args.quiet)
    moltype = sourmash_args.calculate_moltype(args)
    picklist = sourmash_args.load_picklist(args)
    pattern_search = sourmash_args.load_include_exclude_db_patterns(args)

    inp_files = list(args.signatures)
    if args.from_file:
        more_files = sourmash_args.load_pathlist_from_file(args.from_file)
        inp_files.extend(more_files)

    progress = sourmash_args.SignatureLoadingProgress()

    # load in the various signatures
    siglist = []
    ksizes = set()
    moltypes = set()
    size_may_be_inaccurate = False
    for filename in inp_files:
        notify(f"loading '{filename}'", end="\r")
        loaded = sourmash_args.load_file_as_signatures(
            filename,
            ksize=args.ksize,
            select_moltype=moltype,
            picklist=picklist,
            yield_all_files=args.force,
            progress=progress,
            pattern=pattern_search,
        )
        loaded = list(loaded)
        if not loaded:
            notify(
                f"\nwarning: no signatures loaded at given ksize/molecule type/picklist from {filename}"
            )

        # add to siglist; track ksizes/moltypes
        s = None
        for s in loaded:
            siglist.append((s, filename))
            ksizes.add(s.minhash.ksize)
            moltypes.add(sourmash_args.get_moltype(s))

        if s is None:
            notify(
                f"\nwarning: no signatures loaded at given ksize/molecule type/picklist from {filename}"
            )

        # error out while loading if we have more than one ksize/moltype
        if len(ksizes) > 1 or len(moltypes) > 1:
            break

    if not siglist:
        error("no signatures found! exiting.")
        sys.exit(-1)

    # check ksizes and type
    if len(ksizes) > 1:
        error("multiple k-mer sizes loaded; please specify one with -k.")
        ksizes = sorted(ksizes)
        error("(saw k-mer sizes {})".format(", ".join(map(str, ksizes))))
        sys.exit(-1)

    if len(moltypes) > 1:
        error("multiple molecule types loaded; please specify --dna, --protein")
        sys.exit(-1)

    notify(" " * 79, end="\r")
    notify(f"loaded {format(len(siglist))} signatures total.")

    if picklist:
        sourmash_args.report_picklist(args, picklist)

    # check to make sure they're potentially compatible - either using
    # scaled, or not.
    scaled_sigs = [s.minhash.scaled for (s, _) in siglist]
    is_scaled = all(scaled_sigs)
    is_scaled_2 = any(scaled_sigs)

    # complain if it's not all one or the other
    if is_scaled != is_scaled_2:
        error("ERROR: cannot mix scaled signatures with num signatures")
        sys.exit(-1)

    is_containment = False
    if args.containment or args.max_containment or args.avg_containment:
        is_containment = True

        containment_args = [
            args.containment,
            args.max_containment,
            args.avg_containment,
        ]
        if sum(containment_args) > 1:
            notify("ERROR: cannot specify more than one containment argument!")
            sys.exit(-1)

    # complain if --containment and not is_scaled
    if is_containment and not is_scaled:
        error(
            "must use scaled signatures with --containment, --max-containment, and --avg-containment"
        )
        sys.exit(-1)

    # complain if --ani and not is_scaled
    return_ani = False
    if args.estimate_ani:
        return_ani = True

    if return_ani and not is_scaled:
        error("must use scaled signatures with --estimate-ani")
        sys.exit(-1)

    # notify about implicit --ignore-abundance:
    if is_containment or return_ani:
        track_abundances = any(s.minhash.track_abundance for s, _ in siglist)
        if track_abundances:
            notify(
                "NOTE: --containment, --max-containment, --avg-containment, and --estimate-ani ignore signature abundances."
            )

    # CTB: note, up to this point, we could do everything with manifests
    # w/o actually loading any signatures. I'm not sure the manifest
    # API allows it tho.

    # if using scaled sketches or --scaled, downsample to common max scaled.
    printed_scaled_msg = False
    if is_scaled:
        max_scaled = max(s.minhash.scaled for s, _ in siglist)
        if args.scaled:
            args.scaled = int(args.scaled)

            max_scaled = max(max_scaled, args.scaled)
            if max_scaled > args.scaled:
                notify(
                    f"WARNING: --scaled specified {args.scaled}, but max scaled of sketches is {max_scaled}"
                )
                notify(f"WARNING: continuing with scaled value of {max_scaled}.")

        new_siglist = []
        for s, filename in siglist:
            if not size_may_be_inaccurate and not s.minhash.size_is_accurate():
                size_may_be_inaccurate = True
            if s.minhash.scaled != max_scaled:
                if not printed_scaled_msg:
                    notify(
                        f"NOTE: downsampling to scaled value of {format(max_scaled)}"
                    )
                    printed_scaled_msg = True
                with s.update() as s:
                    s.minhash = s.minhash.downsample(scaled=max_scaled)
                new_siglist.append((s, filename))
            else:
                new_siglist.append((s, filename))
        siglist = new_siglist
    elif args.scaled is not None:
        error("ERROR: cannot specify --scaled with non-scaled signatures.")
        sys.exit(-1)

    if len(siglist) == 0:
        error("no signatures!")
        sys.exit(-1)

    notify("")

    # build the distance matrix
    numpy.set_printoptions(precision=3, suppress=True)

    # do all-by-all calculation

    labeltext = [str(ss) for ss, _ in siglist]
    sigsonly = [ss for ss, _ in siglist]
    if args.containment:
        similarity = compare_serial_containment(sigsonly, return_ani=return_ani)
    elif args.max_containment:
        similarity = compare_serial_max_containment(sigsonly, return_ani=return_ani)
    elif args.avg_containment:
        similarity = compare_serial_avg_containment(sigsonly, return_ani=return_ani)
    else:
        similarity = compare_all_pairs(
            sigsonly,
            args.ignore_abundance,
            n_jobs=args.processes,
            return_ani=return_ani,
        )

    # if distance matrix desired, switch to 1-similarity
    if args.distance_matrix:
        matrix = 1 - similarity
    else:
        matrix = similarity

    if len(siglist) < 30:
        for i, (ss, filename) in enumerate(siglist):
            # for small matrices, pretty-print some output
            name_num = f"{i}-{str(ss)}"
            if len(name_num) > 20:
                name_num = name_num[:17] + "..."
            print_results(
                "{:20s}\t{}".format(
                    name_num,
                    matrix[
                        i,
                        :,
                    ],
                )
            )

    if args.distance_matrix:
        print_results("max distance in matrix: {:.3f}", numpy.max(matrix))
    else:
        print_results("min similarity in matrix: {:.3f}", numpy.min(matrix))

    # shall we output a matrix to stdout?
    if args.output:
        labeloutname = args.output + ".labels.txt"
        notify(f"saving labels to: {labeloutname}")
        with open(labeloutname, "w") as fp:
            fp.write("\n".join(labeltext))

        notify(f"saving comparison matrix to: {args.output}")
        with open(args.output, "wb") as fp:
            numpy.save(fp, matrix)

    # output labels information via --labels-to?
    if args.labels_to:
        labeloutname = args.labels_to
        notify(f"saving labels to: {labeloutname}")
        with sourmash_args.FileOutputCSV(labeloutname) as fp:
            w = csv.writer(fp)
            w.writerow(
                ["sort_order", "md5", "label", "name", "filename", "signature_file"]
            )

            for n, (ss, location) in enumerate(siglist):
                md5 = ss.md5sum()
                sigfile = location
                label = str(ss)
                name = ss.name
                filename = ss.filename

                w.writerow([str(n + 1), md5, label, name, filename, sigfile])

    # output CSV?
    if args.csv:
        with FileOutputCSV(args.csv) as csv_fp:
            w = csv.writer(csv_fp)
            w.writerow(labeltext)

            for i in range(len(labeltext)):
                y = []
                for j in range(len(labeltext)):
                    y.append(str(matrix[i][j]))
                w.writerow(y)

    if size_may_be_inaccurate:
        if args.distance_matrix:
            notify(
                "WARNING: size estimation for at least one of these sketches may be inaccurate. ANI distances will be set to 1 for these comparisons."
            )
        else:
            notify(
                "WARNING: size estimation for at least one of these sketches may be inaccurate. ANI values will be set to 1 for these comparisons."
            )


def plot(args):
    "Produce a clustering matrix and plot."
    import matplotlib as mpl

    mpl.use("Agg")
    import numpy
    import pylab
    import scipy.cluster.hierarchy as sch
    from . import fig as sourmash_fig

    # load files
    D_filename = args.distances

    notify(f"loading comparison matrix from {D_filename}...")
    with open(D_filename, "rb") as f:
        D = numpy.load(f)
    # not sure how to change this to use f-strings
    notify("...got {} x {} matrix.", *D.shape)

    # see sourmash#2790 for details :)
    if args.labeltext or args.labels or args.labels_from:
        if args.labeltext and args.labels_from:
            notify("ERROR: cannot supply both --labeltext and --labels-from")
            sys.exit(-1)
        display_labels = True
        args.labels = True  # override => labels always true
    elif args.labels is None and not args.indices:
        # default to labels
        args.labels = True
        display_labels = True
    elif args.indices or (not args.labels and args.indices is None):
        # turn on indices only, not label names
        args.indices = True
        display_labels = True
    else:
        display_labels = False

    if args.labels_from:
        labelfilename = args.labels_from
        notify(f"loading labels from CSV file '{labelfilename}'")

        labeltext = []
        with sourmash_args.FileInputCSV(labelfilename) as r:
            for row in r:
                order, label = row["sort_order"], row["label"]
                labeltext.append((int(order), label))
        labeltext.sort()
        labeltext = [t[1] for t in labeltext]
    elif args.labels:
        if args.labeltext:
            labelfilename = args.labeltext
        else:
            labelfilename = D_filename + ".labels.txt"

        notify(f"loading labels from text file '{labelfilename}'")
        with open(labelfilename) as f:
            labeltext = [x.strip() for x in f]

        if len(labeltext) != D.shape[0]:
            error("{} labels != matrix size, exiting", len(labeltext))
            sys.exit(-1)
    elif args.indices:
        # construct integer labels
        labeltext = [str(i + 1) for i in range(D.shape[0])]
    else:
        assert not display_labels
        labeltext = [""] * D.shape[0]

    if args.pdf:
        ext = ".pdf"
    else:
        ext = ".png"

    # build filenames, decide on PDF/PNG output
    dendrogram_out = os.path.basename(D_filename) + ".dendro" + ext
    matrix_out = os.path.basename(D_filename) + ".matrix" + ext
    hist_out = os.path.basename(D_filename) + ".hist" + ext

    # output to a different directory?
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        dendrogram_out = os.path.join(args.output_dir, dendrogram_out)
        matrix_out = os.path.join(args.output_dir, matrix_out)
        hist_out = os.path.join(args.output_dir, hist_out)

    # make the histogram
    notify(f"saving histogram of matrix values => {hist_out}")
    fig = pylab.figure(figsize=(8, 5))
    pylab.hist(numpy.array(D.flat), bins=100)
    fig.savefig(hist_out)

    ### make the dendrogram:
    fig = pylab.figure(figsize=(8, 5))
    ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax1.set_xticks([])
    ax1.set_yticks([])

    # subsample?
    if args.subsample:
        numpy.random.seed(args.subsample_seed)

        sample_idx = list(range(len(labeltext)))
        numpy.random.shuffle(sample_idx)
        sample_idx = sample_idx[: args.subsample]

        np_idx = numpy.array(sample_idx)
        D = D[numpy.ix_(np_idx, np_idx)]
        labeltext = [labeltext[idx] for idx in sample_idx]

    ### do clustering
    Y = sch.linkage(D, method="single")
    sch.dendrogram(
        Y, orientation="right", labels=labeltext, no_labels=not display_labels
    )
    fig.savefig(dendrogram_out)
    notify(f"wrote dendrogram to: {dendrogram_out}")

    ### make the dendrogram+matrix:
    (fig, rlabels, rmat) = sourmash_fig.plot_composite_matrix(
        D,
        labeltext,
        show_labels=display_labels,
        vmin=args.vmin,
        vmax=args.vmax,
        force=args.force,
    )
    fig.savefig(matrix_out)
    notify(f"wrote numpy distance matrix to: {matrix_out}")

    if len(labeltext) < 30:
        # for small matrices, print out sample numbering for FYI.
        for i, name in enumerate(labeltext):
            print_results("{}\t{}", i, name)

    # write out re-ordered matrix and labels
    if args.csv:
        with FileOutputCSV(args.csv) as csv_fp:
            w = csv.writer(csv_fp)
            w.writerow(rlabels)

            for i in range(len(rlabels)):
                y = []
                for j in range(len(rlabels)):
                    y.append(f"{rmat[i][j]}")
                w.writerow(y)
        notify(f"Wrote clustered matrix and labels out to {args.csv}")


def import_csv(args):
    "Import a CSV file full of signatures/hashes."

    with open(args.mash_csvfile, newline="") as fp:
        reader = csv.reader(fp)
        siglist = []
        for row in reader:
            hashfn = row[0]
            hashseed = int(row[1])

            # only support a limited import type, for now ;)
            assert hashfn == "murmur64"
            assert hashseed == 42

            _, _, ksize, name, hashes = row
            ksize = int(ksize)

            hashes = hashes.strip()
            hashes = list(map(int, hashes.split(" ")))

            e = MinHash(len(hashes), ksize)
            e.add_many(hashes)
            s = sig.SourmashSignature(e, filename=name)
            siglist.append(s)
            notify(f"loaded signature: {name} {s.md5sum()[:8]}")

        notify(f"saving {len(siglist)} signatures to JSON")
        with SaveSignaturesToLocation(args.output) as save_sig:
            save_sig.add_many(siglist)


def sbt_combine(args):
    inp_files = list(args.sbts)
    notify(f"combining {len(inp_files)} SBTs")

    tree = load_sbt_index(inp_files.pop(0))

    for f in inp_files:
        new_tree = load_sbt_index(f)
        # TODO: check if parameters are the same for both trees!
        tree.combine(new_tree)

    notify(f'saving SBT under "{args.sbt_name}".')
    tree.save(args.sbt_name)


def index(args):
    """
    Build an on-disk index of the given signatures. Currently supports
    SBT and RocksDB inverted indices.
    """
    set_quiet(args.quiet)
    moltype = sourmash_args.calculate_moltype(args)
    picklist = sourmash_args.load_picklist(args)

    index_type = args.index_type
    add_sketch = None
    output_name = args.name

    # check input options
    if index_type != "SBT":
        if args.append:
            error("cannot only use --append with an SBT index type")
            sys.exit(-1)
        if args.sparseness > 0.0:
            error("cannot use use --sparseness with an SBT index type")
            sys.exit(-1)

    # open writing
    match index_type:
        case EnumIndexType.SBT:
            if args.sparseness < 0 or args.sparseness > 1.0:
                error("sparseness must be in range [0.0, 1.0].")
                sys.exit(-1)

            if args.append:
                tree = load_sbt_index(args.name)
            else:
                tree = create_sbt_index(args.bf_size, n_children=args.n_children)

            def add_sketch(sigobj):
                tree.insert(sigobj)
        case EnumIndexType.ZIP:
            save_sigs = sourmash_args.SaveSignaturesToLocation(output_name)
            save_sigs.open()

            def add_sketch(sigobj):
                save_sigs.add(sigobj)
        case EnumIndexType.ROCKSDB:
            full_siglist = []

            def add_sketch(sigobj):
                full_siglist.append(sigobj)
        case _:
            error(f"ERROR: unknown index type '{index_type}'; quitting.")
            sys.exit(-1)

    if args.scaled:
        args.scaled = int(args.scaled)
        notify(f"downsampling signatures to scaled={args.scaled}")

    inp_files = list(args.signatures)
    if args.from_file:
        more_files = sourmash_args.load_pathlist_from_file(args.from_file)
        inp_files.extend(more_files)

    if not inp_files:
        error("ERROR: no files to index!? Supply on command line or use --from-file")
        sys.exit(-1)

    notify(f"loading {len(inp_files)} files into {index_type} index")

    progress = sourmash_args.SignatureLoadingProgress()

    n = 0
    ksizes = set()
    moltypes = set()
    nums = set()
    scaleds = set()
    full_siglist = []
    for f in inp_files:
        siglist = sourmash_args.load_file_as_signatures(
            f,
            ksize=args.ksize,
            select_moltype=moltype,
            yield_all_files=args.force,
            picklist=picklist,
            progress=progress,
        )

        # load all matching signatures in this file
        ss = None
        for ss in siglist:
            ksizes.add(ss.minhash.ksize)
            moltypes.add(sourmash_args.get_moltype(ss))
            nums.add(ss.minhash.num)

            with ss.update() as ss:
                if args.scaled:
                    ss.minhash = ss.minhash.downsample(scaled=args.scaled)
                if ss.minhash.track_abundance:
                    ss.minhash = ss.minhash.flatten()

            scaleds.add(ss.minhash.scaled)

            add_sketch(ss)
            n += 1

        if not ss:
            continue

        # check to make sure we aren't loading incompatible signatures
        if len(ksizes) > 1 or len(moltypes) > 1:
            error("multiple k-mer sizes or molecule types present; fail.")
            error("specify --dna/--protein and --ksize as necessary")
            error(
                "ksizes: {}; moltypes: {}",
                ", ".join(map(str, ksizes)),
                ", ".join(moltypes),
            )
            sys.exit(-1)

        if nums == {0} and len(scaleds) == 1:
            pass  # good
        elif scaleds == {0} and len(nums) == 1:
            pass  # also good
        else:
            error("trying to build an index with incompatible signatures.")
            error("nums = {}; scaleds = {}", repr(nums), repr(scaleds))
            sys.exit(-1)

    notify("")

    # did we load any!?
    if n == 0:
        error("no signatures found to load into index!? failing.")
        sys.exit(-1)

    if picklist:
        sourmash_args.report_picklist(args, picklist)

    match index_type:
        case EnumIndexType.SBT:
            notify(f'loaded {n} sigs; saving SBT under "{output_name}"')
            tree.save(output_name, sparseness=args.sparseness)
            if tree.storage:
                tree.storage.close()
        case EnumIndexType.ZIP:
            notify(f'loaded {n} sigs; saving zip file under "{output_name}"')
            save_sigs.close()
        case EnumIndexType.ROCKSDB:
            notify(f'loaded {n} sigs; saving rocksdb index under "{output_name}"')
            DiskRevIndex.create_from_sigs(full_siglist, output_name)


def search(args):
    from .search import (
        search_databases_with_flat_query,
        search_databases_with_abund_query,
    )

    set_quiet(args.quiet, args.debug)
    moltype = sourmash_args.calculate_moltype(args)
    picklist = sourmash_args.load_picklist(args)
    pattern_search = sourmash_args.load_include_exclude_db_patterns(args)

    # set up the query.
    query = sourmash_args.load_query_signature(
        args.query, ksize=args.ksize, select_moltype=moltype, select_md5=args.md5
    )
    notify(
        f"loaded query: {str(query)[:30]}... (k={query.minhash.ksize}, {sourmash_args.get_moltype(query)})"
    )

    if args.scaled:
        if not query.minhash.scaled:
            error("cannot downsample a signature not created with --scaled")
            sys.exit(-1)
        if args.scaled != query.minhash.scaled:
            notify(
                f"downsampling query from scaled={query.minhash.scaled} to {int(args.scaled)}"
            )
            with query.update() as query:
                query.minhash = query.minhash.downsample(scaled=args.scaled)

    # set up the search databases
    is_containment = args.containment or args.max_containment
    if is_containment:
        if args.containment and args.max_containment:
            notify("ERROR: cannot specify both --containment and --max-containment!")
            sys.exit(-1)

    databases = sourmash_args.load_dbs_and_sigs(
        args.databases,
        query,
        not is_containment,
        picklist=picklist,
        pattern=pattern_search,
        fail_on_empty_database=args.fail_on_empty_database,
    )

    # handle signatures with abundance
    if query.minhash.track_abundance:
        if args.ignore_abundance:
            if query.minhash.track_abundance:
                # abund sketch + ignore abundance => flatten sketch.
                with query.update() as query:
                    query.minhash = query.minhash.flatten()
        elif args.containment or args.max_containment:
            # abund sketch + keep abundance => no containment searches
            notify(
                "ERROR: cannot do containment searches on an abund signature; maybe specify --ignore-abundance?"
            )
            sys.exit(-1)
    else:
        # forcibly ignore abundances if query has no abundances
        args.ignore_abundance = True

    # do the actual search
    if query.minhash.track_abundance:
        try:
            results = search_databases_with_abund_query(
                query,
                databases,
                threshold=args.threshold,
                do_containment=args.containment,
                do_max_containment=args.max_containment,
                best_only=args.best_only,
                unload_data=True,
            )
        except TypeError as exc:
            error(f"ERROR: {str(exc)}")
            sys.exit(-1)
    else:
        results = search_databases_with_flat_query(
            query,
            databases,
            threshold=args.threshold,
            do_containment=args.containment,
            do_max_containment=args.max_containment,
            best_only=args.best_only,
            unload_data=True,
            estimate_ani_ci=args.estimate_ani_ci,
        )

    n_matches = len(results)
    if args.best_only:
        args.num_results = 1

    if not args.num_results or n_matches <= args.num_results:
        print_results(f"{len(results)} matches above threshold {args.threshold:0.3f}:")
    else:
        print_results(
            f"{len(results)} matches above threshold {args.threshold:0.3f}; showing first {args.num_results}:"
        )

        n_matches = args.num_results

    size_may_be_inaccurate = False
    jaccard_ani_untrustworthy = False

    # output!
    print_results("similarity   match")
    print_results("----------   -----")
    for sr in results[:n_matches]:
        pct = f"{sr.similarity * 100:.1f}%"
        name = sr.match._display_name(60)
        print_results("{:>6}       {}", pct, name)
        if sr.cmp_scaled is not None:
            if not size_may_be_inaccurate and sr.size_may_be_inaccurate:
                size_may_be_inaccurate = True
            if not is_containment and sr.cmp.jaccard_ani_untrustworthy:
                jaccard_ani_untrustworthy = True

    if args.best_only:
        notify("** reporting only one match because --best-only was set")

    writer = None
    if args.output:
        with FileOutputCSV(args.output) as fp:
            for sr in results:
                # if this is the first result we're writing, initialize the csv, return writer
                if writer is None:
                    writer = sr.init_dictwriter(fp)
                sr.write(writer)

    # save matching signatures upon request
    if args.save_matches:
        notify(f'saving all matched signatures to "{args.save_matches}"')

        with SaveSignaturesToLocation(args.save_matches) as save_sig:
            for sr in results:
                save_sig.add(sr.match)

    if picklist:
        sourmash_args.report_picklist(args, picklist)

    if size_may_be_inaccurate:
        notify(
            "WARNING: size estimation for at least one of these sketches may be inaccurate. ANI values will not be reported for these comparisons."
        )
    if jaccard_ani_untrustworthy:
        notify(
            "WARNING: Jaccard estimation for at least one of these comparisons is likely inaccurate. Could not estimate ANI for these comparisons."
        )


def categorize(args):
    "Use a database to find the best match to many signatures."
    from .index import MultiIndex
    from .search import make_jaccard_search_query

    set_quiet(args.quiet)
    moltype = sourmash_args.calculate_moltype(args)

    # eliminate names we've already categorized
    already_names = set()
    if args.load_csv:
        with open(args.load_csv, newline="") as fp:
            r = csv.reader(fp)
            for row in r:
                already_names.add(row[0])

    # load search database
    db = sourmash_args.load_file_as_index(args.database)
    if args.ksize or moltype:
        db = db.select(ksize=args.ksize, moltype=moltype)

    # utility function to load & select relevant signatures.
    def _yield_all_sigs(queries, ksize, moltype):
        for filename in queries:
            mi = MultiIndex.load_from_path(filename, False)
            mi = mi.select(ksize=ksize, moltype=moltype)
            yield from mi.signatures_with_location()

    csv_w = None
    csv_fp = None
    if args.csv:
        csv_fp = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_fp)

    search_obj = make_jaccard_search_query(threshold=args.threshold)
    for orig_query, loc in _yield_all_sigs(args.queries, args.ksize, moltype):
        # skip if we've already done signatures from this file.
        if loc in already_names:
            continue

        notify(
            f"loaded query: {str(orig_query)[:30]}... (k={orig_query.minhash.ksize}, {orig_query.minhash.moltype})"
        )

        if args.ignore_abundance and orig_query.minhash.track_abundance:
            query = orig_query.copy()
            with query.update() as query:
                query.minhash = query.minhash.flatten()
        else:
            if orig_query.minhash.track_abundance:
                notify(
                    "ERROR: this search cannot be done on signatures calculated with abundance."
                )
                notify("ERROR: please specify --ignore-abundance.")
                sys.exit(-1)

            query = orig_query.copy()

        results = []
        for sr in db.find(search_obj, query):
            match = sr.signature
            if match.md5sum() != query.md5sum():  # ignore self.
                results.append((orig_query.similarity(match), match))

        if results:
            results.sort(key=lambda x: -x[0])  # reverse sort on similarity
            best_hit_sim, best_hit_query = results[0]
            notify(f"for {query}, found: {best_hit_sim:.2f} {best_hit_query}")
            best_hit_query_name = best_hit_query.name
            if csv_w:
                csv_w.writerow([loc, query, best_hit_query_name, best_hit_sim])
        else:
            notify(f"for {query}, no match found")

    if csv_fp:
        csv_fp.close()


def gather(args):
    from .search import GatherDatabases, format_bp

    set_quiet(args.quiet, args.debug)
    moltype = sourmash_args.calculate_moltype(args)
    picklist = sourmash_args.load_picklist(args)
    pattern_search = sourmash_args.load_include_exclude_db_patterns(args)

    # load the query signature & figure out all the things
    query = sourmash_args.load_query_signature(
        args.query, ksize=args.ksize, select_moltype=moltype, select_md5=args.md5
    )
    notify(
        f"loaded query: {str(query)[:30]}... (k={query.minhash.ksize}, {sourmash_args.get_moltype(query)})"
    )

    # verify signature was computed right.
    if not query.minhash.scaled:
        error("query signature needs to be created with --scaled")
        sys.exit(-1)

    if args.scaled and args.scaled != query.minhash.scaled:
        notify(
            f"downsampling query from scaled={query.minhash.scaled} to {int(args.scaled)}"
        )
        with query.update() as query:
            query.minhash = query.minhash.downsample(scaled=args.scaled)

    # empty?
    if not len(query.minhash):
        error("no query hashes!? exiting.")
        sys.exit(-1)

    # set up the search databases
    cache_size = args.cache_size
    if args.cache_size == 0:
        cache_size = None
    databases = sourmash_args.load_dbs_and_sigs(
        args.databases,
        query,
        False,
        cache_size=cache_size,
        picklist=picklist,
        pattern=pattern_search,
        fail_on_empty_database=args.fail_on_empty_database,
    )

    if args.linear:  # force linear traversal?
        databases = [LazyLinearIndex(db) for db in databases]

    size_may_be_inaccurate = False
    if args.prefetch:  # note: on by default!
        notify("Starting prefetch sweep across databases.")
        prefetch_query = query.copy()
        if prefetch_query.minhash.track_abundance:
            with prefetch_query.update() as prefetch_query:
                prefetch_query.minhash = prefetch_query.minhash.flatten()

        noident_mh = prefetch_query.minhash.to_mutable()
        total_prefetch = 0
        save_prefetch = SaveSignaturesToLocation(args.save_prefetch)
        save_prefetch.open()
        # set up prefetch CSV output
        prefetch_csvout_fp = None
        prefetch_csvout_w = None
        if args.save_prefetch_csv:
            prefetch_csvout_fp = FileOutputCSV(args.save_prefetch_csv).open()

            query_mh = prefetch_query.minhash
            scaled = query_mh.scaled

        counters = []
        ident_mh = noident_mh.copy_and_clear()
        for db in databases:
            counter = None
            try:
                counter = db.counter_gather(prefetch_query, args.threshold_bp)
            except ValueError:
                # catch "no signatures to search" ValueError if empty db.
                continue

            total_prefetch += len(counter)
            if args.save_prefetch:
                save_prefetch.add_many(counter.signatures())

            # update found/not found hashes from the union/intersection of
            # found.
            union_found = counter.union_found
            ident_mh.add_many(union_found)
            noident_mh.remove_many(union_found)

            # optionally calculate and output prefetch info to csv
            if prefetch_csvout_fp:
                for found_sig in counter.signatures():
                    # calculate intersection stats and info
                    prefetch_result = PrefetchResult(
                        prefetch_query,
                        found_sig,
                        cmp_scaled=scaled,
                        threshold_bp=args.threshold_bp,
                        estimate_ani_ci=args.estimate_ani_ci,
                    )
                    if prefetch_csvout_w is None:
                        prefetch_csvout_w = prefetch_result.init_dictwriter(
                            prefetch_csvout_fp
                        )
                    prefetch_result.write(prefetch_csvout_w)

            counters.append(counter)

            # flush csvout so that things get saved progressively
            if prefetch_csvout_fp:
                prefetch_csvout_fp.flush()

        display_bp = format_bp(args.threshold_bp)
        notify(
            f"Prefetch found {total_prefetch} signatures with overlap >= {display_bp}."
        )
        save_prefetch.close()
        if prefetch_csvout_fp:
            prefetch_csvout_fp.close()
    else:
        counters = databases
        # we can't track unidentified hashes w/o prefetch
        noident_mh = None
        ident_mh = None

    ## ok! now do gather -
    notify("Doing gather to generate minimum metagenome cover.")

    found = 0
    is_abundance = query.minhash.track_abundance and not args.ignore_abundance
    orig_query_mh = query.minhash
    if not orig_query_mh.size_is_accurate():
        size_may_be_inaccurate = True
    gather_iter = GatherDatabases(
        query,
        counters,
        threshold_bp=args.threshold_bp,
        ignore_abundance=args.ignore_abundance,
        noident_mh=noident_mh,
        ident_mh=ident_mh,
        estimate_ani_ci=args.estimate_ani_ci,
    )

    screen_width = _get_screen_width()
    sum_f_uniq_found = 0.0
    result = None

    ### open output handles as needed for (1) saving CSV (2) saving matches

    # save matching signatures?
    if args.save_matches:
        notify(f"saving all matches to '{args.save_matches}'")
        save_sig_obj = SaveSignaturesToLocation(args.save_matches)
        save_sig = save_sig_obj.__enter__()
    else:
        save_sig_obj = None
        save_sig = None

    # save CSV?
    csv_outfp = io.StringIO()
    csv_writer = None

    try:
        for result in gather_iter:
            found += 1
            sum_f_uniq_found += result.f_unique_to_query

            if found == 1:  # first result? print header.
                if is_abundance:
                    print_results("")
                    print_results("overlap     p_query p_match avg_abund")
                    print_results("---------   ------- ------- ---------")
                else:
                    print_results("")
                    print_results("overlap     p_query p_match")
                    print_results("---------   ------- -------")

            # print interim result & save in `found` list for later use
            pct_query = f"{result.f_unique_weighted * 100:.1f}%"
            pct_genome = f"{result.f_match * 100:.1f}%"

            if is_abundance:
                name = result.match._display_name(screen_width - 41)
                average_abund = f"{result.average_abund:.1f}"
                print_results(
                    "{:9}   {:>7} {:>7} {:>9}    {}",
                    format_bp(result.intersect_bp),
                    pct_query,
                    pct_genome,
                    average_abund,
                    name,
                )
            else:
                name = result.match._display_name(screen_width - 31)
                print_results(
                    "{:9}   {:>7} {:>7}    {}",
                    format_bp(result.intersect_bp),
                    pct_query,
                    pct_genome,
                    name,
                )

            # write out CSV
            if args.output:
                if csv_writer is None:
                    csv_writer = result.init_dictwriter(csv_outfp)
                result.write(csv_writer)

            # save matches?
            if save_sig is not None:
                save_sig.add(result.match)

            if args.num_results and found >= args.num_results:
                break
    finally:
        if save_sig_obj:
            save_sig_obj.close()
            save_sig_obj = None
            save_sig = None

    # report on thresholding -
    if gather_iter.query:
        # if still a query, then we failed the threshold.
        notify(f"found less than {format_bp(args.threshold_bp)} in common. => exiting")

    # basic reporting:
    if found:
        print_results(f"\nfound {found} matches total;")
        if found == args.num_results:
            print_results(
                f"(truncated gather because --num-results={args.num_results})"
            )
    else:
        display_bp = format_bp(args.threshold_bp)
        notify(f"\nNo matches found for --threshold-bp at {display_bp}.")

    if found:
        if is_abundance and result:
            p_covered = result.sum_weighted_found / result.total_weighted_hashes
            p_covered *= 100
            print_results(
                f"the recovered matches hit {p_covered:.1f}% of the abundance-weighted query."
            )

        print_results(
            f"the recovered matches hit {sum_f_uniq_found * 100:.1f}% of the query k-mers (unweighted)."
        )

    print_results("")
    if gather_iter.scaled != query.minhash.scaled:
        print_results(
            f"WARNING: final scaled was {gather_iter.scaled}, vs query scaled of {query.minhash.scaled}"
        )

    # save CSV?
    if (found and args.output) or args.create_empty_results:
        with FileOutputCSV(args.output) as fp:
            fp.write(csv_outfp.getvalue())

    # save unassigned hashes?
    if args.output_unassigned:
        remaining_query = gather_iter.query
        if not (remaining_query.minhash or noident_mh):
            notify("no unassigned hashes to save with --output-unassigned!")
        else:
            notify(f"saving unassigned hashes to '{args.output_unassigned}'")

            if noident_mh:
                remaining_mh = remaining_query.minhash.to_mutable()
                remaining_mh += noident_mh
                remaining_query.minhash = remaining_mh

            if is_abundance:
                abund_query_mh = remaining_query.minhash.inflate(orig_query_mh)
                remaining_query.minhash = abund_query_mh

            with SaveSignaturesToLocation(args.output_unassigned) as save_sig:
                save_sig.add(remaining_query)

    if picklist:
        sourmash_args.report_picklist(args, picklist)

    if size_may_be_inaccurate:
        notify(
            "WARNING: size estimation for at least one of these sketches may be inaccurate. ANI values will not be reported for these comparisons."
        )
    # DONE w/gather function.


def multigather(args):
    "Gather many signatures against multiple databases."
    from .search import GatherDatabases, format_bp

    set_quiet(args.quiet)
    moltype = sourmash_args.calculate_moltype(args)

    if not args.db:
        error("Error! must specify at least one database with --db")
        sys.exit(-1)

    if not args.query and not args.query_from_file:
        error("Error! must specify at least one query signature with --query")
        sys.exit(-1)

    # flatten --db and --query
    args.db = [item for sublist in args.db for item in sublist]
    inp_files = [item for sublist in args.query for item in sublist]
    if args.query_from_file:
        more_files = sourmash_args.load_pathlist_from_file(args.query_from_file)
        inp_files.extend(more_files)

    # need a query to get ksize, moltype for db loading
    query = next(
        iter(
            sourmash_args.load_file_as_signatures(
                inp_files[0], ksize=args.ksize, select_moltype=moltype
            )
        )
    )

    notify(
        f"loaded first query: {str(query)[:30]}... (k={query.minhash.ksize}, {sourmash_args.get_moltype(query)})"
    )

    databases = sourmash_args.load_dbs_and_sigs(
        args.db, query, False, fail_on_empty_database=args.fail_on_empty_database
    )

    # run gather on all the queries.
    n = 0
    size_may_be_inaccurate = False
    output_base_tracking = set()  # make sure we are not reusing 'output_base'
    for queryfile in inp_files:
        # load the query signature(s) & figure out all the things
        for query in sourmash_args.load_file_as_signatures(
            queryfile, ksize=args.ksize, select_moltype=moltype
        ):
            notify(
                f"loaded query: {str(query)[:30]}... (k={query.minhash.ksize}, {sourmash_args.get_moltype(query)})"
            )

            # verify signature was computed right.
            if not query.minhash.scaled:
                error("query signature needs to be created with --scaled; skipping")
                continue

            if args.scaled and args.scaled != query.minhash.scaled:
                notify(
                    f"downsampling query from scaled={query.minhash.scaled} to {int(args.scaled)}"
                )
                with query.update() as query:
                    query.minhash = query.minhash.downsample(scaled=args.scaled)

            # empty?
            if not len(query.minhash):
                error("no query hashes!? skipping to next..")
                continue

            counters = []
            prefetch_query = query.copy()
            if prefetch_query.minhash.track_abundance:
                with prefetch_query.update() as prefetch_query:
                    prefetch_query.minhash = prefetch_query.minhash.flatten()

            ident_mh = prefetch_query.minhash.copy_and_clear()
            noident_mh = prefetch_query.minhash.to_mutable()

            counters = []
            for db in databases:
                try:
                    counter = db.counter_gather(prefetch_query, args.threshold_bp)
                except ValueError:
                    # catch "no signatures to search" ValueError if empty db.
                    continue
                counters.append(counter)

                # track found/not found hashes
                union_found = counter.union_found
                noident_mh.remove_many(union_found)
                ident_mh.add_many(union_found)

            found = 0
            is_abundance = query.minhash.track_abundance and not args.ignore_abundance
            orig_query_mh = query.minhash
            gather_iter = GatherDatabases(
                query,
                counters,
                threshold_bp=args.threshold_bp,
                ignore_abundance=args.ignore_abundance,
                noident_mh=noident_mh,
                ident_mh=ident_mh,
            )

            screen_width = _get_screen_width()
            sum_f_uniq_found = 0.0
            result = None

            query_filename = query.filename
            if not query_filename or query_filename == "-":
                # use md5sum if query.filename not properly set
                output_base = query.md5sum()
            elif args.output_add_query_md5sum:
                # Uniquify the output file if all signatures were made from the same file (e.g. with --singleton)
                assert query_filename and query_filename != "-"  # first branch
                output_base = os.path.basename(query_filename) + "." + query.md5sum()
            else:
                output_base = os.path.basename(query_filename)

            if args.output_dir:
                output_base = os.path.join(args.output_dir, output_base)

            # track overwrites of output files!
            if output_base in output_base_tracking:
                error(
                    f"ERROR: detected overwritten outputs! '{output_base}' has already been used. Failing."
                )
                if args.force_allow_overwrite_output:
                    error("continuing because --force-allow-overwrite was specified")
                else:
                    error(
                        "Consider using '-U/--output-add-query-md5sum' to build unique outputs"
                    )
                    error("and/or '--force-allow-overwrite-output'")
                    sys.exit(-1)

            output_base_tracking.add(output_base)

            output_matches = output_base + ".matches.sig"
            save_sig_obj = SaveSignaturesToLocation(output_matches)
            save_sig = save_sig_obj.__enter__()
            notify(f"saving all matching signatures to '{output_matches}'")

            # write out basic CSV file
            output_csv = output_base + ".csv"
            notify(f'saving all CSV matches to "{output_csv}"')
            csv_out_obj = FileOutputCSV(output_csv)
            csv_outfp = csv_out_obj.__enter__()
            csv_writer = None

            for result in gather_iter:
                found += 1
                sum_f_uniq_found += result.f_unique_to_query
                if found == 1:  # first result? print header.
                    if is_abundance:
                        print_results("")
                        print_results("overlap     p_query p_match avg_abund")
                        print_results("---------   ------- ------- ---------")
                    else:
                        print_results("")
                        print_results("overlap     p_query p_match")
                        print_results("---------   ------- -------")

                # print interim result & save in a list for later use
                pct_query = f"{result.f_unique_weighted * 100:.1f}%"
                pct_genome = f"{result.f_match * 100:.1f}%"

                if is_abundance:
                    name = result.match._display_name(screen_width - 41)
                    average_abund = f"{result.average_abund:.1f}"
                    print_results(
                        "{:9}   {:>7} {:>7} {:>9}    {}",
                        format_bp(result.intersect_bp),
                        pct_query,
                        pct_genome,
                        average_abund,
                        name,
                    )
                else:
                    name = result.match._display_name(screen_width - 31)
                    print_results(
                        "{:9}   {:>7} {:>7}    {}",
                        format_bp(result.intersect_bp),
                        pct_query,
                        pct_genome,
                        name,
                    )

                if csv_writer is None:
                    csv_writer = result.init_dictwriter(csv_outfp)
                result.write(csv_writer)

                save_sig.add(result.match)

                # check for size estimation accuracy, which impacts ANI estimation
                if not size_may_be_inaccurate and result.size_may_be_inaccurate:
                    size_may_be_inaccurate = True

            # report on thresholding -
            if gather_iter.query.minhash:
                # if still a query, then we failed the threshold.
                notify(
                    f"found less than {format_bp(args.threshold_bp)} in common. => exiting"
                )

            # basic reporting
            print_results("\nfound {} matches total;", found)

            # close saving etc.
            save_sig_obj.close()
            save_sig_obj = save_sig = None

            csv_out_obj.close()
            csv_out_obj = csv_outfp = csv_writer = None

            if is_abundance and result:
                p_covered = result.sum_weighted_found / result.total_weighted_hashes
                p_covered *= 100
                print_results(
                    f"the recovered matches hit {p_covered:.1f}% of the abundance-weighted query."
                )

            print_results(
                f"the recovered matches hit {sum_f_uniq_found * 100:.1f}% of the query k-mers (unweighted)."
            )
            print_results("")

            if found == 0:
                notify("nothing found... skipping.")
                continue

            output_unassigned = output_base + f".unassigned{args.extension}"
            remaining_query = gather_iter.query
            if noident_mh:
                remaining_mh = remaining_query.minhash.to_mutable()
                remaining_mh += noident_mh.downsample(scaled=remaining_mh.scaled)
                remaining_query.minhash = remaining_mh

            if is_abundance:
                abund_query_mh = remaining_query.minhash.inflate(orig_query_mh)
                remaining_query.minhash = abund_query_mh

            if found == 0:
                notify("nothing found - entire query signature unassigned.")
            elif not remaining_query:
                notify("no unassigned hashes! not saving.")
            else:
                notify(f'saving unassigned hashes to "{output_unassigned}"')

            with SaveSignaturesToLocation(output_unassigned) as save_sig:
                save_sig.add(remaining_query)

            n += 1

        # fini, next query!

    # done! report at end.
    notify(f"\nconducted gather searches on {n} signatures")
    if size_may_be_inaccurate:
        notify(
            "WARNING: size estimation for at least one of these sketches may be inaccurate. ANI values will not be reported for these comparisons."
        )


def watch(args):
    "Build a signature from raw FASTA/FASTQ coming in on stdin, search."
    set_quiet(args.quiet)

    if args.input_is_protein and args.dna:
        notify("WARNING: input is protein, turning off nucleotide hashing.")
        args.dna = False
        args.protein = True

    if args.dna and args.protein:
        notify('ERROR: cannot use "watch" with both nucleotide and protein.')

    if args.dna:
        moltype = "DNA"
        is_protein = False
        dayhoff = False
        hp = False
    elif args.protein:
        moltype = "protein"
        is_protein = True
        dayhoff = False
        hp = False
    elif args.dayhoff:
        moltype = "dayhoff"
        is_protein = True
        dayhoff = True
        hp = False
    else:
        moltype = "hp"
        is_protein = True
        dayhoff = False
        hp = True

    tree = load_sbt_index(args.sbt_name)

    # check ksize from the SBT we are loading
    ksize = args.ksize
    if ksize is None:
        leaf = next(iter(tree.leaves()))
        tree_mh = leaf.data.minhash
        ksize = tree_mh.ksize

    E = MinHash(
        ksize=ksize, n=args.num_hashes, is_protein=is_protein, dayhoff=dayhoff, hp=hp
    )

    notify(f"Computing signature for k={ksize}, {moltype} from stdin")

    def do_search():
        results = []
        streamsig = sig.SourmashSignature(E, filename="stdin", name=args.name)
        for similarity, match, _ in tree.search(
            streamsig,
            threshold=args.threshold,
            best_only=True,
            ignore_abundance=True,
            do_containment=False,
        ):
            results.append((similarity, match))

        return results

    notify("reading sequences from stdin")
    watermark = WATERMARK_SIZE

    # iterate over input records
    n = 0
    with screed.open(args.inp_file) as screed_iter:
        for n, record in enumerate(screed_iter):
            # at each watermark, print status & check cardinality
            if n >= watermark:
                notify(f"\r... read {n} sequences", end="")
                watermark += WATERMARK_SIZE

                if do_search():
                    break

            if args.input_is_protein:
                E.add_protein(record.sequence)
            else:
                E.add_sequence(record.sequence, False)

    results = do_search()
    if not results:
        notify(f"... read {n} sequences, no matches found.")
    else:
        results.sort(key=lambda x: -x[0])  # take best
        similarity, found_sig = results[0]
        print_results("FOUND: {}, at {:.3f}", found_sig, similarity)

    if args.output:
        notify(f"saving signature to '{args.output}'")
        streamsig = sig.SourmashSignature(E, filename="stdin", name=args.name)
        with SaveSignaturesToLocation(args.output) as save_sig:
            save_sig.add(streamsig)


def migrate(args):
    "Migrate an SBT database to the latest version."
    tree = load_sbt_index(args.sbt_name, print_version_warning=False)

    notify(f'saving SBT under "{args.sbt_name}".')
    tree.save(args.sbt_name, structure_only=True)


def prefetch(args):
    "Output the 'raw' results of a containment/overlap search."

    # load databases from files, too.
    if args.db_from_file:
        more_db = sourmash_args.load_pathlist_from_file(args.db_from_file)
        args.databases.extend(more_db)

    if not args.databases:
        notify("ERROR: no databases or signatures to search!?")
        sys.exit(-1)

    if not (
        args.save_unmatched_hashes
        or args.save_matching_hashes
        or args.save_matches
        or args.output
    ):
        notify(
            "WARNING: no output(s) specified! Nothing will be saved from this prefetch!"
        )

    # figure out what k-mer size and molecule type we're looking for here
    ksize = args.ksize
    moltype = sourmash_args.calculate_moltype(args)
    picklist = sourmash_args.load_picklist(args)
    pattern_search = sourmash_args.load_include_exclude_db_patterns(args)

    # load the query signature & figure out all the things
    query = sourmash_args.load_query_signature(
        args.query, ksize=args.ksize, select_moltype=moltype, select_md5=args.md5
    )
    notify(
        f"loaded query: {str(query)[:30]}... (k={query.minhash.ksize}, {sourmash_args.get_moltype(query)})"
    )

    # verify signature was computed with scaled.
    if not query.minhash.scaled:
        error("query signature needs to be created with --scaled")
        sys.exit(-1)

    # if with track_abund, flatten me
    query_mh = query.minhash
    orig_query_mh = query_mh
    if query_mh.track_abundance:
        query_mh = query_mh.flatten()

    if args.scaled and args.scaled != query_mh.scaled:
        notify(
            f"downsampling query from scaled={query_mh.scaled} to {int(args.scaled)}"
        )
        query_mh = query_mh.downsample(scaled=args.scaled)

    notify(
        f"query sketch has scaled={query_mh.scaled}; will be dynamically downsampled as needed."
    )
    common_scaled = query_mh.scaled

    # empty?
    if not len(query_mh):
        error("no query hashes!? exiting.")
        sys.exit(-1)

    with query.update() as query:
        query.minhash = query_mh
    ksize = query_mh.ksize

    # set up CSV output, write headers, etc.
    csvout_fp = None
    csvout_w = None
    if args.output:
        csvout_fp = FileOutputCSV(args.output).open()

    # track & maybe save matches progressively
    matches_out = SaveSignaturesToLocation(args.save_matches)
    matches_out.open()
    if args.save_matches:
        notify(f"saving all matching database signatures to '{args.save_matches}'")

    # iterate over signatures in db one at a time, for each db;
    # find those with sufficient overlap
    ident_mh = query_mh.copy_and_clear()
    noident_mh = query_mh.to_mutable()

    did_a_search = False  # track whether we did _any_ search at all!
    size_may_be_inaccurate = False
    total_signatures_loaded = 0
    sum_signatures_after_select = 0
    for dbfilename in args.databases:
        notify(f"loading signatures from '{dbfilename}'", end="\r")

        db = sourmash_args.load_file_as_index(dbfilename)
        total_signatures_loaded += len(db)

        # force linear traversal?
        if args.linear:
            db = LazyLinearIndex(db)

        db = db.select(ksize=ksize, moltype=moltype, containment=True)

        sum_signatures_after_select += len(db)

        db = sourmash_args.apply_picklist_and_pattern(db, picklist, pattern_search)

        if not db:
            notify(f"...no compatible signatures in '{dbfilename}'; skipping")
            continue

        for result in prefetch_database(
            query, db, args.threshold_bp, estimate_ani_ci=args.estimate_ani_ci
        ):
            match = result.match

            # ensure we're all on the same page wrt scaled resolution:
            common_scaled = max(
                match.minhash.scaled, query.minhash.scaled, common_scaled
            )

            query_mh = query.minhash.downsample(scaled=common_scaled)
            match_mh = match.minhash.downsample(scaled=common_scaled)

            if ident_mh.scaled != common_scaled:
                ident_mh = ident_mh.downsample(scaled=common_scaled)
            if noident_mh.scaled != common_scaled:
                noident_mh = noident_mh.downsample(scaled=common_scaled)

            # track found & "untouched" hashes.
            ident_mh += query_mh & match_mh.flatten()
            noident_mh.remove_many(match_mh)

            # output match info as we go
            if csvout_fp:
                if csvout_w is None:
                    csvout_w = result.init_dictwriter(csvout_fp)
                result.write(csvout_w)

            # output match signatures as we go (maybe)
            matches_out.add(match)

            if matches_out.count % 10 == 0:
                notify(
                    f"total of {matches_out.count} matching signatures so far.",
                    end="\r",
                )

            # keep track of inaccurate size estimation
            if not size_may_be_inaccurate and result.size_may_be_inaccurate:
                size_may_be_inaccurate = True

        did_a_search = True

        # flush csvout so that things get saved progressively
        if csvout_fp:
            csvout_fp.flush()

        # delete db explicitly ('cause why not)
        del db

    notify("--")
    notify(
        f"loaded {total_signatures_loaded} total signatures from {len(args.databases)} locations."
    )
    notify(
        f"after selecting signatures compatible with search, {sum_signatures_after_select} remain."
    )

    if not did_a_search:
        notify(
            "ERROR in prefetch: after picklists and patterns, no signatures to search!?"
        )
        sys.exit(-1)

    notify("--")
    notify(f"total of {matches_out.count} matching signatures.")
    matches_out.close()

    if csvout_fp:
        notify(f"saved {matches_out.count} matches to CSV file '{args.output}'")
        csvout_fp.close()

    assert len(query_mh) == len(ident_mh) + len(noident_mh)
    notify(
        f"of {len(query_mh)} distinct query hashes, {len(ident_mh)} were found in matches above threshold."
    )
    notify(f"a total of {len(noident_mh)} query hashes remain unmatched.")
    notify(f"final scaled value (max across query and all matches) is {common_scaled}")

    if args.save_matching_hashes:
        filename = args.save_matching_hashes
        notify(f"saving {len(ident_mh)} matched hashes to '{filename}'")

        sig_name = ""
        if query.name:
            sig_name = f"{query.name}-known"

        # restore abundances, if present in original query
        if orig_query_mh.track_abundance:
            ident_mh = ident_mh.inflate(orig_query_mh)

        ss = sig.SourmashSignature(ident_mh, name=sig_name)
        with SaveSignaturesToLocation(filename) as save_sig:
            save_sig.add(ss)

    if args.save_unmatched_hashes:
        filename = args.save_unmatched_hashes

        sig_name = ""
        if query.name:
            sig_name = f"{query.name}-unknown"

        notify(f"saving {len(noident_mh)} unmatched hashes to '{filename}'")

        # restore abundances, if present in original query
        if orig_query_mh.track_abundance:
            noident_mh = noident_mh.inflate(orig_query_mh)

        ss = sig.SourmashSignature(noident_mh, name=sig_name)
        with SaveSignaturesToLocation(filename) as save_sig:
            save_sig.add(ss)

    if picklist:
        sourmash_args.report_picklist(args, picklist)

    if size_may_be_inaccurate:
        notify(
            "WARNING: size estimation for at least one of these sketches may be inaccurate. ANI values will not be reported for these comparisons."
        )

    return 0
