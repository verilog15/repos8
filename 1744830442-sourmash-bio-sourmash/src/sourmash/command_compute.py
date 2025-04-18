"""
Functions implementing the 'compute' command and related functions.
"""

import os
import os.path
import sys
import random
import screed

from . import sourmash_args
from .signature import SourmashSignature
from .logging import notify, error, set_quiet
from .utils import RustObject
from ._lowlevel import ffi, lib


from .command_sketch import (
    _compute_individual,
    _compute_merged,
    ComputeParameters,
    add_seq,
    set_sig_name,
    DEFAULT_MMHASH_SEED,
)


def compute(args):
    """Compute the signature for one or more files.

    Use cases:
        sourmash compute multiseq.fa              => multiseq.fa.sig, etc.
        sourmash compute genome.fa --singleton    => genome.fa.sig
        sourmash compute file1.fa file2.fa -o file.sig
            => creates one output file file.sig, with one signature for each
               input file.
        sourmash compute file1.fa file2.fa --merge merged -o file.sig
            => creates one output file file.sig, with all sequences from
               file1.fa and file2.fa combined into one signature.
    """
    set_quiet(args.quiet)

    if args.license != "CC0":
        error("error: sourmash only supports CC0-licensed signatures. sorry!")
        sys.exit(-1)

    if args.input_is_protein and args.dna:
        notify("WARNING: input is protein, turning off nucleotide hashing")
        args.dna = False
        args.protein = True

    if args.scaled:
        if args.scaled < 1:
            error("ERROR: --scaled value must be >= 1")
            sys.exit(-1)
        if args.scaled != round(args.scaled, 0):
            error("ERROR: --scaled value must be integer value")
            sys.exit(-1)
        if args.scaled >= 1e9:
            notify("WARNING: scaled value is nonsensical!? Continuing anyway.")

        if args.num_hashes != 0:
            notify("setting num_hashes to 0 because --scaled is set")
            args.num_hashes = 0

    notify("computing signatures for files: {}", ", ".join(args.filenames))

    if args.randomize:
        notify("randomizing file list because of --randomize")
        random.shuffle(args.filenames)

    # get list of k-mer sizes for which to compute sketches
    ksizes = args.ksizes

    notify("Computing signature for ksizes: {}", str(ksizes))
    num_sigs = 0
    if args.dna and args.protein:
        notify("Computing both nucleotide and protein signatures.")
        num_sigs = 2 * len(ksizes)
    elif args.dna and args.dayhoff:
        notify("Computing both nucleotide and Dayhoff-encoded protein signatures.")
        num_sigs = 2 * len(ksizes)
    elif args.dna and args.hp:
        notify("Computing both nucleotide and hp-encoded protein signatures.")
        num_sigs = 2 * len(ksizes)
    elif args.dna:
        notify("Computing only nucleotide (and not protein) signatures.")
        num_sigs = len(ksizes)
    elif args.protein:
        notify("Computing only protein (and not nucleotide) signatures.")
        num_sigs = len(ksizes)
    elif args.dayhoff:
        notify(
            "Computing only Dayhoff-encoded protein (and not nucleotide) signatures."
        )
        num_sigs = len(ksizes)
    elif args.hp:
        notify("Computing only hp-encoded protein (and not nucleotide) signatures.")
        num_sigs = len(ksizes)

    if args.protein or args.dayhoff or args.hp:
        notify("")
        notify(
            "WARNING: you are using 'compute' to make a protein/dayhoff/hp signature,"
        )
        notify("WARNING: but the meaning of ksize has changed in 4.0. Please see the")
        notify("WARNING: migration guide to sourmash v4.0 at http://sourmash.rtfd.io/")
        notify("")
        bad_ksizes = [str(k) for k in ksizes if k % 3 != 0]
        if bad_ksizes:
            error("protein ksizes must be divisible by 3, sorry!")
            error("bad ksizes: {}", ", ".join(bad_ksizes))
            sys.exit(-1)

    notify("Computing a total of {} signature(s) for each input.", num_sigs)

    if num_sigs == 0:
        error("...nothing to calculate!? Exiting!")
        sys.exit(-1)

    if args.merge and not args.output:
        error("ERROR: must specify -o with --merge")
        sys.exit(-1)

    if args.output and args.output_dir:
        error("ERROR: --output-dir doesn't make sense with -o/--output")
        sys.exit(-1)

    if args.track_abundance:
        notify("Tracking abundance of input k-mers.")

    signatures_factory = _signatures_for_compute_factory(args)

    if args.merge:  # single name specified - combine all
        _compute_merged(args, signatures_factory)
    else:  # compute individual signatures
        _compute_individual(args, signatures_factory)


class _signatures_for_compute_factory:
    "Build signatures on demand, based on args input to 'compute'."

    def __init__(self, args):
        self.args = args

    def __call__(self):
        args = self.args
        params = ComputeParameters(
            ksizes=args.ksizes,
            seed=args.seed,
            protein=args.protein,
            dayhoff=args.dayhoff,
            hp=args.hp,
            dna=args.dna,
            num_hashes=args.num_hashes,
            track_abundance=args.track_abundance,
            scaled=args.scaled,
        )
        sig = SourmashSignature.from_params(params)
        return [sig]


def _compute_individual(args, signatures_factory):
    # this is where output signatures will go.
    save_sigs = None

    # track: is this the first file? in cases where we have empty inputs,
    # we don't want to open any outputs.
    first_file_for_output = True

    # if args.output is set, we are aggregating all output to a single file.
    # do not open a new output file for each input.
    open_output_each_time = True
    if args.output:
        open_output_each_time = False

    for filename in args.filenames:
        if open_output_each_time:
            # for each input file, construct output filename
            sigfile = os.path.basename(filename) + ".sig"
            if args.output_dir:
                sigfile = os.path.join(args.output_dir, sigfile)

            # does it already exist? skip if so.
            if os.path.exists(sigfile) and not args.force:
                notify("skipping {} - already done", filename)
                continue  # go on to next file.

            # nope? ok, let's save to it.
            assert not save_sigs
            save_sigs = sourmash_args.SaveSignaturesToLocation(sigfile)

        #
        # calculate signatures!
        #

        # now, set up to iterate over sequences.
        with screed.open(filename) as screed_iter:
            if not screed_iter:
                notify(f"no sequences found in '{filename}'?!")
                continue

            # open output for signatures
            if open_output_each_time:
                save_sigs.open()
            # or... is this the first time to write something to args.output?
            elif first_file_for_output:
                save_sigs = sourmash_args.SaveSignaturesToLocation(args.output)
                save_sigs.open()
                first_file_for_output = False

            # make a new signature for each sequence?
            if args.singleton:
                n_calculated = 0
                for n, record in enumerate(screed_iter):
                    sigs = signatures_factory()
                    try:
                        add_seq(
                            sigs,
                            record.sequence,
                            args.input_is_protein,
                            args.check_sequence,
                        )
                    except ValueError as exc:
                        error(f"ERROR when reading from '{filename}' - ")
                        error(str(exc))
                        sys.exit(-1)

                    n_calculated += len(sigs)
                    set_sig_name(sigs, filename, name=record.name)
                    save_sigs_to_location(sigs, save_sigs)

                notify(
                    "calculated {} signatures for {} sequences in {}",
                    n_calculated,
                    n + 1,
                    filename,
                )

            # nope; make a single sig for the whole file
            else:
                sigs = signatures_factory()

                # consume & calculate signatures
                notify(f"... reading sequences from {filename}")
                name = None
                for n, record in enumerate(screed_iter):
                    if n % 10000 == 0:
                        if n:
                            notify("\r...{} {}", filename, n, end="")
                        elif args.name_from_first:
                            name = record.name

                    try:
                        add_seq(
                            sigs,
                            record.sequence,
                            args.input_is_protein,
                            args.check_sequence,
                        )
                    except ValueError as exc:
                        error(f"ERROR when reading from '{filename}' - ")
                        error(str(exc))
                        sys.exit(-1)

                notify("...{} {} sequences", filename, n, end="")

                set_sig_name(sigs, filename, name)
                save_sigs_to_location(sigs, save_sigs)

                notify(
                    f"calculated {len(sigs)} signatures for {n + 1} sequences in {filename}"
                )

        # if not args.output, close output for every input filename.
        if open_output_each_time:
            save_sigs.close()
            notify(
                f"saved {len(save_sigs)} signature(s) to '{save_sigs.location}'. Note: signature license is CC0."
            )
            save_sigs = None

    # if --output-dir specified, all collected signatures => args.output,
    # and we need to close here.
    if args.output and save_sigs is not None:
        save_sigs.close()
        notify(
            f"saved {len(save_sigs)} signature(s) to '{save_sigs.location}'. Note: signature license is CC0."
        )


def _compute_merged(args, signatures_factory):
    # make a signature for the whole file
    sigs = signatures_factory()

    total_seq = 0
    for filename in args.filenames:
        # consume & calculate signatures
        notify("... reading sequences from {}", filename)

        n = None
        with screed.open(filename) as f:
            for n, record in enumerate(f):
                if n % 10000 == 0 and n:
                    notify("\r... {} {}", filename, n, end="")

                add_seq(
                    sigs, record.sequence, args.input_is_protein, args.check_sequence
                )
        if n is not None:
            notify("... {} {} sequences", filename, n + 1)
            total_seq += n + 1
        else:
            notify(f"no sequences found in '{filename}'?!")

    if total_seq:
        set_sig_name(sigs, filename, name=args.merge)
        notify(
            "calculated 1 signature for {} sequences taken from {} files",
            total_seq,
            len(args.filenames),
        )

        # at end, save!
        save_siglist(sigs, args.output)


def add_seq(sigs, seq, input_is_protein, check_sequence):
    for sig in sigs:
        if input_is_protein:
            sig.add_protein(seq)
        else:
            sig.add_sequence(seq, not check_sequence)


def set_sig_name(sigs, filename, name=None):
    if filename == "-":  # if stdin, set filename to empty.
        filename = ""
    for sig in sigs:
        if name is not None:
            sig._name = name

        sig.filename = filename


def save_siglist(siglist, sigfile_name):
    "Save multiple signatures to a filename."

    # save!
    with sourmash_args.SaveSignaturesToLocation(sigfile_name) as save_sig:
        for ss in siglist:
            save_sig.add(ss)

        notify(f"saved {len(save_sig)} signature(s) to '{save_sig.location}'")


def save_sigs_to_location(siglist, save_sig):
    "Save multiple signatures to an already-open location."
    import sourmash

    for ss in siglist:
        save_sig.add(ss)


class ComputeParameters(RustObject):
    __dealloc_func__ = lib.computeparams_free

    def __init__(
        self,
        *,
        ksizes=(21, 31, 51),
        seed=42,
        protein=False,
        dayhoff=False,
        hp=False,
        dna=True,
        num_hashes=500,
        track_abundance=False,
        scaled=0,
    ):
        self._objptr = lib.computeparams_new()

        self.seed = seed
        self.ksizes = ksizes
        self.protein = protein
        self.dayhoff = dayhoff
        self.hp = hp
        self.dna = dna
        self.num_hashes = num_hashes
        self.track_abundance = track_abundance
        self.scaled = scaled

    @classmethod
    def from_manifest_row(cls, row):
        "convert a CollectionManifest row into a ComputeParameters object"
        is_dna = is_protein = is_dayhoff = is_hp = False
        if row["moltype"] == "DNA":
            is_dna = True
        elif row["moltype"] == "protein":
            is_protein = True
        elif row["moltype"] == "hp":
            is_hp = True
        elif row["moltype"] == "dayhoff":
            is_dayhoff = True
        else:
            assert 0

        if is_dna:
            ksize = row["ksize"]
        else:
            ksize = row["ksize"] * 3

        p = cls(
            ksizes=[ksize],
            seed=DEFAULT_MMHASH_SEED,
            protein=is_protein,
            dayhoff=is_dayhoff,
            hp=is_hp,
            dna=is_dna,
            num_hashes=row["num"],
            track_abundance=row["with_abundance"],
            scaled=row["scaled"],
        )

        return p

    def to_param_str(self):
        "Convert object to equivalent params str."
        pi = []

        if self.dna:
            pi.append("dna")
        elif self.protein:
            pi.append("protein")
        elif self.hp:
            pi.append("hp")
        elif self.dayhoff:
            pi.append("dayhoff")
        else:
            assert 0  # must be one of the previous

        if self.dna:
            kstr = [f"k={k}" for k in self.ksizes]
        else:
            # for protein, divide ksize by three.
            kstr = [f"k={k // 3}" for k in self.ksizes]
        assert kstr
        pi.extend(kstr)

        if self.num_hashes != 0:
            pi.append(f"num={self.num_hashes}")
        elif self.scaled != 0:
            pi.append(f"scaled={self.scaled}")
        else:
            assert 0

        if self.track_abundance:
            pi.append("abund")
        # noabund is default

        if self.seed != DEFAULT_MMHASH_SEED:
            pi.append(f"seed={self.seed}")
        # self.seed

        return ",".join(pi)

    def __repr__(self):
        return f"ComputeParameters(ksizes={self.ksizes}, seed={self.seed}, protein={self.protein}, dayhoff={self.dayhoff}, hp={self.hp}, dna={self.dna}, num_hashes={self.num_hashes}, track_abundance={self.track_abundance}, scaled={self.scaled})"

    def __eq__(self, other):
        return (
            self.ksizes == other.ksizes
            and self.seed == other.seed
            and self.protein == other.protein
            and self.dayhoff == other.dayhoff
            and self.hp == other.hp
            and self.dna == other.dna
            and self.num_hashes == other.num_hashes
            and self.track_abundance == other.track_abundance
            and self.scaled == other.scaled
        )

    @staticmethod
    def from_args(args):
        ptr = lib.computeparams_new()
        ret = ComputeParameters._from_objptr(ptr)

        for arg, value in vars(args).items():
            try:
                getattr(type(ret), arg).fset(ret, value)
            except AttributeError:
                pass

        return ret

    @property
    def seed(self):
        return self._methodcall(lib.computeparams_seed)

    @seed.setter
    def seed(self, v):
        return self._methodcall(lib.computeparams_set_seed, v)

    @property
    def ksizes(self):
        size = ffi.new("uintptr_t *")
        ksizes_ptr = self._methodcall(lib.computeparams_ksizes, size)
        size = size[0]
        ksizes = ffi.unpack(ksizes_ptr, size)
        lib.computeparams_ksizes_free(ksizes_ptr, size)
        return ksizes

    @ksizes.setter
    def ksizes(self, v):
        return self._methodcall(lib.computeparams_set_ksizes, list(v), len(v))

    @property
    def protein(self):
        return self._methodcall(lib.computeparams_protein)

    @protein.setter
    def protein(self, v):
        return self._methodcall(lib.computeparams_set_protein, v)

    @property
    def dayhoff(self):
        return self._methodcall(lib.computeparams_dayhoff)

    @dayhoff.setter
    def dayhoff(self, v):
        return self._methodcall(lib.computeparams_set_dayhoff, v)

    @property
    def hp(self):
        return self._methodcall(lib.computeparams_hp)

    @hp.setter
    def hp(self, v):
        return self._methodcall(lib.computeparams_set_hp, v)

    @property
    def dna(self):
        return self._methodcall(lib.computeparams_dna)

    @dna.setter
    def dna(self, v):
        return self._methodcall(lib.computeparams_set_dna, v)

    @property
    def moltype(self):
        if self.dna:
            moltype = "DNA"
        elif self.protein:
            moltype = "protein"
        elif self.hp:
            moltype = "hp"
        elif self.dayhoff:
            moltype = "dayhoff"
        else:
            assert 0

        return moltype

    @property
    def num_hashes(self):
        return self._methodcall(lib.computeparams_num_hashes)

    @num_hashes.setter
    def num_hashes(self, v):
        return self._methodcall(lib.computeparams_set_num_hashes, v)

    @property
    def track_abundance(self):
        return self._methodcall(lib.computeparams_track_abundance)

    @track_abundance.setter
    def track_abundance(self, v):
        return self._methodcall(lib.computeparams_set_track_abundance, v)

    @property
    def scaled(self):
        return self._methodcall(lib.computeparams_scaled)

    @scaled.setter
    def scaled(self, v):
        return self._methodcall(lib.computeparams_set_scaled, int(v))
