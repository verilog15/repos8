# Prepared databases

```{contents}
```

We provide a number of pre-built collections and indexed databases
that you can use with sourmash.

## Types of databases

For each k-mer size, three types of databases may be available: Zipfile (`.zip`), SBT (`.sbt.zip`), and LCA (`.lca.jzon.gz`). The Zipfile and SBT databases are built with scaled=1000, and then LCA databases are built with scaled=10,000.
We recommend using the Zipfile databases for `sourmash gather` and the SBT databases for `sourmash search`. You must use the LCA databases for `sourmash lca` operations.

You can read more about the different database and index types [here](https://sourmash.readthedocs.io/en/latest/command-line.html#indexed-databases).

Note that the SBT and LCA databases can be used with sourmash v3.5 and later, while Zipfile collections can only be used with sourmash v4.1 and up.

## Taxonomic Information (for non-LCA databases)

For each prepared database, we have also made taxonomic information available linking each genome with its assigned lineage (`GTDB` or `NCBI` as appropriate).
For private databases, users can create their own `taxonomy` files: the critical columns are `ident`, containing the genome accession (e.g. `GCA_1234567.1`) and
a column for each taxonomic rank, `superkingdom` to `species`. If a `strain` column is provided, it will also be used.
As of v4.8, we can also use LIN taxonomic information in tax commands that accept the `--lins` flag. If used, `sourmash tax` commands will require a `lin` column in the taxonomy file which should contain `;`-separated LINs, preferably with a standard number of positions (e.g. all 20 positions in length or all 10 positions in length). Some taxonomy commands also accept a `lingroups` file, which is a two-column file (`name`, `lin`) describing the name and LIN prefix of LINgroups to be used for taxonomic summarization.

## Downloading and using the databases

All databases below can be downloaded via the command line with `curl -JLO <url>`, where `<url>` is the URL below. This will download an appropriately named file; you can name it yourself by specify `'-o <output>` to specify the local filename.

The databases do not need to be unpacked or prepared in any way after download.

You can verify that they've been successfully downloaded (and view database properties such as `ksize` and `scaled`) with `sourmash sig summarize <output>`.

## Sketches for human and animal genomes

These sketches are of the latest releases of a number of animal
genomes. Among other uses, they can be used to detect host
contamination in microbial metagenomes.

Each file includes sketches at k=21, k=31, and k=51, at a scaled of
1000, and is under 50 MB.

* Human (hg38) - [hg38.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/hg38.sig.zip)
* Cow (bosTau9) - [bosTau9.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/bosTau9.sig.zip)
* Dog (canFam6) - [canFam6.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/canFam6.sig.zip)
* Horse (equCab3) - [equCab3.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/equCab3.sig.zip)
* Cat (felCat9) - [felCat9.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/felCat9.sig.zip)
* Chicken (galGAl6) - [galGal6.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/galGal6.sig.zip)
* Mouse (mm39) - [mm39.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/mm39.sig.zip)
* Goat (oviAri4) - [oviAri4.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/oviAri4.sig.zip)
* Pig (susCr11) - [susScr11.sig.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/host/susScr11.sig.zip)

## Sketches for plant genomes

These sketches are for the plant genomes available in GenBank as of 2024-07.

| K-mer size | Zipfile collection |
| -------- | -------- |
| k21 | [download (7G)](https://farm.cse.ucdavis.edu/\~ctbrown/sourmash-db/genbank-plant-2024-07/genbank-plants-2024-07.k21.zip) |
| k31 | [download (8.8G)](https://farm.cse.ucdavis.edu/\~ctbrown/sourmash-db/genbank-plant-2024-07/genbank-plants-2024-07.k31.zip) |
| k51 | [download (11G)](https://farm.cse.ucdavis.edu/\~ctbrown/sourmash-db/genbank-plant-2024-07/genbank-plants-2024-07.k51.zip) |

Lineage spreadsheet for sourmash `tax` commands: [download](https://farm.cse.ucdavis.edu/\~ctbrown/sourmash-db/genbank-plant-2024-07/genbank-plants-2024-07.lineages.csv.gz)

## GTDB R08-RS214 - DNA databases

[GTDB R08-RS214](https://forum.gtdb.ecogenomic.org/t/announcing-gtdb-r08-rs214/456) consists of 402,709 genomes organized into 85,205 species clusters.

The lineage spreadsheet (for `sourmash tax` commands) is available [for the genome database (402k)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214.lineages.csv.gz).

### GTDB R08-RS214 genomic representatives (85k)

The GTDB genomic representatives are a low-redundancy subset of Genbank genomes, with 85,205 species-level genomes.

| K-mer size | Zipfile collection | SBT | LCA |
| -------- | -------- | -------- | ---- |
| 21 | [download (2.2 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k21.zip) | [download (4.4 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k21.sbt.zip) | [download (189 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k21.lca.json.gz) |
| 31 | [download (2.2 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k31.zip) | [download (4.4 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k31.sbt.zip) | [download (221 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k31.lca.json.gz) |
| 51 | [download (2.2 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k51.zip) | [download (4.4 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k51.sbt.zip) | [download (230 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k51.lca.json.gz) |

### GTDB R08-RS214 all genomes (403k)

These are databases for the full GTDB release, each containing 402,709  genomes.

| K-mer size | Zipfile collection | SBT | LCA |
| -------- | -------- | -------- | ---- |
| 21 | [download (12 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k21.zip) | [download (23 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k21.sbt.zip) | [download (406 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k21.lca.json.gz) |
| 31 | [download (12 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k31.zip) | [download (23 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k31.sbt.zip) | [download (438 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k31.lca.json.gz) |
| 51 | [download (12 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k51.zip) | [download (23 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k51.sbt.zip) | [download (460 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-k51.lca.json.gz) |

## Genbank genomes from March 2022

The below zip files contain different subsets of the signatures for
all microbial Genbank genomes. The databases were built in March 2022,
and are based on the assembly_summary files provided
[here](https://ftp.ncbi.nlm.nih.gov/genomes/genbank/).

Since some of the files are extremely large, we only provide them in
Zip format (which is our smallest and most flexible format).

Note that all of the sourmash search commands support multiple
databases on the command line, so you can search multiple subsets
simply by providing them all on the command line, e.g. `sourmash
search query.sig genbank-2022.03-{viral,protozoa}-k31.zip`.

Taxonomic spreadsheets for each domain are provided below as well.

### Genbank viral

47,952 genomes:

[genbank-2022.03-viral-k21.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-viral-k21.zip)

[genbank-2022.03-viral-k31.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-viral-k31.zip)

[genbank-2022.03-viral-k51.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-viral-k51.zip)

[genbank-2022.03-viral.lineages.csv.gz](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-viral.lineages.csv.gz)

### Genbank archaeal

8,750 genomes:

[genbank-2022.03-archaea-k21.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-archaea-k21.zip)

[genbank-2022.03-archaea-k31.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-archaea-k31.zip)

[genbank-2022.03-archaea-k51.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-archaea-k51.zip)

[genbank-2022.03-archaea.lineages.csv.gz](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-archaea.lineages.csv.gz)


### Genbank protozoa

1193 genomes:

[genbank-2022.03-protozoa-k21.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-protozoa-k21.zip)

[genbank-2022.03-protozoa-k31.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-protozoa-k31.zip)

[genbank-2022.03-protozoa-k51.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-protozoa-k51.zip)

[genbank-2022.03-protozoa.lineages.csv.gz](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-protozoa.lineages.csv.gz)


### Genbank fungi

10,286 genomes:

[genbank-2022.03-fungi-k21.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-fungi-k21.zip)

[genbank-2022.03-fungi-k31.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-fungi-k31.zip)

[genbank-2022.03-fungi-k51.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-fungi-k51.zip)

[genbank-2022.03-fungi.lineages.csv.gz](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-fungi.lineages.csv.gz)

### Genbank bacterial:

1,148,011 genomes:

[genbank-2022.03-bacteria-k21.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-bacteria-k21.zip)

[genbank-2022.03-bacteria-k31.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-bacteria-k31.zip)

[genbank-2022.03-bacteria-k51.zip](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-bacteria-k51.zip)

[genbank-2022.03-bacteria.lineages.csv.gz](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/genbank-2022.03/genbank-2022.03-bacteria.lineages.csv.gz)

## GTDB R07-RS207 - DNA databases

[GTDB R07-RS207](https://forum.gtdb.ecogenomic.org/t/announcing-gtdb-r07-rs207/264) consists of 317,542 genomes organized into 65,703 species clusters.

The lineage spreadsheet (for `sourmash tax` commands) is available [for the species database (65k)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.taxonomy.reps.csv.gz) and [for the genome database (317k)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.taxonomy.with-strain.csv.gz).

### GTDB R07-RS207 genomic representatives (66k)

The GTDB genomic representatives are a low-redundancy subset of Genbank genomes, with 65,703 species-level genomes.

| K-mer size | Zipfile collection | SBT | LCA |
| -------- | -------- | -------- | ---- |
| 21 | [download (1.7 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k21.zip) | [download (3.5 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k21.sbt.zip) | [download (181 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k21.lca.json.gz) |
| 31 | [download (1.7 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k31.zip) | [download (3.5 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k31.sbt.zip) | [download (181 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k31.lca.json.gz) |
| 51 | [download (1.7 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k51.zip) | [download (3.5 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k51.sbt.zip) | [download (181 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic-reps.dna.k51.lca.json.gz) |

### GTDB R07-RS207 all genomes (318k)

These are databases for the full GTDB release, each containing 317,542 genomes.

| K-mer size | Zipfile collection | SBT | LCA |
| -------- | -------- | -------- | ---- |
| 21 | [download (9.4 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k21.zip) | [download (19 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k21.sbt.zip) | [download (351 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k21.lca.json.gz) |
| 31 | [download (9.4 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k31.zip) | [download (19 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k31.sbt.zip) | [download (351 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k31.lca.json.gz) |
| 51 | [download (9.4 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k51.zip) | [download (19 GB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k51.sbt.zip) | [download (351 MB)](https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs207/gtdb-rs207.genomic.k51.lca.json.gz) |


## GTDB R06-RS202 - DNA databases

All files below are available under https://osf.io/wxf9z/. The GTDB taxonomy spreadsheet (in a format suitable for `sourmash lca index`) is available [here](https://osf.io/p6z3w/).

### GTDB R06-RS202 genomic representatives (47.8k)

The GTDB genomic representatives are a low-redundancy subset of Genbank genomes.

| K-mer size | Zipfile collection | SBT | LCA |
| -------- | -------- | -------- | ---- |
| 21 | [download (1.3 GB)](https://osf.io/jp5zh/download) | [download (2.6 GB)](https://osf.io/py92w/download) | [download (114 MB)](https://osf.io/gk2za/download) | 
| 31 | [download (1.3 GB)](https://osf.io/nqmau/download) | [download (2.6 GB)](https://osf.io/w4bcm/download) | [download (131 MB)](https://osf.io/ypsjq/download) | 
| 51 | [download (1.3 GB)](https://osf.io/px6qd/download) | [download (2.6 GB)](https://osf.io/rv9zp/download) | [download (137 MB)](https://osf.io/297dp/download) | 

### GTDB R06-RS202 all genomes (258k)

These databases contain the complete GTDB collection of 258,406 genomes.

| K-mer size | Zipfile collection | SBT | LCA |
| -------- | -------- | -------- | ---- |
| 21 | [download (7.8 GB)](https://osf.io/vgex4/download) | [download (15 GB)](https://osf.io/ar67j/download) | [download (266 MB)](https://osf.io/hm3c4/download) | 
| 31 | [download (7.8 GB)](https://osf.io/94mzh/download) | [download (15 GB)](https://osf.io/dmsz8/download) | [download (286 MB)](https://osf.io/9xdg2/download) | 
| 51 | [download (7.8 GB)](https://osf.io/x9cdp/download) | [download (15 GB)](https://osf.io/8fc3t/download) | [download (299 MB)](https://osf.io/3cdp6/download)  | 

## Appendix: database use and construction details

Database release workflows are being archived at [sourmash-bio/database-releases](https://github.com/sourmash-bio/database-releases).

Some more details on database use and construction:

* Zipfile collections can be used for a linear search. The signatures were calculated with a scaled of 1000, which robustly supports searches for ~10kb or larger matches.
* SBT databases are indexed versions of the Zipfile collections that support faster search. They are also indexed with scaled=1000.
* LCA databases are indexed versions of the Zipfile collections that also contain taxonomy information and can be used with regular search as well as with [the `lca` subcommands for taxonomic analysis](https://sourmash.readthedocs.io/en/latest/command-line.html#sourmash-lca-subcommands-for-taxonomic-classification). They are indexed with scaled=10,000, which robustly supports searches for 100kb or larger matches.

## Appendix: Memory and time requirements

The detailed memory usage of sourmash depends on the type of search, the query, and the database you're searching, but to help guide you here is a range of numbers:

| Search type | Query | Database | Max RAM | Time |
| -------- | -------- | -------- | -------- | -------- |
| gather | Bacterial genome | GTDB complete (280k) | 1 GB | 6 minutes |
| gather | Simple metagenome | GTDB reps .zip (65k)     |   2 GB   | 6 minutes |
| gather | Real metagenome | All Genbank (1.2m) | 100 GB | 3 hours
| lca summarize     |Simple metagenome | GTDB reps .sql (65k)     |   400 MB   | 20 seconds |
| lca summarize | Simple metagenome | GTDB reps .json (65k) | 6.2 GB | 1m 20 seconds |


Please see [sourmash#1958](https://github.com/sourmash-bio/sourmash/issues/1958) for detailed GTDB numbers and [gather paper#47](https://github.com/dib-lab/2020-paper-sourmash-gather/issues/47) for detailed Genbank numbers.

## Appendix: legacy databases

Legacy databases are available [here](legacy-databases.md).
