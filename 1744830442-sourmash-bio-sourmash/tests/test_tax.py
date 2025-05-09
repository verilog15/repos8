"""
Tests for the 'sourmash tax' command line and high level API.
"""

import os
import csv
import pytest
import gzip
from collections import Counter
from pathlib import Path

import sourmash
import sourmash_tst_utils as utils
from sourmash.tax import tax_utils
from sourmash.lca import lca_utils
from sourmash_tst_utils import SourmashCommandFailed

from sourmash import sqlite_utils
from sourmash.exceptions import IndexNotSupported
from sourmash import sourmash_args


## command line tests
def test_run_sourmash_tax():
    status, out, err = utils.runscript("sourmash", ["tax"], fail_ok=True)
    assert status != 0  # no args provided, ok ;)


def test_metagenome_stdout_0(runtmp):
    # test basic metagenome
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    c.run_sourmash("tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax)

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.204,d__Bacteria,md5,test1.sig,0.131,1024000"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.116,d__Bacteria;p__Bacteroidota,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.088,d__Bacteria;p__Proteobacteria,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,class,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,class,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,class,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,order,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,order,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,order,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,family,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,family,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,family,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.028,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola,md5,test1.sig,0.016,138000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.028,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus,md5,test1.sig,0.016,138000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )


def test_metagenome_stdout_0_db(runtmp):
    # test basic metagenome with sqlite database
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.db")

    c.run_sourmash("tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax)

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.204,d__Bacteria,md5,test1.sig,0.131,1024000"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.116,d__Bacteria;p__Bacteroidota,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.088,d__Bacteria;p__Proteobacteria,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,class,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,class,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,class,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,order,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,order,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,order,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,family,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,family,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,family,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.028,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola,md5,test1.sig,0.016,138000"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.028,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus,md5,test1.sig,0.016,138000"
        in c.last_result.out
    )
    assert (
        "test1,species,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )


def test_metagenome_summary_csv_out(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    sum_csv = csv_base + ".summarized.csv"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-dir",
        outdir,
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)

    sum_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    assert f"saving 'csv_summary' output to '{csvout}'" in runtmp.last_result.err
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in sum_gather_results[0]
    )
    assert (
        "test1,superkingdom,0.2042281611487834,d__Bacteria,md5,test1.sig,0.13080306238801107,1024000"
        in sum_gather_results[1]
    )
    assert (
        "test1,superkingdom,0.7957718388512166,unclassified,md5,test1.sig,0.8691969376119889,3990000"
        in sum_gather_results[2]
    )
    assert (
        "test1,phylum,0.11607499002792182,d__Bacteria;p__Bacteroidota,md5,test1.sig,0.07265026877341586,582000"
        in sum_gather_results[3]
    )
    assert (
        "test1,phylum,0.08815317112086159,d__Bacteria;p__Proteobacteria,md5,test1.sig,0.05815279361459521,442000"
        in sum_gather_results[4]
    )
    assert (
        "test1,phylum,0.7957718388512166,unclassified,md5,test1.sig,0.8691969376119889,3990000"
        in sum_gather_results[5]
    )
    assert (
        "test1,class,0.11607499002792182,d__Bacteria;p__Bacteroidota;c__Bacteroidia,md5,test1.sig,0.07265026877341586,582000"
        in sum_gather_results[6]
    )
    assert (
        "test1,class,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria,md5,test1.sig,0.05815279361459521,442000"
        in sum_gather_results[7]
    )
    assert (
        "test1,class,0.7957718388512166,unclassified,md5,test1.sig,0.8691969376119889,3990000"
        in sum_gather_results[8]
    )
    assert (
        "test1,order,0.11607499002792182,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales,md5,test1.sig,0.07265026877341586,582000"
        in sum_gather_results[9]
    )
    assert (
        "test1,order,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales,md5,test1.sig,0.05815279361459521,442000"
        in sum_gather_results[10]
    )
    assert (
        "test1,order,0.7957718388512166,unclassified,md5,test1.sig,0.8691969376119889,3990000"
        in sum_gather_results[11]
    )
    assert (
        "test1,family,0.11607499002792182,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test1.sig,0.07265026877341586,582000"
        in sum_gather_results[12]
    )
    assert (
        "test1,family,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae,md5,test1.sig,0.05815279361459521,442000"
        in sum_gather_results[13]
    )
    assert (
        "test1,family,0.7957718388512166,unclassified,md5,test1.sig,0.8691969376119889,3990000"
        in sum_gather_results[14]
    )
    assert (
        "test1,genus,0.0885520542481053,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella,md5,test1.sig,0.05701254275940707,444000"
        in sum_gather_results[15]
    )
    assert (
        "test1,genus,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia,md5,test1.sig,0.05815279361459521,442000"
        in sum_gather_results[16]
    )
    assert (
        "test1,genus,0.027522935779816515,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola,md5,test1.sig,0.015637726014008795,138000"
        in sum_gather_results[17]
    )
    assert (
        "test1,genus,0.7957718388512166,unclassified,md5,test1.sig,0.8691969376119889,3990000"
        in sum_gather_results[18]
    )
    assert (
        "test1,species,0.0885520542481053,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.05701254275940707,444000"
        in sum_gather_results[19]
    )
    assert (
        "test1,species,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli,md5,test1.sig,0.05815279361459521,442000"
        in sum_gather_results[20]
    )
    assert (
        "test1,species,0.027522935779816515,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus,md5,test1.sig,0.015637726014008795,138000"
        in sum_gather_results[21]
    )
    assert (
        "test1,species,0.7957718388512166,unclassified,md5,test1.sig,0.8691969376119889,3990000"
        in sum_gather_results[22]
    )


def test_metagenome_summary_csv_out_empty_gather_force(runtmp):
    # test multiple -g, empty -g file, and --force
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    sum_csv = csv_base + ".summarized.csv"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    gather_empty = runtmp.output("g.csv")
    with open(gather_empty, "w") as fp:
        fp.write("")
    print("g_csv: ", gather_empty)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_csv,
        "-g",
        gather_empty,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-dir",
        outdir,
        "-f",
    )
    sum_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    assert f"saving 'csv_summary' output to '{csvout}'" in runtmp.last_result.err
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in sum_gather_results[0]
    )
    assert (
        "test1,superkingdom,0.2042281611487834,d__Bacteria,md5,test1.sig,0.13080306238801107,1024000"
        in sum_gather_results[1]
    )


def test_metagenome_kreport_out(runtmp):
    # test 'kreport' kraken output format
    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    sum_csv = csv_base + ".kreport.txt"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-dir",
        outdir,
        "-F",
        "kreport",
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)

    kreport_results = [
        x.rstrip().split("\t") for x in Path(csvout).read_text().splitlines()
    ]
    assert f"saving 'kreport' output to '{csvout}'" in runtmp.last_result.err
    print(kreport_results)
    assert ["13.08", "1605999", "0", "D", "", "d__Bacteria"] == kreport_results[0]
    assert [
        "86.92",
        "10672000",
        "10672000",
        "U",
        "",
        "unclassified",
    ] == kreport_results[1]
    assert ["7.27", "892000", "0", "P", "", "p__Bacteroidota"] == kreport_results[2]
    assert ["5.82", "714000", "0", "P", "", "p__Proteobacteria"] == kreport_results[3]
    assert ["7.27", "892000", "0", "C", "", "c__Bacteroidia"] == kreport_results[4]
    assert [
        "5.82",
        "714000",
        "0",
        "C",
        "",
        "c__Gammaproteobacteria",
    ] == kreport_results[5]
    assert ["7.27", "892000", "0", "O", "", "o__Bacteroidales"] == kreport_results[6]
    assert ["5.82", "714000", "0", "O", "", "o__Enterobacterales"] == kreport_results[7]
    assert ["7.27", "892000", "0", "F", "", "f__Bacteroidaceae"] == kreport_results[8]
    assert ["5.82", "714000", "0", "F", "", "f__Enterobacteriaceae"] == kreport_results[
        9
    ]
    assert ["5.70", "700000", "0", "G", "", "g__Prevotella"] == kreport_results[10]
    assert ["5.82", "714000", "0", "G", "", "g__Escherichia"] == kreport_results[11]
    assert ["1.56", "192000", "0", "G", "", "g__Phocaeicola"] == kreport_results[12]
    assert [
        "5.70",
        "700000",
        "700000",
        "S",
        "",
        "s__Prevotella copri",
    ] == kreport_results[13]
    assert [
        "5.82",
        "714000",
        "714000",
        "S",
        "",
        "s__Escherichia coli",
    ] == kreport_results[14]
    assert [
        "1.56",
        "192000",
        "192000",
        "S",
        "",
        "s__Phocaeicola vulgatus",
    ] == kreport_results[15]


def test_metagenome_kreport_ncbi_taxid_out(runtmp):
    # test NCBI taxid output from kreport
    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.ncbi-taxonomy.csv")
    csv_base = "out"
    sum_csv = csv_base + ".kreport.txt"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-dir",
        outdir,
        "-F",
        "kreport",
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)

    kreport_results = [
        x.rstrip().split("\t") for x in Path(csvout).read_text().splitlines()
    ]
    assert f"saving 'kreport' output to '{csvout}'" in runtmp.last_result.err
    print(kreport_results)
    assert ["13.08", "1605999", "0", "D", "2", "Bacteria"] == kreport_results[0]
    assert [
        "86.92",
        "10672000",
        "10672000",
        "U",
        "",
        "unclassified",
    ] == kreport_results[1]
    assert ["7.27", "892000", "0", "P", "976", "Bacteroidota"] == kreport_results[2]
    assert ["5.82", "714000", "0", "P", "1224", "Pseudomonadota"] == kreport_results[3]
    assert ["7.27", "892000", "0", "C", "200643", "Bacteroidia"] == kreport_results[4]
    assert [
        "5.82",
        "714000",
        "0",
        "C",
        "1236",
        "Gammaproteobacteria",
    ] == kreport_results[5]
    assert ["7.27", "892000", "0", "O", "171549", "Bacteroidales"] == kreport_results[6]
    assert ["5.82", "714000", "0", "O", "91347", "Enterobacterales"] == kreport_results[
        7
    ]
    assert ["5.70", "700000", "0", "F", "171552", "Prevotellaceae"] == kreport_results[
        8
    ]
    assert ["5.82", "714000", "0", "F", "543", "Enterobacteriaceae"] == kreport_results[
        9
    ]
    assert ["1.56", "192000", "0", "F", "815", "Bacteroidaceae"] == kreport_results[10]
    assert ["5.70", "700000", "0", "G", "838", "Prevotella"] == kreport_results[11]
    assert ["5.82", "714000", "0", "G", "561", "Escherichia"] == kreport_results[12]
    assert ["1.56", "192000", "0", "G", "909656", "Phocaeicola"] == kreport_results[13]
    assert [
        "5.70",
        "700000",
        "700000",
        "S",
        "165179",
        "Prevotella copri",
    ] == kreport_results[14]
    assert [
        "5.82",
        "714000",
        "714000",
        "S",
        "562",
        "Escherichia coli",
    ] == kreport_results[15]
    assert [
        "1.56",
        "192000",
        "192000",
        "S",
        "821",
        "Phocaeicola vulgatus",
    ] == kreport_results[16]


def test_metagenome_kreport_out_lemonade(runtmp):
    # test 'kreport' kraken output format against lemonade output
    g_csv = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.csv")
    tax = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.tax.csv")
    csv_base = "out"
    sum_csv = csv_base + ".kreport.txt"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-dir",
        outdir,
        "-F",
        "kreport",
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)

    kreport_results = [
        x.rstrip().split("\t") for x in Path(csvout).read_text().splitlines()
    ]
    assert f"saving 'kreport' output to '{csvout}'" in runtmp.last_result.err
    print(kreport_results)
    assert ["5.35", "116000", "0", "D", "", "d__Bacteria"] == kreport_results[0]
    assert ["94.65", "2054000", "2054000", "U", "", "unclassified"] == kreport_results[
        1
    ]
    assert ["5.35", "116000", "0", "P", "", "p__Bacteroidota"] == kreport_results[2]
    assert ["5.35", "116000", "0", "C", "", "c__Chlorobia"] == kreport_results[3]
    assert ["5.35", "116000", "0", "O", "", "o__Chlorobiales"] == kreport_results[4]
    assert ["5.35", "116000", "0", "F", "", "f__Chlorobiaceae"] == kreport_results[5]
    assert ["5.35", "116000", "0", "G", "", "g__Prosthecochloris"] == kreport_results[6]
    assert [
        "5.35",
        "116000",
        "116000",
        "S",
        "",
        "s__Prosthecochloris vibrioformis",
    ] == kreport_results[7]


def test_metagenome_kreport_out_fail(runtmp):
    # kreport cannot be generated with gather results from < v4.5.0
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    sum_csv = csv_base + ".kreport.txt"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    with pytest.raises(SourmashCommandFailed):
        runtmp.run_sourmash(
            "tax",
            "metagenome",
            "--gather-csv",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-o",
            csv_base,
            "--output-dir",
            outdir,
            "-F",
            "kreport",
        )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert (
        "ERROR: cannot produce 'kreport' format from gather results before sourmash v4.5.0"
        in runtmp.last_result.err
    )


def test_metagenome_bioboxes_stdout(runtmp):
    # test CAMI bioboxes format output
    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.ncbi-taxonomy.csv")

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-F",
        "bioboxes",
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0

    assert "# Taxonomic Profiling Output" in runtmp.last_result.out
    assert "@SampleID:test1" in runtmp.last_result.out
    assert "@Version:0.10.0" in runtmp.last_result.out
    assert (
        "@Ranks:superkingdom|phylum|class|order|family|genus|species|strain"
        in runtmp.last_result.out
    )
    assert "@__program__:sourmash" in runtmp.last_result.out
    assert "2	superkingdom	2	Bacteria	13.08" in runtmp.last_result.out
    assert (
        "976	phylum	2|976	Bacteria|Bacteroidota	7.27"
        in runtmp.last_result.out
    )
    assert (
        "1224	phylum	2|1224	Bacteria|Pseudomonadota	5.82"
        in runtmp.last_result.out
    )
    assert (
        "200643	class	2|976|200643	Bacteria|Bacteroidota|Bacteroidia	7.27"
        in runtmp.last_result.out
    )
    assert (
        "1236	class	2|1224|1236	Bacteria|Pseudomonadota|Gammaproteobacteria	5.82"
        in runtmp.last_result.out
    )
    assert (
        "171549	order	2|976|200643|171549	Bacteria|Bacteroidota|Bacteroidia|Bacteroidales	7.27"
        in runtmp.last_result.out
    )
    assert (
        "91347	order	2|1224|1236|91347	Bacteria|Pseudomonadota|Gammaproteobacteria|Enterobacterales	5.82"
        in runtmp.last_result.out
    )
    assert (
        "171552	family	2|976|200643|171549|171552	Bacteria|Bacteroidota|Bacteroidia|Bacteroidales|Prevotellaceae	5.70"
        in runtmp.last_result.out
    )
    assert (
        "543	family	2|1224|1236|91347|543	Bacteria|Pseudomonadota|Gammaproteobacteria|Enterobacterales|Enterobacteriaceae	5.82"
        in runtmp.last_result.out
    )
    assert (
        "815	family	2|976|200643|171549|815	Bacteria|Bacteroidota|Bacteroidia|Bacteroidales|Bacteroidaceae	1.56"
        in runtmp.last_result.out
    )
    assert (
        "838	genus	2|976|200643|171549|171552|838	Bacteria|Bacteroidota|Bacteroidia|Bacteroidales|Prevotellaceae|Prevotella	5.70"
        in runtmp.last_result.out
    )
    assert (
        "561	genus	2|1224|1236|91347|543|561	Bacteria|Pseudomonadota|Gammaproteobacteria|Enterobacterales|Enterobacteriaceae|Escherichia	5.82"
        in runtmp.last_result.out
    )
    assert (
        "909656	genus	2|976|200643|171549|815|909656	Bacteria|Bacteroidota|Bacteroidia|Bacteroidales|Bacteroidaceae|Phocaeicola	1.56"
        in runtmp.last_result.out
    )
    assert (
        "165179	species	2|976|200643|171549|171552|838|165179	Bacteria|Bacteroidota|Bacteroidia|Bacteroidales|Prevotellaceae|Prevotella|Prevotella copri	5.70"
        in runtmp.last_result.out
    )
    assert (
        "562	species	2|1224|1236|91347|543|561|562	Bacteria|Pseudomonadota|Gammaproteobacteria|Enterobacterales|Enterobacteriaceae|Escherichia|Escherichia coli	5.82"
        in runtmp.last_result.out
    )
    assert (
        "821	species	2|976|200643|171549|815|909656|821	Bacteria|Bacteroidota|Bacteroidia|Bacteroidales|Bacteroidaceae|Phocaeicola|Phocaeicola vulgatus	1.56"
        in runtmp.last_result.out
    )


def test_metagenome_bioboxes_outfile(runtmp):
    # test CAMI bioboxes format output
    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.ncbi-taxonomy.csv")
    csv_base = "out"
    sum_csv = csv_base + ".bioboxes.profile"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-F",
        "bioboxes",
        "-o",
        csv_base,
        "--output-dir",
        outdir,
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0

    bb_results = [x.rstrip().split("\t") for x in Path(csvout).read_text().splitlines()]
    assert f"saving 'bioboxes' output to '{csvout}'" in runtmp.last_result.err
    print(bb_results)
    assert ["# Taxonomic Profiling Output"] == bb_results[0]
    assert ["@SampleID:test1"] == bb_results[1]
    assert ["2", "superkingdom", "2", "Bacteria", "13.08"] == bb_results[6]
    assert [
        "838",
        "genus",
        "2|976|200643|171549|171552|838",
        "Bacteria|Bacteroidota|Bacteroidia|Bacteroidales|Prevotellaceae|Prevotella",
        "5.70",
    ] == bb_results[16]


def test_metagenome_krona_tsv_out(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    kr_csv = csv_base + ".krona.tsv"
    csvout = runtmp.output(kr_csv)
    outdir = os.path.dirname(csvout)
    print("csvout: ", csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-format",
        "krona",
        "--rank",
        "genus",
        "--output-dir",
        outdir,
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)
    assert f"saving 'krona' output to '{csvout}'" in runtmp.last_result.err

    gn_krona_results = [
        x.rstrip().split("\t") for x in Path(csvout).read_text().splitlines()
    ]
    print("species krona results: \n", gn_krona_results)
    assert [
        "fraction",
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
    ] == gn_krona_results[0]
    assert [
        "0.0885520542481053",
        "d__Bacteria",
        "p__Bacteroidota",
        "c__Bacteroidia",
        "o__Bacteroidales",
        "f__Bacteroidaceae",
        "g__Prevotella",
    ] == gn_krona_results[1]
    assert [
        "0.08815317112086159",
        "d__Bacteria",
        "p__Proteobacteria",
        "c__Gammaproteobacteria",
        "o__Enterobacterales",
        "f__Enterobacteriaceae",
        "g__Escherichia",
    ] == gn_krona_results[2]
    assert [
        "0.027522935779816515",
        "d__Bacteria",
        "p__Bacteroidota",
        "c__Bacteroidia",
        "o__Bacteroidales",
        "f__Bacteroidaceae",
        "g__Phocaeicola",
    ] == gn_krona_results[3]
    assert [
        "0.7957718388512166",
        "unclassified",
        "unclassified",
        "unclassified",
        "unclassified",
        "unclassified",
        "unclassified",
    ] == gn_krona_results[4]


def test_metagenome_lineage_summary_out(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    lin_csv = csv_base + ".lineage_summary.tsv"
    csvout = runtmp.output(lin_csv)
    outdir = os.path.dirname(csvout)
    print("csvout: ", csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-format",
        "lineage_summary",
        "--rank",
        "genus",
        "--output-dir",
        outdir,
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)
    assert f"saving 'lineage_summary' output to '{csvout}'" in runtmp.last_result.err

    gn_lineage_summary = [
        x.rstrip().split("\t") for x in Path(csvout).read_text().splitlines()
    ]
    print("species lineage summary results: \n", gn_lineage_summary)
    assert ["lineage", "test1"] == gn_lineage_summary[0]
    assert [
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola",
        "0.027522935779816515",
    ] == gn_lineage_summary[1]
    assert [
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella",
        "0.0885520542481053",
    ] == gn_lineage_summary[2]
    assert [
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia",
        "0.08815317112086159",
    ] == gn_lineage_summary[3]
    assert ["unclassified", "0.7957718388512166"] == gn_lineage_summary[4]


def test_metagenome_human_format_out(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    csvout = runtmp.output(csv_base + ".human.txt")
    outdir = os.path.dirname(csvout)
    print("csvout: ", csvout)

    runtmp.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        csv_base,
        "--output-format",
        "human",
        "--rank",
        "genus",
        "--output-dir",
        outdir,
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)
    assert f"saving 'human' output to '{csvout}'" in runtmp.last_result.err

    with open(csvout) as fp:
        outp = fp.readlines()

    assert len(outp) == 6
    outp = [x.strip() for x in outp]
    print(outp)

    assert outp[0] == "sample name    proportion   cANI   lineage"
    assert outp[1] == "-----------    ----------   ----   -------"
    assert outp[2] == "test1             86.9%     -      unclassified"
    assert (
        outp[3]
        == "test1              5.8%     92.5%  d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia"
    )
    assert (
        outp[4]
        == "test1              5.7%     92.5%  d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella"
    )
    assert (
        outp[5]
        == "test1              1.6%     89.1%  d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola"
    )


def test_metagenome_no_taxonomy_fail(runtmp):
    c = runtmp
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash("tax", "metagenome", "-g", g_csv)
    assert "error: the following arguments are required: -t/--taxonomy-csv" in str(
        exc.value
    )


def test_metagenome_no_rank_lineage_summary(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-o",
            csv_base,
            "--output-format",
            "lineage_summary",
        )
    print(str(exc.value))
    assert (
        "Rank (--rank) is required for krona, lineage_summary output formats."
        in str(exc.value)
    )


def test_metagenome_no_rank_krona(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-o",
            csv_base,
            "--output-format",
            "krona",
        )
    print(str(exc.value))
    assert (
        "Rank (--rank) is required for krona, lineage_summary output formats."
        in str(exc.value)
    )


def test_metagenome_bad_rank_krona(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-o",
            csv_base,
            "--output-format",
            "krona",
            "--rank",
            "NotARank",
        )
    print(str(exc.value))
    assert (
        "Invalid '--rank'/'--position' input: 'NotARank'. Please choose: 'strain', 'species', 'genus', 'family', 'order', 'class', 'phylum', 'superkingdom'"
        in runtmp.last_result.err
    )

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-o",
            csv_base,
            "--output-format",
            "krona",
            "--rank",
            "5",
        )
    print(str(exc.value))
    assert (
        "Invalid '--rank'/'--position' input: '5'. Please choose: 'strain', 'species', 'genus', 'family', 'order', 'class', 'phylum', 'superkingdom'"
        in runtmp.last_result.err
    )


def test_metagenome_ictv(runtmp):
    # test basic metagenome
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.ictv-taxonomy.csv")

    c.run_sourmash("tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax, "--ictv")

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    print(c.last_result.out)
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,realm,0.204,Riboviria,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,realm,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,kingdom,0.204,Riboviria;;Orthornavirae,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,kingdom,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.204,Riboviria;;Orthornavirae;;Negarnaviricota,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,subphylum,0.204,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,subphylum,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,class,0.204,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,class,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,order,0.204,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,order,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,family,0.204,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,family,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.204,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus,md5,test1.sig,0.131,1024000,0.950,0"
        in c.last_result.out
    )
    assert (
        "test1,genus,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,species,0.088,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus bundibugyoense,md5,test1.sig,0.058,442000,0.925,0"
        in c.last_result.out
    )
    assert (
        "test1,species,0.078,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus taiense,md5,test1.sig,0.050,390000,0.921,0"
        in c.last_result.out
    )
    assert (
        "test1,species,0.028,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus bombaliense,md5,test1.sig,0.016,138000,0.891,0"
        in c.last_result.out
    )
    assert (
        "test1,species,0.011,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus restonense,md5,test1.sig,0.007,54000,0.864,0"
        in c.last_result.out
    )
    assert (
        "test1,species,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )
    assert (
        "test1,name,0.088,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus bundibugyoense;Bundibugyo virus,md5,test1.sig,0.058,442000,0.925,0"
        in c.last_result.out
    )
    assert (
        "test1,name,0.078,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus taiense;Taï Forest virus,md5,test1.sig,0.050,390000,0.921,0"
        in c.last_result.out
    )
    assert (
        "test1,name,0.028,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus bombaliense;Bombali virus,md5,test1.sig,0.016,138000,0.891,0"
        in c.last_result.out
    )
    assert (
        "test1,name,0.011,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus restonense;Reston virus,md5,test1.sig,0.007,54000,0.864,0"
        in c.last_result.out
    )
    assert (
        "test1,name,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )


def test_genome_no_rank_krona(runtmp):
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-o",
            csv_base,
            "--output-format",
            "krona",
        )
    assert "ERROR: Rank (--rank) is required for krona output formats" in str(exc.value)


def test_metagenome_rank_not_available(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax, "--rank", "strain"
        )

    print(str(exc.value))

    assert c.last_result.status == -1
    assert (
        "No taxonomic information provided for rank strain: cannot summarize at this rank"
        in str(exc.value)
    )


def test_metagenome_duplicated_taxonomy_fail(runtmp):
    c = runtmp
    # write temp taxonomy with duplicates
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    duplicated_csv = runtmp.output("duplicated_taxonomy.csv")
    with open(duplicated_csv, "w") as dup:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax.append(tax[1] + "FOO")  # add first tax_assign again
        dup.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", duplicated_csv
        )

    assert "cannot read taxonomy" in str(exc.value)
    assert "multiple lineages for identifier GCF_001881345" in str(exc.value)


def test_metagenome_duplicated_taxonomy_force(runtmp):
    c = runtmp
    # write temp taxonomy with duplicates
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    duplicated_csv = runtmp.output("duplicated_taxonomy.csv")
    with open(duplicated_csv, "w") as dup:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax.append(tax[1])  # add first tax_assign again
        dup.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", duplicated_csv, "--force"
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    # same as stdout test - just check the first few lines
    assert c.last_result.status == 0
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.204,d__Bacteria,md5,test1.sig,0.131,1024000"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.116,d__Bacteria;p__Bacteroidota,md5,test1.sig,0.073,582000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.088,d__Bacteria;p__Proteobacteria,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.796,unclassified,md5,test1.sig,0.869,3990000"
        in c.last_result.out
    )


def test_metagenome_missing_taxonomy(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        subset.write("\n".join(tax[:4]))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash("tax", "metagenome", "-g", g_csv, "--taxonomy-csv", subset_csv)
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "The following are missing from the taxonomy information: GCF_003471795"
        in c.last_result.err
    )

    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.193,d__Bacteria,md5,test1.sig,0.124,970000"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.807,unclassified,md5,test1.sig,0.876,4044000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.105,d__Bacteria;p__Bacteroidota,md5,test1.sig,0.066,528000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.088,d__Bacteria;p__Proteobacteria,md5,test1.sig,0.058,442000"
        in c.last_result.out
    )
    assert (
        "test1,phylum,0.807,unclassified,md5,test1.sig,0.876,4044000"
        in c.last_result.out
    )
    assert (
        "test1,class,0.105,d__Bacteria;p__Bacteroidota;c__Bacteroidia,md5,test1.sig,0.066,528000"
        in c.last_result.out
    )


def test_metagenome_missing_fail_taxonomy(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        subset.write("\n".join(tax[:4]))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            subset_csv,
            "--fail-on-missing-taxonomy",
        )

    print(str(exc.value))

    assert "ident 'GCF_003471795' is not in the taxonomy database." in str(exc.value)
    assert "Failing, as requested via --fail-on-missing-taxonomy" in str(exc.value)
    assert c.last_result.status == -1


def test_metagenome_multiple_taxonomy_files_missing(runtmp):
    c = runtmp
    # write temp taxonomy with duplicates
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")

    # gather against mult databases
    g_csv = utils.get_test_data("tax/test1_x_gtdbrs202_genbank_euks.gather.csv")

    c.run_sourmash(
        "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", taxonomy_csv, "--force"
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert (
        "of 6 gather results, lineage assignments for 2 results were missed"
        in c.last_result.err
    )
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.204,d__Bacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.131,1024000"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.796,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.116,d__Bacteria;p__Bacteroidota,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.088,d__Bacteria;p__Proteobacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.058,442000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.796,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.869,3990000"
        in c.last_result.out
    )
    assert (
        "multtest,class,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )
    assert (
        "multtest,class,0.088,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.058,442000"
        in c.last_result.out
    )
    assert (
        "multtest,class,0.796,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.869,3990000"
        in c.last_result.out
    )


def test_metagenome_multiple_taxonomy_files(runtmp):
    c = runtmp
    # write temp taxonomy with duplicates
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    protozoa_genbank = utils.get_test_data("tax/protozoa_genbank_lineage.csv")
    bacteria_refseq = utils.get_test_data("tax/bacteria_refseq_lineage.csv")

    # gather against mult databases
    g_csv = utils.get_test_data("tax/test1_x_gtdbrs202_genbank_euks.gather.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        taxonomy_csv,
        protozoa_genbank,
        bacteria_refseq,
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.204,Bacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.131,1024000"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.051,Eukaryota,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.245,258000"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.744,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.624,3732000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.116,Bacteria;Bacteroidetes,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.088,Bacteria;Proteobacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.058,442000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.051,Eukaryota;Apicomplexa,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.245,258000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.744,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.624,3732000"
        in c.last_result.out
    )
    assert (
        "multtest,class,0.116,Bacteria;Bacteroidetes;Bacteroidia,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )


def test_metagenome_multiple_taxonomy_files_multiple_taxonomy_args(runtmp):
    c = runtmp
    # pass in mult tax files using mult tax arguments
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    protozoa_genbank = utils.get_test_data("tax/protozoa_genbank_lineage.csv")
    bacteria_refseq = utils.get_test_data("tax/bacteria_refseq_lineage.csv")

    # gather against mult databases
    g_csv = utils.get_test_data("tax/test1_x_gtdbrs202_genbank_euks.gather.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        taxonomy_csv,
        "-t",
        protozoa_genbank,
        "-t",
        bacteria_refseq,
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.204,Bacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.131,1024000"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.051,Eukaryota,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.245,258000"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.744,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.624,3732000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.116,Bacteria;Bacteroidetes,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.088,Bacteria;Proteobacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.058,442000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.051,Eukaryota;Apicomplexa,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.245,258000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.744,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.624,3732000"
        in c.last_result.out
    )
    assert (
        "multtest,class,0.116,Bacteria;Bacteroidetes;Bacteroidia,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )


def test_metagenome_multiple_taxonomy_files_multiple_taxonomy_args_empty_force(runtmp):
    # pass in mult tax files using mult tax arguments, with one empty,
    # and use --force
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    protozoa_genbank = utils.get_test_data("tax/protozoa_genbank_lineage.csv")
    bacteria_refseq = utils.get_test_data("tax/bacteria_refseq_lineage.csv")

    tax_empty = runtmp.output("t.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    # gather against mult databases
    g_csv = utils.get_test_data("tax/test1_x_gtdbrs202_genbank_euks.gather.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        taxonomy_csv,
        "-t",
        protozoa_genbank,
        "-t",
        bacteria_refseq,
        "-t",
        tax_empty,
        "--force",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.204,Bacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.131,1024000"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.051,Eukaryota,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.245,258000"
        in c.last_result.out
    )
    assert (
        "multtest,superkingdom,0.744,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.624,3732000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.116,Bacteria;Bacteroidetes,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.088,Bacteria;Proteobacteria,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.058,442000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.051,Eukaryota;Apicomplexa,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.245,258000"
        in c.last_result.out
    )
    assert (
        "multtest,phylum,0.744,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.624,3732000"
        in c.last_result.out
    )
    assert (
        "multtest,class,0.116,Bacteria;Bacteroidetes;Bacteroidia,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.073,582000"
        in c.last_result.out
    )


def test_metagenome_empty_gather_results(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    # creates empty gather result
    g_csv = runtmp.output("g.csv")
    with open(g_csv, "w") as fp:
        fp.write("")
    print("g_csv: ", g_csv)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax)

    assert f"Cannot read gather results from '{g_csv}'. Is file empty?" in str(
        exc.value
    )
    assert runtmp.last_result.status == -1


def test_metagenome_bad_gather_header(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    bad_g_csv = runtmp.output("g.csv")

    # creates bad gather result
    bad_g = [
        x.replace("query_bp", "nope") + "\n"
        for x in Path(g_csv).read_text().splitlines()
    ]
    with open(bad_g_csv, "w") as fp:
        fp.writelines(bad_g)
    print("bad_gather_results: \n", bad_g)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "metagenome", "-g", bad_g_csv, "--taxonomy-csv", tax)

    print(str(exc.value))
    assert "is missing columns needed for taxonomic summarization." in str(exc.value)
    assert runtmp.last_result.status == -1


def test_metagenome_empty_tax_lineage_input(runtmp):
    # test an empty tax CSV
    tax_empty = runtmp.output("t.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax_empty
        )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status != 0
    assert "cannot read taxonomy assignments from" in str(exc.value)


def test_metagenome_empty_tax_lineage_input_force(runtmp):
    # test an empty tax CSV with --force
    tax_empty = runtmp.output("t.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax_empty, "--force"
        )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status != 0
    assert "ERROR: No taxonomic assignments loaded" in str(exc.value)


def test_metagenome_perfect_match_warning(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    perfect_g_csv = runtmp.output("g.csv")

    # create a perfect gather result
    with open(g_csv) as fp:
        r = csv.DictReader(fp, delimiter=",")
        header = r.fieldnames
        print(header)
        with open(perfect_g_csv, "w") as out_fp:
            w = csv.DictWriter(out_fp, header)
            w.writeheader()
            for n, row in enumerate(r):
                if n == 0:
                    # make a perfect match
                    row["f_unique_to_query"] = 1.0
                else:
                    # set the rest to 0
                    row["f_unique_to_query"] = 0.0
                w.writerow(row)
                print(row)

    runtmp.run_sourmash("tax", "metagenome", "-g", perfect_g_csv, "--taxonomy-csv", tax)

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert (
        "WARNING: 100% match! Is query 'test1' identical to its database match, 'GCF_001881345'?"
        in runtmp.last_result.err
    )


def test_metagenome_over100percent_error(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    perfect_g_csv = runtmp.output("g.csv")

    # create a perfect gather result
    with open(g_csv) as fp:
        r = csv.DictReader(fp, delimiter=",")
        header = r.fieldnames
        print(header)
        with open(perfect_g_csv, "w") as out_fp:
            w = csv.DictWriter(out_fp, header)
            w.writeheader()
            for n, row in enumerate(r):
                if n == 0:
                    row["f_unique_to_query"] = 1.0
                # let the rest stay as they are (should be > 100% match now)
                w.writerow(row)
                print(row)

    with pytest.raises(SourmashCommandFailed):
        runtmp.run_sourmash(
            "tax", "metagenome", "-g", perfect_g_csv, "--taxonomy-csv", tax
        )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == -1
    assert (
        "fraction is > 100% of the query! This should not be possible."
        in runtmp.last_result.err
    )


def test_metagenome_gather_duplicate_query(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # different filename, contents identical to test1
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        fp.write(Path(g_res).read_text())

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "metagenome",
            "--gather-csv",
            g_res,
            g_res2,
            "--taxonomy-csv",
            taxonomy_csv,
        )

    assert c.last_result.status == -1
    print(str(exc.value))
    assert (
        "Gather query test1 was found in more than one CSV. Cannot load from "
        in str(exc.value)
    )


def test_metagenome_gather_duplicate_query_force(runtmp):
    # do not load same query from multiple files.
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # different filename, contents identical to test1
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        fp.write(Path(g_res).read_text())

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "--gather-csv",
            g_res,
            g_res2,
            "--taxonomy-csv",
            taxonomy_csv,
            "--force",
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1

    assert "Gather query test1 was found in more than one CSV." in c.last_result.err
    assert "Cannot force past duplicated gather query. Exiting." in c.last_result.err


def test_metagenome_two_queries_human_output(runtmp):
    # do not load same query from multiple files.
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make a second query with same output
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        for line in Path(g_res).read_text().splitlines():
            line = line.replace("test1", "test2") + "\n"
            fp.write(line)

    c.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_res,
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "-F",
        "human",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert "test1             86.9%     -      unclassified" in c.last_result.out
    assert (
        "test1              5.8%     92.5%  d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
        in c.last_result.out
    )
    assert "test2             86.9%     -      unclassified" in c.last_result.out
    assert (
        "test2              5.8%     92.5%  d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
        in c.last_result.out
    )
    assert "test2              5.7%     92.5%  d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
    assert "test2              1.6%     89.1%  d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus"


def test_metagenome_two_queries_csv_summary_output(runtmp):
    # remove single-query outputs when working with multiple queries
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make a second query with same output
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        for line in Path(g_res).read_text().splitlines():
            line = line.replace("test1", "test2") + "\n"
            fp.write(line)

    csv_summary_out = runtmp.output("tst.summarized.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_res,
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "-F",
        "csv_summary",
        "--rank",
        "phylum",
        "-o",
        "tst",
    )

    assert os.path.exists(csv_summary_out)

    assert c.last_result.status == 0
    assert "loaded results for 2 queries from 2 gather CSVs" in c.last_result.err
    assert (
        f"saving 'csv_summary' output to '{os.path.basename(csv_summary_out)}'"
        in runtmp.last_result.err
    )
    sum_gather_results = [
        x.rstrip() for x in Path(csv_summary_out).read_text().splitlines()
    ]
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in sum_gather_results[0]
    )
    # check both queries exist in csv_summary results; check several
    assert (
        "test1,superkingdom,0.2042281611487834,d__Bacteria,md5,test1.sig,0.13080306238801107,1024000,0.9500482567175479,0"
        in sum_gather_results[1]
    )
    assert (
        "test2,superkingdom,0.2042281611487834,d__Bacteria,md5,test2.sig,0.13080306238801107,1024000,0.9500482567175479,0"
        in sum_gather_results[23]
    )
    assert (
        "test2,phylum,0.11607499002792182,d__Bacteria;p__Bacteroidota,md5,test2.sig,0.07265026877341586,582000"
        in sum_gather_results[25]
    )
    assert (
        "test2,phylum,0.08815317112086159,d__Bacteria;p__Proteobacteria,md5,test2.sig,0.05815279361459521,442000"
        in sum_gather_results[26]
    )
    assert (
        "test2,phylum,0.7957718388512166,unclassified,md5,test2.sig,0.8691969376119889,3990000"
        in sum_gather_results[27]
    )
    assert (
        "test2,class,0.11607499002792182,d__Bacteria;p__Bacteroidota;c__Bacteroidia,md5,test2.sig,0.07265026877341586,582000"
        in sum_gather_results[28]
    )
    assert (
        "test2,class,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria,md5,test2.sig,0.05815279361459521,442000"
        in sum_gather_results[29]
    )
    assert (
        "test2,class,0.7957718388512166,unclassified,md5,test2.sig,0.8691969376119889,3990000"
        in sum_gather_results[30]
    )
    assert (
        "test2,order,0.11607499002792182,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales,md5,test2.sig,0.07265026877341586,582000"
        in sum_gather_results[31]
    )
    assert (
        "test2,order,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales,md5,test2.sig,0.05815279361459521,442000"
        in sum_gather_results[32]
    )
    assert (
        "test2,order,0.7957718388512166,unclassified,md5,test2.sig,0.8691969376119889,3990000"
        in sum_gather_results[33]
    )
    assert (
        "test2,family,0.11607499002792182,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test2.sig,0.07265026877341586,582000"
        in sum_gather_results[34]
    )
    assert (
        "test2,family,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae,md5,test2.sig,0.05815279361459521,442000"
        in sum_gather_results[35]
    )
    assert (
        "test2,family,0.7957718388512166,unclassified,md5,test2.sig,0.8691969376119889,3990000"
        in sum_gather_results[36]
    )
    assert (
        "test2,genus,0.0885520542481053,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella,md5,test2.sig,0.05701254275940707,444000"
        in sum_gather_results[37]
    )
    assert (
        "test2,genus,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia,md5,test2.sig,0.05815279361459521,442000"
        in sum_gather_results[38]
    )
    assert (
        "test2,genus,0.027522935779816515,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola,md5,test2.sig,0.015637726014008795,138000"
        in sum_gather_results[39]
    )
    assert (
        "test2,genus,0.7957718388512166,unclassified,md5,test2.sig,0.8691969376119889,3990000"
        in sum_gather_results[40]
    )
    assert (
        "test2,species,0.0885520542481053,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test2.sig,0.05701254275940707,444000"
        in sum_gather_results[41]
    )
    assert (
        "test2,species,0.08815317112086159,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli,md5,test2.sig,0.05815279361459521,442000"
        in sum_gather_results[42]
    )
    assert (
        "test2,species,0.027522935779816515,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus,md5,test2.sig,0.015637726014008795,138000"
        in sum_gather_results[43]
    )


def test_metagenome_two_queries_with_single_query_output_formats_fail(runtmp):
    # fail on multiple queries with single query output formats
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make a second query with same output
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        for line in Path(g_res).read_text().splitlines():
            line = line.replace("test1", "test2") + "\n"
            fp.write(line)

    runtmp.output("tst.summarized.csv")
    bioboxes_out = runtmp.output("tst.bioboxes.out")
    kreport_out = runtmp.output("tst.kreport.txt")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "metagenome",
            "--gather-csv",
            g_res,
            g_res2,
            "--taxonomy-csv",
            taxonomy_csv,
            "-F",
            "bioboxes",
            "kreport",
            "--rank",
            "phylum",
            "-o",
            "tst",
        )
    print(str(exc.value))

    assert not os.path.exists(bioboxes_out)
    assert not os.path.exists(kreport_out)

    assert c.last_result.status == -1
    assert "loaded results for 2 queries from 2 gather CSVs" in c.last_result.err
    assert (
        "WARNING: found results for multiple gather queries. Can only output multi-query result formats: skipping bioboxes, kreport"
        in c.last_result.err
    )
    assert "ERROR: No output formats remaining." in c.last_result.err


def test_metagenome_two_queries_skip_single_query_output_formats(runtmp):
    # remove single-query outputs when working with multiple queries
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make a second query with same output
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        for line in Path(g_res).read_text().splitlines():
            line = line.replace("test1", "test2") + "\n"
            fp.write(line)

    csv_summary_out = runtmp.output("tst.summarized.csv")
    kreport_out = runtmp.output("tst.kreport.txt")
    bioboxes_out = runtmp.output("tst.bioboxes.txt")
    lineage_summary_out = runtmp.output("tst.lineage_summary.tsv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_res,
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "-F",
        "csv_summary",
        "bioboxes",
        "kreport",
        "lineage_summary",
        "--rank",
        "phylum",
        "-o",
        "tst",
    )

    assert not os.path.exists(kreport_out)
    assert not os.path.exists(bioboxes_out)
    assert os.path.exists(csv_summary_out)
    assert os.path.exists(lineage_summary_out)

    assert c.last_result.status == 0
    assert "loaded results for 2 queries from 2 gather CSVs" in c.last_result.err
    assert (
        "WARNING: found results for multiple gather queries. Can only output multi-query result formats: skipping bioboxes, kreport"
        in c.last_result.err
    )

    assert (
        f"saving 'csv_summary' output to '{os.path.basename(csv_summary_out)}'"
        in runtmp.last_result.err
    )
    sum_gather_results = [
        x.rstrip() for x in Path(csv_summary_out).read_text().splitlines()
    ]
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in sum_gather_results[0]
    )
    # check both queries exist in csv_summary results
    assert (
        "test1,superkingdom,0.2042281611487834,d__Bacteria,md5,test1.sig,0.13080306238801107,1024000,0.9500482567175479,0"
        in sum_gather_results[1]
    )
    assert (
        "test2,superkingdom,0.2042281611487834,d__Bacteria,md5,test2.sig,0.13080306238801107,1024000,0.9500482567175479,0"
        in sum_gather_results[23]
    )


def test_metagenome_two_queries_krona(runtmp):
    # for now, we enable multi-query krona. Is this desired?
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make a second query with same output
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        for line in Path(g_res).read_text().splitlines():
            line = line.replace("test1", "test2") + "\n"
            fp.write(line)

    c.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_res,
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "-F",
        "krona",
        "--rank",
        "superkingdom",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "WARNING: results from more than one query found. Krona summarization not recommended."
        in c.last_result.err
    )
    assert (
        "Percentage assignment will be normalized by the number of queries to maintain range 0-100%"
        in c.last_result.err
    )
    assert "fraction	superkingdom" in c.last_result.out
    assert "0.2042281611487834	d__Bacteria" in c.last_result.out
    assert "0.7957718388512166	unclassified" in c.last_result.out


def test_metagenome_gather_duplicate_filename(runtmp):
    # test that a duplicate filename is properly flagged, when passed in
    # twice to a single -g argument.
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_res,
        g_res,
        "--taxonomy-csv",
        taxonomy_csv,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert f"ignoring duplicated reference to file: {g_res}"
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.204,d__Bacteria,md5,test1.sig,0.131,1024000"
        in c.last_result.out
    )


def test_metagenome_gather_duplicate_filename_2(runtmp):
    # test that a duplicate filename is properly flagged, with -g a -g b
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "--gather-csv",
        g_res,
        "-g",
        g_res,
        "--taxonomy-csv",
        taxonomy_csv,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert f"ignoring duplicated reference to file: {g_res}"
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.204,d__Bacteria,md5,test1.sig,0.131,1024000"
        in c.last_result.out
    )


def test_metagenome_gather_duplicate_filename_from_file(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")
    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")
        f_csv.write(f"{g_res}\n")

    c.run_sourmash(
        "tax", "metagenome", "--from-file", g_from_file, "--taxonomy-csv", taxonomy_csv
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert f"ignoring duplicated reference to file: {g_res}"
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,superkingdom,0.204,d__Bacteria,md5,test1.sig,0.131,1024000"
        in c.last_result.out
    )


def test_genome_empty_gather_results(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    # creates empty gather result
    g_csv = runtmp.output("g.csv")
    with open(g_csv, "w") as fp:
        fp.write("")
    print("g_csv: ", g_csv)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "genome", "-g", g_csv, "--taxonomy-csv", tax)

    assert runtmp.last_result.status == -1
    print(runtmp.last_result.err)
    print(runtmp.last_result.out)
    assert f"Cannot read gather results from '{g_csv}'. Is file empty?" in str(
        exc.value
    )


def test_genome_bad_gather_header(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    bad_g_csv = runtmp.output("g.csv")

    # creates bad gather result
    bad_g = [
        x.replace("f_unique_to_query", "nope") + "\n"
        for x in Path(g_csv).read_text().splitlines()
    ]
    with open(bad_g_csv, "w") as fp:
        fp.writelines(bad_g)
    print("bad_gather_results: \n", bad_g)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "genome", "-g", bad_g_csv, "--taxonomy-csv", tax)

    assert "is missing columns needed for taxonomic summarization." in str(exc.value)
    assert runtmp.last_result.status == -1


def test_genome_empty_tax_lineage_input(runtmp):
    # test an empty tax csv
    tax_empty = runtmp.output("t.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "genome", "-g", g_csv, "--taxonomy-csv", tax_empty)

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status != 0
    assert "cannot read taxonomy assignments from" in str(exc.value)


def test_genome_rank_stdout_0(runtmp):
    # test basic genome
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    c.run_sourmash(
        "tax",
        "genome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_rank_stdout_0_db(runtmp):
    # test basic genome with sqlite database
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.db")

    c.run_sourmash(
        "tax",
        "genome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )

    # too stringent of containment threshold:
    c.run_sourmash(
        "tax",
        "genome",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--rank",
        "species",
        "--containment-threshold",
        "1.0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "test1,below_threshold,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000,"
        in c.last_result.out
    )


def test_genome_rank_csv_0(runtmp):
    # test basic genome - output csv
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    cl_csv = csv_base + ".classifications.csv"
    csvout = runtmp.output(cl_csv)
    outdir = os.path.dirname(csvout)
    print("csvout: ", csvout)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--rank",
        "species",
        "-o",
        csv_base,
        "--containment-threshold",
        "0",
        "--output-dir",
        outdir,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert f"saving 'classification' output to '{csvout}'" in runtmp.last_result.err
    assert c.last_result.status == 0
    cl_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in cl_results[0]
    )
    assert (
        "test1,match,species,0.0885520542481053,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.05701254275940707,444000"
        in cl_results[1]
    )


def test_genome_rank_krona(runtmp):
    # test basic genome - output csv
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    cl_csv = csv_base + ".krona.tsv"
    csvout = runtmp.output(cl_csv)
    outdir = os.path.dirname(csvout)
    print("csvout: ", csvout)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--rank",
        "species",
        "-o",
        csv_base,
        "--containment-threshold",
        "0",
        "--output-format",
        "krona",
        "--output-dir",
        outdir,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert f"saving 'krona' output to '{csvout}'" in runtmp.last_result.err
    assert c.last_result.status == 0
    kr_results = [x.rstrip().split("\t") for x in Path(csvout).read_text().splitlines()]
    print(kr_results)
    assert [
        "fraction",
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ] == kr_results[0]
    assert [
        "0.0885520542481053",
        "d__Bacteria",
        "p__Bacteroidota",
        "c__Bacteroidia",
        "o__Bacteroidales",
        "f__Bacteroidaceae",
        "g__Prevotella",
        "s__Prevotella copri",
    ] == kr_results[1]


def test_genome_rank_human_output(runtmp):
    # test basic genome - output csv
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    csvout = runtmp.output(csv_base + ".human.txt")
    outdir = os.path.dirname(csvout)
    print("csvout: ", csvout)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--rank",
        "species",
        "-o",
        csv_base,
        "--containment-threshold",
        "0",
        "--output-format",
        "human",
        "--output-dir",
        outdir,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert f"saving 'human' output to '{csvout}'" in runtmp.last_result.err
    assert c.last_result.status == 0

    with open(csvout) as fp:
        outp = fp.readlines()
        print(outp)

    assert len(outp) == 3
    outp = [x.strip() for x in outp]

    assert outp[0] == "sample name    status    proportion   cANI   lineage"
    assert outp[1] == "-----------    ------    ----------   ----   -------"
    assert (
        outp[2]
        == "test1             match     5.7%     92.5%  d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
    )


def test_genome_rank_lineage_csv_output(runtmp):
    # test basic genome - output csv
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csv_base = "out"
    csvout = runtmp.output(csv_base + ".lineage.csv")
    outdir = os.path.dirname(csvout)
    print("csvout: ", csvout)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--rank",
        "species",
        "-o",
        csv_base,
        "--containment-threshold",
        "0",
        "--output-format",
        "lineage_csv",
        "--output-dir",
        outdir,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert f"saving 'lineage_csv' output to '{csvout}'" in runtmp.last_result.err
    assert c.last_result.status == 0
    with open(csvout) as fp:
        outp = fp.readlines()

    assert len(outp) == 2
    outp = [x.strip() for x in outp]

    assert outp[0] == "ident,superkingdom,phylum,class,order,family,genus,species"
    assert (
        outp[1]
        == "test1,d__Bacteria,p__Bacteroidota,c__Bacteroidia,o__Bacteroidales,f__Bacteroidaceae,g__Prevotella,s__Prevotella copri"
    )


def test_genome_gather_from_file_rank(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")
    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")

    c.run_sourmash(
        "tax",
        "genome",
        "--from-file",
        g_from_file,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_gather_two_files(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make test2 results (identical to test1 except query_name and filename)
    g_res2 = runtmp.output("test2.gather.csv")
    test2_results = [
        x.replace("test1", "test2") + "\n" for x in Path(g_res).read_text().splitlines()
    ]
    with open(g_res2, "w") as fp:
        fp.writelines(test2_results)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res,
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert (
        "test2,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test2.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_gather_two_files_empty_force(runtmp):
    # make test2 results (identical to test1 except query_name and filename)
    # add an empty file too, with --force -> should work
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    g_empty_csv = runtmp.output("g_empty.csv")
    with open(g_empty_csv, "w") as fp:
        fp.write("")
    print("g_csv: ", g_empty_csv)

    g_res2 = runtmp.output("test2.gather.csv")
    test2_results = [
        x.replace("test1", "test2") + "\n" for x in Path(g_res).read_text().splitlines()
    ]
    with open(g_res2, "w") as fp:
        fp.writelines(test2_results)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res,
        g_res2,
        "-g",
        g_empty_csv,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
        "--force",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert (
        "test2,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test2.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_gather_two_files_one_classif_fail(runtmp):
    # if one query cant be classified still get classif for second
    # no --force = fail but still write file
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make test2 results (identical to test1 except query_name and filename)
    g_res2 = runtmp.output("test2.gather.csv")
    test2_results = [
        x.replace("test1", "test2") + "\n" for x in Path(g_res).read_text().splitlines()
    ]
    test2_results[1] = test2_results[1].replace(
        "0.08815317112086159", "1.1"
    )  # make test2 f_unique_to_query sum to >1
    for line in test2_results:
        print(line)
    with open(g_res2, "w") as fp:
        fp.writelines(test2_results)

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_res,
            g_res2,
            "--taxonomy-csv",
            taxonomy_csv,
            "--rank",
            "species",
            "--containment-threshold",
            "0",
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert "test2" not in c.last_result.out
    assert (
        "ERROR: Summarized fraction is > 100% of the query! This should not be possible. Please check that your input files come directly from a single gather run per query."
        in c.last_result.err
    )


def test_genome_gather_two_files_one_classif(runtmp):
    # if one query cant be classified, still get classif for second
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # make test2 results (identical to test1 except query_name and filename)
    g_res2 = runtmp.output("test2.gather.csv")
    test2_results = [
        x.replace("test1", "test2") + "\n" for x in Path(g_res).read_text().splitlines()
    ]
    test2_results[1] = test2_results[1].replace(
        "0.08815317112086159", "1.1"
    )  # make test2 f_unique_to_query sum to >1
    for line in test2_results:
        print(line)
    with open(g_res2, "w") as fp:
        fp.writelines(test2_results)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res,
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
        "--force",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert "test2" not in c.last_result.out
    assert (
        "ERROR: Summarized fraction is > 100% of the query! This should not be possible. Please check that your input files come directly from a single gather run per query."
        in c.last_result.err
    )


def test_genome_gather_duplicate_filename(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax",
        "genome",
        "--gather-csv",
        g_res,
        "-g",
        g_res,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert f"ignoring duplicated reference to file: {g_res}"
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_gather_from_file_duplicate_filename(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")
    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")
        f_csv.write(f"{g_res}\n")

    c.run_sourmash(
        "tax",
        "genome",
        "--from-file",
        g_from_file,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert f"ignoring duplicated reference to file: {g_res}"
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_gather_from_file_duplicate_query(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # different filename, contents identical to test1
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        fp.write(Path(g_res).read_text())

    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")
        f_csv.write(f"{g_res2}\n")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "--from-file",
            g_from_file,
            "--taxonomy-csv",
            taxonomy_csv,
            "--rank",
            "species",
            "--containment-threshold",
            "0",
        )
    assert c.last_result.status == -1
    print(str(exc.value))
    assert (
        "Gather query test1 was found in more than one CSV. Cannot load from "
        in str(exc.value)
    )


def test_genome_gather_from_file_duplicate_query_force(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")

    # different filename, contents identical to test1
    g_res2 = runtmp.output("test2.gather.csv")
    with open(g_res2, "w") as fp:
        fp.write(Path(g_res).read_text())

    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")
        f_csv.write(f"{g_res2}\n")

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "genome",
            "--from-file",
            g_from_file,
            "--taxonomy-csv",
            taxonomy_csv,
            "--rank",
            "species",
            "--containment-threshold",
            "0",
            "--force",
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1

    assert "Gather query test1 was found in more than one CSV." in c.last_result.err
    assert "Cannot force past duplicated gather query. Exiting." in c.last_result.err


def test_genome_gather_cli_and_from_file(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")
    g_from_file = runtmp.output("tmp-from-file.txt")

    # make test2 results (identical to test1 except query_name)
    g_res2 = runtmp.output("test2.gather.csv")
    test2_results = [
        x.replace("test1", "test2") + "\n" for x in Path(g_res).read_text().splitlines()
    ]
    with open(g_res2, "w") as fp:
        fp.writelines(test2_results)

    # write test2 csv to a text file for input
    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res2}\n")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res,
        "--from-file",
        g_from_file,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )
    assert (
        "test2,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test2.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_gather_cli_and_from_file_duplicate_filename(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")
    g_from_file = runtmp.output("tmp-from-file.txt")

    # also write test1 csv to a text file for input
    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res,
        "--from-file",
        g_from_file,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert f"ignoring duplicated reference to file: {g_res}" in c.last_result.err
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_gather_from_file_below_threshold(runtmp):
    # What do we want the results from this to be? I think I initially thought we shouldn't report anything,
    # but wouldn't a "below_threshold" + superkingdom result (here, 0.204) be helpful information?
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/test1.gather.csv")
    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")

    c.run_sourmash(
        "tax",
        "genome",
        "--from-file",
        g_from_file,
        "--taxonomy-csv",
        taxonomy_csv,
        "--containment-threshold",
        "1",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert "query_name,status,rank,fraction,lineage" in c.last_result.out
    assert "test1,below_threshold,superkingdom,0.204," in c.last_result.out


def test_genome_gather_two_queries(runtmp):
    """
    This checks for initial bug where classification
    would only happen for one genome per rank when
    doing --containment-threshold classification
    """
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    g_res = utils.get_test_data("tax/47+63_x_gtdb-rs202.gather.csv")

    # split 47+63 into two fake queries: q47, q63
    g_res2 = runtmp.output("two-queries.gather.csv")
    q2_results = [x + "\n" for x in Path(g_res).read_text().splitlines()]
    # rename queries
    q2_results[1] = q2_results[1].replace("47+63", "q47")
    q2_results[2] = q2_results[2].replace("47+63", "q63")
    with open(g_res2, "w") as fp:
        for line in q2_results:
            print(line)
            fp.write(line)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "--containment-threshold",
        "0",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert "query_name,status,rank,fraction,lineage" in c.last_result.out
    assert (
        "q63,match,species,0.336,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Shewanellaceae;g__Shewanella;s__Shewanella baltica,491c0a81,"
        in c.last_result.out
    )
    assert (
        "q47,match,species,0.664,d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Shewanellaceae;g__Shewanella;s__Shewanella baltica,"
        in c.last_result.out
    )


def test_genome_gather_ictv(runtmp):
    """
    test genome classification with ictv taxonomy
    """
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.ictv-taxonomy.csv")
    g_res = utils.get_test_data("tax/47+63_x_gtdb-rs202.gather.csv")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res,
        "--taxonomy-csv",
        taxonomy_csv,
        "--containment-threshold",
        "0",
        "--ictv",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert "query_name,status,rank,fraction,lineage" in c.last_result.out
    assert (
        "47+63,match,name,0.664,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus sudanense;Sudan virus,491c0a81,,0.664,5238000,0.987"
        in c.last_result.out
    )


def test_genome_gather_ictv_twoqueries(runtmp):
    """
    test genome classification with ictv taxonomy
    """
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.ictv-taxonomy.csv")
    g_res = utils.get_test_data("tax/47+63_x_gtdb-rs202.gather.csv")

    # split 47+63 into two fake queries: q47, q63
    g_res2 = runtmp.output("two-queries.gather.csv")
    q2_results = [x + "\n" for x in Path(g_res).read_text().splitlines()]
    # rename queries
    q2_results[1] = q2_results[1].replace("47+63", "q47")
    q2_results[2] = q2_results[2].replace("47+63", "q63")
    with open(g_res2, "w") as fp:
        for line in q2_results:
            print(line)
            fp.write(line)

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_res2,
        "--taxonomy-csv",
        taxonomy_csv,
        "--containment-threshold",
        "0",
        "--ictv",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    print(c.last_result.out)
    assert "query_name,status,rank,fraction,lineage" in c.last_result.out
    assert (
        "q47,match,name,0.664,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus sudanense;Sudan virus,491c0a81,,0.664,5238000,0.987"
        in c.last_result.out
    )
    assert (
        "q63,match,name,0.336,Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus zairense;Ebola virus,491c0a81,,0.336,2648000,0.965"
        in c.last_result.out
    )


def test_genome_gather_ictv_fail(runtmp):
    """
    test genome classification with ictv taxonomy
    """
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.ictv-taxonomy.csv")
    tax2_csv = runtmp.output("ictv-taxfail")
    # copy taxonomy csv to new file, but remove one of the columns
    with open(taxonomy_csv) as inF:
        with open(tax2_csv, "w") as outF:
            for line in inF.readlines():
                line = line.rsplit(",", 1)[0]
                outF.write(f"{line}\n")

    g_res = utils.get_test_data("tax/47+63_x_gtdb-rs202.gather.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_res,
            "--taxonomy-csv",
            tax2_csv,
            "--containment-threshold",
            "0",
            "--ictv",
        )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status != 0
    print(c.last_result.out)
    assert "Not all taxonomy ranks present" in str(exc.value)


def test_genome_rank_duplicated_taxonomy_fail(runtmp):
    c = runtmp
    # write temp taxonomy with duplicates
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    duplicated_csv = runtmp.output("duplicated_taxonomy.csv")
    with open(duplicated_csv, "w") as dup:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax.append(tax[1] + "FOO")  # add first tax_assign again
        dup.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            duplicated_csv,
            "--rank",
            "species",
        )
    assert "cannot read taxonomy assignments" in str(exc.value)
    assert "multiple lineages for identifier GCF_001881345" in str(exc.value)


def test_genome_rank_duplicated_taxonomy_fail_lineages(runtmp):
    # write temp taxonomy with duplicates => lineages-style file
    c = runtmp

    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    taxdb = tax_utils.LineageDB.load(taxonomy_csv)

    for k, v in taxdb.items():
        print(k, v)

    lineage_csv = runtmp.output("lin.csv")
    with open(lineage_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "lineage"])
        for k, v in taxdb.items():
            linstr = lca_utils.display_lineage(v)
            w.writerow([k, linstr])

            # duplicate each row, changing something (truncate species, here)
            v = v[:-1]
            linstr = lca_utils.display_lineage(v)
            w.writerow([k, linstr])

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash("tax", "summarize", lineage_csv)
        print(c.last_result.out)
        print(c.last_result.err)

    assert "cannot read taxonomy assignments" in str(exc.value)
    assert "multiple lineages for identifier GCF_001881345" in str(exc.value)


def test_genome_rank_duplicated_taxonomy_force(runtmp):
    # write temp taxonomy with duplicates
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    duplicated_csv = runtmp.output("duplicated_taxonomy.csv")
    with open(duplicated_csv, "w") as dup:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax.append(tax[1])  # add first tax_assign again
        dup.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        duplicated_csv,
        "--rank",
        "species",
        "--force",
        "--containment-threshold",
        "0",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_missing_taxonomy_ignore_threshold(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax = [tax[0]] + tax[2:]  # remove the best match (1st tax entry)
        subset.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        subset_csv,
        "--containment-threshold",
        "0",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "The following are missing from the taxonomy information: GCF_001881345"
        in c.last_result.err
    )
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_missing_taxonomy_recover_with_second_tax_file(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax = [tax[0]] + tax[2:]  # remove the best match (1st tax entry)
        subset.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        subset_csv,
        "-t",
        taxonomy_csv,
        "--containment-threshold",
        "0",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "The following are missing from the taxonomy information: GCF_001881345"
        not in c.last_result.err
    )
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_missing_taxonomy_ignore_rank(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax = [tax[0]] + tax[2:]  # remove the best match (1st tax entry)
        subset.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    c.run_sourmash(
        "tax", "genome", "-g", g_csv, "--taxonomy-csv", subset_csv, "--rank", "species"
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "The following are missing from the taxonomy information: GCF_001881345"
        in c.last_result.err
    )
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,below_threshold,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_multiple_taxonomy_files(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax = [tax[0]] + tax[2:]  # remove the best match (1st tax entry)
        subset.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    # using mult -t args
    c.run_sourmash(
        "tax", "genome", "-g", g_csv, "--taxonomy-csv", subset_csv, "-t", taxonomy_csv
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "The following are missing from the taxonomy information: GCF_001881345"
        not in c.last_result.err
    )
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,family,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test1.sig,0.073,582000,"
        in c.last_result.out
    )
    # using single -t arg
    c.run_sourmash(
        "tax", "genome", "-g", g_csv, "--taxonomy-csv", subset_csv, taxonomy_csv
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "The following are missing from the taxonomy information: GCF_001881345"
        not in c.last_result.err
    )
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,family,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test1.sig,0.073,582000,"
        in c.last_result.out
    )


def test_genome_multiple_taxonomy_files_empty_force(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry, as well as an empty file,
    # and use force
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax = [tax[0]] + tax[2:]  # remove the best match (1st tax entry)
        subset.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    empty_tax = runtmp.output("tax_empty.txt")
    with open(empty_tax, "w") as fp:
        fp.write("")

    # using mult -t args
    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        subset_csv,
        "-t",
        taxonomy_csv,
        "-t",
        empty_tax,
        "--force",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "The following are missing from the taxonomy information: GCF_001881345"
        not in c.last_result.err
    )
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,family,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test1.sig,0.073,582000,"
        in c.last_result.out
    )


def test_genome_missing_taxonomy_fail_threshold(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax = [tax[0]] + tax[2:]  # remove the best match (1st tax entry)
        subset.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            subset_csv,
            "--fail-on-missing-taxonomy",
            "--containment-threshold",
            "0",
        )

    print(str(exc.value))
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert "ident 'GCF_001881345' is not in the taxonomy database." in str(exc.value)
    assert "Failing, as requested via --fail-on-missing-taxonomy" in str(exc.value)
    assert c.last_result.status == -1


def test_genome_missing_taxonomy_fail_rank(runtmp):
    c = runtmp
    # write temp taxonomy with missing entry
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")
    subset_csv = runtmp.output("subset_taxonomy.csv")
    with open(subset_csv, "w") as subset:
        tax = [x.rstrip() for x in Path(taxonomy_csv).read_text().splitlines()]
        tax = [tax[0]] + tax[2:]  # remove the best match (1st tax entry)
        subset.write("\n".join(tax))

    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            subset_csv,
            "--fail-on-missing-taxonomy",
            "--rank",
            "species",
        )

    print(str(exc.value))
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert "ident 'GCF_001881345' is not in the taxonomy database." in str(exc.value)
    assert "Failing, as requested via --fail-on-missing-taxonomy" in str(exc.value)
    assert c.last_result.status == -1


def test_genome_rank_not_available(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--rank",
            "strain",
            "--containment-threshold",
            "0",
        )

    print(str(exc.value))
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1
    assert (
        "No taxonomic information provided for rank strain: cannot classify at this rank"
        in str(exc.value)
    )


def test_genome_empty_gather_results_with_header_single(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    gather_results = [x for x in Path(g_csv).read_text().splitlines()]
    empty_gather_with_header = runtmp.output("g_header.csv")
    # write temp empty gather results (header only)
    with open(empty_gather_with_header, "w") as fp:
        fp.write(gather_results[0])

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            empty_gather_with_header,
            "--taxonomy-csv",
            taxonomy_csv,
        )

    print(str(exc.value))
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1
    assert f"No gather results loaded from {empty_gather_with_header}." in str(
        exc.value
    )
    assert "Exiting." in str(exc.value)


def test_genome_empty_gather_results_single(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")

    # write temp empty gather results
    empty_tax = runtmp.output("tax_header.csv")
    with open(empty_tax, "w") as fp:
        fp.write("")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash("tax", "genome", "-g", empty_tax, "--taxonomy-csv", taxonomy_csv)

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1
    assert f"Cannot read gather results from '{empty_tax}'. Is file empty?" in str(
        exc.value
    )
    assert "Exiting." in c.last_result.err


def test_genome_empty_gather_results_single_force(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")

    # write temp empty gather results (header only)
    empty_tax = runtmp.output("tax_header.csv")
    with open(empty_tax, "w") as fp:
        fp.write("")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax", "genome", "-g", empty_tax, "--taxonomy-csv", taxonomy_csv, "--force"
        )

    print(str(exc.value))
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1
    assert (
        "--force is set. Attempting to continue to next set of gather results."
        in str(exc.value)
    )
    assert "No results for classification. Exiting." in str(exc.value)


def test_genome_empty_gather_results_with_empty_csv_force(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")

    # write temp empty gather results
    empty_tax = runtmp.output("tax_empty.txt")
    with open(empty_tax, "w") as fp:
        fp.write("")

    g_from_file = runtmp.output("tmp-from-csv.csv")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{empty_tax}\n")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            empty_tax,
            "--from-file",
            g_from_file,
            "--taxonomy-csv",
            taxonomy_csv,
            "--rank",
            "species",
            "--force",
        )

    print(str(exc.value))
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == -1
    assert (
        "--force is set. Attempting to continue to next set of gather results."
        in str(exc.value)
    )
    assert "No results for classification. Exiting." in str(exc.value)


def test_genome_empty_gather_results_with_csv_force(runtmp):
    c = runtmp
    taxonomy_csv = utils.get_test_data("tax/test.taxonomy.csv")

    g_res = utils.get_test_data("tax/test1.gather.csv")
    g_from_file = runtmp.output("tmp-from-file.txt")
    with open(g_from_file, "w") as f_csv:
        f_csv.write(f"{g_res}\n")

    # write temp empty gather results
    empty_tax = runtmp.output("tax_empty.csv")
    with open(empty_tax, "w") as fp:
        fp.write("")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        empty_tax,
        "--from-file",
        g_from_file,
        "--taxonomy-csv",
        taxonomy_csv,
        "--rank",
        "species",
        "--containment-threshold",
        "0",
        "--force",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "--force is set. Attempting to continue to next set of gather results."
        in c.last_result.err
    )
    assert "loaded results for 1 queries from 1 gather CSVs" in c.last_result.err
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )


def test_genome_containment_threshold_bounds(runtmp):
    c = runtmp
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    below_threshold = "-1"

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            tax,
            "--taxonomy-csv",
            tax,
            "--containment-threshold",
            below_threshold,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    assert "ERROR: Argument must be >0 and <1" in str(exc.value)

    above_threshold = "1.1"
    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--containment-threshold",
            above_threshold,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    assert "ERROR: Argument must be >0 and <1" in str(exc.value)


def test_genome_containment_threshold_type(runtmp):
    c = runtmp
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    not_a_float = "str"

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--containment-threshold",
            not_a_float,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    assert "ERROR: Must be a floating point number" in str(exc.value)


def test_genome_over100percent_error(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    perfect_g_csv = runtmp.output("g.csv")

    # create an impossible gather result
    with open(g_csv) as fp:
        r = csv.DictReader(fp, delimiter=",")
        header = r.fieldnames
        print(header)
        with open(perfect_g_csv, "w") as out_fp:
            w = csv.DictWriter(out_fp, header)
            w.writeheader()
            for n, row in enumerate(r):
                if n == 0:
                    row["f_unique_to_query"] = 1.1
                w.writerow(row)
                print(row)

    with pytest.raises(SourmashCommandFailed):
        runtmp.run_sourmash("tax", "genome", "-g", perfect_g_csv, "--taxonomy-csv", tax)

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == -1
    assert (
        "fraction is > 100% of the query! This should not be possible."
        in runtmp.last_result.err
    )


def test_genome_ani_threshold_input_errors(runtmp):
    c = runtmp
    g_csv = utils.get_test_data("tax/test1.gather_old.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    below_threshold = "-1"

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            tax,
            "--taxonomy-csv",
            tax,
            "--ani-threshold",
            below_threshold,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    assert "ERROR: Argument must be >0 and <1" in str(exc.value)

    above_threshold = "1.1"
    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--ani-threshold",
            above_threshold,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    assert "ERROR: Argument must be >0 and <1" in str(exc.value)

    not_a_float = "str"

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--ani-threshold",
            not_a_float,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    assert "ERROR: Must be a floating point number" in str(exc.value)


def test_genome_ani_threshold(runtmp):
    c = runtmp
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    c.run_sourmash(
        "tax", "genome", "-g", g_csv, "--taxonomy-csv", tax, "--ani-threshold", "0.93"
    )  # note: I think this was previously a bug, if 0.95 produced the result below...

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,family,0.116,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae,md5,test1.sig,0.073,582000,0.93"
        in c.last_result.out
    )

    # more lax threshold
    c.run_sourmash(
        "tax", "genome", "-g", g_csv, "--taxonomy-csv", tax, "--ani-threshold", "0.9"
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "test1,match,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000"
        in c.last_result.out
    )

    # too stringent of threshold (using rank)
    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--ani-threshold",
        "1.0",
        "--rank",
        "species",
    )
    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    assert (
        "test1,below_threshold,species,0.089,d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri,md5,test1.sig,0.057,444000,0.92"
        in c.last_result.out
    )


def test_genome_ani_oldgather(runtmp):
    # now fail if using gather <4.4
    c = runtmp
    g_csv = utils.get_test_data("tax/test1.gather_old.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    with pytest.raises(SourmashCommandFailed) as exc:
        c.run_sourmash("tax", "genome", "-g", g_csv, "--taxonomy-csv", tax)
    assert (
        "is missing columns needed for taxonomic summarization. Please run gather with sourmash >= 4.4."
        in str(exc.value)
    )
    assert c.last_result.status == -1


def test_genome_ani_lemonade_classify(runtmp):
    # test a complete MAG classification with lemonade MAG from STAMPS 2022
    # (real data!)
    c = runtmp

    ## first run gather
    genome = utils.get_test_data("tax/lemonade-MAG3.sig.gz")
    matches = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.zip")

    c.run_sourmash("gather", genome, matches, "--threshold-bp=5000", "-o", "gather.csv")

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0

    this_gather_file = c.output("gather.csv")
    this_gather = Path(this_gather_file).read_text().splitlines()

    assert len(this_gather) == 4

    ## now run 'tax genome' with human output
    taxonomy_file = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.tax.csv")
    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        this_gather_file,
        "-t",
        taxonomy_file,
        "--ani",
        "0.8",
        "-F",
        "human",
    )

    output = c.last_result.out
    assert (
        "MAG3_1            match     5.3%     91.0%  d__Bacteria;p__Bacteroidota;c__Chlorobia;o__Chlorobiales;f__Chlorobiaceae;g__Prosthecochloris;s__Prosthecochloris vibrioformis"
        in output
    )

    # aaand classify to lineage_csv
    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        this_gather_file,
        "-t",
        taxonomy_file,
        "--ani",
        "0.8",
        "-F",
        "lineage_csv",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    output = c.last_result.out
    assert "ident,superkingdom,phylum,class,order,family,genus,species" in output
    assert (
        "MAG3_1,d__Bacteria,p__Bacteroidota,c__Chlorobia,o__Chlorobiales,f__Chlorobiaceae,g__Prosthecochloris,s__Prosthecochloris vibrioformis"
        in output
    )


def test_genome_ani_lemonade_classify_estimate_ani_ci(runtmp):
    # test a complete MAG classification with lemonade MAG from STAMPS 2022
    # (real data!)
    c = runtmp

    ## first run gather
    genome = utils.get_test_data("tax/lemonade-MAG3.sig.gz")
    matches = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.zip")

    c.run_sourmash(
        "gather",
        genome,
        matches,
        "--threshold-bp=5000",
        "-o",
        "gather.csv",
        "--estimate-ani",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0

    this_gather_file = c.output("gather.csv")
    this_gather = Path(this_gather_file).read_text().splitlines()

    assert len(this_gather) == 4

    ## now run 'tax genome' with human output
    taxonomy_file = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.tax.csv")
    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        this_gather_file,
        "-t",
        taxonomy_file,
        "--ani",
        "0.8",
        "-F",
        "human",
    )

    output = c.last_result.out
    assert (
        "MAG3_1            match     5.3%     91.0%  d__Bacteria;p__Bacteroidota;c__Chlorobia;o__Chlorobiales;f__Chlorobiaceae;g__Prosthecochloris;s__Prosthecochloris vibrioformis"
        in output
    )

    # aaand classify to lineage_csv
    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        this_gather_file,
        "-t",
        taxonomy_file,
        "--ani",
        "0.8",
        "-F",
        "lineage_csv",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)
    output = c.last_result.out
    assert "ident,superkingdom,phylum,class,order,family,genus,species" in output
    assert (
        "MAG3_1,d__Bacteria,p__Bacteroidota,c__Chlorobia,o__Chlorobiales,f__Chlorobiaceae,g__Prosthecochloris,s__Prosthecochloris vibrioformis"
        in output
    )


def test_metagenome_no_gather_csv(runtmp):
    # test tax metagenome with no -g
    taxonomy_file = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.tax.csv")
    with pytest.raises(SourmashCommandFailed):
        runtmp.run_sourmash("tax", "metagenome", "-t", taxonomy_file)

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)


def test_genome_no_gather_csv(runtmp):
    # test tax genome with no -g
    taxonomy_file = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.tax.csv")
    with pytest.raises(SourmashCommandFailed):
        runtmp.run_sourmash("tax", "genome", "-t", taxonomy_file)

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)


def test_annotate_no_gather_csv(runtmp):
    # test tax annotate with no -g
    taxonomy_file = utils.get_test_data("tax/lemonade-MAG3.x.gtdb.matches.tax.csv")
    with pytest.raises(SourmashCommandFailed):
        runtmp.run_sourmash("tax", "annotate", "-t", taxonomy_file)

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)


def test_genome_LIN(runtmp):
    # test basic genome with LIN taxonomy
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "--ani-threshold",
        "0.93",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank,query_ani_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,below_threshold,0,0.089,1,md5,test1.sig,0.057,444000,0.925"
        in c.last_result.out
    )

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "--ani-threshold",
        "0.924",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank,query_ani_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,19,0.088,0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,md5,test1.sig,0.058,442000,0.925"
        in c.last_result.out
    )

    c.run_sourmash(
        "tax", "genome", "-g", g_csv, "--taxonomy-csv", tax, "--lins", "--rank", "4"
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank,query_ani_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,below_threshold,4,0.088,0;0;0;0;0,md5,test1.sig,0.058,442000,0.925"
        in c.last_result.out
    )


def test_genome_LIN_lingroups(runtmp):
    # test basic genome with LIN taxonomy
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    lg_file = runtmp.output("test.lg.csv")

    with open(lg_file, "w") as out:
        out.write("lin,name\n")
        out.write("0;0;0,lg1\n")
        out.write("1;0;0,lg2\n")
        out.write("2;0;0,lg3\n")
        out.write("1;0;1,lg3\n")
        # write a 19 so we can check the end
        out.write("0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,lg4\n")

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "--lingroup",
        lg_file,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank,query_ani_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,below_threshold,2,0.088,0;0;0,md5,test1.sig,0.058,442000,0.925"
        in c.last_result.out
    )

    c.run_sourmash(
        "tax",
        "genome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "--lingroup",
        lg_file,
        "--ani-threshold",
        "0.924",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,status,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank,query_ani_at_rank"
        in c.last_result.out
    )
    assert (
        "test1,match,19,0.088,0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,md5,test1.sig,0.058,442000,0.925"
        in c.last_result.out
    )


def test_annotate_0(runtmp):
    # test annotate basics
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csvout = runtmp.output("test1.gather.with-lineages.csv")
    out_dir = os.path.dirname(csvout)

    c.run_sourmash(
        "tax", "annotate", "--gather-csv", g_csv, "--taxonomy-csv", tax, "-o", out_dir
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert os.path.exists(csvout)

    lin_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    print("\n".join(lin_gather_results))
    assert f"saving 'annotate' output to '{csvout}'" in runtmp.last_result.err

    assert "lineage" in lin_gather_results[0]
    assert (
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
        in lin_gather_results[1]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
        in lin_gather_results[2]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus"
        in lin_gather_results[3]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
        in lin_gather_results[4]
    )


def test_annotate_gzipped_gather(runtmp):
    # test annotate basics
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    # rewrite gather_csv as gzipped csv
    gz_gather = runtmp.output("test1.gather.csv.gz")
    with open(g_csv, "rb") as f_in, gzip.open(gz_gather, "wb") as f_out:
        f_out.writelines(f_in)

    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csvout = runtmp.output("test1.gather.with-lineages.csv")
    out_dir = os.path.dirname(csvout)

    c.run_sourmash(
        "tax",
        "annotate",
        "--gather-csv",
        gz_gather,
        "--taxonomy-csv",
        tax,
        "-o",
        out_dir,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert os.path.exists(csvout)

    lin_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    print("\n".join(lin_gather_results))
    assert f"saving 'annotate' output to '{csvout}'" in runtmp.last_result.err

    assert "lineage" in lin_gather_results[0]
    assert (
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
        in lin_gather_results[1]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
        in lin_gather_results[2]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus"
        in lin_gather_results[3]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
        in lin_gather_results[4]
    )


def test_annotate_0_ictv(runtmp):
    # test annotate basics
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.ictv-taxonomy.csv")
    csvout = runtmp.output("test1.gather.with-lineages.csv")
    out_dir = os.path.dirname(csvout)

    c.run_sourmash(
        "tax",
        "annotate",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        out_dir,
        "--ictv",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert os.path.exists(csvout)

    gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    print("\n".join(gather_results))
    assert f"saving 'annotate' output to '{csvout}'" in runtmp.last_result.err

    assert "lineage" in gather_results[0]
    assert (
        "Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus bundibugyoense;Bundibugyo virus"
        in gather_results[1]
    )
    assert (
        "Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus taiense;Taï Forest virus"
        in gather_results[2]
    )
    assert (
        "Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus bombaliense;Bombali virus"
        in gather_results[3]
    )
    assert (
        "Riboviria;;Orthornavirae;;Negarnaviricota;Haploviricotina;Monjiviricetes;;Mononegavirales;;Filoviridae;;Orthoebolavirus;;Orthoebolavirus restonense;Reston virus"
        in gather_results[4]
    )


def test_annotate_0_LIN(runtmp):
    # test annotate basics
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")
    csvout = runtmp.output("test1.gather.with-lineages.csv")
    out_dir = os.path.dirname(csvout)

    c.run_sourmash(
        "tax",
        "annotate",
        "--gather-csv",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        out_dir,
        "--lins",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert os.path.exists(csvout)

    lin_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    print("\n".join(lin_gather_results))
    assert f"saving 'annotate' output to '{csvout}'" in runtmp.last_result.err

    assert "lineage" in lin_gather_results[0]
    assert "0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0" in lin_gather_results[1]
    assert "1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0" in lin_gather_results[2]
    assert "2;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0" in lin_gather_results[3]
    assert "1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0" in lin_gather_results[4]


def test_annotate_gather_argparse(runtmp):
    # test annotate with two gather CSVs, second one empty, and --force.
    # this tests argparse handling w/extend.
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csvout = runtmp.output("test1.gather.with-lineages.csv")
    out_dir = os.path.dirname(csvout)

    g_empty_csv = runtmp.output("g_empty.csv")
    with open(g_empty_csv, "w") as fp:
        fp.write("")
    print("g_csv: ", g_empty_csv)

    c.run_sourmash(
        "tax",
        "annotate",
        "--gather-csv",
        g_csv,
        "-g",
        g_empty_csv,
        "--taxonomy-csv",
        tax,
        "-o",
        out_dir,
        "--force",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert os.path.exists(csvout)

    lin_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    print("\n".join(lin_gather_results))
    assert f"saving 'annotate' output to '{csvout}'" in runtmp.last_result.err

    assert "lineage" in lin_gather_results[0]
    assert (
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
        in lin_gather_results[1]
    )


def test_annotate_0_db(runtmp):
    # test annotate with sqlite db
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.db")
    csvout = runtmp.output("test1.gather.with-lineages.csv")
    out_dir = os.path.dirname(csvout)

    c.run_sourmash(
        "tax", "annotate", "--gather-csv", g_csv, "--taxonomy-csv", tax, "-o", out_dir
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0

    lin_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    print("\n".join(lin_gather_results))
    assert f"saving 'annotate' output to '{csvout}'" in runtmp.last_result.err

    assert "lineage" in lin_gather_results[0]
    assert (
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
        in lin_gather_results[1]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
        in lin_gather_results[2]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola;s__Phocaeicola vulgatus"
        in lin_gather_results[3]
    )
    assert (
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella;s__Prevotella copri"
        in lin_gather_results[4]
    )


def test_annotate_empty_gather_results(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")

    # creates empty gather result
    g_csv = runtmp.output("g.csv")
    with open(g_csv, "w") as fp:
        fp.write("")
    print("g_csv: ", g_csv)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "annotate", "-g", g_csv, "--taxonomy-csv", tax)

    assert f"Cannot read from '{g_csv}'. Is file empty?" in str(exc.value)
    assert runtmp.last_result.status == -1


def test_annotate_prefetch_or_other_header(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    alt_csv = runtmp.output("g.csv")
    for alt_col in ["match_name", "ident", "accession"]:
        # modify 'name' to other acceptable id_columns result
        alt_g = [
            x.replace("name", alt_col) + "\n"
            for x in Path(g_csv).read_text().splitlines()
        ]
        with open(alt_csv, "w") as fp:
            fp.writelines(alt_g)

        runtmp.run_sourmash("tax", "annotate", "-g", alt_csv, "--taxonomy-csv", tax)

        assert runtmp.last_result.status == 0
        print(runtmp.last_result.out)
        print(runtmp.last_result.err)
        assert (
            f"Starting annotation on '{alt_csv}'. Using ID column: '{alt_col}'"
            in runtmp.last_result.err
        )
        assert f"Annotated 4 of 4 total rows from '{alt_csv}'" in runtmp.last_result.err


def test_annotate_bad_header(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    bad_g_csv = runtmp.output("g.csv")

    # creates bad gather result
    bad_g = [
        x.replace("name", "nope") + "\n" for x in Path(g_csv).read_text().splitlines()
    ]
    with open(bad_g_csv, "w") as fp:
        fp.writelines(bad_g)
    # print("bad_gather_results: \n", bad_g)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "annotate", "-g", bad_g_csv, "--taxonomy-csv", tax)

    assert (
        f"ERROR: Cannot find taxonomic identifier column in '{bad_g_csv}'. Tried: name, match_name, ident, accession"
        in str(exc.value)
    )
    assert runtmp.last_result.status == -1
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)


def test_annotate_no_tax_matches(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    bad_g_csv = runtmp.output("g.csv")

    # mess up tax idents
    bad_g = [
        x.replace("GCF_", "GGG_") + "\n" for x in Path(g_csv).read_text().splitlines()
    ]
    with open(bad_g_csv, "w") as fp:
        fp.writelines(bad_g)
    # print("bad_gather_results: \n", bad_g)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "annotate", "-g", bad_g_csv, "--taxonomy-csv", tax)

    assert f"ERROR: Could not annotate any rows from '{bad_g_csv}'" in str(exc.value)
    assert runtmp.last_result.status == -1
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    runtmp.run_sourmash(
        "tax", "annotate", "-g", bad_g_csv, "--taxonomy-csv", tax, "--force"
    )

    assert runtmp.last_result.status == 0
    assert f"Could not annotate any rows from '{bad_g_csv}'" in runtmp.last_result.err
    assert (
        "--force is set. Attempting to continue to next file." in runtmp.last_result.err
    )
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)


def test_annotate_missed_tax_matches(runtmp):
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    bad_g_csv = runtmp.output("g.csv")

    with open(g_csv) as gather_lines, open(bad_g_csv, "w") as fp:
        for n, line in enumerate(gather_lines):
            if n > 2:
                # mess up tax idents of lines 3, 4
                line = line.replace("GCF_", "GGG_")
            fp.write(line)
    # print("bad_gather_results: \n", bad_g)

    runtmp.run_sourmash("tax", "annotate", "-g", bad_g_csv, "--taxonomy-csv", tax)

    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert f"Annotated 2 of 4 total rows from '{bad_g_csv}'." in runtmp.last_result.err


def test_annotate_empty_tax_lineage_input(runtmp):
    tax_empty = runtmp.output("t.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash("tax", "annotate", "-g", g_csv, "--taxonomy-csv", tax_empty)

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status != 0
    assert "cannot read taxonomy assignments from" in str(exc.value)


def test_annotate_empty_tax_lineage_input_recover_with_second_taxfile(runtmp):
    tax_empty = runtmp.output("t.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    runtmp.run_sourmash(
        "tax",
        "annotate",
        "-g",
        g_csv,
        "-t",
        tax_empty,
        "--taxonomy-csv",
        tax,
        "--force",
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0


def test_annotate_empty_tax_lineage_input_recover_with_second_taxfile_2(runtmp):
    # test with empty tax second, to check on argparse handling
    tax_empty = runtmp.output("t.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    g_csv = utils.get_test_data("tax/test1.gather.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    runtmp.run_sourmash(
        "tax",
        "annotate",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "-t",
        tax_empty,
        "--force",
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0


def test_tax_prepare_1_csv_to_csv(runtmp, keep_identifiers, keep_versions):
    # CSV -> CSV; same assignments
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    taxout = runtmp.output("out.csv")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_1_combine_csv(runtmp):
    # multiple CSVs to a single combined CSV
    tax1 = utils.get_test_data("tax/test.taxonomy.csv")
    tax2 = utils.get_test_data("tax/protozoa_genbank_lineage.csv")

    taxout = runtmp.output("out.csv")

    runtmp.sourmash("tax", "prepare", "-t", tax1, tax2, "-F", "csv", "-o", taxout)

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert not out
    assert "...loaded 8 entries" in err

    out = Path(taxout).read_text().splitlines()
    assert len(out) == 9


def test_tax_prepare_1_csv_to_csv_empty_ranks(runtmp, keep_identifiers, keep_versions):
    # CSV -> CSV; same assignments, even when trailing ranks are empty
    tax = utils.get_test_data("tax/test-empty-ranks.taxonomy.csv")
    taxout = runtmp.output("out.csv")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_1_csv_to_csv_empty_file(runtmp, keep_identifiers, keep_versions):
    # CSV -> CSV with an empty input file and --force
    # tests argparse extend
    tax = utils.get_test_data("tax/test-empty-ranks.taxonomy.csv")
    tax_empty = runtmp.output("t.csv")
    taxout = runtmp.output("out.csv")

    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args
            )
        return

    runtmp.run_sourmash(
        "tax",
        "prepare",
        "-t",
        tax,
        "-t",
        tax_empty,
        "-o",
        taxout,
        "-F",
        "csv",
        *args,
        "--force",
    )
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_1_csv_to_csv_empty_ranks_2(
    runtmp, keep_identifiers, keep_versions
):
    # CSV -> CSV; same assignments for situations with empty internal ranks
    tax = utils.get_test_data("tax/test-empty-ranks-2.taxonomy.csv")
    taxout = runtmp.output("out.csv")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_1_csv_to_csv_empty_ranks_3(
    runtmp, keep_identifiers, keep_versions
):
    # CSV -> CSV; same assignments for situations with empty internal ranks
    tax = utils.get_test_data("tax/test-empty-ranks-3.taxonomy.csv")
    taxout = runtmp.output("out.csv")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "csv", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_2_csv_to_sql(runtmp, keep_identifiers, keep_versions):
    # CSV -> SQL; same assignments?
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    taxout = runtmp.output("out.db")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )
    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)

    # cannot overwrite -
    with pytest.raises(SourmashCommandFailed) as exc:
        runtmp.run_sourmash(
            "tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args
        )
    assert "taxonomy table already exists" in str(exc.value)


def test_tax_prepare_2_csv_to_sql_empty_ranks(runtmp, keep_identifiers, keep_versions):
    # CSV -> SQL with some empty ranks in the taxonomy file
    tax = utils.get_test_data("tax/test-empty-ranks.taxonomy.csv")
    taxout = runtmp.output("out.db")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )
    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_3_db_to_csv(runtmp):
    # SQL -> CSV; same assignments
    taxcsv = utils.get_test_data("tax/test.taxonomy.csv")
    taxdb = utils.get_test_data("tax/test.taxonomy.db")
    taxout = runtmp.output("out.csv")

    runtmp.run_sourmash("tax", "prepare", "-t", taxdb, "-o", taxout, "-F", "csv")
    assert os.path.exists(taxout)
    with open(taxout) as fp:
        print(fp.read())

    db1 = tax_utils.MultiLineageDB.load(
        [taxcsv], keep_full_identifiers=False, keep_identifier_versions=False
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])
    db3 = tax_utils.MultiLineageDB.load(
        [taxdb], keep_full_identifiers=False, keep_identifier_versions=False
    )
    assert set(db1) == set(db2)
    assert set(db1) == set(db3)


def test_tax_prepare_3_db_to_csv_gz(runtmp):
    # SQL -> CSV; same assignments
    taxcsv = utils.get_test_data("tax/test.taxonomy.csv")
    taxdb = utils.get_test_data("tax/test.taxonomy.db")
    taxout = runtmp.output("out.csv.gz")

    runtmp.run_sourmash("tax", "prepare", "-t", taxdb, "-o", taxout, "-F", "csv")
    assert os.path.exists(taxout)
    with gzip.open(taxout, "rt") as fp:
        print(fp.read())

    db1 = tax_utils.MultiLineageDB.load(
        [taxcsv], keep_full_identifiers=False, keep_identifier_versions=False
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])
    db3 = tax_utils.MultiLineageDB.load(
        [taxdb], keep_full_identifiers=False, keep_identifier_versions=False
    )
    assert set(db1) == set(db2)
    assert set(db1) == set(db3)


def test_tax_prepare_2_csv_to_sql_empty_ranks_2(
    runtmp, keep_identifiers, keep_versions
):
    # CSV -> SQL with some empty internal ranks in the taxonomy file
    tax = utils.get_test_data("tax/test-empty-ranks-2.taxonomy.csv")
    taxout = runtmp.output("out.db")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )
    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_2_csv_to_sql_empty_ranks_3(
    runtmp, keep_identifiers, keep_versions
):
    # CSV -> SQL with some empty internal ranks in the taxonomy file
    tax = utils.get_test_data("tax/test-empty-ranks-3.taxonomy.csv")
    taxout = runtmp.output("out.db")

    args = []
    if keep_identifiers:
        args.append("--keep-full-identifiers")
    if keep_versions:
        args.append("--keep-identifier-versions")

    # this is an error - can't strip versions if not splitting identifiers
    if keep_identifiers and not keep_versions:
        with pytest.raises(SourmashCommandFailed):
            runtmp.run_sourmash(
                "tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args
            )
        return

    runtmp.run_sourmash("tax", "prepare", "-t", tax, "-o", taxout, "-F", "sql", *args)
    assert os.path.exists(taxout)

    db1 = tax_utils.MultiLineageDB.load(
        [tax],
        keep_full_identifiers=keep_identifiers,
        keep_identifier_versions=keep_versions,
    )
    db2 = tax_utils.MultiLineageDB.load([taxout])

    assert set(db1) == set(db2)


def test_tax_prepare_3_db_to_csv_empty_ranks(runtmp):
    # SQL -> CSV; same assignments, with empty ranks
    taxcsv = utils.get_test_data("tax/test-empty-ranks.taxonomy.csv")
    taxdb = utils.get_test_data("tax/test-empty-ranks.taxonomy.db")
    taxout = runtmp.output("out.csv")

    runtmp.run_sourmash("tax", "prepare", "-t", taxdb, "-o", taxout, "-F", "csv")
    assert os.path.exists(taxout)
    with open(taxout) as fp:
        print(fp.read())

    db1 = tax_utils.MultiLineageDB.load(
        [taxcsv], keep_full_identifiers=False, keep_identifier_versions=False
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])
    db3 = tax_utils.MultiLineageDB.load(
        [taxdb], keep_full_identifiers=False, keep_identifier_versions=False
    )
    assert set(db1) == set(db2)
    assert set(db1) == set(db3)


def test_tax_prepare_3_db_to_csv_empty_ranks_2(runtmp):
    # SQL -> CSV; same assignments, with empty ranks
    taxcsv = utils.get_test_data("tax/test-empty-ranks-2.taxonomy.csv")
    taxdb = utils.get_test_data("tax/test-empty-ranks-2.taxonomy.db")
    taxout = runtmp.output("out.csv")

    runtmp.run_sourmash("tax", "prepare", "-t", taxdb, "-o", taxout, "-F", "csv")
    assert os.path.exists(taxout)
    with open(taxout) as fp:
        print(fp.read())

    db1 = tax_utils.MultiLineageDB.load(
        [taxcsv], keep_full_identifiers=False, keep_identifier_versions=False
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])
    db3 = tax_utils.MultiLineageDB.load(
        [taxdb], keep_full_identifiers=False, keep_identifier_versions=False
    )
    assert set(db1) == set(db2)
    assert set(db1) == set(db3)


def test_tax_prepare_3_db_to_csv_empty_ranks_3(runtmp):
    # SQL -> CSV; same assignments, with empty ranks
    taxcsv = utils.get_test_data("tax/test-empty-ranks-3.taxonomy.csv")
    taxdb = utils.get_test_data("tax/test-empty-ranks-3.taxonomy.db")
    taxout = runtmp.output("out.csv")

    runtmp.run_sourmash("tax", "prepare", "-t", taxdb, "-o", taxout, "-F", "csv")
    assert os.path.exists(taxout)
    with open(taxout) as fp:
        print(fp.read())

    db1 = tax_utils.MultiLineageDB.load(
        [taxcsv], keep_full_identifiers=False, keep_identifier_versions=False
    )

    db2 = tax_utils.MultiLineageDB.load([taxout])
    db3 = tax_utils.MultiLineageDB.load(
        [taxdb], keep_full_identifiers=False, keep_identifier_versions=False
    )
    assert set(db1) == set(db2)
    assert set(db1) == set(db3)


def test_tax_prepare_sqlite_lineage_version(runtmp):
    # test bad sourmash_internals version for SqliteLineage
    taxcsv = utils.get_test_data("tax/test.taxonomy.csv")
    taxout = runtmp.output("out.db")

    runtmp.run_sourmash("tax", "prepare", "-t", taxcsv, "-o", taxout, "-F", "sql")
    assert os.path.exists(taxout)

    # set bad version
    conn = sqlite_utils.open_sqlite_db(taxout)
    c = conn.cursor()
    c.execute("UPDATE sourmash_internal SET value='0.9' WHERE key='SqliteLineage'")

    conn.commit()
    conn.close()

    with pytest.raises(IndexNotSupported):
        tax_utils.MultiLineageDB.load([taxout])


def test_tax_prepare_sqlite_no_lineage():
    # no lineage table at all
    sqldb = utils.get_test_data("sqlite/index.sqldb")

    with pytest.raises(ValueError):
        tax_utils.MultiLineageDB.load([sqldb])


def test_tax_grep_exists(runtmp):
    # test that 'tax grep' exists

    with pytest.raises(SourmashCommandFailed):
        runtmp.sourmash("tax", "grep")

    err = runtmp.last_result.err
    assert "usage:" in err


def test_tax_grep_search_shew(runtmp):
    # test 'tax grep Shew'
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "grep", "Shew", "-t", taxfile)

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    lines = [x.strip() for x in out.splitlines()]
    lines = [x.split(",") for x in lines]
    assert lines[0][0] == "ident"
    assert lines[1][0] == "GCF_000017325.1"
    assert lines[2][0] == "GCF_000021665.1"
    assert len(lines) == 3

    assert "searching 1 taxonomy files for 'Shew'" in err
    assert "found 2 matches; saved identifiers to picklist" in err


def test_tax_grep_search_shew_out(runtmp):
    # test 'tax grep Shew', save result to a file
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "grep", "Shew", "-t", taxfile, "-o", "pick.csv")

    err = runtmp.last_result.err

    out = Path(runtmp.output("pick.csv")).read_text()
    lines = [x.strip() for x in out.splitlines()]
    lines = [x.split(",") for x in lines]
    assert lines[0][0] == "ident"
    assert lines[1][0] == "GCF_000017325.1"
    assert lines[2][0] == "GCF_000021665.1"
    assert len(lines) == 3

    assert "searching 1 taxonomy files for 'Shew'" in err
    assert "found 2 matches; saved identifiers to picklist" in err


def test_tax_grep_search_shew_sqldb_out(runtmp):
    # test 'tax grep Shew' on a sqldb, save result to a file
    taxfile = utils.get_test_data("tax/test.taxonomy.db")

    runtmp.sourmash("tax", "grep", "Shew", "-t", taxfile, "-o", "pick.csv")

    err = runtmp.last_result.err

    out = Path(runtmp.output("pick.csv")).read_text()
    lines = [x.strip() for x in out.splitlines()]
    lines = [x.split(",") for x in lines]
    assert lines[0][0] == "ident"
    assert lines[1][0] == "GCF_000017325"
    assert lines[2][0] == "GCF_000021665"
    assert len(lines) == 3

    assert "searching 1 taxonomy files for 'Shew'" in err
    assert "found 2 matches; saved identifiers to picklist" in err


def test_tax_grep_search_shew_lowercase(runtmp):
    # test 'tax grep shew' (lowercase), save result to a file
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "grep", "shew", "-t", taxfile, "-o", "pick.csv")

    err = runtmp.last_result.err
    assert "searching 1 taxonomy files for 'shew'" in err
    assert "found 0 matches; saved identifiers to picklist" in err

    runtmp.sourmash("tax", "grep", "-i", "shew", "-t", taxfile, "-o", "pick.csv")

    err = runtmp.last_result.err
    assert "searching 1 taxonomy files for 'shew'" in err
    assert "found 2 matches; saved identifiers to picklist" in err

    out = Path(runtmp.output("pick.csv")).read_text()
    lines = [x.strip() for x in out.splitlines()]
    lines = [x.split(",") for x in lines]
    assert lines[0][0] == "ident"
    assert lines[1][0] == "GCF_000017325.1"
    assert lines[2][0] == "GCF_000021665.1"
    assert len(lines) == 3


def test_tax_grep_search_shew_out_use_picklist(runtmp):
    # test 'tax grep Shew', output to a picklist, use picklist
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")
    dbfile = utils.get_test_data("tax/gtdb-tax-grep.sigs.zip")

    runtmp.sourmash("tax", "grep", "Shew", "-t", taxfile, "-o", "pick.csv")

    runtmp.sourmash(
        "sig", "cat", dbfile, "--picklist", "pick.csv:ident:ident", "-o", "pick-out.zip"
    )

    all_sigs = sourmash.load_file_as_index(dbfile)
    assert len(all_sigs) == 3

    pick_sigs = sourmash.load_file_as_index(runtmp.output("pick-out.zip"))
    assert len(pick_sigs) == 2

    names = [ss.name.split()[0] for ss in pick_sigs.signatures()]
    assert len(names) == 2
    assert "GCF_000017325.1" in names
    assert "GCF_000021665.1" in names


def test_tax_grep_search_shew_invert(runtmp):
    # test 'tax grep -v Shew'
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "grep", "-v", "Shew", "-t", taxfile)

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert (
        "-v/--invert-match specified; returning only lineages that do not match." in err
    )

    lines = [x.strip() for x in out.splitlines()]
    lines = [x.split(",") for x in lines]
    assert lines[0][0] == "ident"
    assert lines[1][0] == "GCF_001881345.1"
    assert lines[2][0] == "GCF_003471795.1"
    assert len(lines) == 5

    assert "searching 1 taxonomy files for 'Shew'" in err
    assert "found 4 matches; saved identifiers to picklist" in err

    all_names = set([x[0] for x in lines])
    assert "GCF_000017325.1" not in all_names
    assert "GCF_000021665.1" not in all_names


def test_tax_grep_search_shew_invert_select_phylum(runtmp):
    # test 'tax grep -v Shew -r phylum'
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "grep", "-v", "Shew", "-t", taxfile, "-r", "phylum")

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert (
        "-v/--invert-match specified; returning only lineages that do not match." in err
    )
    assert "limiting matches to phylum"

    lines = [x.strip() for x in out.splitlines()]
    lines = [x.split(",") for x in lines]
    assert lines[0][0] == "ident"
    assert len(lines) == 7

    assert "searching 1 taxonomy files for 'Shew'" in err
    assert "found 6 matches; saved identifiers to picklist" in err

    all_names = set([x[0] for x in lines])
    assert "GCF_000017325.1" in all_names
    assert "GCF_000021665.1" in all_names


def test_tax_grep_search_shew_invert_select_bad_rank(runtmp):
    # test 'tax grep -v Shew -r badrank' - should fail
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    with pytest.raises(SourmashCommandFailed):
        runtmp.sourmash("tax", "grep", "-v", "Shew", "-t", taxfile, "-r", "badrank")

    err = runtmp.last_result.err

    print(err)
    assert "error: argument -r/--rank: invalid choice:" in err


def test_tax_grep_search_shew_count(runtmp):
    # test 'tax grep Shew --count'
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "grep", "Shew", "-t", taxfile, "-c")

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert not out.strip()

    assert "searching 1 taxonomy files for 'Shew'" in err
    assert "found 2 matches; saved identifiers to picklist" not in err


def test_tax_grep_multiple_csv(runtmp):
    # grep on multiple CSVs
    tax1 = utils.get_test_data("tax/test.taxonomy.csv")
    tax2 = utils.get_test_data("tax/protozoa_genbank_lineage.csv")

    taxout = runtmp.output("out.csv")

    runtmp.sourmash("tax", "grep", "Toxo|Gamma", "-t", tax1, tax2, "-o", taxout)

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert not out
    assert "found 4 matches" in err

    lines = Path(taxout).read_text().splitlines()
    assert len(lines) == 5

    names = set([x.split(",")[0] for x in lines])
    assert "GCA_000256725" in names
    assert "GCF_000017325.1" in names
    assert "GCF_000021665.1" in names
    assert "GCF_001881345.1" in names


def test_tax_grep_multiple_csv_empty_force(runtmp):
    # grep on multiple CSVs, one empty, with --force
    tax1 = utils.get_test_data("tax/test.taxonomy.csv")
    tax2 = utils.get_test_data("tax/protozoa_genbank_lineage.csv")
    tax_empty = runtmp.output("t.csv")

    taxout = runtmp.output("out.csv")
    with open(tax_empty, "w") as fp:
        fp.write("")
    print("t_csv: ", tax_empty)

    runtmp.sourmash(
        "tax",
        "grep",
        "Toxo|Gamma",
        "-t",
        tax1,
        tax2,
        "-t",
        tax_empty,
        "-o",
        taxout,
        "--force",
    )

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert not out
    assert "found 4 matches" in err

    lines = Path(taxout).read_text().splitlines()
    assert len(lines) == 5

    names = set([x.split(",")[0] for x in lines])
    assert "GCA_000256725" in names
    assert "GCF_000017325.1" in names
    assert "GCF_000021665.1" in names
    assert "GCF_001881345.1" in names


def test_tax_grep_duplicate_csv(runtmp):
    # grep on duplicates => should collapse to uniques on identifiers
    tax1 = utils.get_test_data("tax/test.taxonomy.csv")

    taxout = runtmp.output("out.csv")

    runtmp.sourmash("tax", "grep", "Gamma", "-t", tax1, tax1, "-o", taxout)

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert not out
    assert "found 3 matches" in err

    lines = Path(taxout).read_text().splitlines()
    assert len(lines) == 4

    names = set([x.split(",")[0] for x in lines])
    assert "GCF_000017325.1" in names
    assert "GCF_000021665.1" in names
    assert "GCF_001881345.1" in names


def test_tax_summarize(runtmp):
    # test basic operation with summarize
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "summarize", taxfile)

    out = runtmp.last_result.out

    assert "number of distinct taxonomic lineages: 6" in out
    assert "rank superkingdom:        1 distinct taxonomic lineages" in out
    assert "rank phylum:              2 distinct taxonomic lineages" in out
    assert "rank class:               2 distinct taxonomic lineages" in out
    assert "rank order:               2 distinct taxonomic lineages" in out
    assert "rank family:              3 distinct taxonomic lineages" in out
    assert "rank genus:               4 distinct taxonomic lineages" in out
    assert "rank species:             4 distinct taxonomic lineages" in out


def test_tax_summarize_multiple(runtmp):
    # test basic operation with summarize on multiple files
    tax1 = utils.get_test_data("tax/bacteria_refseq_lineage.csv")
    tax2 = utils.get_test_data("tax/protozoa_genbank_lineage.csv")

    runtmp.sourmash("tax", "summarize", tax1, tax2)

    out = runtmp.last_result.out

    assert "number of distinct taxonomic lineages: 6" in out
    assert "rank superkingdom:        2 distinct taxonomic lineages" in out
    assert "rank phylum:              3 distinct taxonomic lineages" in out
    assert "rank class:               4 distinct taxonomic lineages" in out
    assert "rank order:               4 distinct taxonomic lineages" in out
    assert "rank family:              5 distinct taxonomic lineages" in out
    assert "rank genus:               5 distinct taxonomic lineages" in out
    assert "rank species:             5 distinct taxonomic lineages" in out


def test_tax_summarize_empty_line(runtmp):
    # test basic operation with summarize on a file w/empty line
    taxfile = utils.get_test_data("tax/test-empty-line.taxonomy.csv")

    runtmp.sourmash("tax", "summarize", taxfile)

    out = runtmp.last_result.out

    assert "number of distinct taxonomic lineages: 6" in out
    assert "rank superkingdom:        1 distinct taxonomic lineages" in out
    assert "rank phylum:              2 distinct taxonomic lineages" in out
    assert "rank class:               2 distinct taxonomic lineages" in out
    assert "rank order:               2 distinct taxonomic lineages" in out
    assert "rank family:              3 distinct taxonomic lineages" in out
    assert "rank genus:               4 distinct taxonomic lineages" in out
    assert "rank species:             4 distinct taxonomic lineages" in out


def test_tax_summarize_empty(runtmp):
    # test failure on empty file
    taxfile = runtmp.output("no-exist")

    with pytest.raises(SourmashCommandFailed):
        runtmp.sourmash("tax", "summarize", taxfile)

    err = runtmp.last_result.err
    assert "ERROR while loading taxonomies" in err


def test_tax_summarize_csv(runtmp):
    # test basic operation w/csv output
    taxfile = utils.get_test_data("tax/test.taxonomy.csv")

    runtmp.sourmash("tax", "summarize", taxfile, "-o", "ranks.csv")

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert "number of distinct taxonomic lineages: 6" in out
    assert "saved 18 lineage counts to 'ranks.csv'" in err

    csv_out = runtmp.output("ranks.csv")

    with sourmash_args.FileInputCSV(csv_out) as r:
        # count number across ranks as a cheap consistency check
        c = Counter()
        for row in r:
            val = row["lineage_count"]
            c[val] += 1

        assert c["3"] == 7
        assert c["2"] == 5
        assert c["1"] == 5


def test_tax_summarize_on_annotate(runtmp):
    # test summarize on output of annotate basics
    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.taxonomy.csv")
    csvout = runtmp.output("test1.gather.with-lineages.csv")
    out_dir = os.path.dirname(csvout)

    runtmp.run_sourmash(
        "tax", "annotate", "--gather-csv", g_csv, "--taxonomy-csv", tax, "-o", out_dir
    )

    print(runtmp.last_result.status)
    print(runtmp.last_result.out)
    print(runtmp.last_result.err)

    assert runtmp.last_result.status == 0
    assert os.path.exists(csvout)

    # so far so good - now see if we can run summarize!

    runtmp.run_sourmash("tax", "summarize", csvout)
    out = runtmp.last_result.out
    err = runtmp.last_result.err

    print(out)
    print(err)

    assert "number of distinct taxonomic lineages: 4" in out
    assert "rank superkingdom:        1 distinct taxonomic lineages" in out
    assert "rank phylum:              2 distinct taxonomic lineages" in out
    assert "rank class:               2 distinct taxonomic lineages" in out
    assert "rank order:               2 distinct taxonomic lineages" in out
    assert "rank family:              2 distinct taxonomic lineages" in out
    assert "rank genus:               3 distinct taxonomic lineages" in out
    assert "rank species:             3 distinct taxonomic lineages" in out


def test_tax_summarize_strain_csv(runtmp):
    # test basic operation w/csv output on taxonomy with strains
    taxfile = utils.get_test_data("tax/test-strain.taxonomy.csv")

    runtmp.sourmash("tax", "summarize", taxfile, "-o", "ranks.csv")

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert "number of distinct taxonomic lineages: 6" in out
    assert "saved 24 lineage counts to 'ranks.csv'" in err

    csv_out = runtmp.output("ranks.csv")

    with sourmash_args.FileInputCSV(csv_out) as r:
        # count number across ranks as a cheap consistency check
        c = Counter()
        for row in r:
            print(row)
            val = row["lineage_count"]
            c[val] += 1

        print(list(c.most_common()))

        assert c["3"] == 7
        assert c["2"] == 5
        assert c["6"] == 1
        assert c["1"] == 11


def test_tax_summarize_strain_csv_with_lineages(runtmp):
    # test basic operation w/csv output on lineages-style file w/strain csv
    taxfile = utils.get_test_data("tax/test-strain.taxonomy.csv")
    lineage_csv = runtmp.output("lin-with-strains.csv")

    taxdb = tax_utils.LineageDB.load(taxfile)
    with open(lineage_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "lineage"])
        for k, v in taxdb.items():
            linstr = lca_utils.display_lineage(v)
            w.writerow([k, linstr])

    runtmp.sourmash("tax", "summarize", lineage_csv, "-o", "ranks.csv")

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert "number of distinct taxonomic lineages: 6" in out
    assert "saved 24 lineage counts to" in err

    csv_out = runtmp.output("ranks.csv")

    with sourmash_args.FileInputCSV(csv_out) as r:
        # count number across ranks as a cheap consistency check
        c = Counter()
        for row in r:
            print(row)
            val = row["lineage_count"]
            c[val] += 1

        print(list(c.most_common()))

        assert c["3"] == 7
        assert c["2"] == 5
        assert c["6"] == 1
        assert c["1"] == 11


def test_tax_summarize_ictv(runtmp):
    # test basic operation w/csv output on lineages-style file w/strain csv
    taxfile = utils.get_test_data("tax/test.ictv-taxonomy.csv")
    lineage_csv = runtmp.output("ictv-lins.csv")

    taxdb = tax_utils.LineageDB.load(taxfile)
    with open(lineage_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "lineage"])
        for k, v in taxdb.items():
            linstr = lca_utils.display_lineage(v)
            w.writerow([k, linstr])

    runtmp.sourmash("tax", "summarize", lineage_csv, "-o", "ranks.csv", "--ictv")

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    assert "number of distinct taxonomic lineages: 7" in out
    assert "saved 14 lineage counts to" in err

    csv_out = runtmp.output("ranks.csv")

    with sourmash_args.FileInputCSV(csv_out) as r:
        # count number across ranks as a cheap consistency check
        c = Counter()
        for row in r:
            print(row)
            val = row["lineage_count"]
            c[val] += 1

        print(list(c.most_common()))
        print(c)
        assert c["1"] == 8
        assert c["7"] == 5
        assert c["6"] == 1


def test_tax_summarize_LINS(runtmp):
    # test basic operation w/LINs
    taxfile = utils.get_test_data("tax/test.LIN-taxonomy.csv")
    lineage_csv = runtmp.output("annotated-lin.csv")

    taxdb = tax_utils.LineageDB.load(taxfile, lins=True)
    with open(lineage_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "lineage"])
        for k, v in taxdb.items():
            lin = tax_utils.LINLineageInfo(lineage=v)
            linstr = lin.display_lineage(truncate_empty=False)
            print(linstr)
            w.writerow([k, linstr])

    runtmp.sourmash("tax", "summarize", lineage_csv, "-o", "ranks.csv", "--lins")

    out = runtmp.last_result.out
    err = runtmp.last_result.err

    print(out)
    print(err)

    assert "number of distinct taxonomic lineages: 6" in out
    assert "saved 91 lineage counts to" in err

    csv_out = runtmp.output("ranks.csv")

    with sourmash_args.FileInputCSV(csv_out) as r:
        # count number across ranks as a cheap consistency check
        c = Counter()
        for row in r:
            print(row)
            val = row["lineage_count"]
            c[val] += 1

        print(list(c.most_common()))

        assert c["1"] == 77
        assert c["2"] == 1
        assert c["3"] == 11
        assert c["4"] == 2


def test_metagenome_LIN(runtmp):
    # test basic metagenome with LIN taxonomy
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    c.run_sourmash("tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax, "--lins")

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in c.last_result.out
    )
    # 0th rank/position
    assert "test1,0,0.089,1,md5,test1.sig,0.057,444000,0.925,0" in c.last_result.out
    assert "test1,0,0.088,0,md5,test1.sig,0.058,442000,0.925,0" in c.last_result.out
    assert "test1,0,0.028,2,md5,test1.sig,0.016,138000,0.891,0" in c.last_result.out
    assert (
        "test1,0,0.796,unclassified,md5,test1.sig,0.869,3990000,,0" in c.last_result.out
    )
    # 1st rank/position
    assert "test1,1,0.089,1;0,md5,test1.sig,0.057,444000,0.925,0" in c.last_result.out
    assert "test1,1,0.088,0;0,md5,test1.sig,0.058,442000,0.925,0" in c.last_result.out
    assert "test1,1,0.028,2;0,md5,test1.sig,0.016,138000,0.891,0" in c.last_result.out
    assert (
        "test1,1,0.796,unclassified,md5,test1.sig,0.869,3990000,,0" in c.last_result.out
    )
    # 2nd rank/position
    assert "test1,2,0.088,0;0;0,md5,test1.sig,0.058,442000,0.925,0" in c.last_result.out
    assert "test1,2,0.078,1;0;0,md5,test1.sig,0.050,390000,0.921,0" in c.last_result.out
    assert "test1,2,0.028,2;0;0,md5,test1.sig,0.016,138000,0.891,0" in c.last_result.out
    assert "test1,2,0.011,1;0;1,md5,test1.sig,0.007,54000,0.864,0" in c.last_result.out
    assert (
        "test1,2,0.796,unclassified,md5,test1.sig,0.869,3990000,,0" in c.last_result.out
    )
    # 19th rank/position
    assert (
        "test1,19,0.088,0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,md5,test1.sig,0.058,442000,0.925,0"
        in c.last_result.out
    )
    assert (
        "test1,19,0.078,1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,md5,test1.sig,0.050,390000,0.921,0"
        in c.last_result.out
    )
    assert (
        "test1,19,0.028,2;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,md5,test1.sig,0.016,138000,0.891,0"
        in c.last_result.out
    )
    assert (
        "test1,19,0.011,1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,md5,test1.sig,0.007,54000,0.864,0"
        in c.last_result.out
    )
    assert (
        "test1,19,0.796,unclassified,md5,test1.sig,0.869,3990000,,0"
        in c.last_result.out
    )


def test_metagenome_LIN_lingroups(runtmp):
    # test lingroups output
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    lg_file = runtmp.output("test.lg.csv")
    with open(lg_file, "w") as out:
        out.write("lin,name\n")
        out.write("0;0;0,lg1\n")
        out.write("1;0;0,lg2\n")
        out.write("2;0;0,lg3\n")
        out.write("1;0;1,lg3\n")
        # write a 19 so we can check the end
        out.write("1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,lg4\n")

    c.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "--lingroup",
        lg_file,
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "Read 5 lingroup rows and found 5 distinct lingroup prefixes."
        in c.last_result.err
    )
    assert "name	lin	percent_containment	num_bp_contained" in c.last_result.out
    assert "lg1	0;0;0	5.82	714000" in c.last_result.out
    assert "lg2	1;0;0	5.05	620000" in c.last_result.out
    assert "lg3	2;0;0	1.56	192000" in c.last_result.out
    assert "lg3	1;0;1	0.65	80000" in c.last_result.out
    assert (
        "lg4	1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0	0.65	80000"
        in c.last_result.out
    )


def test_metagenome_LIN_lingroups_summary(runtmp):
    # test lingroups summary file. Can no longer output stdout, b/c will produce 2 files
    c = runtmp
    csv_base = "out"
    sum_csv = csv_base + ".summarized.csv"
    csvout = runtmp.output(sum_csv)
    outdir = os.path.dirname(csvout)

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    lg_file = runtmp.output("test.lg.csv")
    with open(lg_file, "w") as out:
        out.write("lin,name\n")
        out.write("0;0;0,lg1\n")
        out.write("1;0;0,lg2\n")
        out.write("2;0;0,lg3\n")
        out.write("1;0;1,lg3\n")
        # write a 19 so we can check the end
        out.write("1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0,lg4\n")

    c.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "--lingroup",
        lg_file,
        "-o",
        csv_base,
        "--output-dir",
        outdir,
        "-F",
        "csv_summary",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert (
        "Read 5 lingroup rows and found 5 distinct lingroup prefixes."
        in c.last_result.err
    )
    assert os.path.exists(csvout)
    sum_gather_results = [x.rstrip() for x in Path(csvout).read_text().splitlines()]
    print(sum_gather_results)
    assert f"saving 'csv_summary' output to '{csvout}'" in runtmp.last_result.err
    assert (
        "query_name,rank,fraction,lineage,query_md5,query_filename,f_weighted_at_rank,bp_match_at_rank"
        in sum_gather_results[0]
    )
    assert (
        "test1,2,0.08815317112086159,lg1,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.05815279361459521,442000,0.9246458342627294,6139"
        in sum_gather_results[1]
    )
    assert (
        "test1,2,0.07778220981252493,lg2,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.050496823586903404,390000,0.920920083987624,6139"
        in sum_gather_results[2]
    )
    assert (
        "test1,2,0.027522935779816515,lg3,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.015637726014008795,138000,0.8905689983332759,6139"
        in sum_gather_results[3]
    )
    assert (
        "test1,2,0.010769844435580374,lg3,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.006515719172503665,54000,0.8640181883213995,6139"
        in sum_gather_results[4]
    )
    assert (
        "test1,2,0.7957718388512166,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.8691969376119889,3990000,,6139"
        in sum_gather_results[5]
    )
    assert (
        "test1,19,0.010769844435580374,lg4,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.006515719172503665,54000,0.8640181883213995,6139"
        in sum_gather_results[6]
    )
    assert (
        "test1,19,0.7957718388512166,unclassified,9687eeed,outputs/abundtrim/HSMA33MX.abundtrim.fq.gz,0.8691969376119889,3990000,,6139"
        in sum_gather_results[7]
    )


def test_metagenome_LIN_human_summary_no_lin_position(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    c.run_sourmash(
        "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax, "--lins", "-F", "human"
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert "sample name    proportion   cANI   lineage" in c.last_result.out
    assert "-----------    ----------   ----   -------" in c.last_result.out
    assert "test1             86.9%     -      unclassified" in c.last_result.out
    assert (
        "test1              5.8%     92.5%  0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0"
        in c.last_result.out
    )
    assert (
        "test1              5.0%     92.1%  1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0"
        in c.last_result.out
    )
    assert (
        "test1              1.6%     89.1%  2;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0"
        in c.last_result.out
    )
    assert (
        "test1              0.7%     86.4%  1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0"
        in c.last_result.out
    )


def test_metagenome_LIN_human_summary_lin_position_5(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "-F",
        "human",
        "--lin-position",
        "5",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert "sample name    proportion   cANI   lineage" in c.last_result.out
    assert "-----------    ----------   ----   -------" in c.last_result.out
    assert "test1             86.9%     -      unclassified" in c.last_result.out
    assert "test1              5.8%     92.5%  0;0;0;0;0;0" in c.last_result.out
    assert "test1              5.0%     92.1%  1;0;0;0;0;0" in c.last_result.out
    assert "test1              1.6%     89.1%  2;0;0;0;0;0" in c.last_result.out
    assert "test1              0.7%     86.4%  1;0;1;0;0;0" in c.last_result.out


def test_metagenome_LIN_krona_lin_position_5(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    c.run_sourmash(
        "tax",
        "metagenome",
        "-g",
        g_csv,
        "--taxonomy-csv",
        tax,
        "--lins",
        "-F",
        "krona",
        "--lin-position",
        "5",
    )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status == 0
    assert "fraction	0	1	2	3	4	5" in c.last_result.out
    assert "0.08815317112086159	0	0	0	0	0	0" in c.last_result.out
    assert "0.07778220981252493	1	0	0	0	0	0" in c.last_result.out
    assert "0.027522935779816515	2	0	0	0	0	0" in c.last_result.out
    assert "0.010769844435580374	1	0	1	0	0	0" in c.last_result.out
    assert (
        "0.7957718388512166	unclassified	unclassified	unclassified	unclassified	unclassified	unclassified"
        in c.last_result.out
    )


def test_metagenome_LIN_krona_bad_rank(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--lins",
            "-F",
            "krona",
            "--lin-position",
            "strain",
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status != 0
    assert (
        "Invalid '--rank'/'--position' input: 'strain'. '--lins' is specified. Rank must be an integer corresponding to a LIN position."
        in c.last_result.err
    )


def test_metagenome_LIN_lingroups_empty_lg_file(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    lg_file = runtmp.output("test.lg.csv")
    with open(lg_file, "w") as out:
        out.write("")

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--lins",
            "--lingroup",
            lg_file,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status != 0
    assert (
        f"Cannot read lingroups from '{lg_file}'. Is file empty?" in c.last_result.err
    )


def test_metagenome_LIN_lingroups_bad_cli_inputs(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    lg_file = runtmp.output("test.lg.csv")
    with open(lg_file, "w") as out:
        out.write("")

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--lins",
            "-F",
            "lingroup",
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status != 0
    assert (
        "Must provide lingroup csv via '--lingroup' in order to output a lingroup report."
        in c.last_result.err
    )

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax", "metagenome", "-g", g_csv, "--taxonomy-csv", tax, "-F", "lingroup"
        )
    print(c.last_result.err)
    assert c.last_result.status != 0
    assert (
        "Must enable LIN taxonomy via '--lins' in order to use lingroups."
        in c.last_result.err
    )

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--lingroup",
            lg_file,
        )
    print(c.last_result.err)
    assert c.last_result.status != 0

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--lins",
            "-F",
            "bioboxes",
        )
    print(c.last_result.err)
    assert c.last_result.status != 0
    assert (
        "ERROR: The following outputs are incompatible with '--lins': : bioboxes, kreport"
        in c.last_result.err
    )


def test_metagenome_mult_outputs_stdout_fail(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-F",
            "kreport",
            "csv_summary",
        )

    print(c.last_result.err)
    assert c.last_result.status != 0
    assert (
        "Writing to stdout is incompatible with multiple output formats ['kreport', 'csv_summary']"
        in c.last_result.err
    )


def test_genome_mult_outputs_stdout_fail(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "genome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "-F",
            "lineage_csv",
            "csv_summary",
        )

    print(c.last_result.err)
    assert c.last_result.status != 0
    assert (
        "Writing to stdout is incompatible with multiple output formats ['lineage_csv', 'csv_summary']"
        in c.last_result.err
    )


def test_metagenome_LIN_lingroups_lg_only_header(runtmp):
    c = runtmp

    g_csv = utils.get_test_data("tax/test1.gather.v450.csv")
    tax = utils.get_test_data("tax/test.LIN-taxonomy.csv")

    lg_file = runtmp.output("test.lg.csv")
    with open(lg_file, "w") as out:
        out.write("lin,name\n")

    with pytest.raises(SourmashCommandFailed):
        c.run_sourmash(
            "tax",
            "metagenome",
            "-g",
            g_csv,
            "--taxonomy-csv",
            tax,
            "--lins",
            "--lingroup",
            lg_file,
        )

    print(c.last_result.status)
    print(c.last_result.out)
    print(c.last_result.err)

    assert c.last_result.status != 0
    assert f"No lingroups loaded from {lg_file}" in c.last_result.err
