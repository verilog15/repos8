use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use proptest::collection::vec;
use proptest::num::u64;
use proptest::proptest;
use sourmash::encodings::HashFunctions;
use sourmash::prelude::ToWriter;
use sourmash::signature::SeqToHashes;
use sourmash::signature::{Signature, SigsTrait};
use sourmash::sketch::minhash::{
    max_hash_for_scaled, scaled_for_max_hash, KmerMinHash, KmerMinHashBTree,
};
use sourmash::sketch::Sketch;
use sourmash::ScaledType;

// TODO: use f64::EPSILON when we bump MSRV
const EPSILON: f64 = 0.01;

#[test]
fn throws_error() {
    let mut mh = KmerMinHash::new(0, 4, HashFunctions::Murmur64Dna, 42, false, 1);

    assert!(
        mh.add_sequence(b"ATGR", false).is_err(),
        "R is not a valid DNA character"
    );
}

#[test]
fn merge() {
    let mut a = KmerMinHash::new(0, 10, HashFunctions::Murmur64Dna, 42, false, 20);
    let mut b = KmerMinHash::new(0, 10, HashFunctions::Murmur64Dna, 42, false, 20);

    a.add_sequence(b"TGCCGCCCAGCA", false).unwrap();
    b.add_sequence(b"TGCCGCCCAGCA", false).unwrap();

    a.add_sequence(b"GTCCGCCCAGTGA", false).unwrap();
    b.add_sequence(b"GTCCGCCCAGTGG", false).unwrap();

    a.merge(&b).unwrap();
    assert_eq!(
        a.to_vec(),
        vec![
            2996412506971915891,
            4448613756639084635,
            8373222269469409550,
            9390240264282449587,
            11085758717695534616,
            11668188995231815419,
            11760449009842383350,
            14682565545778736889,
        ]
    );
}

#[test]
fn invalid_dna() {
    let mut a = KmerMinHash::new(0, 3, HashFunctions::Murmur64Dna, 42, false, 20);

    a.add_sequence(b"AAANNCCCTN", true).unwrap();
    assert_eq!(a.mins().len(), 3);

    let mut b = KmerMinHash::new(0, 3, HashFunctions::Murmur64Dna, 42, false, 20);
    b.add_sequence(b"NAAA", true).unwrap();
    assert_eq!(b.mins().len(), 1);
}

#[test]
fn similarity() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = KmerMinHash::new(0, 20, HashFunctions::Murmur64Hp, 42, true, 5);
    let mut b = KmerMinHash::new(0, 20, HashFunctions::Murmur64Hp, 42, true, 5);

    a.add_hash(1);
    b.add_hash(1);
    b.add_hash(2);

    assert!((a.similarity(&a, false, false)? - 1.0).abs() < EPSILON);
    assert!((a.similarity(&b, false, false)? - 0.5).abs() < EPSILON);

    Ok(())
}

#[test]
fn similarity_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = KmerMinHash::new(0, 5, HashFunctions::Murmur64Dna, 42, true, 5);
    let mut b = KmerMinHash::new(0, 5, HashFunctions::Murmur64Dna, 42, true, 5);

    a.add_sequence(b"ATGGA", false)?;
    a.add_sequence(b"GGACA", false)?;

    a.add_sequence(b"ATGGA", false)?;
    b.add_sequence(b"ATGGA", false)?;

    assert!(
        (a.similarity(&b, false, false)? - 0.705).abs() < EPSILON,
        "{}",
        a.similarity(&b, false, false)?
    );

    Ok(())
}

#[test]
fn similarity_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = KmerMinHash::new(0, 20, HashFunctions::Murmur64Dayhoff, 42, true, 5);
    let mut b = KmerMinHash::new(0, 20, HashFunctions::Murmur64Dayhoff, 42, true, 5);

    a.add_hash(1);
    a.add_hash(1);
    a.add_hash(5);
    a.add_hash(5);

    b.add_hash(1);
    b.add_hash(2);
    b.add_hash(3);
    b.add_hash(4);

    assert!((a.similarity(&a, false, false)? - 1.0).abs() < EPSILON);
    assert!((a.similarity(&b, false, false)? - 0.23).abs() < EPSILON);

    assert!((a.similarity(&a, true, false)? - 1.0).abs() < EPSILON);
    assert!((a.similarity(&b, true, false)? - 0.2).abs() < EPSILON);

    Ok(())
}

#[test]
fn angular_similarity_requires_abundance() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = KmerMinHash::new(0, 20, HashFunctions::Murmur64Dayhoff, 42, false, 5);
    let mut b = KmerMinHash::new(0, 20, HashFunctions::Murmur64Dayhoff, 42, false, 5);

    a.add_hash(1);
    b.add_hash(1);

    assert!(a.angular_similarity(&b).is_err());

    Ok(())
}

#[test]
fn angular_similarity_btree_requires_abundance() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = KmerMinHashBTree::new(0, 20, HashFunctions::Murmur64Dayhoff, 42, false, 5);
    let mut b = KmerMinHashBTree::new(0, 20, HashFunctions::Murmur64Dayhoff, 42, false, 5);

    a.add_hash(1);
    b.add_hash(1);

    assert!(a.angular_similarity(&b).is_err());

    Ok(())
}

#[test]
fn dayhoff() {
    let mut a = KmerMinHash::new(0, 6, HashFunctions::Murmur64Dayhoff, 42, false, 10);
    let mut b = KmerMinHash::new(0, 6, HashFunctions::Murmur64Protein, 42, false, 10);

    a.add_sequence(b"ACTGAC", false).unwrap();
    b.add_sequence(b"ACTGAC", false).unwrap();

    assert_eq!(a.size(), 2);
    assert_eq!(b.size(), 2);
}

#[test]
fn hp() {
    let mut a = KmerMinHash::new(0, 6, HashFunctions::Murmur64Hp, 42, false, 10);
    let mut b = KmerMinHash::new(0, 6, HashFunctions::Murmur64Protein, 42, false, 10);

    a.add_sequence(b"ACTGAC", false).unwrap();
    b.add_sequence(b"ACTGAC", false).unwrap();

    assert_eq!(a.size(), 2);
    assert_eq!(b.size(), 2);
}

#[test]
fn max_for_scaled() {
    assert_eq!(max_hash_for_scaled(100), 184467440737095520);
}

proptest! {
#[test]
fn oracle_mins(hashes in vec(u64::ANY, 1..10000)) {
    let mut a = KmerMinHash::new(0, 21, HashFunctions::Murmur64Protein, 42, true, 1000);
    let mut b = KmerMinHashBTree::new(0, 21, HashFunctions::Murmur64Protein, 42, true, 1000);

    let mut c: KmerMinHash = Default::default();
    c.set_hash_function(HashFunctions::Murmur64Protein).unwrap();
    c.enable_abundance().unwrap();

    let mut d: KmerMinHashBTree = Default::default();
    d.set_hash_function(HashFunctions::Murmur64Protein).unwrap();
    d.enable_abundance().unwrap();

    let mut to_remove = vec![];
    for hash in &hashes {
        a.add_hash(*hash);
        b.add_hash(*hash);

        if hash % 2 == 0 {
            to_remove.push(*hash);
        }
    }

    c.add_from(&a).unwrap();
    c.remove_many(to_remove.iter().copied()).unwrap();

    d.add_from(&b).unwrap();
    d.remove_many(to_remove.iter().copied()).unwrap();

    assert_eq!(a.mins(), b.mins());
    assert_eq!(c.mins(), d.mins());

    assert_eq!(a.count_common(&c, false).unwrap(), b.count_common(&d, false).unwrap());
    assert_eq!(a.count_common(&c, true).unwrap(), b.count_common(&d, true).unwrap());

    assert_eq!(a.abunds(), b.abunds());
    assert_eq!(c.abunds(), d.abunds());

    assert!((a.similarity(&c, false, false).unwrap() - b.similarity(&d, false, false).unwrap()).abs() < EPSILON);
}
}

proptest! {
#[test]
fn oracle_mins_scaled(hashes in vec(u64::ANY, 1..10000)) {
    let scaled = 100;
    let mut a = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);
    let mut b = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);

    let mut c = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);
    let mut d = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);

    let mut to_remove = vec![];
    for hash in &hashes {
        a.add_hash(*hash);
        b.add_hash(*hash);

        if hash % 2 == 0 {
            to_remove.push(*hash);
        }
    }

    c.add_many(&hashes).unwrap();
    d.add_many(&hashes).unwrap();

    c.remove_many(to_remove.iter().copied()).unwrap();
    d.remove_many(to_remove.iter().copied()).unwrap();

    a.remove_hash(hashes[0]);
    b.remove_hash(hashes[0]);

    assert_eq!(a.mins(), b.mins());
    assert_eq!(c.mins(), d.mins());

    assert_eq!(a.md5sum(), b.md5sum());
    assert_eq!(c.md5sum(), d.md5sum());

    assert_eq!(a.is_protein(), b.is_protein());
    assert_eq!(a.num(), b.num());
    assert_eq!(a.seed(), b.seed());
    assert_eq!(a.ksize(), b.ksize());
    assert_eq!(a.scaled(), b.scaled());
    assert_eq!(a.track_abundance(), b.track_abundance());
    assert_eq!(a.hash_function(), b.hash_function());

    assert_eq!(a.abunds(), b.abunds());
    assert_eq!(c.abunds(), d.abunds());

    assert!((a.similarity(&c, false, false).unwrap() - b.similarity(&d, false, false).unwrap()).abs() < EPSILON);
    assert!((c.similarity(&a, false, false).unwrap() - d.similarity(&b, false, false).unwrap()).abs() < EPSILON);
    assert!((a.similarity(&c, true, false).unwrap() - b.similarity(&d, true, false).unwrap()).abs() < EPSILON);
    assert!((c.similarity(&a, true, false).unwrap() - d.similarity(&b, true, false).unwrap()).abs() < EPSILON);

    assert_eq!(a.count_common(&c, false).unwrap(), b.count_common(&d, false).unwrap());
    assert_eq!(c.count_common(&a, false).unwrap(), d.count_common(&b, false).unwrap());
    assert_eq!(a.count_common(&c, true).unwrap(), b.count_common(&d, true).unwrap());
    assert_eq!(c.count_common(&a, true).unwrap(), d.count_common(&b, true).unwrap());

    let mut e = a.downsample_max_hash(100).unwrap();
    let scaled = scaled_for_max_hash(100);
    let mut f = b.downsample_scaled(scaled).unwrap();
    assert_eq!(f.scaled(), scaled);

    // Can't compare different scaled without explicit downsample
    assert!(c.similarity(&e, false, false).is_err());
    assert!(d.similarity(&f, false, false).is_err());
    assert!(c.similarity(&e, true, false).is_err());
    assert!(d.similarity(&f, true, false).is_err());

    assert!((c.similarity(&e, true, true).unwrap() - d.similarity(&f, true, true).unwrap()).abs() < EPSILON);
    assert!((e.similarity(&c, true, true).unwrap() - f.similarity(&d, true, true).unwrap()).abs() < EPSILON);
    assert!((c.similarity(&e, false, true).unwrap() - d.similarity(&f, false, true).unwrap()).abs() < EPSILON);
    assert!((e.similarity(&c, false, true).unwrap() - f.similarity(&d, false, true).unwrap()).abs() < EPSILON);

    // Can't compare different scaled without explicit downsample
    assert!(e.count_common(&c, false).is_err());
    assert!(f.count_common(&d, false).is_err());

    assert_eq!(e.count_common(&c, true).unwrap(), f.count_common(&d, true).unwrap());
    assert_eq!(c.count_common(&e, true).unwrap(), d.count_common(&f, true).unwrap());

    // disable abundances
    e.disable_abundance();
    f.disable_abundance();

    // Can't compare different scaled without explicit downsample
    assert!(c.similarity(&e, false, false).is_err());
    assert!(d.similarity(&f, false, false).is_err());
    assert!(c.similarity(&e, true, false).is_err());
    assert!(d.similarity(&f, true, false).is_err());

    assert!((c.similarity(&e, true, true).unwrap() - d.similarity(&f, true, true).unwrap()).abs() < EPSILON);
    assert!((e.similarity(&c, true, true).unwrap() - f.similarity(&d, true, true).unwrap()).abs() < EPSILON);
    assert!((c.similarity(&e, false, true).unwrap() - d.similarity(&f, false, true).unwrap()).abs() < EPSILON);
    assert!((e.similarity(&c, false, true).unwrap() - f.similarity(&d, false, true).unwrap()).abs() < EPSILON);

    // Can't compare different scaled without explicit downsample
    assert!(e.count_common(&c, false).is_err());
    assert!(f.count_common(&d, false).is_err());

    assert_eq!(e.count_common(&c, true).unwrap(), f.count_common(&d, true).unwrap());
    assert_eq!(c.count_common(&e, true).unwrap(), d.count_common(&f, true).unwrap());
}
}

proptest! {
#[test]
fn prop_merge(seq1 in "[ACGT]{6,100}", seq2 in "[ACGT]{6,200}") {
    let scaled: ScaledType = 10;
    let mut a = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);
    let mut b = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);

    let mut c = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);
    let mut d = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);

    a.add_sequence(seq1.as_bytes(), false).unwrap();
    b.add_sequence(seq1.as_bytes(), false).unwrap();

    c.add_sequence(seq2.as_bytes(), false).unwrap();
    d.add_sequence(seq2.as_bytes(), false).unwrap();

    a.merge(&c).unwrap();
    b.merge(&d).unwrap();

    assert_eq!(a.mins(), b.mins());
    assert_eq!(c.mins(), d.mins());

    assert_eq!(a.abunds(), b.abunds());
    assert_eq!(c.abunds(), d.abunds());

    assert_eq!(a.intersection_size(&c).unwrap(), b.intersection_size(&d).unwrap());
    assert_eq!(c.intersection(&a).unwrap(), d.intersection(&b).unwrap());

    assert!((a.similarity(&c, false, false).unwrap() - b.similarity(&d, false, false).unwrap()).abs() < EPSILON);
    assert!((a.similarity(&c, true, false).unwrap() - b.similarity(&d, true, false).unwrap()).abs() < EPSILON);

    let mut e = a.downsample_max_hash(100).unwrap();
    let scaled = scaled_for_max_hash(100);
    let mut f = b.downsample_scaled(scaled).unwrap();

    assert!((e.similarity(&c, false, true).unwrap() - f.similarity(&d, false, true).unwrap()).abs() < EPSILON);
    assert!((e.similarity(&c, true, true).unwrap() - f.similarity(&d, true, true).unwrap()).abs() < EPSILON);

    e.disable_abundance();
    f.disable_abundance();

    assert!((e.similarity(&c, false, true).unwrap() - f.similarity(&d, false, true).unwrap()).abs() < EPSILON);
    assert!((e.similarity(&c, true, true).unwrap() - f.similarity(&d, true, true).unwrap()).abs() < EPSILON);

    e.clear();
    f.clear();

    assert!(e.is_empty());
    assert!(f.is_empty());
}
}

#[test]
fn load_save_minhash_sketches() {
    let mut filename = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    filename.push("../../tests/test-data/genome-s10+s11.sig");

    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let sigs = Signature::from_reader(reader).expect("Loading error");

    let sig = sigs.get(0).unwrap();
    let sketches = sig.sketches();
    let mut buffer = vec![];

    if let Sketch::MinHash(mh) = &sketches[0] {
        let bmh: KmerMinHashBTree = mh.clone().into();
        {
            bmh.to_writer(&mut buffer).unwrap();
        }

        let new_mh = KmerMinHash::from_reader(&buffer[..]).unwrap();
        let new_bmh = KmerMinHashBTree::from_reader(&buffer[..]).unwrap();

        assert_eq!(mh.md5sum(), new_mh.md5sum());
        assert_eq!(bmh.md5sum(), new_bmh.md5sum());
        assert_eq!(bmh.md5sum(), new_mh.md5sum());
        assert_eq!(mh.md5sum(), new_bmh.md5sum());

        assert_eq!(mh.mins(), new_mh.mins());
        assert_eq!(bmh.mins(), new_bmh.mins());
        assert_eq!(bmh.mins(), new_mh.mins());
        assert_eq!(mh.mins(), new_bmh.mins());

        assert_eq!(mh.abunds(), new_mh.abunds());
        assert_eq!(bmh.abunds(), new_bmh.abunds());
        assert_eq!(bmh.abunds(), new_mh.abunds());
        assert_eq!(mh.abunds(), new_bmh.abunds());

        assert!(
            (mh.similarity(&new_mh, false, false).unwrap()
                - bmh.similarity(&new_bmh, false, false).unwrap())
            .abs()
                < EPSILON
        );

        assert!(
            (mh.similarity(&new_mh, true, false).unwrap()
                - bmh.similarity(&new_bmh, true, false).unwrap())
            .abs()
                < EPSILON
        );

        buffer.clear();
        let imh: KmerMinHash = bmh.clone().into();
        {
            imh.to_writer(&mut buffer).unwrap();
        }

        let new_mh = KmerMinHash::from_reader(&buffer[..]).unwrap();
        let new_bmh = KmerMinHashBTree::from_reader(&buffer[..]).unwrap();

        assert_eq!(mh.md5sum(), new_mh.md5sum());
        assert_eq!(bmh.md5sum(), new_bmh.md5sum());
        assert_eq!(bmh.md5sum(), new_mh.md5sum());
        assert_eq!(mh.md5sum(), new_bmh.md5sum());

        assert_eq!(mh.mins(), new_mh.mins());
        assert_eq!(bmh.mins(), new_bmh.mins());
        assert_eq!(bmh.mins(), new_mh.mins());
        assert_eq!(mh.mins(), new_bmh.mins());

        assert_eq!(mh.abunds(), new_mh.abunds());
        assert_eq!(bmh.abunds(), new_bmh.abunds());
        assert_eq!(bmh.abunds(), new_mh.abunds());
        assert_eq!(mh.abunds(), new_bmh.abunds());

        assert_eq!(mh.to_vec(), new_mh.to_vec());
        assert_eq!(bmh.to_vec(), new_bmh.to_vec());
        assert_eq!(bmh.to_vec(), new_mh.to_vec());
        assert_eq!(mh.to_vec(), new_bmh.to_vec());

        assert_eq!(mh.to_vec_abunds(), new_mh.to_vec_abunds());
        assert_eq!(bmh.to_vec_abunds(), new_bmh.to_vec_abunds());
        assert_eq!(bmh.to_vec_abunds(), new_mh.to_vec_abunds());
        assert_eq!(mh.to_vec_abunds(), new_bmh.to_vec_abunds());

        assert!(
            (mh.similarity(&new_mh, false, false).unwrap()
                - bmh.similarity(&new_bmh, false, false).unwrap())
            .abs()
                < EPSILON
        );

        assert!(
            (mh.similarity(&new_mh, true, false).unwrap()
                - bmh.similarity(&new_bmh, true, false).unwrap())
            .abs()
                < EPSILON
        );
    }
}

#[test]
fn load_save_minhash_sketches_abund() {
    let mut filename = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    filename.push("../../tests/test-data/gather-abund/reads-s10-s11.sig");

    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let sigs = Signature::from_reader(reader).expect("Loading error");

    let sig = sigs.get(0).unwrap();
    let sketches = sig.sketches();
    let mut buffer = vec![];

    if let Sketch::MinHash(mh) = &sketches[0] {
        let bmh: KmerMinHashBTree = mh.clone().into();
        {
            bmh.to_writer(&mut buffer).unwrap();
        }

        let new_mh = KmerMinHash::from_reader(&buffer[..]).unwrap();
        let new_bmh = KmerMinHashBTree::from_reader(&buffer[..]).unwrap();

        assert_eq!(mh.md5sum(), new_mh.md5sum());
        assert_eq!(bmh.md5sum(), new_bmh.md5sum());
        assert_eq!(bmh.md5sum(), new_mh.md5sum());
        assert_eq!(mh.md5sum(), new_bmh.md5sum());

        assert_eq!(mh.mins(), new_mh.mins());
        assert_eq!(bmh.mins(), new_bmh.mins());
        assert_eq!(bmh.mins(), new_mh.mins());
        assert_eq!(mh.mins(), new_bmh.mins());

        assert_eq!(mh.abunds(), new_mh.abunds());
        assert_eq!(bmh.abunds(), new_bmh.abunds());
        assert_eq!(bmh.abunds(), new_mh.abunds());
        assert_eq!(mh.abunds(), new_bmh.abunds());

        assert_eq!(mh.to_vec(), new_mh.to_vec());
        assert_eq!(bmh.to_vec(), new_bmh.to_vec());
        assert_eq!(bmh.to_vec(), new_mh.to_vec());
        assert_eq!(mh.to_vec(), new_bmh.to_vec());

        assert_eq!(mh.to_vec_abunds(), new_mh.to_vec_abunds());
        assert_eq!(bmh.to_vec_abunds(), new_bmh.to_vec_abunds());
        assert_eq!(bmh.to_vec_abunds(), new_mh.to_vec_abunds());
        assert_eq!(mh.to_vec_abunds(), new_bmh.to_vec_abunds());

        assert!(
            (mh.similarity(&new_mh, false, false).unwrap()
                - bmh.similarity(&new_bmh, false, false).unwrap())
            .abs()
                < EPSILON
        );

        assert!(
            (mh.similarity(&new_mh, true, false).unwrap()
                - bmh.similarity(&new_bmh, true, false).unwrap())
            .abs()
                < EPSILON
        );

        buffer.clear();
        let imh: KmerMinHash = bmh.clone().into();
        {
            imh.to_writer(&mut buffer).unwrap();
        }

        let new_mh = KmerMinHash::from_reader(&buffer[..]).unwrap();
        let new_bmh = KmerMinHashBTree::from_reader(&buffer[..]).unwrap();

        assert_eq!(mh.md5sum(), new_mh.md5sum());
        assert_eq!(bmh.md5sum(), new_bmh.md5sum());
        assert_eq!(bmh.md5sum(), new_mh.md5sum());
        assert_eq!(mh.md5sum(), new_bmh.md5sum());

        assert_eq!(mh.mins(), new_mh.mins());
        assert_eq!(bmh.mins(), new_bmh.mins());
        assert_eq!(bmh.mins(), new_mh.mins());
        assert_eq!(mh.mins(), new_bmh.mins());

        assert_eq!(mh.abunds(), new_mh.abunds());
        assert_eq!(bmh.abunds(), new_bmh.abunds());
        assert_eq!(bmh.abunds(), new_mh.abunds());
        assert_eq!(mh.abunds(), new_bmh.abunds());

        assert!(
            (mh.similarity(&new_mh, false, false).unwrap()
                - bmh.similarity(&new_bmh, false, false).unwrap())
            .abs()
                < EPSILON
        );

        assert!(
            (mh.similarity(&new_mh, true, false).unwrap()
                - bmh.similarity(&new_bmh, true, false).unwrap())
            .abs()
                < EPSILON
        );
    }
}

#[test]
fn merge_empty_scaled() {
    let scaled = 10;
    let mut a = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);
    let mut b = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);

    let c = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);
    let d = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 42, true, 0);

    a.merge(&c).unwrap();
    b.merge(&d).unwrap();

    assert!(a.is_empty());
    assert!(b.is_empty());

    a.add_hash_with_abundance(0, 0);
    assert!(a.is_empty());
    b.add_hash_with_abundance(0, 0);
    assert!(b.is_empty());

    a.clear();
    assert!(a.is_empty());
    b.clear();
    assert!(b.is_empty());
}

#[test]
fn check_errors() {
    let scaled = 10;
    let mut a = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 42, false, 0);
    let mut b = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 42, false, 0);

    // sequence too short: OK
    assert!(a.add_sequence(b"AC", false).is_ok());
    assert!(b.add_sequence(b"AC", false).is_ok());

    // invalid base, throw error
    assert!(a.add_sequence(b"ACTGNN", false).is_err());
    assert!(b.add_sequence(b"ACTGNN", false).is_err());

    a.add_hash(1);
    b.add_hash(1);

    // Can't set abundance after something was inserted
    assert!(a.enable_abundance().is_err());
    assert!(b.enable_abundance().is_err());

    // Can't change hash function after insertion
    assert!(a.set_hash_function(HashFunctions::Murmur64Hp).is_err());
    assert!(b.set_hash_function(HashFunctions::Murmur64Hp).is_err());

    // setting to the same hash function is fine
    assert!(a.set_hash_function(HashFunctions::Murmur64Dna).is_ok());
    assert!(b.set_hash_function(HashFunctions::Murmur64Dna).is_ok());

    let c = KmerMinHash::new(scaled, 7, HashFunctions::Murmur64Dna, 42, true, 0);
    let d = KmerMinHashBTree::new(scaled, 7, HashFunctions::Murmur64Dna, 42, true, 0);

    // different ksize
    assert!(a.check_compatible(&c).is_err());
    assert!(b.check_compatible(&d).is_err());

    let c = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Protein, 42, true, 0);
    let d = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Protein, 42, true, 0);

    // different hash_function
    assert!(a.check_compatible(&c).is_err());
    assert!(b.check_compatible(&d).is_err());

    let c = KmerMinHash::new(scaled, 6, HashFunctions::Murmur64Dna, 31, true, 0);
    let d = KmerMinHashBTree::new(scaled, 6, HashFunctions::Murmur64Dna, 31, true, 0);

    // different seed
    assert!(a.check_compatible(&c).is_err());
    assert!(b.check_compatible(&d).is_err());
}

//fn prop_merge(seq1 in "[ACGT]{6,100}", seq2 in "[ACGT]{6,200}") {

proptest! {
#[test]
fn load_save_minhash_dayhoff(seq in "FLYS*CWLPGQRMTHINKVADER{0,1000}") {
    let scaled = 10;
    let mut a = KmerMinHash::new(scaled, 3, HashFunctions::Murmur64Dayhoff, 42, true, 0);
    let mut b = KmerMinHashBTree::new(scaled, 3, HashFunctions::Murmur64Dayhoff, 42, true, 0);

    a.add_protein(seq.as_bytes()).unwrap();
    b.add_protein(seq.as_bytes()).unwrap();

    let mut buffer_a = vec![];
    let mut buffer_b = vec![];

    {
        a.to_writer(&mut buffer_a).unwrap();
        b.to_writer(&mut buffer_b).unwrap();
    }

    assert_eq!(buffer_a, buffer_b);

    let c = KmerMinHash::from_reader(&buffer_b[..]).unwrap();
    let d = KmerMinHashBTree::from_reader(&buffer_a[..]).unwrap();

    assert!((a.similarity(&c, false, false).unwrap() - b.similarity(&d, false, false).unwrap()).abs() < EPSILON);
    assert!((a.similarity(&c, true, false).unwrap() - b.similarity(&d, true, false).unwrap()).abs() < EPSILON);
}
}

proptest! {
#[test]
fn load_save_minhash_hp(seq in "FLYS*CWLPGQRMTHINKVADER{0,1000}") {
    let scaled = 10;
    let mut a = KmerMinHash::new(scaled, 3, HashFunctions::Murmur64Hp, 42, true, 0);
    let mut b = KmerMinHashBTree::new(scaled, 3, HashFunctions::Murmur64Hp, 42, true, 0);

    a.add_protein(seq.as_bytes()).unwrap();
    b.add_protein(seq.as_bytes()).unwrap();

    let mut buffer_a = vec![];
    let mut buffer_b = vec![];

    {
        a.to_writer(&mut buffer_a).unwrap();
        b.to_writer(&mut buffer_b).unwrap();
    }

    assert_eq!(buffer_a, buffer_b);

    let c = KmerMinHash::from_reader(&buffer_b[..]).unwrap();
    let d = KmerMinHashBTree::from_reader(&buffer_a[..]).unwrap();

    assert!((a.similarity(&c, false, false).unwrap() - b.similarity(&d, false, false).unwrap()).abs() < EPSILON);
    assert!((a.similarity(&c, true, false).unwrap() - b.similarity(&d, true, false).unwrap()).abs() < EPSILON);
}
}

proptest! {
#[test]
fn load_save_minhash_dna(seq in "ACGTN{0,1000}") {
    let scaled = 10;
    let mut a = KmerMinHash::new(scaled, 21, HashFunctions::Murmur64Dna, 42, true, 0);
    let mut b = KmerMinHashBTree::new(scaled, 21, HashFunctions::Murmur64Dna, 42, true, 0);

    a.add_sequence(seq.as_bytes(), true).unwrap();
    b.add_sequence(seq.as_bytes(), true).unwrap();

    let mut buffer_a = vec![];
    let mut buffer_b = vec![];

    {
        a.to_writer(&mut buffer_a).unwrap();
        b.to_writer(&mut buffer_b).unwrap();
    }

    assert_eq!(buffer_a, buffer_b);

    let c = KmerMinHash::from_reader(&buffer_b[..]).unwrap();
    let d = KmerMinHashBTree::from_reader(&buffer_a[..]).unwrap();

    assert!((a.similarity(&c, false, false).unwrap() - b.similarity(&d, false, false).unwrap()).abs() < EPSILON);
    assert!((a.similarity(&c, true, false).unwrap() - b.similarity(&d, true, false).unwrap()).abs() < EPSILON);
}
}

proptest! {
#[test]
fn seq_to_hashes(seq in "ACGTGTAGCTAGACACTGACTGACTGAC") {

    let scaled = 1;
    let mut mh = KmerMinHash::new(scaled, 21, HashFunctions::Murmur64Dna, 42, true, 0);
    mh.add_sequence(seq.as_bytes(), false)?; // .unwrap();

    let mut hashes: Vec<u64> = Vec::new();

    let ready_hashes = SeqToHashes::new(seq.as_bytes(), mh.ksize(), false, false, mh.hash_function(), mh.seed())?;

    for hash_value in ready_hashes{
        match hash_value{
            Ok(0) => continue,
            Ok(x) => hashes.push(x),
            Err(_) => (),
        }
    }

    mh.mins().sort_unstable();
    hashes.sort_unstable();
    assert_eq!(mh.mins(), hashes);

}

#[test]
fn seq_to_hashes_2(seq in "QRMTHINK") {

    let scaled = 1;
    let mut mh = KmerMinHash::new(scaled, 3, HashFunctions::Murmur64Protein, 42, true, 0);
    mh.add_protein(seq.as_bytes())?; // .unwrap();

    let mut hashes: Vec<u64> = Vec::new();

    let ready_hashes = SeqToHashes::new(seq.as_bytes(), mh.ksize(), false, true, mh.hash_function(), mh.seed())?;

    for hash_value in ready_hashes {
        match hash_value{
            Ok(0) => continue,
            Ok(x) => hashes.push(x),
            Err(_) => (),
        }
    }

    mh.mins().sort_unstable();
    hashes.sort_unstable();
    assert_eq!(mh.mins(), hashes);

}

}

#[test]
fn test_inflate() {
    // Setup minhash_a with some mins but no abundances
    let mut a = KmerMinHash::new(5, 3, HashFunctions::Murmur64Hp, 42, true, 0);
    a.add_hash(10);
    a.add_hash(20);
    a.add_hash(30);

    // Setup minhash_b with mins, some of which match a, and with abundances
    let mut b = KmerMinHash::new(5, 3, HashFunctions::Murmur64Hp, 42, true, 0);
    b.add_hash_with_abundance(10, 2);
    b.add_hash_with_abundance(20, 4);
    b.add_hash_with_abundance(40, 6); // Non-matching hash

    // Attempt to inflate minhash_a using minhash_b's abundances
    assert!(a.inflate(&b).is_ok());

    a.inflate(&b).unwrap();
    eprintln!("{:?}", a.to_vec_abunds());
    assert_eq!(a.to_vec_abunds(), vec![(10, 2), (20, 4)]);
}

#[test]
fn test_inflated_abundances() {
    // Setup minhash_a with some mins but no abundances
    let mut a = KmerMinHash::new(5, 3, HashFunctions::Murmur64Hp, 42, false, 0);
    a.add_hash(10);
    a.add_hash(20);
    a.add_hash(30);

    // Setup minhash_b with mins, some of which match a, and with abundances
    let mut b = KmerMinHash::new(5, 3, HashFunctions::Murmur64Hp, 42, true, 0);
    b.add_hash_with_abundance(10, 2);
    b.add_hash_with_abundance(20, 4);
    b.add_hash_with_abundance(40, 9); // Non-matching hash

    // Attempt to inflate minhash_a using minhash_b's abundances
    assert!(a.inflate(&b).is_ok());

    let (abunds, total_abund) = a.inflated_abundances(&b).unwrap();
    assert_eq!(abunds, vec![2, 4]);
    assert_eq!(total_abund, 6);
}

#[test]
fn test_inflate_noabund() {
    // Setup minhash a with some mins but no abundances
    let mut a = KmerMinHash::new(5, 3, HashFunctions::Murmur64Dna, 42, false, 0);
    a.add_hash(10);
    a.add_hash(20);
    a.add_hash(30);
    let b = a.clone();
    let result = a.inflate(&b);
    assert!(matches!(
        result,
        Err(sourmash::Error::NeedsAbundanceTracking)
    ));
}

#[test]
fn test_inflated_abunds_noabund() {
    // Setup minhash a with some mins but no abundances
    let mut a = KmerMinHash::new(5, 3, HashFunctions::Murmur64Dna, 42, false, 0);
    a.add_hash(10);
    a.add_hash(20);
    a.add_hash(30);
    let result = a.inflated_abundances(&a);
    assert!(matches!(
        result,
        Err(sourmash::Error::NeedsAbundanceTracking)
    ));
}

#[test]
fn test_sum_abunds() {
    let mut a = KmerMinHash::new(5, 3, HashFunctions::Murmur64Dna, 42, true, 0);
    a.add_hash_with_abundance(10, 2);
    a.add_hash_with_abundance(20, 4);
    a.add_hash_with_abundance(40, 9);
    assert_eq!(a.sum_abunds(), 15);
}

#[test]
fn test_sum_abunds_noabund() {
    let mut a = KmerMinHash::new(5, 3, HashFunctions::Murmur64Dna, 42, false, 0);
    a.add_hash(10);
    a.add_hash(20);
    a.add_hash(30);
    assert_eq!(a.sum_abunds(), 3);
}

#[test]
fn test_n_unique_kmers() {
    let mut mh = KmerMinHash::new(10, 21, HashFunctions::Murmur64Dna, 42, true, 0);
    mh.add_hash(10);
    mh.add_hash(20);
    mh.add_hash(30);
    assert_eq!(mh.n_unique_kmers(), 30)
}

#[test]
fn test_scaled_downsampling_kmerminhash() {
    let mh = KmerMinHash::new(10, 21, HashFunctions::Murmur64Dna, 42, true, 0);

    // downsampling to same scaled is OK:
    let new_mh = mh.clone().downsample_scaled(10).unwrap();
    assert_eq!(new_mh.scaled(), 10);

    // downsampling is OK:
    let new_mh = mh.clone().downsample_scaled(100).unwrap();
    assert_eq!(new_mh.scaled(), 100);

    // upsampling not ok
    let e = mh.clone().downsample_scaled(1).unwrap_err();
    assert!(matches!(e, sourmash::Error::CannotUpsampleScaled));
}

#[test]
fn test_scaled_downsampling_kmerminhashbtree() {
    let mh = KmerMinHashBTree::new(10, 21, HashFunctions::Murmur64Dna, 42, true, 0);

    // downsampling to same scaled is OK:
    let new_mh = mh.clone().downsample_scaled(10).unwrap();
    assert_eq!(new_mh.scaled(), 10);

    // downsampling is OK:
    let new_mh = mh.clone().downsample_scaled(100).unwrap();
    assert_eq!(new_mh.scaled(), 100);

    // upsampling not ok
    let e = mh.clone().downsample_scaled(1).unwrap_err();
    assert!(matches!(e, sourmash::Error::CannotUpsampleScaled));
}
