//! # Compute, compare and search signatures for nucleotide (DNA/RNA) and protein sequences.
//!
//! sourmash is a command-line tool and Python library for computing
//! [MinHash sketches][0] from DNA sequences, comparing them to each other,
//! and plotting the results.
//! This allows you to estimate sequence similarity between even very
//! large data sets quickly and accurately.
//!
//! [0]: https://en.wikipedia.org/wiki/MinHash
//!
//! sourmash can be used to quickly search large databases of genomes
//! for matches to query genomes and metagenomes.
//!
//! sourmash also includes k-mer based taxonomic exploration and
//! classification routines for genome and metagenome analysis. These
//! routines can use the NCBI taxonomy but do not depend on it in any way.
//! Documentation and further examples for each module can be found in the module descriptions below.

// TODO: remove this line and update all the appropriate type names for 1.0
#![allow(clippy::upper_case_acronyms)]

pub mod errors;
pub use errors::SourmashError as Error;
pub type Result<T> = std::result::Result<T, Error>;

pub mod prelude;

pub mod cmd;

pub mod ani_utils;
pub mod collection;
pub mod encodings;
pub mod index;
pub mod manifest;
pub mod selection;
pub mod signature;
pub mod sketch;
pub mod storage;

use cfg_if::cfg_if;
use murmurhash3::murmurhash3_x64_128;

cfg_if! {
    if #[cfg(all(target_arch = "wasm32", target_os = "unknown"))] {
        // Explicitly keeping emscripten and wasi out of this
        pub mod wasm;
    } else {
        pub mod ffi;
    }
}

type HashIntoType = u64;
pub type ScaledType = u32;

pub fn _hash_murmur(kmer: &[u8], seed: u64) -> u64 {
    murmurhash3_x64_128(kmer, seed).0
}
