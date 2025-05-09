use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::fmt::Write;
use std::io;
use std::iter::Peekable;
use std::str;
use std::sync::Mutex;

use itertools::Itertools;
use serde::de::Deserializer;
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::encodings::HashFunctions;
use crate::prelude::ToWriter;
use crate::signature::SigsTrait;
use crate::sketch::hyperloglog::HyperLogLog;
use crate::Error;
use crate::{ScaledType, _hash_murmur};

pub fn max_hash_for_scaled(scaled: ScaledType) -> u64 {
    match scaled {
        0 => 0, // scaled == 0 indicates this is a num minhash
        1 => u64::MAX,
        _ => (u64::MAX as f64 / scaled as f64) as u64,
    }
}

pub fn scaled_for_max_hash(max_hash: u64) -> ScaledType {
    match max_hash {
        0 => 0, // scaled == 0 indicates this is a num minhash
        _ => (u64::MAX as f64 / max_hash as f64) as ScaledType,
    }
}

#[derive(Debug, TypedBuilder)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Serialize, rkyv::Deserialize, rkyv::Archive)
)]
pub struct KmerMinHash {
    num: u32,
    ksize: u32,

    #[builder(setter(into), default = HashFunctions::Murmur64Dna)]
    hash_function: HashFunctions,

    #[builder(default = 42u64)]
    seed: u64,

    #[builder(default = u64::MAX)]
    max_hash: u64,

    #[builder(default)]
    mins: Vec<u64>,

    #[builder(default)]
    abunds: Option<Vec<u64>>,

    #[builder(default)]
    #[cfg_attr(feature = "rkyv", with(rkyv::with::Skip))]
    md5sum: Mutex<Option<String>>,
}

impl PartialEq for KmerMinHash {
    fn eq(&self, other: &KmerMinHash) -> bool {
        // TODO: check all other fields?
        self.md5sum() == other.md5sum()
    }
}

impl Clone for KmerMinHash {
    fn clone(&self) -> Self {
        KmerMinHash {
            num: self.num,
            ksize: self.ksize,
            hash_function: self.hash_function.clone(),
            seed: self.seed,
            max_hash: self.max_hash,
            mins: self.mins.clone(),
            abunds: self.abunds.clone(),
            md5sum: Mutex::new(Some(self.md5sum())),
        }
    }
}

impl Default for KmerMinHash {
    fn default() -> KmerMinHash {
        KmerMinHash {
            num: 1000,
            ksize: 21,
            hash_function: HashFunctions::Murmur64Dna,
            seed: 42,
            max_hash: 0,
            mins: Vec::with_capacity(1000),
            abunds: None,
            md5sum: Mutex::new(None),
        }
    }
}

impl Serialize for KmerMinHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let n_fields = match &self.abunds {
            Some(_) => 8,
            _ => 7,
        };

        let mut partial = serializer.serialize_struct("KmerMinHash", n_fields)?;
        partial.serialize_field("num", &self.num)?;
        partial.serialize_field("ksize", &self.ksize)?;
        partial.serialize_field("seed", &self.seed)?;
        partial.serialize_field("max_hash", &self.max_hash)?;
        partial.serialize_field("mins", &self.mins)?;
        partial.serialize_field("md5sum", &self.md5sum())?;

        if let Some(abunds) = &self.abunds {
            partial.serialize_field("abundances", abunds)?;
        }

        partial.serialize_field("molecule", &self.hash_function.to_string())?;

        partial.end()
    }
}

impl<'de> Deserialize<'de> for KmerMinHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TempSig {
            num: u32,
            ksize: u32,
            seed: u64,
            max_hash: u64,
            md5sum: String,
            mins: Vec<u64>,
            abundances: Option<Vec<u64>>,
            molecule: String,
        }

        let tmpsig = TempSig::deserialize(deserializer)?;

        let num = if tmpsig.max_hash != 0 { 0 } else { tmpsig.num };

        // Set the hash function based on the molecule string. This will panic if
        // the molecule string is not a valid.
        let hash_function =
            HashFunctions::try_from(tmpsig.molecule.as_str()).map_err(serde::de::Error::custom)?;

        // This shouldn't be necessary, but at some point we
        // created signatures with unordered mins =(
        let (mins, abunds) = if let Some(abunds) = tmpsig.abundances {
            let mut values: Vec<(_, _)> = tmpsig.mins.iter().zip(abunds.iter()).collect();
            values.sort();
            let mins = values.iter().map(|(v, _)| **v).collect();
            let abunds = values.iter().map(|(_, v)| **v).collect();
            (mins, Some(abunds))
        } else {
            let mut values: Vec<_> = tmpsig.mins.into_iter().collect();
            values.sort_unstable();
            (values, None)
        };

        Ok(KmerMinHash {
            num,
            ksize: tmpsig.ksize,
            seed: tmpsig.seed,
            max_hash: tmpsig.max_hash,
            md5sum: Mutex::new(Some(tmpsig.md5sum)),
            mins,
            abunds,
            hash_function,
        })
    }
}

impl ToWriter for KmerMinHash {
    fn to_writer<W>(&self, writer: &mut W) -> Result<(), Error>
    where
        W: io::Write,
    {
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }
}

impl KmerMinHash {
    pub fn new(
        scaled: ScaledType,
        ksize: u32,
        hash_function: HashFunctions,
        seed: u64,
        track_abundance: bool,
        num: u32,
    ) -> KmerMinHash {
        let mins = if num > 0 {
            Vec::with_capacity(num as usize)
        } else {
            Vec::with_capacity(1000)
        };

        let abunds = if track_abundance {
            Some(Vec::with_capacity(mins.capacity()))
        } else {
            None
        };

        let max_hash = max_hash_for_scaled(scaled);

        KmerMinHash {
            num,
            ksize,
            hash_function,
            seed,
            max_hash,
            mins,
            abunds,
            md5sum: Mutex::new(None),
        }
    }

    pub fn num(&self) -> u32 {
        self.num
    }

    pub fn is_protein(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Protein
    }

    pub fn max_hash(&self) -> u64 {
        self.max_hash
    }

    pub fn scaled(&self) -> ScaledType {
        scaled_for_max_hash(self.max_hash)
    }

    pub fn clear(&mut self) {
        self.mins.clear();
        if let Some(ref mut abunds) = self.abunds {
            abunds.clear();
        }
    }

    pub fn is_empty(&self) -> bool {
        self.mins.is_empty()
    }

    pub fn set_hash_function(&mut self, h: HashFunctions) -> Result<(), Error> {
        if self.hash_function == h {
            return Ok(());
        }

        if !self.is_empty() {
            return Err(Error::NonEmptyMinHash {
                message: "hash_function".into(),
            });
        }

        self.hash_function = h;
        Ok(())
    }

    pub fn track_abundance(&self) -> bool {
        self.abunds.is_some()
    }

    pub fn enable_abundance(&mut self) -> Result<(), Error> {
        if !self.mins.is_empty() {
            return Err(Error::NonEmptyMinHash {
                message: "track_abundance=True".into(),
            });
        }

        self.abunds = Some(vec![]);

        Ok(())
    }

    pub fn disable_abundance(&mut self) {
        self.abunds = None;
    }

    fn reset_md5sum(&self) {
        let mut data = self.md5sum.lock().unwrap();
        if data.is_some() {
            *data = None;
        }
    }

    pub fn md5sum(&self) -> String {
        let mut data = self.md5sum.lock().unwrap();
        if data.is_none() {
            let mut buffer = String::with_capacity(20);

            let mut md5_ctx = md5::Context::new();
            write!(&mut buffer, "{}", self.ksize()).unwrap();
            md5_ctx.consume(&buffer);
            buffer.clear();
            for x in &self.mins {
                write!(&mut buffer, "{}", x).unwrap();
                md5_ctx.consume(&buffer);
                buffer.clear();
            }
            *data = Some(format!("{:x}", md5_ctx.compute()));
        }
        data.clone().unwrap()
    }

    pub fn add_hash(&mut self, hash: u64) {
        self.add_hash_with_abundance(hash, 1);
    }

    pub fn add_hash_with_abundance(&mut self, hash: u64, abundance: u64) {
        let current_max = match self.mins.last() {
            Some(&x) => x,
            None => u64::MAX,
        };

        if hash > self.max_hash && self.max_hash != 0 {
            // This is a scaled minhash, and we don't need to add the new hash
            return;
        }

        if self.num == 0 && self.max_hash == 0 {
            // why did you create this minhash? it will always be empty...
            return;
        }

        if abundance == 0 {
            self.remove_hash(hash);
            return;
        }

        // From this point on, hash is within scaled (or no scaled specified).

        // empty mins? add it.
        if self.mins.is_empty() {
            self.mins.push(hash);
            if let Some(ref mut abunds) = self.abunds {
                abunds.push(abundance);
                self.reset_md5sum();
            }
            return;
        }

        if hash <= self.max_hash || hash <= current_max || (self.mins.len() as u32) < self.num {
            // "good" hash - within range, smaller than current entry, or
            // still have space available
            let pos = match self.mins.binary_search(&hash) {
                Ok(p) => p,
                Err(p) => p,
            };

            if pos == self.mins.len() {
                // at end - must still be growing, we know the list won't
                // get too long
                self.mins.push(hash);
                self.reset_md5sum();
                if let Some(ref mut abunds) = self.abunds {
                    abunds.push(abundance);
                }
            } else if self.mins[pos] != hash {
                // didn't find hash in mins, so inserting somewhere
                // in the middle; shrink list if needed.
                self.mins.insert(pos, hash);
                if let Some(ref mut abunds) = self.abunds {
                    abunds.insert(pos, abundance);
                }

                // is it too big now?
                if self.num != 0 && self.mins.len() > (self.num as usize) {
                    self.mins.pop();
                    if let Some(ref mut abunds) = self.abunds {
                        abunds.pop();
                    }
                }
                self.reset_md5sum();
            } else if let Some(ref mut abunds) = self.abunds {
                // pos == hash: hash value already in mins, inc count by abundance
                abunds[pos] += abundance;
            }
        }
    }

    pub fn set_hash_with_abundance(&mut self, hash: u64, abundance: u64) {
        let mut found = false;
        if let Ok(pos) = self.mins.binary_search(&hash) {
            if self.mins[pos] == hash {
                found = true;
                if let Some(ref mut abunds) = self.abunds {
                    abunds[pos] = abundance;
                }
            }
        }

        if !found {
            self.add_hash_with_abundance(hash, abundance);
        }
    }

    pub fn add_word(&mut self, word: &[u8]) {
        let hash = _hash_murmur(word, self.seed);
        self.add_hash(hash);
    }

    pub fn remove_hash(&mut self, hash: u64) {
        if let Ok(pos) = self.mins.binary_search(&hash) {
            if self.mins[pos] == hash {
                self.mins.remove(pos);
                self.reset_md5sum();
                if let Some(ref mut abunds) = self.abunds {
                    abunds.remove(pos);
                }
            }
        };
    }

    pub fn remove_from(&mut self, other: &KmerMinHash) -> Result<(), Error> {
        for min in &other.mins {
            self.remove_hash(*min);
        }
        Ok(())
    }

    pub fn remove_many<T: IntoIterator<Item = u64>>(&mut self, hashes: T) -> Result<(), Error> {
        for min in hashes {
            self.remove_hash(min);
        }
        Ok(())
    }

    pub fn merge(&mut self, other: &KmerMinHash) -> Result<(), Error> {
        self.check_compatible(other)?;
        let max_size = self.mins.len() + other.mins.len();

        let mut merged: Vec<u64> = Vec::with_capacity(max_size);
        let mut merged_abunds: Option<Vec<u64>> = if self.abunds.is_some() && other.abunds.is_some()
        {
            Some(Vec::with_capacity(max_size))
        } else {
            None
        };

        let mut self_iter = self.mins.iter();
        let mut other_iter = other.mins.iter();

        let mut self_abunds_iter = self.abunds.iter().flatten();
        let mut other_abunds_iter = other.abunds.iter().flatten();

        let mut self_value = self_iter.next();
        let mut other_value = other_iter.next();
        while self_value.is_some() {
            let value = self_value.unwrap();
            match other_value {
                None => {
                    merged.push(*value);
                    merged.extend(self_iter);
                    if let Some(v) = merged_abunds.as_mut() {
                        v.extend(self_abunds_iter)
                    }
                    break;
                }
                Some(x) if x < value => {
                    merged.push(*x);
                    other_value = other_iter.next();
                    if let Some(v) = other_abunds_iter.next() {
                        if let Some(n) = merged_abunds.as_mut() {
                            n.push(*v)
                        }
                    }
                }
                Some(x) if x == value => {
                    merged.push(*x);
                    other_value = other_iter.next();
                    self_value = self_iter.next();

                    if let (Some(v), Some(s)) = (other_abunds_iter.next(), self_abunds_iter.next())
                    {
                        if let Some(n) = merged_abunds.as_mut() {
                            n.push(*v + *s)
                        }
                    }
                }
                Some(x) if x > value => {
                    merged.push(*value);
                    self_value = self_iter.next();

                    if let Some(v) = self_abunds_iter.next() {
                        if let Some(n) = merged_abunds.as_mut() {
                            n.push(*v)
                        }
                    }
                }
                Some(_) => {}
            }
        }
        if let Some(value) = other_value {
            merged.push(*value);
        }
        merged.extend(other_iter);
        if let Some(n) = merged_abunds.as_mut() {
            n.extend(other_abunds_iter)
        }

        if merged.len() > (self.num as usize) && (self.num as usize) != 0 {
            merged.truncate(self.num as usize);
            if let Some(v) = merged_abunds.as_mut() {
                v.truncate(self.num as usize)
            }
        }
        self.mins = merged;
        self.abunds = merged_abunds;

        self.reset_md5sum();
        Ok(())
    }

    pub fn add_from(&mut self, other: &KmerMinHash) -> Result<(), Error> {
        for min in &other.mins {
            self.add_hash(*min);
        }
        Ok(())
    }

    pub fn add_many(&mut self, hashes: &[u64]) -> Result<(), Error> {
        for min in hashes {
            self.add_hash(*min);
        }
        Ok(())
    }

    pub fn add_many_with_abund(&mut self, hashes: &[(u64, u64)]) -> Result<(), Error> {
        for item in hashes {
            self.add_hash_with_abundance(item.0, item.1);
        }
        Ok(())
    }

    pub fn count_common(&self, other: &KmerMinHash, downsample: bool) -> Result<u64, Error> {
        if downsample && self.scaled() != other.scaled() {
            let (first, second) = if self.scaled() > other.scaled() {
                (self, other)
            } else {
                (other, self)
            };
            let downsampled_mh = second.clone().downsample_scaled(first.scaled())?;
            first.count_common(&downsampled_mh, false)
        } else {
            self.check_compatible(other)?;
            let iter = if self.size() < other.size() {
                Intersection::new(self.mins.iter(), other.mins.iter())
            } else {
                Intersection::new(other.mins.iter(), self.mins.iter())
            };

            Ok(iter.count() as u64)
        }
    }

    pub fn intersection(&self, other: &KmerMinHash) -> Result<(Vec<u64>, u64), Error> {
        self.check_compatible(other)?;

        if self.num != 0 {
            // Intersection for regular MinHash sketches
            let mut combined_mh = KmerMinHash::new(
                self.scaled(),
                self.ksize,
                self.hash_function.clone(),
                self.seed,
                self.abunds.is_some(),
                self.num,
            );

            combined_mh.merge(self)?;
            combined_mh.merge(other)?;

            let it1 = Intersection::new(self.mins.iter(), other.mins.iter());

            // TODO: there is probably a way to avoid this Vec here,
            // and pass the it1 as left in it2.
            let i1: Vec<u64> = it1.cloned().collect();
            let it2 = Intersection::new(i1.iter(), combined_mh.mins.iter());

            let common: Vec<u64> = it2.cloned().collect();
            Ok((common, combined_mh.mins.len() as u64))
        } else {
            Ok(intersection(self.mins.iter(), other.mins.iter()))
        }
    }

    // FIXME: intersection_size and count_common should be the same?
    // (for scaled minhashes)
    pub fn intersection_size(&self, other: &KmerMinHash) -> Result<(u64, u64), Error> {
        self.check_compatible(other)?;

        if self.num != 0 {
            // Intersection for regular MinHash sketches
            let mut combined_mh = KmerMinHash::new(
                self.scaled(),
                self.ksize,
                self.hash_function.clone(),
                self.seed,
                self.abunds.is_some(),
                self.num,
            );

            combined_mh.merge(self)?;
            combined_mh.merge(other)?;

            let it1 = Intersection::new(self.mins.iter(), other.mins.iter());

            // TODO: there is probably a way to avoid this Vec here,
            // and pass the it1 as left in it2.
            let i1: Vec<u64> = it1.cloned().collect();
            let it2 = Intersection::new(i1.iter(), combined_mh.mins.iter());

            Ok((it2.count() as u64, combined_mh.mins.len() as u64))
        } else {
            Ok(intersection_size(self.mins.iter(), other.mins.iter()))
        }
    }

    // calculate Jaccard similarity, ignoring abundance.
    pub fn jaccard(&self, other: &KmerMinHash) -> Result<f64, Error> {
        self.check_compatible(other)?;
        if let Ok((common, size)) = self.intersection_size(other) {
            Ok(common as f64 / u64::max(1, size) as f64)
        } else {
            Ok(0.0)
        }
    }

    // compare two minhashes, with abundance;
    // calculate their angular similarity.
    pub fn angular_similarity(&self, other: &KmerMinHash) -> Result<f64, Error> {
        self.check_compatible(other)?;

        if self.abunds.is_none() || other.abunds.is_none() {
            return Err(Error::NeedsAbundanceTracking);
        }

        // TODO: check which one is smaller, swap around if needed

        let abunds = self.abunds.as_ref().unwrap();
        let other_abunds = other.abunds.as_ref().unwrap();

        let mut prod = 0;
        let mut other_iter = other.mins.iter().enumerate();
        let mut next_hash = other_iter.next();
        let a_sq: u64 = abunds.iter().map(|a| (a * a)).sum();
        let b_sq: u64 = other_abunds.iter().map(|a| (a * a)).sum();

        for (i, hash) in self.mins.iter().enumerate() {
            while let Some((j, k)) = next_hash {
                match k.cmp(hash) {
                    Ordering::Less => next_hash = other_iter.next(),
                    Ordering::Equal => {
                        // Calling `get_unchecked` here is safe since
                        // both `i` and `j` are valid indices
                        // (`i` and `j` came from valid iterator calls)
                        unsafe {
                            prod += abunds.get_unchecked(i) * other_abunds.get_unchecked(j);
                        }
                        break;
                    }
                    Ordering::Greater => break,
                }
            }
        }

        let norm_a = (a_sq as f64).sqrt();
        let norm_b = (b_sq as f64).sqrt();

        if norm_a == 0. || norm_b == 0. {
            return Ok(0.0);
        }
        let prod = f64::min(prod as f64 / (norm_a * norm_b), 1.);
        let distance = 2. * prod.acos() / PI;
        Ok(1. - distance)
    }

    pub fn similarity(
        &self,
        other: &KmerMinHash,
        ignore_abundance: bool,
        downsample: bool,
    ) -> Result<f64, Error> {
        if downsample && self.scaled() != other.scaled() {
            // downsample to larger of two scaled
            let (first, second) = if self.scaled() > other.scaled() {
                (self, other)
            } else {
                (other, self)
            };
            let downsampled_mh = second.clone().downsample_scaled(first.scaled())?;
            first.similarity(&downsampled_mh, ignore_abundance, false)
        } else if ignore_abundance || self.abunds.is_none() || other.abunds.is_none() {
            self.jaccard(other)
        } else {
            self.angular_similarity(other)
        }
    }

    pub fn dayhoff(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Dayhoff
    }

    pub fn hp(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Hp
    }

    pub fn skipm1n3(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Skipm1n3
    }

    pub fn skipm2n3(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Skipm2n3
    }

    pub fn mins(&self) -> Vec<u64> {
        self.mins.clone()
    }

    pub fn iter_mins(&self) -> impl Iterator<Item = &u64> {
        self.mins.iter()
    }

    pub fn abunds(&self) -> Option<Vec<u64>> {
        self.abunds.clone()
    }

    // create a downsampled copy of self
    pub fn downsample_max_hash(self, max_hash: u64) -> Result<KmerMinHash, Error> {
        if self.max_hash == 0 {
            // CTB: this is a num minhash. Should we just blithely return?
            Ok(self)
        } else {
            let scaled = scaled_for_max_hash(max_hash);
            self.downsample_scaled(scaled)
        }
    }

    pub fn sum_abunds(&self) -> u64 {
        if let Some(abunds) = &self.abunds {
            abunds.iter().sum()
        } else {
            self.size() as u64
        }
    }

    pub fn to_vec_abunds(&self) -> Vec<(u64, u64)> {
        if let Some(abunds) = &self.abunds {
            self.mins
                .iter()
                .cloned()
                .zip(abunds.iter().cloned())
                .collect()
        } else {
            self.mins
                .iter()
                .cloned()
                .zip(std::iter::repeat(1))
                .collect()
        }
    }

    pub fn as_hll(&self) -> HyperLogLog {
        let mut hll = HyperLogLog::with_error_rate(0.01, self.ksize()).unwrap();

        for h in &self.mins {
            hll.add_hash(*h)
        }

        hll
    }

    // Approximate total number of kmers
    // this could be improved by generating an HLL estimate while sketching instead
    // (for scaled minhashes)
    pub fn n_unique_kmers(&self) -> u64 {
        self.size() as u64 * self.scaled() as u64 // + (self.ksize - 1) for bp estimation
    }

    // create a downsampled copy of self
    pub fn downsample_scaled(self, scaled: ScaledType) -> Result<KmerMinHash, Error> {
        if self.scaled() == scaled || self.scaled() == 0 {
            Ok(self)
        } else if self.scaled() > scaled {
            Err(Error::CannotUpsampleScaled)
        } else {
            let mut new_mh = KmerMinHash::new(
                scaled,
                self.ksize,
                self.hash_function.clone(),
                self.seed,
                self.abunds.is_some(),
                self.num,
            );
            if self.abunds.is_some() {
                new_mh.add_many_with_abund(&self.to_vec_abunds())?;
            } else {
                new_mh.add_many(&self.mins)?;
            }
            Ok(new_mh)
        }
    }

    pub fn inflate(&mut self, abunds_from: &KmerMinHash) -> Result<(), Error> {
        self.check_compatible(abunds_from)?;

        // check that abunds_from has abundances
        if abunds_from.abunds.is_none() {
            return Err(Error::NeedsAbundanceTracking);
        }

        let self_iter = self.mins.iter();
        let abunds_iter = abunds_from.abunds.as_ref().unwrap().iter();
        let abunds_from_iter = abunds_from.mins.iter().zip(abunds_iter);

        let (mins, abunds): (Vec<u64>, Vec<u64>) = self_iter
            .merge_join_by(abunds_from_iter, |&self_val, &(other_val, _)| {
                self_val.cmp(other_val)
            })
            .filter_map(|either| match either {
                itertools::EitherOrBoth::Both(self_val, (_other_val, &other_abund)) => {
                    Some((self_val, other_abund))
                }
                _ => None,
            })
            .unzip();

        self.mins = mins;
        self.abunds = Some(abunds);

        self.reset_md5sum();
        Ok(())
    }

    pub fn inflated_abundances(&self, abunds_from: &KmerMinHash) -> Result<(Vec<u64>, u64), Error> {
        self.check_compatible(abunds_from)?;
        // check that abunds_from has abundances
        if abunds_from.abunds.is_none() {
            return Err(Error::NeedsAbundanceTracking);
        }

        let self_iter = self.mins.iter();
        let abunds_iter = abunds_from.abunds.as_ref().unwrap().iter();
        let abunds_from_iter = abunds_from.mins.iter().zip(abunds_iter);

        let (abundances, total_abundance): (Vec<u64>, u64) = self_iter
            .merge_join_by(abunds_from_iter, |&self_val, &(other_val, _)| {
                self_val.cmp(other_val)
            })
            .filter_map(|either| match either {
                itertools::EitherOrBoth::Both(_self_val, (_other_val, other_abund)) => {
                    Some(*other_abund)
                }
                _ => None,
            })
            .fold((Vec::new(), 0u64), |(mut acc_vec, acc_sum), abund| {
                acc_vec.push(abund);
                (acc_vec, acc_sum + abund)
            });

        Ok((abundances, total_abundance))
    }

    pub fn from_reader<R>(rdr: R) -> Result<KmerMinHash, Error>
    where
        R: std::io::Read,
    {
        let (rdr, _format) = niffler::get_reader(Box::new(rdr))?;

        let mh: KmerMinHash = serde_json::from_reader(rdr)?;
        Ok(mh)
    }
}

impl SigsTrait for KmerMinHash {
    fn size(&self) -> usize {
        self.mins.len()
    }

    fn to_vec(&self) -> Vec<u64> {
        self.mins.clone()
    }

    fn ksize(&self) -> usize {
        self.ksize as usize
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn hash_function(&self) -> HashFunctions {
        self.hash_function.clone()
    }

    fn add_hash(&mut self, hash: u64) {
        self.add_hash_with_abundance(hash, 1);
    }

    fn check_compatible(&self, other: &KmerMinHash) -> Result<(), Error> {
        /*
        if self.num != other.num {
            return Err(Error::MismatchNum {
                n1: self.num,
                n2: other.num,
            }
            .into());
        }
        */
        if self.ksize != other.ksize {
            return Err(Error::MismatchKSizes);
        }
        if self.hash_function != other.hash_function {
            // TODO: fix this error
            return Err(Error::MismatchDNAProt);
        }
        // TODO: if supporting downsampled to be compatible
        //if self.max_hash < other.max_hash {
        if self.max_hash != other.max_hash {
            return Err(Error::MismatchScaled);
        }
        if self.seed != other.seed {
            return Err(Error::MismatchSeed);
        }
        Ok(())
    }
}

struct Intersection<T, I: Iterator<Item = T>> {
    iter: Peekable<I>,
    other: Peekable<I>,
}

impl<T, I: Iterator<Item = T>> Intersection<T, I> {
    pub fn new(left: I, right: I) -> Self {
        Intersection {
            iter: left.peekable(),
            other: right.peekable(),
        }
    }
}

impl<T: Ord, I: Iterator<Item = T>> Iterator for Intersection<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            let res = match (self.iter.peek(), self.other.peek()) {
                (Some(ref left_key), Some(ref right_key)) => left_key.cmp(right_key),
                _ => return None,
            };

            match res {
                Ordering::Less => {
                    self.iter.next();
                }
                Ordering::Greater => {
                    self.other.next();
                }
                Ordering::Equal => {
                    self.other.next();
                    return self.iter.next();
                }
            }
        }
    }
}

//#############
// A MinHash implementation for low scaled or large cardinalities

#[derive(Debug, TypedBuilder)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Serialize, rkyv::Deserialize, rkyv::Archive)
)]
pub struct KmerMinHashBTree {
    num: u32,
    ksize: u32,

    #[builder(setter(into), default = HashFunctions::Murmur64Dna)]
    hash_function: HashFunctions,

    #[builder(default = 42u64)]
    seed: u64,

    #[builder(default = u64::MAX)]
    max_hash: u64,

    #[builder(default)]
    mins: BTreeSet<u64>,

    #[builder(default)]
    abunds: Option<BTreeMap<u64, u64>>,

    #[builder(default = 0u64)]
    current_max: u64,

    #[builder(default)]
    #[cfg_attr(feature = "rkyv", with(rkyv::with::Skip))]
    md5sum: Mutex<Option<String>>,
}

impl PartialEq for KmerMinHashBTree {
    fn eq(&self, other: &KmerMinHashBTree) -> bool {
        // TODO: check all other fields?
        self.md5sum() == other.md5sum()
    }
}

impl Clone for KmerMinHashBTree {
    fn clone(&self) -> Self {
        KmerMinHashBTree {
            num: self.num,
            ksize: self.ksize,
            hash_function: self.hash_function.clone(),
            seed: self.seed,
            max_hash: self.max_hash,
            mins: self.mins.clone(),
            abunds: self.abunds.clone(),
            current_max: self.current_max,
            md5sum: Mutex::new(self.md5sum.lock().unwrap().clone()),
        }
    }
}

impl Default for KmerMinHashBTree {
    fn default() -> KmerMinHashBTree {
        KmerMinHashBTree {
            num: 1000,
            ksize: 21,
            hash_function: HashFunctions::Murmur64Dna,
            seed: 42,
            max_hash: 0,
            mins: Default::default(),
            abunds: None,
            current_max: 0,
            md5sum: Mutex::new(None),
        }
    }
}

impl Serialize for KmerMinHashBTree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let n_fields = match &self.abunds {
            Some(_) => 8,
            _ => 7,
        };

        let mut partial = serializer.serialize_struct("KmerMinHashBTree", n_fields)?;
        partial.serialize_field("num", &self.num)?;
        partial.serialize_field("ksize", &self.ksize)?;
        partial.serialize_field("seed", &self.seed)?;
        partial.serialize_field("max_hash", &self.max_hash)?;
        partial.serialize_field("mins", &self.mins)?;
        partial.serialize_field("md5sum", &self.md5sum())?;

        if let Some(abunds) = &self.abunds {
            let abs: Vec<u64> = abunds.values().cloned().collect();
            partial.serialize_field("abundances", &abs)?;
        }

        partial.serialize_field("molecule", &self.hash_function.to_string())?;

        partial.end()
    }
}

impl<'de> Deserialize<'de> for KmerMinHashBTree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TempSig {
            num: u32,
            ksize: u32,
            seed: u64,
            max_hash: u64,
            md5sum: String,
            mins: Vec<u64>,
            abundances: Option<Vec<u64>>,
            molecule: String,
        }

        let tmpsig = TempSig::deserialize(deserializer)?;

        let num = if tmpsig.max_hash != 0 { 0 } else { tmpsig.num };
        let hash_function = match tmpsig.molecule.to_lowercase().as_ref() {
            "protein" => HashFunctions::Murmur64Protein,
            "dayhoff" => HashFunctions::Murmur64Dayhoff,
            "hp" => HashFunctions::Murmur64Hp,
            "dna" => HashFunctions::Murmur64Dna,
            _ => unimplemented!(), // TODO: throw error here
        };

        let current_max;
        // This shouldn't be necessary, but at some point we
        // created signatures with unordered mins =(
        let (mins, abunds) = if let Some(abunds) = tmpsig.abundances {
            let mut values: Vec<(_, _)> = tmpsig.mins.iter().zip(abunds.iter()).collect();
            values.sort();
            let mins: BTreeSet<_> = values.iter().map(|(v, _)| **v).collect();
            let abunds = values.into_iter().map(|(v, x)| (*v, *x)).collect();
            current_max = *mins.iter().next_back().unwrap_or(&0);
            (mins, Some(abunds))
        } else {
            current_max = 0;
            (tmpsig.mins.into_iter().collect(), None)
        };

        Ok(KmerMinHashBTree {
            num,
            ksize: tmpsig.ksize,
            seed: tmpsig.seed,
            max_hash: tmpsig.max_hash,
            md5sum: Mutex::new(Some(tmpsig.md5sum)),
            mins,
            abunds,
            hash_function,
            current_max,
        })
    }
}

impl ToWriter for KmerMinHashBTree {
    fn to_writer<W>(&self, writer: &mut W) -> Result<(), Error>
    where
        W: io::Write,
    {
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }
}

impl KmerMinHashBTree {
    pub fn new(
        scaled: ScaledType,
        ksize: u32,
        hash_function: HashFunctions,
        seed: u64,
        track_abundance: bool,
        num: u32,
    ) -> KmerMinHashBTree {
        let mins = Default::default();

        let abunds = if track_abundance {
            Some(Default::default())
        } else {
            None
        };

        let max_hash = max_hash_for_scaled(scaled);

        KmerMinHashBTree {
            num,
            ksize,
            hash_function,
            seed,
            max_hash,
            mins,
            abunds,
            current_max: 0,
            md5sum: Mutex::new(None),
        }
    }

    pub fn num(&self) -> u32 {
        self.num
    }

    pub fn is_protein(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Protein
    }

    pub fn max_hash(&self) -> u64 {
        self.max_hash
    }

    pub fn scaled(&self) -> ScaledType {
        scaled_for_max_hash(self.max_hash)
    }

    pub fn clear(&mut self) {
        self.mins.clear();
        if let Some(ref mut abunds) = self.abunds {
            abunds.clear();
        }
        self.current_max = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.mins.is_empty()
    }

    pub fn set_hash_function(&mut self, h: HashFunctions) -> Result<(), Error> {
        if self.hash_function == h {
            return Ok(());
        }

        if !self.is_empty() {
            return Err(Error::NonEmptyMinHash {
                message: "hash_function".into(),
            });
        }

        self.hash_function = h;
        Ok(())
    }

    pub fn track_abundance(&self) -> bool {
        self.abunds.is_some()
    }

    pub fn enable_abundance(&mut self) -> Result<(), Error> {
        if !self.mins.is_empty() {
            return Err(Error::NonEmptyMinHash {
                message: "track_abundance=True".into(),
            });
        }

        self.abunds = Some(Default::default());

        Ok(())
    }

    pub fn disable_abundance(&mut self) {
        self.abunds = None;
    }

    fn reset_md5sum(&self) {
        let mut data = self.md5sum.lock().unwrap();
        if data.is_some() {
            *data = None;
        }
    }

    pub fn md5sum(&self) -> String {
        let mut data = self.md5sum.lock().unwrap();
        if data.is_none() {
            let mut buffer = String::with_capacity(20);

            let mut md5_ctx = md5::Context::new();
            write!(&mut buffer, "{}", self.ksize()).unwrap();
            md5_ctx.consume(&buffer);
            buffer.clear();
            for x in &self.mins {
                write!(&mut buffer, "{}", x).unwrap();
                md5_ctx.consume(&buffer);
                buffer.clear();
            }
            *data = Some(format!("{:x}", md5_ctx.compute()));
        }
        data.clone().unwrap()
    }

    pub fn add_hash_with_abundance(&mut self, hash: u64, abundance: u64) {
        if hash > self.max_hash && self.max_hash != 0 {
            // This is a scaled minhash, and we don't need to add the new hash
            return;
        }

        if self.num == 0 && self.max_hash == 0 {
            // why did you create this minhash? it will always be empty...
            return;
        }

        if abundance == 0 {
            // well, don't add it.
            return;
        }

        // From this point on, hash is within scaled (or no scaled specified).

        // empty mins? add it.
        if self.mins.is_empty() {
            self.mins.insert(hash);
            self.reset_md5sum();
            if let Some(ref mut abunds) = self.abunds {
                abunds.insert(hash, abundance);
            }
            self.current_max = hash;
            return;
        }

        if hash <= self.max_hash || hash <= self.current_max || (self.mins.len() as u32) < self.num
        {
            // "good" hash - within range, smaller than current entry, or
            // still have space available
            if self.mins.insert(hash) {
                self.reset_md5sum();
                if hash > self.current_max {
                    self.current_max = hash;
                }
            }
            if let Some(ref mut abunds) = self.abunds {
                *abunds.entry(hash).or_insert(0) += abundance;
            }

            // is it too big now?
            if self.num != 0 && self.mins.len() > (self.num as usize) {
                let last = *self.mins.iter().next_back().unwrap();
                self.mins.remove(&last);
                self.reset_md5sum();
                if let Some(ref mut abunds) = self.abunds {
                    abunds.remove(&last);
                }
                self.current_max = *self.mins.iter().next_back().unwrap();
            }
        }
    }

    pub fn add_word(&mut self, word: &[u8]) {
        let hash = _hash_murmur(word, self.seed);
        self.add_hash(hash);
    }

    pub fn remove_hash(&mut self, hash: u64) {
        if self.mins.remove(&hash) {
            self.reset_md5sum();
            if let Some(ref mut abunds) = self.abunds {
                abunds.remove(&hash);
            }
        }
        if hash == self.current_max {
            self.current_max = *self.mins.iter().next_back().unwrap_or(&0);
        }
    }

    pub fn remove_many<T: IntoIterator<Item = u64>>(&mut self, hashes: T) -> Result<(), Error> {
        for min in hashes {
            self.remove_hash(min);
        }
        Ok(())
    }

    pub fn merge(&mut self, other: &KmerMinHashBTree) -> Result<(), Error> {
        self.check_compatible(other)?;
        let union = self.mins.union(&other.mins);

        let to_take = if self.num == 0 {
            usize::MAX
        } else {
            self.num as usize
        };

        self.mins = union.take(to_take).cloned().collect();

        if let Some(abunds) = &self.abunds {
            if let Some(oabunds) = &other.abunds {
                let mut new_abunds = BTreeMap::new();

                for hash in &self.mins {
                    *new_abunds.entry(*hash).or_insert(0) +=
                        abunds.get(hash).unwrap_or(&0) + oabunds.get(hash).unwrap_or(&0);
                }
                self.abunds = Some(new_abunds)
            }
        }
        // Better safe than sorry, but could check in other places to avoid
        // always resetting
        self.reset_md5sum();

        Ok(())
    }

    pub fn add_from(&mut self, other: &KmerMinHashBTree) -> Result<(), Error> {
        for min in &other.mins {
            self.add_hash(*min);
        }
        Ok(())
    }

    pub fn add_many(&mut self, hashes: &[u64]) -> Result<(), Error> {
        for min in hashes {
            self.add_hash(*min);
        }
        Ok(())
    }

    pub fn add_many_with_abund(&mut self, hashes: &[(u64, u64)]) -> Result<(), Error> {
        for item in hashes {
            self.add_hash_with_abundance(item.0, item.1);
        }
        Ok(())
    }

    pub fn count_common(&self, other: &KmerMinHashBTree, downsample: bool) -> Result<u64, Error> {
        if downsample && self.scaled() != other.scaled() {
            // downsample to the larger of the two scaled values
            let (first, second) = if self.scaled() > other.scaled() {
                (self, other)
            } else {
                (other, self)
            };
            let downsampled_mh = second.clone().downsample_scaled(first.scaled())?;
            first.count_common(&downsampled_mh, false)
        } else {
            self.check_compatible(other)?;
            let iter = if self.size() < other.size() {
                Intersection::new(self.mins.iter(), other.mins.iter())
            } else {
                Intersection::new(other.mins.iter(), self.mins.iter())
            };

            Ok(iter.count() as u64)
        }
    }

    pub fn intersection(&self, other: &KmerMinHashBTree) -> Result<(Vec<u64>, u64), Error> {
        self.check_compatible(other)?;

        if self.num != 0 {
            let mut combined_mh = KmerMinHashBTree::new(
                self.scaled(),
                self.ksize,
                self.hash_function.clone(),
                self.seed,
                self.abunds.is_some(),
                self.num,
            );

            combined_mh.merge(self)?;
            combined_mh.merge(other)?;

            let it1 = Intersection::new(self.mins.iter(), other.mins.iter());

            // TODO: there is probably a way to avoid this Vec here,
            // and pass the it1 as left in it2.
            let i1: Vec<u64> = it1.cloned().collect();
            let i2: Vec<u64> = combined_mh.mins.iter().cloned().collect();
            let it2 = Intersection::new(i1.iter(), i2.iter());

            let common: Vec<u64> = it2.cloned().collect();
            Ok((common, combined_mh.mins.len() as u64))
        } else {
            // Intersection for scaled MinHash sketches
            Ok(intersection(self.mins.iter(), other.mins.iter()))
        }
    }

    pub fn intersection_size(&self, other: &KmerMinHashBTree) -> Result<(u64, u64), Error> {
        self.check_compatible(other)?;

        if self.num != 0 {
            let mut combined_mh = KmerMinHashBTree::new(
                self.scaled(),
                self.ksize,
                self.hash_function.clone(),
                self.seed,
                self.abunds.is_some(),
                self.num,
            );

            combined_mh.merge(self)?;
            combined_mh.merge(other)?;

            let it1 = Intersection::new(self.mins.iter(), other.mins.iter());

            // TODO: there is probably a way to avoid this Vec here,
            // and pass the it1 as left in it2.
            let i1: Vec<u64> = it1.cloned().collect();
            let i2: Vec<u64> = combined_mh.mins.iter().cloned().collect();
            let it2 = Intersection::new(i1.iter(), i2.iter());

            Ok((it2.count() as u64, combined_mh.mins.len() as u64))
        } else {
            Ok(intersection_size(self.mins.iter(), other.mins.iter()))
        }
    }

    // calculate Jaccard similarity, ignoring abundance.
    pub fn jaccard(&self, other: &KmerMinHashBTree) -> Result<f64, Error> {
        self.check_compatible(other)?;
        if let Ok((common, size)) = self.intersection_size(other) {
            Ok(common as f64 / u64::max(1, size) as f64)
        } else {
            Ok(0.0)
        }
    }

    // compare two minhashes, with abundance;
    // calculate their angular similarity.
    pub fn angular_similarity(&self, other: &KmerMinHashBTree) -> Result<f64, Error> {
        self.check_compatible(other)?;

        if self.abunds.is_none() || other.abunds.is_none() {
            return Err(Error::NeedsAbundanceTracking);
        }

        let abunds = self.abunds.as_ref().unwrap();
        let other_abunds = other.abunds.as_ref().unwrap();

        let mut prod = 0;
        let a_sq: u64 = abunds.values().map(|a| (a * a)).sum();
        let b_sq: u64 = other_abunds.values().map(|a| (a * a)).sum();

        for (hash, value) in abunds.iter() {
            if let Some(oa) = other_abunds.get(hash) {
                prod += value * oa
            }
        }

        let norm_a = (a_sq as f64).sqrt();
        let norm_b = (b_sq as f64).sqrt();

        if norm_a == 0. || norm_b == 0. {
            return Ok(0.0);
        }
        let prod = f64::min(prod as f64 / (norm_a * norm_b), 1.);
        let distance = 2. * prod.acos() / PI;
        Ok(1. - distance)
    }

    pub fn similarity(
        &self,
        other: &KmerMinHashBTree,
        ignore_abundance: bool,
        downsample: bool,
    ) -> Result<f64, Error> {
        if downsample && self.scaled() != other.scaled() {
            // downsample to larger of two scaled
            let (first, second) = if self.scaled() > other.scaled() {
                (self, other)
            } else {
                (other, self)
            };
            let downsampled_mh = second.clone().downsample_scaled(first.scaled())?;
            first.similarity(&downsampled_mh, ignore_abundance, false)
        } else if ignore_abundance || self.abunds.is_none() || other.abunds.is_none() {
            self.jaccard(other)
        } else {
            self.angular_similarity(other)
        }
    }

    pub fn dayhoff(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Dayhoff
    }

    pub fn hp(&self) -> bool {
        self.hash_function == HashFunctions::Murmur64Hp
    }

    pub fn hash_function(&self) -> HashFunctions {
        self.hash_function.clone()
    }

    pub fn mins(&self) -> Vec<u64> {
        self.mins.iter().cloned().collect()
    }

    pub fn iter_mins(&self) -> impl Iterator<Item = &u64> {
        self.mins.iter()
    }

    pub fn abunds(&self) -> Option<Vec<u64>> {
        self.abunds
            .as_ref()
            .map(|abunds| abunds.values().cloned().collect())
    }

    // create a downsampled copy of self
    pub fn downsample_max_hash(self, max_hash: u64) -> Result<KmerMinHashBTree, Error> {
        if self.max_hash == 0 {
            // CTB: this is a num minhash. Just blithely return.
            Ok(self)
        } else {
            let scaled = scaled_for_max_hash(max_hash);
            self.downsample_scaled(scaled)
        }
    }

    // create a downsampled copy of self
    pub fn downsample_scaled(self, scaled: ScaledType) -> Result<KmerMinHashBTree, Error> {
        if self.scaled() == scaled || self.scaled() == 0 {
            Ok(self)
        } else if self.scaled() > scaled {
            Err(Error::CannotUpsampleScaled)
        } else {
            let mut new_mh = KmerMinHashBTree::new(
                scaled,
                self.ksize,
                self.hash_function.clone(),
                self.seed,
                self.abunds.is_some(),
                self.num,
            );
            if self.abunds.is_some() {
                new_mh.add_many_with_abund(&self.to_vec_abunds())?;
            } else {
                new_mh.add_many(&self.mins())?;
            }
            Ok(new_mh)
        }
    }

    pub fn to_vec_abunds(&self) -> Vec<(u64, u64)> {
        if let Some(abunds) = &self.abunds {
            abunds.iter().map(|(a, b)| (*a, *b)).collect()
        } else {
            self.mins
                .iter()
                .cloned()
                .zip(std::iter::repeat(1))
                .collect()
        }
    }

    pub fn sum_abunds(&self) -> u64 {
        if let Some(abunds) = &self.abunds {
            abunds.values().sum()
        } else {
            self.size() as u64
        }
    }

    pub fn from_reader<R>(rdr: R) -> Result<KmerMinHashBTree, Error>
    where
        R: std::io::Read,
    {
        let (rdr, _format) = niffler::get_reader(Box::new(rdr))?;

        let mh: KmerMinHashBTree = serde_json::from_reader(rdr)?;
        Ok(mh)
    }
}

impl SigsTrait for KmerMinHashBTree {
    fn size(&self) -> usize {
        self.mins.len()
    }

    fn to_vec(&self) -> Vec<u64> {
        self.mins()
    }

    fn ksize(&self) -> usize {
        self.ksize as usize
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn hash_function(&self) -> HashFunctions {
        self.hash_function.clone()
    }

    fn add_hash(&mut self, hash: u64) {
        self.add_hash_with_abundance(hash, 1);
    }

    fn check_compatible(&self, other: &KmerMinHashBTree) -> Result<(), Error> {
        /*
        if self.num != other.num {
            return Err(Error::MismatchNum {
                n1: self.num,
                n2: other.num,
            }
            .into());
        }
        */
        if self.ksize != other.ksize {
            return Err(Error::MismatchKSizes);
        }
        if self.hash_function != other.hash_function {
            // TODO: fix this error
            return Err(Error::MismatchDNAProt);
        }
        if self.max_hash != other.max_hash {
            return Err(Error::MismatchScaled);
        }
        if self.seed != other.seed {
            return Err(Error::MismatchSeed);
        }
        Ok(())
    }
}

impl From<KmerMinHashBTree> for KmerMinHash {
    fn from(other: KmerMinHashBTree) -> KmerMinHash {
        let mut new_mh = KmerMinHash::new(
            other.scaled(),
            other.ksize() as u32,
            other.hash_function(),
            other.seed(),
            other.track_abundance(),
            other.num(),
        );

        let mins = other.mins.into_iter().collect();
        let abunds = other
            .abunds
            .map(|abunds| abunds.values().cloned().collect());

        new_mh.mins = mins;
        new_mh.abunds = abunds;

        new_mh.md5sum = other.md5sum;

        new_mh
    }
}

impl From<&KmerMinHashBTree> for KmerMinHash {
    fn from(other: &KmerMinHashBTree) -> KmerMinHash {
        let mut new_mh = KmerMinHash::new(
            other.scaled(),
            other.ksize() as u32,
            other.hash_function(),
            other.seed(),
            other.track_abundance(),
            other.num(),
        );

        let mins = other.mins.iter().copied().collect();
        let abunds = other
            .abunds
            .as_ref()
            .map(|abunds| abunds.values().cloned().collect());

        new_mh.mins = mins;
        new_mh.abunds = abunds;

        new_mh.md5sum = Mutex::new(other.md5sum.lock().unwrap().clone());

        new_mh
    }
}

impl From<KmerMinHash> for KmerMinHashBTree {
    fn from(other: KmerMinHash) -> KmerMinHashBTree {
        let mut new_mh = KmerMinHashBTree::new(
            other.scaled(),
            other.ksize() as u32,
            other.hash_function(),
            other.seed(),
            other.track_abundance(),
            other.num(),
        );

        let mins: BTreeSet<u64> = other.mins.into_iter().collect();
        let abunds = other
            .abunds
            .map(|abunds| mins.iter().cloned().zip(abunds).collect());

        new_mh.mins = mins;
        new_mh.abunds = abunds;

        new_mh.md5sum = other.md5sum;

        new_mh
    }
}

fn intersection<'a>(
    me_iter: impl Iterator<Item = &'a u64>,
    other_iter: impl Iterator<Item = &'a u64>,
) -> (Vec<u64>, u64) {
    let mut me = me_iter.peekable();
    let mut other = other_iter.peekable();
    let mut common: Vec<u64> = vec![];
    let mut union_size = 0;

    loop {
        match (me.peek(), other.peek()) {
            (Some(ref left_key), Some(ref right_key)) => {
                let res = left_key.cmp(right_key);
                match res {
                    Ordering::Less => {
                        me.next();
                        union_size += 1;
                    }
                    Ordering::Greater => {
                        other.next();
                        union_size += 1;
                    }
                    Ordering::Equal => {
                        other.next();
                        common.push(***left_key);
                        me.next();
                        union_size += 1;
                    }
                };
            }
            (None, Some(_)) => {
                other.next();
                union_size += 1;
            }
            (Some(_), None) => {
                me.next();
                union_size += 1;
            }
            _ => break,
        };
    }
    (common, union_size as u64)
}

fn intersection_size<'a>(
    me_iter: impl Iterator<Item = &'a u64>,
    other_iter: impl Iterator<Item = &'a u64>,
) -> (u64, u64) {
    let mut me = me_iter.peekable();
    let mut other = other_iter.peekable();
    let mut common = 0;
    let mut union_size = 0;

    loop {
        match (me.peek(), other.peek()) {
            (Some(ref left_key), Some(ref right_key)) => {
                let res = left_key.cmp(right_key);
                match res {
                    Ordering::Less => {
                        me.next();
                        union_size += 1;
                    }
                    Ordering::Greater => {
                        other.next();
                        union_size += 1;
                    }
                    Ordering::Equal => {
                        other.next();
                        me.next();
                        common += 1;
                        union_size += 1;
                    }
                };
            }
            (None, Some(_)) => {
                other.next();
                union_size += 1;
            }
            (Some(_), None) => {
                me.next();
                union_size += 1;
            }
            _ => break,
        };
    }
    (common as u64, union_size as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    /// Test that a valid KmerMinHash json can be deserialized correctly
    fn test_deserialize_valid() {
        let json_data = r#"
        {
            "num": 1000,
            "ksize": 21,
            "seed": 42,
            "max_hash": 0,
            "md5sum": "test_md5",
            "mins": [1, 2, 3, 4, 5],
            "abundances": [10, 20, 30, 40, 50],
            "molecule": "dna"
        }
        "#;

        let deserialized: KmerMinHash =
            serde_json::from_str(json_data).expect("Failed to deserialize");

        assert_eq!(deserialized.num, 1000);
        assert_eq!(deserialized.ksize, 21);
        assert_eq!(deserialized.seed, 42);
        assert_eq!(deserialized.hash_function, HashFunctions::Murmur64Dna);
        assert_eq!(deserialized.mins, vec![1, 2, 3, 4, 5]);
        assert!(deserialized.abunds.is_some());
    }

    #[test]
    /// Test that a invalid molecule type panics!
    fn test_deserialize_invalid_molecule() {
        let json_data = r#"
        {
            "num": 1000,
            "ksize": 21,
            "seed": 42,
            "max_hash": 0,
            "md5sum": "test_md5",
            "mins": [1, 2, 3, 4, 5],
            "molecule": "unknown_type"
        }
        "#;

        let result: Result<KmerMinHash, _> = serde_json::from_str(json_data);

        // Assert that the result is an error
        assert!(result.is_err());

        // Extract and check the error message
        let error_message = format!("{}", result.unwrap_err());
        assert!(error_message.contains("Invalid hash function"));
    }
}
