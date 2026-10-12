//! USD's binary layer, `.usdc` — Pixar's "crate" file — read into a
//! [`Layer`].
//!
//! # The file
//!
//! A header naming where its table of contents is; the table naming six
//! sections: the **tokens** (every name and word, once, compressed), the
//! **strings** (indices into the tokens), the **fields** (a name and a
//! value's encoding), the **field sets** (runs of fields, one run a spec),
//! the **paths** (a tree, compressed) and the **specs** (a path, its field
//! set and what it is: a prim, an attribute, a relationship, a variant).
//!
//! A value's encoding is 64 bits: whether it is an array, whether it is
//! stored in the encoding itself, whether it is compressed, its type, and
//! either the value or where in the file it is. Arrays of numbers may be
//! integer-coded (deltas from a common value, each as small as it fits) and
//! that compressed again with LZ4; floats may be stored as integers or as a
//! table and indices into it.
//!
//! # What is read
//!
//! What a scene needs: prims and their types, attributes' values (an
//! animated one by its first sample), relationships, references, payloads,
//! variants and the stage's metadata. Dictionaries and anything else are
//! skipped. Every offset, count and size is checked against the file before
//! it is followed: a crate file is a tree of offsets, and a wrong one must
//! be an error, not a read past the end.

use std::collections::HashMap;

use crate::usd_read::{Arc, Layer, Prim, Property, Specifier, Val};

/// Why a binary layer could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsdcError(pub String);

impl std::fmt::Display for UsdcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "USDC: {}", self.0)
    }
}

impl std::error::Error for UsdcError {}

type R<T> = Result<T, UsdcError>;

fn bad<T>(what: &str) -> R<T> {
    Err(UsdcError(what.to_string()))
}

/// The most anything may unpack to: a layer is not larger than this.
const MAX_UNPACKED: usize = 1 << 31;

// --- decompression ---------------------------------------------------------------

/// One LZ4 block, appended to `out`.
fn lz4_block(src: &[u8], out: &mut Vec<u8>, max: usize) -> R<()> {
    let mut i = 0;
    let byte = |i: usize| {
        src.get(i)
            .copied()
            .ok_or_else(|| UsdcError("LZ4 data ends early".into()))
    };
    while i < src.len() {
        let token = byte(i)?;
        i += 1;
        let mut literals = (token >> 4) as usize;
        if literals == 15 {
            loop {
                let b = byte(i)?;
                i += 1;
                literals += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        let lit = src
            .get(i..i + literals)
            .ok_or_else(|| UsdcError("LZ4 literals run past the end".into()))?;
        out.extend_from_slice(lit);
        i += literals;
        if i >= src.len() {
            break;
        }
        let offset = u16::from_le_bytes([byte(i)?, byte(i + 1)?]) as usize;
        i += 2;
        if offset == 0 || offset > out.len() {
            return bad("an LZ4 match points before the start");
        }
        let mut length = (token & 15) as usize;
        if length == 15 {
            loop {
                let b = byte(i)?;
                i += 1;
                length += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        length += 4;
        if out.len() + length > max {
            return bad("LZ4 data unpacks larger than it said");
        }
        let start = out.len() - offset;
        for k in 0..length {
            let c = out[start + k];
            out.push(c);
        }
    }
    Ok(())
}

/// USD's own wrapper round LZ4: a chunk count, then the chunks.
fn decompress(src: &[u8], max: usize) -> R<Vec<u8>> {
    let mut out = Vec::new();
    let (&chunks, rest) = src
        .split_first()
        .ok_or_else(|| UsdcError("empty compressed data".into()))?;
    if chunks == 0 {
        lz4_block(rest, &mut out, max)?;
    } else {
        let mut at = rest;
        for _ in 0..chunks {
            let size = at
                .get(..4)
                .map(|b| i32::from_le_bytes(b.try_into().expect("four")))
                .ok_or_else(|| UsdcError("a chunk's size runs past the end".into()))?;
            let size = usize::try_from(size).map_err(|_| UsdcError("a negative chunk".into()))?;
            let chunk = at
                .get(4..4 + size)
                .ok_or_else(|| UsdcError("a chunk runs past the end".into()))?;
            lz4_block(chunk, &mut out, max)?;
            at = &at[4 + size..];
        }
    }
    Ok(out)
}

/// Integers as USD codes them: a common value, two bits a number saying how
/// each differs from the last (by the common value, or by a small, a medium
/// or a large number), then those numbers.
fn decode_ints(data: &[u8], count: usize, wide: bool) -> R<Vec<i64>> {
    let common_size = if wide { 8 } else { 4 };
    let common = if wide {
        i64::from_le_bytes(
            data.get(..8)
                .ok_or_else(|| UsdcError("short integers".into()))?
                .try_into()
                .expect("eight"),
        )
    } else {
        i32::from_le_bytes(
            data.get(..4)
                .ok_or_else(|| UsdcError("short integers".into()))?
                .try_into()
                .expect("four"),
        ) as i64
    };
    let codes_len = (count * 2).div_ceil(8);
    let codes = data
        .get(common_size..common_size + codes_len)
        .ok_or_else(|| UsdcError("integer codes run past the end".into()))?;
    let mut at = common_size + codes_len;
    let mut out = Vec::with_capacity(count);
    let mut prev = 0i64;
    let mut take = |n: usize| -> R<&[u8]> {
        let s = data
            .get(at..at + n)
            .ok_or_else(|| UsdcError("integers run past the end".into()))?;
        at += n;
        Ok(s)
    };
    for i in 0..count {
        let code = (codes[i / 4] >> ((i % 4) * 2)) & 3;
        let delta = match (code, wide) {
            (0, _) => common,
            (1, false) => take(1)?[0] as i8 as i64,
            (2, false) => i16::from_le_bytes(take(2)?.try_into().expect("two")) as i64,
            (3, false) => i32::from_le_bytes(take(4)?.try_into().expect("four")) as i64,
            (1, true) => i16::from_le_bytes(take(2)?.try_into().expect("two")) as i64,
            (2, true) => i32::from_le_bytes(take(4)?.try_into().expect("four")) as i64,
            _ => i64::from_le_bytes(take(8)?.try_into().expect("eight")),
        };
        prev = prev.wrapping_add(delta);
        out.push(prev);
    }
    Ok(out)
}

// --- the file ----------------------------------------------------------------------

struct Crate<'a> {
    b: &'a [u8],
    version: (u8, u8, u8),
    tokens: Vec<String>,
    strings: Vec<u32>,
    paths: Vec<String>,
}

/// A position in the file, read forward.
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let s = self
            .b
            .get(
                self.at
                    ..self
                        .at
                        .checked_add(n)
                        .ok_or_else(|| UsdcError("a size overflows".into()))?,
            )
            .ok_or_else(|| UsdcError("data runs past the end of the file".into()))?;
        self.at += n;
        Ok(s)
    }
    fn u8(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four")))
    }
    fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("eight")))
    }
    fn i64(&mut self) -> R<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().expect("eight")))
    }
    fn f64(&mut self) -> R<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().expect("eight")))
    }
    /// A count, checked against what is left of the file at `each` bytes a
    /// thing.
    fn count(&mut self, each: usize) -> R<usize> {
        let n = self.u64()? as usize;
        if n.saturating_mul(each.max(1)) > self.b.len().saturating_sub(self.at) {
            return bad("a count larger than the file");
        }
        Ok(n)
    }
    /// Integers, coded and compressed, `count` of them.
    fn ints(&mut self, count: usize, wide: bool) -> R<Vec<i64>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let size = self.u64()? as usize;
        let packed = self.take(size)?;
        let raw = decompress(packed, (count * 9 + 16).min(MAX_UNPACKED))?;
        decode_ints(&raw, count, wide)
    }
}

/// The parts of a value's 64 bits.
#[derive(Clone, Copy, Debug)]
struct Rep(u64);

impl Rep {
    fn array(self) -> bool {
        self.0 & (1 << 63) != 0
    }
    fn inlined(self) -> bool {
        self.0 & (1 << 62) != 0
    }
    fn compressed(self) -> bool {
        self.0 & (1 << 61) != 0
    }
    fn kind(self) -> u8 {
        ((self.0 >> 48) & 0xff) as u8
    }
    fn payload(self) -> u64 {
        self.0 & ((1 << 48) - 1)
    }
}

/// A value as a spec's field holds it.
#[derive(Clone, Debug, PartialEq)]
enum V {
    Nums(Vec<f64>),
    Texts(Vec<String>),
    Paths(Vec<String>),
    Arcs(Vec<Arc>),
    Pairs(Vec<(String, String)>),
    /// Values over time: each time code and its value.
    Samples(Vec<(f64, V)>),
    Specifier(u8),
    None,
}

// Crate's type numbers.
const BOOL: u8 = 1;
const UCHAR: u8 = 2;
const INT: u8 = 3;
const UINT: u8 = 4;
const INT64: u8 = 5;
const UINT64: u8 = 6;
const HALF: u8 = 7;
const FLOAT: u8 = 8;
const DOUBLE: u8 = 9;
const STRING: u8 = 10;
const TOKEN: u8 = 11;
const ASSET: u8 = 12;
const MATRIX2D: u8 = 13;
const MATRIX3D: u8 = 14;
const MATRIX4D: u8 = 15;
const QUATD: u8 = 16;
const QUATF: u8 = 17;
const QUATH: u8 = 18;
const VEC2D: u8 = 19;
const VEC4I: u8 = 30;
const TOKEN_LIST_OP: u8 = 32;
const STRING_LIST_OP: u8 = 33;
const PATH_LIST_OP: u8 = 34;
const REFERENCE_LIST_OP: u8 = 35;
const PATH_VECTOR: u8 = 40;
const TOKEN_VECTOR: u8 = 41;
const SPECIFIER: u8 = 42;
const VARIANT_SELECTION: u8 = 45;
const TIME_SAMPLES: u8 = 46;
const PAYLOAD: u8 = 47;
const DOUBLE_VECTOR: u8 = 48;
const STRING_VECTOR: u8 = 50;
const VALUE_BLOCK: u8 = 51;
const PAYLOAD_LIST_OP: u8 = 55;
const TIME_CODE: u8 = 56;

/// A vector or matrix type: how many numbers, and of what width (8 double,
/// 4 float or int, 2 half). `None` for anything else.
fn shape(kind: u8) -> Option<(usize, usize, bool)> {
    // (components, bytes each, integer)
    Some(match kind {
        BOOL | UCHAR => (1, 1, true),
        INT | UINT => (1, 4, true),
        INT64 | UINT64 => (1, 8, true),
        HALF => (1, 2, false),
        FLOAT => (1, 4, false),
        DOUBLE | TIME_CODE => (1, 8, false),
        MATRIX2D => (4, 8, false),
        MATRIX3D => (9, 8, false),
        MATRIX4D => (16, 8, false),
        QUATD => (4, 8, false),
        QUATF => (4, 4, false),
        QUATH => (4, 2, false),
        VEC2D..=VEC4I => {
            let i = kind - VEC2D;
            let n = 2 + (i / 4) as usize;
            match i % 4 {
                0 => (n, 8, false),
                1 => (n, 4, false),
                2 => (n, 2, false),
                _ => (n, 4, true),
            }
        }
        _ => return None,
    })
}

fn number(bytes: &[u8], width: usize, integer: bool, kind: u8) -> f64 {
    match (width, integer) {
        (1, _) => bytes[0] as f64,
        (2, _) => half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f64(),
        (4, true) if kind == UINT => u32::from_le_bytes(bytes.try_into().expect("four")) as f64,
        (4, true) => i32::from_le_bytes(bytes.try_into().expect("four")) as f64,
        (4, false) => f32::from_le_bytes(bytes.try_into().expect("four")) as f64,
        (8, true) if kind == UINT64 => u64::from_le_bytes(bytes.try_into().expect("eight")) as f64,
        (8, true) => i64::from_le_bytes(bytes.try_into().expect("eight")) as f64,
        _ => f64::from_le_bytes(bytes.try_into().expect("eight")),
    }
}

/// A quaternion as text writes it, real part first; the file has it last.
fn real_first(v: &mut [f64], kind: u8) {
    if matches!(kind, QUATD | QUATF | QUATH) {
        for q in v.as_chunks_mut::<4>().0 {
            q.rotate_right(1);
        }
    }
}

impl<'a> Crate<'a> {
    fn at(&self, offset: u64) -> R<Cursor<'a>> {
        let at = offset as usize;
        if at > self.b.len() {
            return bad("an offset past the end of the file");
        }
        Ok(Cursor { b: self.b, at })
    }

    fn token(&self, i: u64) -> R<String> {
        self.tokens
            .get(i as usize)
            .cloned()
            .ok_or_else(|| UsdcError("a token out of range".into()))
    }

    fn string(&self, i: u64) -> R<String> {
        let t = *self
            .strings
            .get(i as usize)
            .ok_or_else(|| UsdcError("a string out of range".into()))?;
        self.token(t as u64)
    }

    fn path(&self, i: u64) -> R<String> {
        self.paths
            .get(i as usize)
            .cloned()
            .ok_or_else(|| UsdcError("a path out of range".into()))
    }

    fn value(&self, rep: Rep, depth: usize) -> R<V> {
        if depth > 8 {
            return bad("values nested too deep");
        }
        let kind = rep.kind();
        let p = rep.payload();
        if kind == VALUE_BLOCK {
            return Ok(V::None);
        }
        if rep.array() {
            if !rep.inlined() && p == 0 {
                return Ok(V::Nums(Vec::new()));
            }
            return self.array(rep);
        }
        if rep.inlined() {
            return Ok(match kind {
                BOOL | UCHAR | UINT => V::Nums(vec![(p as u32) as f64]),
                INT | INT64 => V::Nums(vec![(p as u32 as i32) as f64]),
                UINT64 => V::Nums(vec![p as f64]),
                HALF => V::Nums(vec![half::f16::from_bits(p as u16).to_f64()]),
                // A double is inlined only when a float holds it exactly.
                FLOAT | DOUBLE | TIME_CODE => V::Nums(vec![f32::from_bits(p as u32) as f64]),
                STRING => V::Texts(vec![self.string(p)?]),
                TOKEN | ASSET => V::Texts(vec![self.token(p)?]),
                SPECIFIER => V::Specifier(p as u8),
                // Varying (0) or uniform (1).
                44 => V::Nums(vec![p as f64]),
                MATRIX2D | MATRIX3D | MATRIX4D => {
                    // The diagonal, as bytes: everything else is zero.
                    let n = match kind {
                        MATRIX2D => 2,
                        MATRIX3D => 3,
                        _ => 4,
                    };
                    let mut m = vec![0.0; n * n];
                    for k in 0..n {
                        m[k * n + k] = ((p >> (8 * k)) as u8 as i8) as f64;
                    }
                    V::Nums(m)
                }
                k => match shape(k) {
                    // A vector of small whole numbers, a byte each.
                    Some((n, _, _)) => {
                        let mut v: Vec<f64> = (0..n)
                            .map(|k| ((p >> (8 * k)) as u8 as i8) as f64)
                            .collect();
                        real_first(&mut v, kind);
                        V::Nums(v)
                    }
                    None => V::None,
                },
            });
        }
        let mut c = self.at(p)?;
        Ok(match kind {
            TIME_SAMPLES => {
                // A jump over the times' own data to their encoding; then a
                // jump to the values' count and their encodings.
                let start = c.at;
                let jump = c.i64()?;
                c.at = usize::try_from(start as i64 + jump)
                    .map_err(|_| UsdcError("a bad jump".into()))?;
                let times = match self.value(Rep(c.u64()?), depth + 1)? {
                    V::Nums(t) => t,
                    _ => Vec::new(),
                };
                let start = c.at;
                let jump = c.i64()?;
                c.at = usize::try_from(start as i64 + jump)
                    .map_err(|_| UsdcError("a bad jump".into()))?;
                let count = c.count(8)?;
                let mut out = Vec::with_capacity(count.min(times.len()));
                for t in times.into_iter().take(count) {
                    let rep = Rep(c.u64()?);
                    out.push((t, self.value(rep, depth + 1)?));
                }
                V::Samples(out)
            }
            TOKEN_VECTOR | PATH_VECTOR | STRING_VECTOR => {
                let n = c.count(4)?;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let i = c.u32()? as u64;
                    out.push(match kind {
                        TOKEN_VECTOR => self.token(i)?,
                        PATH_VECTOR => self.path(i)?,
                        _ => self.string(i)?,
                    });
                }
                if kind == PATH_VECTOR {
                    V::Paths(out)
                } else {
                    V::Texts(out)
                }
            }
            DOUBLE_VECTOR => {
                let n = c.count(8)?;
                V::Nums((0..n).map(|_| c.f64()).collect::<R<_>>()?)
            }
            TOKEN_LIST_OP | STRING_LIST_OP | PATH_LIST_OP => {
                let items = self.list_op(&mut c, |me, c| {
                    let i = c.u32()? as u64;
                    match kind {
                        TOKEN_LIST_OP => me.token(i),
                        STRING_LIST_OP => me.string(i),
                        _ => me.path(i),
                    }
                })?;
                if kind == PATH_LIST_OP {
                    V::Paths(items)
                } else {
                    V::Texts(items)
                }
            }
            REFERENCE_LIST_OP => V::Arcs(
                self.list_op(&mut c, |me, c| {
                    let asset = me.string(c.u32()? as u64)?;
                    let prim = me.path(c.u32()? as u64)?;
                    let _offset = (c.f64()?, c.f64()?);
                    // Custom data: read past when empty, and the rest of the
                    // list given up when not — what it holds is not needed.
                    if c.u64()? != 0 {
                        return bad("a reference with custom data");
                    }
                    Ok(arc(asset, prim))
                })
                .or_else(|_| Ok::<_, UsdcError>(Vec::new()))?,
            ),
            PAYLOAD => {
                let asset = self.string(c.u32()? as u64)?;
                let prim = self.path(c.u32()? as u64)?;
                V::Arcs(vec![arc(asset, prim)])
            }
            PAYLOAD_LIST_OP => {
                let newer = self.version >= (0, 8, 0);
                V::Arcs(self.list_op(&mut c, |me, c| {
                    let asset = me.string(c.u32()? as u64)?;
                    let prim = me.path(c.u32()? as u64)?;
                    if newer {
                        let _ = (c.f64()?, c.f64()?);
                    }
                    Ok(arc(asset, prim))
                })?)
            }
            VARIANT_SELECTION => {
                let n = c.count(8)?;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let set = self.string(c.u32()? as u64)?;
                    let chosen = self.string(c.u32()? as u64)?;
                    out.push((set, chosen));
                }
                V::Pairs(out)
            }
            STRING | TOKEN | ASSET => V::Texts(vec![self.token(c.u32()? as u64)?]),
            k => match shape(k) {
                Some((n, width, integer)) => {
                    let bytes = c.take(n * width)?;
                    let mut v: Vec<f64> = bytes
                        .chunks_exact(width)
                        .map(|b| number(b, width, integer, kind))
                        .collect();
                    real_first(&mut v, kind);
                    V::Nums(v)
                }
                None => V::None,
            },
        })
    }

    /// An array: a count, then its elements — raw, or coded and compressed.
    fn array(&self, rep: Rep) -> R<V> {
        let kind = rep.kind();
        let mut c = self.at(rep.payload())?;
        let count = if self.version < (0, 7, 0) {
            c.u32()? as usize
        } else {
            c.u64()? as usize
        };
        if count > self.b.len() * 8 {
            return bad("an array larger than the file");
        }
        match kind {
            STRING | TOKEN | ASSET => {
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    let i = c.u32()? as u64;
                    out.push(if kind == STRING {
                        self.string(i)?
                    } else {
                        self.token(i)?
                    });
                }
                return Ok(V::Texts(out));
            }
            _ => {}
        }
        let Some((n, width, integer)) = shape(kind) else {
            return Ok(V::None);
        };
        if rep.compressed() && n == 1 {
            if integer {
                let wide = width == 8;
                return Ok(V::Nums(
                    c.ints(count, wide)?.into_iter().map(|i| i as f64).collect(),
                ));
            }
            // Floats: stored as integers, or as a table and where in it.
            let code = c.u8()?;
            return Ok(V::Nums(match code {
                b'i' => c
                    .ints(count, false)?
                    .into_iter()
                    .map(|i| i as f64)
                    .collect(),
                b't' => {
                    let size = c.u32()? as usize;
                    let table: Vec<f64> = c
                        .take(
                            size.checked_mul(width)
                                .ok_or_else(|| UsdcError("a table overflows".into()))?,
                        )?
                        .chunks_exact(width)
                        .map(|b| number(b, width, false, kind))
                        .collect();
                    c.ints(count, false)?
                        .into_iter()
                        .map(|i| table.get(i as usize).copied().unwrap_or(0.0))
                        .collect()
                }
                _ => return bad("compressed floats of a kind not known"),
            }));
        }
        let bytes = c.take(
            count
                .checked_mul(n * width)
                .ok_or_else(|| UsdcError("an array overflows".into()))?,
        )?;
        let mut v: Vec<f64> = bytes
            .chunks_exact(width)
            .map(|b| number(b, width, integer, kind))
            .collect();
        real_first(&mut v, kind);
        Ok(V::Nums(v))
    }

    /// A list edit's items, all of them: explicit, added, prepended,
    /// appended (deleted and reordered ones are read past).
    fn list_op<T>(
        &self,
        c: &mut Cursor<'a>,
        mut item: impl FnMut(&Self, &mut Cursor<'a>) -> R<T>,
    ) -> R<Vec<T>> {
        let header = c.u8()?;
        let mut out = Vec::new();
        // Explicit, added, prepended, appended, deleted, ordered.
        for (bit, keep) in [
            (1 << 1, true),
            (1 << 2, true),
            (1 << 5, true),
            (1 << 6, true),
            (1 << 3, false),
            (1 << 4, false),
        ] {
            if header & bit == 0 {
                continue;
            }
            let n = c.count(4)?;
            for _ in 0..n {
                let v = item(self, c)?;
                if keep {
                    out.push(v);
                }
            }
        }
        Ok(out)
    }
}

fn arc(asset: String, prim: String) -> Arc {
    Arc {
        asset,
        prim_path: (!prim.is_empty() && prim != "/").then_some(prim),
    }
}

/// A path's variant selections taken out: `/A{v=x}B` is `/A/B`, where the
/// composed stage has it.
fn plain(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut inside = false;
    for ch in path.chars() {
        match ch {
            '{' => {
                inside = true;
                out.push('/');
            }
            '}' => inside = false,
            c if !inside => out.push(c),
            _ => {}
        }
    }
    let out = out.replace("//", "/").replace("/.", ".");
    if out.len() > 1 {
        out.trim_end_matches('/').to_string()
    } else {
        out
    }
}

/// Integers as a section stores them, for the writer's tests.
#[cfg(test)]
pub(crate) fn decode_for_test(file: &[u8], count: usize) -> Result<Vec<i64>, UsdcError> {
    Cursor { b: file, at: 0 }.ints(count, false)
}

/// A `.usdc` layer.
pub fn parse_usdc(b: &[u8]) -> Result<Layer, UsdcError> {
    if !b.starts_with(b"PXR-USDC") || b.len() < 88 {
        return bad("not a USD binary layer");
    }
    let version = (b[8], b[9], b[10]);
    if version < (0, 4, 0) {
        return bad(&format!(
            "version {}.{}.{} is older than Kreate reads (0.4.0)",
            version.0, version.1, version.2
        ));
    }
    let mut c = Cursor { b, at: 16 };
    let toc = c.u64()?;
    let mut c = Cursor {
        b,
        at: toc as usize,
    };
    if c.at > b.len() {
        return bad("the table of contents is past the end");
    }
    let sections = c.count(32)?;
    let mut table: HashMap<String, (u64, u64)> = HashMap::new();
    for _ in 0..sections {
        let name = c.take(16)?;
        let name =
            String::from_utf8_lossy(&name[..name.iter().position(|&x| x == 0).unwrap_or(16)])
                .into_owned();
        let start = c.i64()? as u64;
        let size = c.i64()? as u64;
        table.insert(name, (start, size));
    }
    let section = |name: &str| -> R<Cursor<'_>> {
        let (start, _) = table
            .get(name)
            .ok_or_else(|| UsdcError(format!("no {name} section")))?;
        if *start as usize > b.len() {
            return bad("a section past the end");
        }
        Ok(Cursor {
            b,
            at: *start as usize,
        })
    };

    let mut file = Crate {
        b,
        version,
        tokens: Vec::new(),
        strings: Vec::new(),
        paths: Vec::new(),
    };

    // Tokens: compressed, nul-separated.
    {
        let mut c = section("TOKENS")?;
        let count = c.u64()? as usize;
        let size = c.u64()? as usize;
        let packed_size = c.u64()? as usize;
        let packed = c.take(packed_size)?;
        let raw = decompress(packed, size.min(MAX_UNPACKED))?;
        file.tokens = raw
            .split(|&x| x == 0)
            .take(count)
            .map(|t| String::from_utf8_lossy(t).into_owned())
            .collect();
    }
    {
        let mut c = section("STRINGS")?;
        let n = c.count(4)?;
        file.strings = (0..n).map(|_| c.u32()).collect::<R<_>>()?;
    }
    // Fields: a name and a value's encoding each.
    let fields: Vec<(String, Rep)> = {
        let mut c = section("FIELDS")?;
        let n = c.count(1)?;
        let names = c.ints(n, false)?;
        let size = c.u64()? as usize;
        let packed = c.take(size)?;
        let raw = decompress(packed, (n * 8).min(MAX_UNPACKED))?;
        if raw.len() < n * 8 {
            return bad("fields' values are short");
        }
        names
            .iter()
            .zip(raw.as_chunks::<8>().0)
            .map(|(t, r)| Ok((file.token(*t as u32 as u64)?, Rep(u64::from_le_bytes(*r)))))
            .collect::<R<_>>()?
    };
    let fieldsets: Vec<u32> = {
        let mut c = section("FIELDSETS")?;
        let n = c.count(1)?;
        c.ints(n, false)?.into_iter().map(|i| i as u32).collect()
    };
    // Paths: a tree, as each entry's token and how far to its sibling.
    // (parent, element, is a property)
    let mut tree: Vec<(Option<usize>, String, bool)>;
    {
        let mut c = section("PATHS")?;
        let total = c.count(1)?;
        let encoded = c.count(1)?;
        let indexes = c.ints(encoded, false)?;
        let elements = c.ints(encoded, false)?;
        let jumps = c.ints(encoded, false)?;
        file.paths = vec![String::new(); total];
        tree = vec![(None, String::new(), false); total];
        // Iterative form of USD's recursion: (index to read, parent path index).
        let mut stack: Vec<(usize, Option<usize>)> = vec![(0, None)];
        let mut guard = 0;
        while let Some((mut cur, mut parent)) = stack.pop() {
            loop {
                guard += 1;
                if guard > encoded * 2 + 2 {
                    return bad("the path tree loops");
                }
                let this = cur;
                cur += 1;
                let (Some(&pi), Some(&ti), Some(&jump)) =
                    (indexes.get(this), elements.get(this), jumps.get(this))
                else {
                    return bad("the path tree runs past its end");
                };
                let pi = pi as usize;
                if pi >= total {
                    return bad("a path index out of range");
                }
                match parent {
                    None => {
                        file.paths[pi] = "/".into();
                        tree[pi] = (None, String::new(), false);
                    }
                    Some(up) => {
                        let property = ti < 0;
                        let token = file.token(ti.unsigned_abs())?;
                        let base = file.paths[up].clone();
                        file.paths[pi] = if property {
                            format!("{base}.{token}")
                        } else if base == "/" {
                            format!("/{token}")
                        } else if token.starts_with('{') || base.ends_with('}') {
                            format!("{base}{token}")
                        } else {
                            format!("{base}/{token}")
                        };
                        tree[pi] = (Some(up), token, property);
                    }
                }
                let has_child = jump > 0 || jump == -1;
                let has_sibling = jump >= 0;
                if has_child {
                    if has_sibling {
                        let sibling = (this as i64 + jump) as usize;
                        stack.push((sibling, parent));
                    }
                    parent = Some(pi);
                }
                if !(has_child || has_sibling) {
                    break;
                }
            }
        }
    }
    // Specs: what is at each path.
    let specs: Vec<(usize, usize, u32)> = {
        let mut c = section("SPECS")?;
        let n = c.count(1)?;
        let paths = c.ints(n, false)?;
        let sets = c.ints(n, false)?;
        let kinds = c.ints(n, false)?;
        paths
            .iter()
            .zip(&sets)
            .zip(&kinds)
            .map(|((p, s), k)| (*p as usize, *s as usize, *k as u32))
            .collect()
    };

    let fields_of = |set: usize| -> Vec<&(String, Rep)> {
        let mut out = Vec::new();
        let mut i = set;
        while let Some(&f) = fieldsets.get(i) {
            if f == u32::MAX {
                break;
            }
            if let Some(field) = fields.get(f as usize) {
                out.push(field);
            }
            i += 1;
        }
        out
    };

    let mut layer = Layer::default();
    let mut prims: HashMap<usize, Prim> = HashMap::new();
    let mut properties: Vec<(usize, Property)> = Vec::new();
    let mut child_order: HashMap<usize, Vec<String>> = HashMap::new();
    for &(path, set, kind) in &specs {
        if path >= tree.len() {
            continue;
        }
        let fs = fields_of(set);
        let get = |name: &str| fs.iter().find(|(n, _)| n == name).map(|(_, r)| *r);
        let val = |name: &str| -> Option<V> { get(name).and_then(|r| file.value(r, 0).ok()) };
        match kind {
            // The layer itself.
            7 => {
                if let Some(V::Texts(t)) = val("upAxis") {
                    layer.up_axis = t.into_iter().next();
                }
                if let Some(V::Nums(n)) =
                    val("timeCodesPerSecond").or_else(|| val("framesPerSecond"))
                {
                    layer.time_codes_per_second = n.first().copied();
                }
                if let Some(V::Nums(n)) = val("startTimeCode") {
                    layer.start_time_code = n.first().copied();
                }
                if let Some(V::Nums(n)) = val("endTimeCode") {
                    layer.end_time_code = n.first().copied();
                }
                if let Some(V::Nums(n)) = val("metersPerUnit") {
                    layer.meters_per_unit = n.first().copied();
                }
                if let Some(V::Texts(t)) = val("defaultPrim") {
                    layer.default_prim = t.into_iter().next();
                }
                if let Some(V::Texts(t)) = val("subLayers") {
                    layer.sublayers = t;
                }
                if let Some(V::Texts(t)) = val("primChildren") {
                    child_order.insert(path, t);
                }
            }
            // A prim, or a variant's body.
            6 | 10 => {
                let mut prim = Prim {
                    name: tree[path].1.clone(),
                    ..Prim::default()
                };
                if let Some(V::Texts(t)) = val("typeName") {
                    prim.type_name = t.into_iter().next().unwrap_or_default();
                }
                prim.specifier = match val("specifier") {
                    Some(V::Specifier(1)) => Specifier::Over,
                    Some(V::Specifier(2)) => Specifier::Class,
                    _ if kind == 10 => Specifier::Over,
                    _ => Specifier::Def,
                };
                if let Some(V::Arcs(a)) = val("references") {
                    prim.references = a
                        .into_iter()
                        .map(|mut a| {
                            a.prim_path = a.prim_path.map(|p| plain(&p));
                            a
                        })
                        .collect();
                }
                if let Some(V::Arcs(a)) = val("payload") {
                    prim.payloads = a;
                }
                if let Some(V::Paths(p)) = val("inheritPaths") {
                    prim.inherits = p;
                }
                if let Some(V::Pairs(p)) = val("variantSelection") {
                    prim.variant_selection = p;
                }
                if let Some(V::Nums(n)) = val("active") {
                    prim.active = n.first().map(|a| *a != 0.0);
                }
                if let Some(V::Texts(t)) = val("kind") {
                    prim.kind = t.into_iter().next();
                }
                if let Some(V::Texts(t)) = val("apiSchemas") {
                    prim.api_schemas = t;
                }
                if let Some(V::Texts(t)) = val("primChildren") {
                    child_order.insert(path, t);
                }
                prims.insert(path, prim);
            }
            // An attribute, or a relationship.
            1 | 8 => {
                let Some(owner) = tree[path].0 else { continue };
                let name = tree[path].1.clone();
                let mut property = Property {
                    name,
                    ..Property::default()
                };
                if kind == 8 {
                    property.type_name = "rel".into();
                    if let Some(V::Paths(p)) = val("targetPaths") {
                        property.targets = p.iter().map(|p| plain(p)).collect();
                    }
                } else {
                    if let Some(V::Texts(t)) = val("typeName") {
                        property.type_name = t.into_iter().next().unwrap_or_default();
                    }
                    let plain_val = |v: V| match v {
                        V::Nums(n) => Some(Val::Nums(n)),
                        V::Texts(t) => Some(Val::Texts(t)),
                        _ => None,
                    };
                    if let Some(V::Samples(list)) = val("timeSamples") {
                        property.samples = list
                            .into_iter()
                            .filter_map(|(t, v)| Some((t, plain_val(v)?)))
                            .collect();
                    }
                    property.value = val("default")
                        .and_then(plain_val)
                        .or_else(|| property.samples.first().map(|(_, v)| v.clone()));
                    if let Some(V::Texts(t)) = val("interpolation") {
                        property.interpolation = t.into_iter().next();
                    }
                    if let Some(V::Nums(v)) = val("weight") {
                        property.weight = v.first().copied();
                    }
                    if let Some(V::Nums(v)) = val("variability") {
                        property.uniform = v.first() == Some(&1.0);
                    }
                    if let Some(V::Nums(n)) = val("elementSize") {
                        property.element_size = n.first().map(|e| *e as i64);
                    }
                    if let Some(V::Paths(p)) = val("connectionPaths") {
                        property.targets = p.iter().map(|p| plain(p)).collect();
                    }
                }
                properties.push((owner, property));
            }
            _ => {}
        }
    }
    for (owner, property) in properties {
        if let Some(p) = prims.get_mut(&owner) {
            p.properties.push(property);
        }
    }
    // Into a tree, deepest first, so every prim is whole when it is placed.
    let depth = |mut i: usize| {
        let mut d = 0;
        while let Some(up) = tree.get(i).and_then(|t| t.0) {
            i = up;
            d += 1;
            if d > tree.len() {
                break;
            }
        }
        d
    };
    let mut order: Vec<usize> = prims.keys().copied().collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(depth(i)), i));
    for i in order {
        let Some(mut prim) = prims.remove(&i) else {
            continue;
        };
        if let Some(names) = child_order.get(&i) {
            prim.children.sort_by_key(|c| {
                names
                    .iter()
                    .position(|n| *n == c.name)
                    .unwrap_or(usize::MAX)
            });
        }
        let parent = tree[i].0;
        let is_variant = prim.name.starts_with('{');
        match parent {
            Some(up) if file.paths.get(up).is_some_and(|p| p != "/") => {
                let holder = prims.entry(up).or_insert_with(|| Prim {
                    name: tree[up].1.clone(),
                    specifier: Specifier::Over,
                    ..Prim::default()
                });
                if is_variant {
                    // `{set=choice}`: a variant's body, for its set.
                    let inner = prim
                        .name
                        .trim_start_matches('{')
                        .trim_end_matches('}')
                        .to_string();
                    let Some((set, choice)) = inner.split_once('=') else {
                        continue;
                    };
                    if choice.is_empty() {
                        continue;
                    }
                    let set = set.to_string();
                    match holder.variant_sets.iter_mut().find(|(s, _)| *s == set) {
                        Some((_, list)) => list.push((choice.to_string(), prim)),
                        None => holder
                            .variant_sets
                            .push((set, vec![(choice.to_string(), prim)])),
                    }
                } else {
                    holder.children.push(prim);
                }
            }
            _ => layer.prims.push(prim),
        }
    }
    let root = file.paths.iter().position(|p| p == "/").unwrap_or(0);
    if let Some(names) = child_order.get(&root) {
        layer.prims.sort_by_key(|c| {
            names
                .iter()
                .position(|n| *n == c.name)
                .unwrap_or(usize::MAX)
        });
    }
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_copies_literals_and_overlapping_matches() {
        // "abc", then a match four back for eight: "abcabcab" after "abc"...
        // token: 3 literals, match length 4 + 4 = 8.
        let src = [0x34, b'a', b'b', b'c', 3, 0, 0x10, b'!'];
        let mut out = Vec::new();
        lz4_block(&src, &mut out, 100).unwrap();
        assert_eq!(out, b"abcabcabcab!");
        let mut out = Vec::new();
        assert!(
            lz4_block(&[0x10, b'a', 9, 0], &mut out, 100).is_err(),
            "a match before the start"
        );
    }

    #[test]
    fn integers_are_deltas_from_a_common_value() {
        // Common 1; codes: common, small(+5), medium(-300), large(+100000).
        let mut data = 1i32.to_le_bytes().to_vec();
        data.push(0b11_10_01_00);
        data.push(5u8);
        data.extend((-300i16).to_le_bytes());
        data.extend(100_000i32.to_le_bytes());
        assert_eq!(
            decode_ints(&data, 4, false).unwrap(),
            vec![1, 6, -294, 99_706]
        );
        assert!(decode_ints(&data[..6], 4, false).is_err());
    }

    #[test]
    fn variant_selections_leave_paths() {
        assert_eq!(plain("/A{v=x}B"), "/A/B");
        assert_eq!(plain("/A{v=x}B.outputs:surface"), "/A/B.outputs:surface");
        assert_eq!(plain("/A/B"), "/A/B");
    }

    #[test]
    fn what_is_not_a_crate_file_is_said() {
        assert!(parse_usdc(b"PXR-USDC").is_err());
        let mut b = b"PXR-USDC".to_vec();
        b.extend([0, 8, 0, 0, 0, 0, 0, 0]);
        b.extend(1_000_000u64.to_le_bytes());
        b.resize(100, 0);
        assert!(parse_usdc(&b).is_err(), "a table of contents past the end");
    }
}
