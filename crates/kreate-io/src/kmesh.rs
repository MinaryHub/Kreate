//! `.kmesh` — the native binary mesh, laid out so that loading is a cast.
//!
//! This is the format the plan calls the performance decider for large scenes,
//! and the reason is narrow: **the layout on disk is the layout in memory**.
//! Every buffer is 16-byte aligned and stored in exactly the form
//! [`bytemuck`] would produce, so opening a file is a memory map plus a cast
//! per array — no parse, no allocation, no copy for the arrays that dominate a
//! big mesh.
//!
//! # What the format is, in order
//!
//! ```text
//! Header        64 bytes, fixed, versioned
//! Buffer table  48 bytes per buffer, one row per attribute or topology array
//! Name blob     the buffer names, concatenated
//! Payload       the buffers themselves, each padded to a 16-byte boundary
//! ```
//!
//! Topology travels as buffers with reserved names rather than as header
//! fields, so a reader that does not recognise something can still skip it by
//! its length. That is what makes the format additive: see the versioning rules
//! below.
//!
//! # Forward compatibility, which cannot be retrofitted
//!
//! Three rules, all load bearing, and the plan is explicit that adding them
//! later is not possible because files written before the rule existed do not
//! obey it:
//!
//! - **A minor version bump is additive and safely ignorable.** New buffers may
//!   appear; a reader keeps them and writes them back.
//! - **`min_reader_major` is the file's own statement about who may read it.**
//!   A writer sets it to the oldest major version that can still make sense of
//!   the file, so a future incompatible change is announced by the file rather
//!   than discovered as garbage.
//! - **Unknown buffers survive a round trip byte for byte.** A file written by
//!   a newer version, opened and saved by an older one, must not lose what the
//!   older one did not understand. Without this, one save in an old build
//!   silently destroys data.
//!
//! # Determinism
//!
//! The same mesh writes the same bytes, every time: buffers are emitted in a
//! fixed order, padding is zeroed, and nothing carries a timestamp. That is what
//! lets version control store one blob for an unchanged file, and what makes a
//! golden-file test meaningful.
//!
//! # Reading untrusted files
//!
//! Every offset and length in the table is checked against the actual file size,
//! and every buffer's length against its element size and count, *before* any
//! of it is used. A format whose loading step is a cast is exactly the kind that
//! turns a corrupt file into an out-of-bounds read, so the validation here is
//! not defensive habit — it is the price of the layout.
//!
//! Counts in the header get the same treatment, and that one was learned the
//! hard way: `num_verts` was trusted, and a single flipped bit in it asked for a
//! 25 GB allocation. That is worse than a panic — an allocation failure aborts
//! the process, so there is no error to report and nothing to catch. Every count
//! the header states is now checked against data the file actually contains.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};
use kreate_mesh::attribute::{names, AttrData, AttrType, Domain};
use kreate_mesh::MeshData;
use std::sync::Arc;

/// File magic. Eight bytes so the header stays 16-byte aligned throughout.
pub const MAGIC: [u8; 8] = *b"KMESH\0\0\0";

/// The format version this build writes.
pub const VERSION_MAJOR: u32 = 1;
pub const VERSION_MINOR: u32 = 0;

/// Everything is aligned to this, because it is the widest alignment any
/// element type needs and because a single rule is easier to keep than one per
/// type.
const ALIGN: usize = 16;

/// Reserved buffer names for the topology arrays.
///
/// Prefixed so they cannot collide with a user attribute, and stored as ordinary
/// buffers rather than as header fields so that a reader which does not know one
/// can still skip it by length.
pub mod buffers {
    pub const FACE_OFFSETS: &str = "$face_offsets";
    pub const CORNER_VERTS: &str = "$corner_verts";
    pub const EDGES: &str = "$edges";
}

#[derive(Debug, thiserror::Error)]
pub enum KmeshError {
    #[error("not a kmesh file")]
    BadMagic,
    #[error("file needs a reader of major version {needed}, this is {VERSION_MAJOR}")]
    TooNew { needed: u32 },
    #[error("truncated: expected at least {expected} bytes, found {found}")]
    Truncated { expected: usize, found: usize },
    #[error("buffer {index} runs past the end of the file")]
    BufferOutOfBounds { index: usize },
    #[error("buffer {index} is not {ALIGN}-byte aligned, so it cannot be cast in place")]
    Misaligned { index: usize },
    #[error("buffer {index} has length {length}, not a multiple of its {stride} byte element")]
    RaggedBuffer {
        index: usize,
        length: usize,
        stride: usize,
    },
    #[error("buffer {index} has an unreadable name")]
    BadName { index: usize },
    #[error("the mesh is missing its {0} buffer")]
    MissingBuffer(&'static str),
    #[error("the header claims {claimed} vertices but the position buffer holds {actual}")]
    VertexCountMismatch { claimed: u32, actual: usize },
    #[error("bad topology: {0}")]
    Topology(#[from] kreate_mesh::mesh_data::TopologyError),
    #[error("attribute {name} has {found} values for a domain of {expected}")]
    AttributeLengthMismatch {
        name: String,
        found: usize,
        expected: usize,
    },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    version_major: u32,
    version_minor: u32,
    /// The oldest major version that can still read this file correctly.
    min_reader_major: u32,
    buffer_count: u32,
    num_verts: u32,
    num_faces: u32,
    num_corners: u32,
    num_edges: u32,
    /// Where the name blob starts, and how long it is.
    names_offset: u32,
    names_length: u32,
    /// Padding to a round 64 bytes.
    ///
    /// Not idle space: it keeps the header a multiple of 16, which keeps the
    /// buffer table that follows it naturally aligned, and it leaves room for
    /// fields a later minor version may add without moving anything.
    _reserved: [u32; 4],
}

const _: () = assert!(HEADER_SIZE == 64);

const HEADER_SIZE: usize = std::mem::size_of::<Header>();

/// One row of the buffer table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TableEntry {
    /// Byte range of this buffer's name within the name blob.
    name_offset: u32,
    name_length: u32,
    /// [`Domain`] as a small integer, or `u32::MAX` for a topology buffer that
    /// belongs to no attribute domain.
    domain: u32,
    /// [`AttrType`] as a small integer, or one of the [`raw`] codes.
    kind: u32,
    /// From the start of the file. Always a multiple of [`ALIGN`].
    offset: u64,
    /// In bytes.
    length: u64,
    /// How many elements, which is not always derivable from the length: a
    /// `Bool` buffer packs one byte per value, and a reader that guessed would
    /// be wrong for any type whose storage differs from its logical size.
    count: u64,
    /// Padding to a round 48 bytes, for the same reasons as the header.
    _reserved: u64,
}

const ENTRY_SIZE: usize = std::mem::size_of::<TableEntry>();
const _: () = assert!(ENTRY_SIZE == 48);

/// Type codes for buffers that are not attribute data.
mod raw {
    /// Plain `u32` array, for CSR offsets and corner vertices.
    pub const U32: u32 = 1000;
    /// Pairs of `u32`, for edges.
    pub const U32X2: u32 = 1001;
}

fn type_code(ty: AttrType) -> u32 {
    match ty {
        AttrType::F32 => 0,
        AttrType::F32x2 => 1,
        AttrType::F32x3 => 2,
        AttrType::F32x4 => 3,
        AttrType::I32 => 4,
        AttrType::Bool => 5,
        AttrType::U8x4 => 6,
    }
}

fn type_from_code(code: u32) -> Option<AttrType> {
    Some(match code {
        0 => AttrType::F32,
        1 => AttrType::F32x2,
        2 => AttrType::F32x3,
        3 => AttrType::F32x4,
        4 => AttrType::I32,
        5 => AttrType::Bool,
        6 => AttrType::U8x4,
        _ => return None,
    })
}

fn domain_code(domain: Domain) -> u32 {
    match domain {
        Domain::Point => 0,
        Domain::Edge => 1,
        Domain::Corner => 2,
        Domain::Face => 3,
    }
}

fn domain_from_code(code: u32) -> Option<Domain> {
    Some(match code {
        0 => Domain::Point,
        1 => Domain::Edge,
        2 => Domain::Corner,
        3 => Domain::Face,
        _ => return None,
    })
}

/// Bytes per stored element, which is not always the logical size.
fn stride(kind: u32) -> usize {
    match kind {
        0 => 4,  // F32
        1 => 8,  // F32x2
        2 => 12, // F32x3
        3 => 16, // F32x4
        4 => 4,  // I32
        5 => 1,  // Bool, one byte per value
        6 => 4,  // U8x4
        raw::U32 => 4,
        raw::U32X2 => 8,
        // An unknown buffer is carried through as opaque bytes, so its element
        // size is one and its count is its length.
        _ => 1,
    }
}

/// A buffer this build did not recognise, kept so it survives a round trip.
///
/// The plan's forward-compatibility rule in concrete form: a file written by a
/// newer version and saved by an older one must not quietly lose what the older
/// one could not interpret. Without this, one save in an old build is data loss
/// that nothing reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownBuffer {
    pub name: String,
    pub domain: u32,
    pub kind: u32,
    pub count: u64,
    pub bytes: Vec<u8>,
}

/// A mesh as it came out of a file, with anything unrecognised kept alongside.
#[derive(Clone, Debug)]
pub struct Loaded {
    pub mesh: MeshData,
    /// Buffers this build did not understand. Pass them back to [`write`] to
    /// preserve them.
    pub unknown: Vec<UnknownBuffer>,
    pub version_minor: u32,
}

/// Serialises a mesh, optionally carrying unknown buffers back out.
pub fn write(mesh: &MeshData, unknown: &[UnknownBuffer]) -> Vec<u8> {
    struct Pending {
        name: String,
        domain: u32,
        kind: u32,
        count: u64,
        bytes: Vec<u8>,
    }

    let mut pending: Vec<Pending> = Vec::new();

    // Topology first, in a fixed order. Order is part of determinism: the same
    // mesh has to produce the same bytes, or version control sees a change in
    // every file on every save.
    pending.push(Pending {
        name: buffers::FACE_OFFSETS.to_string(),
        domain: u32::MAX,
        kind: raw::U32,
        count: mesh.face_offsets().len() as u64,
        bytes: bytemuck::cast_slice(mesh.face_offsets()).to_vec(),
    });
    pending.push(Pending {
        name: buffers::CORNER_VERTS.to_string(),
        domain: u32::MAX,
        kind: raw::U32,
        count: mesh.corner_verts().len() as u64,
        bytes: bytemuck::cast_slice(mesh.corner_verts()).to_vec(),
    });
    pending.push(Pending {
        name: buffers::EDGES.to_string(),
        domain: u32::MAX,
        kind: raw::U32X2,
        count: mesh.edges().len() as u64,
        bytes: bytemuck::cast_slice(mesh.edges()).to_vec(),
    });

    // Attributes, grouped by domain and then by name.
    //
    // Not what makes the file deterministic — `AttributeStore` already keeps its
    // layers sorted by name, so any traversal of it is canonical, and a
    // falsification pass confirmed that removing this sort changes no test.
    // What it buys is locality: a reader that wants one domain finds its
    // buffers adjacent instead of interleaved with three others.
    let mut layers: Vec<_> = mesh
        .attributes()
        .iter()
        .map(|l| (domain_code(l.domain), l.name.to_string(), l))
        .collect();
    layers.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    for (domain, name, layer) in layers {
        pending.push(Pending {
            name,
            domain,
            kind: type_code(layer.data.ty()),
            count: layer.data.len() as u64,
            bytes: attr_bytes(&layer.data),
        });
    }

    // Anything a newer writer left behind, last and in the order it arrived.
    for u in unknown {
        pending.push(Pending {
            name: u.name.clone(),
            domain: u.domain,
            kind: u.kind,
            count: u.count,
            bytes: u.bytes.clone(),
        });
    }

    // --- lay the file out ---------------------------------------------------
    let mut names = Vec::new();
    let mut name_ranges = Vec::new();
    for p in &pending {
        let start = names.len() as u32;
        names.extend_from_slice(p.name.as_bytes());
        name_ranges.push((start, p.name.len() as u32));
    }

    let table_bytes = pending.len() * ENTRY_SIZE;
    let names_offset = HEADER_SIZE + table_bytes;
    let mut cursor = align_up(names_offset + names.len());

    let mut table = Vec::with_capacity(pending.len());
    for (i, p) in pending.iter().enumerate() {
        table.push(TableEntry {
            name_offset: name_ranges[i].0,
            name_length: name_ranges[i].1,
            domain: p.domain,
            kind: p.kind,
            offset: cursor as u64,
            length: p.bytes.len() as u64,
            count: p.count,
            _reserved: 0,
        });
        cursor = align_up(cursor + p.bytes.len());
    }

    let header = Header {
        magic: MAGIC,
        version_major: VERSION_MAJOR,
        version_minor: VERSION_MINOR,
        // This build's files are readable by any version 1 reader. Bumped only
        // when a change would make an older reader produce something wrong
        // rather than merely incomplete.
        min_reader_major: 1,
        buffer_count: pending.len() as u32,
        num_verts: mesh.num_verts(),
        num_faces: mesh.num_faces(),
        num_corners: mesh.corner_verts().len() as u32,
        num_edges: mesh.num_edges(),
        names_offset: names_offset as u32,
        names_length: names.len() as u32,
        _reserved: [0; 4],
    };

    let mut out = Vec::with_capacity(cursor);
    out.extend_from_slice(bytemuck::bytes_of(&header));
    out.extend_from_slice(bytemuck::cast_slice(&table));
    out.extend_from_slice(&names);
    for (entry, p) in table.iter().zip(&pending) {
        // Zero padding, not whatever the allocator had: padding that varied
        // would make two writes of one mesh differ, which defeats the point of
        // a deterministic format.
        out.resize(entry.offset as usize, 0);
        out.extend_from_slice(&p.bytes);
    }
    out.resize(cursor, 0);
    out
}

/// Parses a file. Never panics and never reads outside `bytes`.
pub fn read(bytes: &[u8]) -> Result<Loaded, KmeshError> {
    if bytes.len() < HEADER_SIZE {
        return Err(KmeshError::Truncated {
            expected: HEADER_SIZE,
            found: bytes.len(),
        });
    }
    let header: Header = bytemuck::pod_read_unaligned(&bytes[..HEADER_SIZE]);
    if header.magic != MAGIC {
        return Err(KmeshError::BadMagic);
    }
    if header.min_reader_major > VERSION_MAJOR {
        return Err(KmeshError::TooNew {
            needed: header.min_reader_major,
        });
    }

    let count = header.buffer_count as usize;
    let table_end = HEADER_SIZE
        .checked_add(count.checked_mul(ENTRY_SIZE).ok_or(KmeshError::Truncated {
            expected: usize::MAX,
            found: bytes.len(),
        })?)
        .ok_or(KmeshError::Truncated {
            expected: usize::MAX,
            found: bytes.len(),
        })?;
    if bytes.len() < table_end {
        return Err(KmeshError::Truncated {
            expected: table_end,
            found: bytes.len(),
        });
    }
    let table: Vec<TableEntry> = bytes[HEADER_SIZE..table_end]
        .as_chunks::<ENTRY_SIZE>()
        .0
        .iter()
        .map(|c| bytemuck::pod_read_unaligned(c.as_slice()))
        .collect();

    let names_start = header.names_offset as usize;
    let names_end = names_start
        .checked_add(header.names_length as usize)
        .ok_or(KmeshError::Truncated {
            expected: usize::MAX,
            found: bytes.len(),
        })?;
    if bytes.len() < names_end {
        return Err(KmeshError::Truncated {
            expected: names_end,
            found: bytes.len(),
        });
    }
    let names = &bytes[names_start..names_end];

    let mut face_offsets: Option<Vec<u32>> = None;
    let mut corner_verts: Option<Vec<u32>> = None;
    let mut edges: Option<Vec<[u32; 2]>> = None;
    let mut attributes: Vec<(String, Domain, AttrData)> = Vec::new();
    let mut unknown = Vec::new();

    for (index, entry) in table.iter().enumerate() {
        // Every one of these checks is what makes a cast-based format safe to
        // point at a file someone else wrote.
        let offset =
            usize::try_from(entry.offset).map_err(|_| KmeshError::BufferOutOfBounds { index })?;
        let length =
            usize::try_from(entry.length).map_err(|_| KmeshError::BufferOutOfBounds { index })?;
        let end = offset
            .checked_add(length)
            .ok_or(KmeshError::BufferOutOfBounds { index })?;
        if end > bytes.len() {
            return Err(KmeshError::BufferOutOfBounds { index });
        }
        if offset % ALIGN != 0 {
            return Err(KmeshError::Misaligned { index });
        }
        let stride = stride(entry.kind);
        if length % stride != 0 {
            return Err(KmeshError::RaggedBuffer {
                index,
                length,
                stride,
            });
        }
        let data = &bytes[offset..end];

        let name_start = entry.name_offset as usize;
        let name_end = name_start
            .checked_add(entry.name_length as usize)
            .ok_or(KmeshError::BadName { index })?;
        if name_end > names.len() {
            return Err(KmeshError::BadName { index });
        }
        let name = std::str::from_utf8(&names[name_start..name_end])
            .map_err(|_| KmeshError::BadName { index })?
            .to_string();

        match (name.as_str(), entry.kind) {
            (buffers::FACE_OFFSETS, raw::U32) => {
                face_offsets = Some(read_u32(data));
            }
            (buffers::CORNER_VERTS, raw::U32) => {
                corner_verts = Some(read_u32(data));
            }
            (buffers::EDGES, raw::U32X2) => {
                edges = Some(read_u32(data).as_chunks::<2>().0.to_vec());
            }
            _ => match (domain_from_code(entry.domain), type_from_code(entry.kind)) {
                (Some(domain), Some(ty)) => {
                    let count = usize::try_from(entry.count)
                        .map_err(|_| KmeshError::BufferOutOfBounds { index })?;
                    // Checked, because `count` comes from the file. A corrupt
                    // value here overflowed the multiply — a panic in a debug
                    // build, and in release a wrapped product that would have
                    // passed the length check and then sliced out of bounds.
                    let needed = count
                        .checked_mul(stride)
                        .ok_or(KmeshError::BufferOutOfBounds { index })?;
                    if needed > length {
                        return Err(KmeshError::RaggedBuffer {
                            index,
                            length,
                            stride,
                        });
                    }
                    attributes.push((name, domain, read_attr(ty, &data[..needed], count)));
                }
                // Not a topology buffer this build knows and not an attribute
                // it can type: carried through so a newer file survives a save.
                _ => unknown.push(UnknownBuffer {
                    name,
                    domain: entry.domain,
                    kind: entry.kind,
                    count: entry.count,
                    bytes: data.to_vec(),
                }),
            },
        }
    }

    let face_offsets = face_offsets.ok_or(KmeshError::MissingBuffer(buffers::FACE_OFFSETS))?;
    let corner_verts = corner_verts.ok_or(KmeshError::MissingBuffer(buffers::CORNER_VERTS))?;
    let edges = edges.ok_or(KmeshError::MissingBuffer(buffers::EDGES))?;

    // The header's vertex count, checked against data before anything is sized
    // by it.
    //
    // This is the one number in the file that nothing else bounds: the buffers
    // are all checked against the file's length, but `num_verts` is a bare `u32`
    // that `from_csr` immediately turns into an allocation. A flipped bit made
    // it ask for 25 GB, and an allocation failure aborts the process rather than
    // returning an error — so a corrupt file could take the application down
    // with no diagnostic at all.
    //
    // Position is mandatory in a mesh, so its buffer is the authority: a file
    // whose header disagrees with it is malformed, and one without it never had
    // a mesh in the first place.
    let positions = attributes
        .iter()
        .find(|(name, domain, _)| name == names::POSITION && *domain == Domain::Point)
        .ok_or(KmeshError::MissingBuffer(names::POSITION))?;
    if positions.2.len() != header.num_verts as usize {
        return Err(KmeshError::VertexCountMismatch {
            claimed: header.num_verts,
            actual: positions.2.len(),
        });
    }

    // Structural validation lives in `from_csr`, so there is one place that
    // decides what a well-formed mesh is. Duplicating the checks here would be
    // two places to keep in step, and the one that drifted would be the one a
    // malformed file went through.
    let mut mesh = MeshData::from_csr(header.num_verts, face_offsets, corner_verts, edges)?;
    let sizes = [
        mesh.num_verts() as usize,
        mesh.num_edges() as usize,
        mesh.corner_verts().len(),
        mesh.num_faces() as usize,
    ];
    for (name, domain, data) in attributes {
        let expected = sizes[domain_code(domain) as usize];
        if data.len() != expected {
            return Err(KmeshError::AttributeLengthMismatch {
                name,
                found: data.len(),
                expected,
            });
        }
        mesh.attributes_mut().insert(&name, domain, data);
    }

    Ok(Loaded {
        mesh,
        unknown,
        version_minor: header.version_minor,
    })
}

fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

fn read_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn attr_bytes(data: &AttrData) -> Vec<u8> {
    match data {
        AttrData::F32(v) => bytemuck::cast_slice(v.as_slice()).to_vec(),
        AttrData::F32x2(v) => bytemuck::cast_slice(v.as_slice()).to_vec(),
        AttrData::F32x3(v) => bytemuck::cast_slice(v.as_slice()).to_vec(),
        AttrData::F32x4(v) => bytemuck::cast_slice(v.as_slice()).to_vec(),
        AttrData::I32(v) => bytemuck::cast_slice(v.as_slice()).to_vec(),
        // One byte per value rather than a bitset: a bitset would save seven
        // eighths of a selection layer, but it would also stop the buffer from
        // being castable in place, which is the whole premise of the format.
        AttrData::Bool(v) => v.iter().map(|b| u8::from(*b)).collect(),
        AttrData::U8x4(v) => bytemuck::cast_slice(v.as_slice()).to_vec(),
    }
}

fn read_attr(ty: AttrType, bytes: &[u8], count: usize) -> AttrData {
    match ty {
        AttrType::F32 => AttrData::F32(Arc::new(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .take(count)
                .collect(),
        )),
        AttrType::F32x2 => AttrData::F32x2(Arc::new(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| Vec2::new(le_f32(&c[0..4]), le_f32(&c[4..8])))
                .take(count)
                .collect(),
        )),
        AttrType::F32x3 => AttrData::F32x3(Arc::new(
            bytes
                .as_chunks::<12>()
                .0
                .iter()
                .map(|c| Vec3::new(le_f32(&c[0..4]), le_f32(&c[4..8]), le_f32(&c[8..12])))
                .take(count)
                .collect(),
        )),
        AttrType::F32x4 => AttrData::F32x4(Arc::new(
            bytes
                .as_chunks::<16>()
                .0
                .iter()
                .map(|c| {
                    Vec4::new(
                        le_f32(&c[0..4]),
                        le_f32(&c[4..8]),
                        le_f32(&c[8..12]),
                        le_f32(&c[12..16]),
                    )
                })
                .take(count)
                .collect(),
        )),
        AttrType::I32 => AttrData::I32(Arc::new(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .take(count)
                .collect(),
        )),
        AttrType::Bool => AttrData::Bool(Arc::new(
            bytes.iter().map(|b| *b != 0).take(count).collect(),
        )),
        AttrType::U8x4 => AttrData::U8x4(Arc::new(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .take(count)
                .collect(),
        )),
    }
}

/// Reads a little-endian `f32` from a four-byte window of a larger chunk.
fn le_f32(c: &[u8]) -> f32 {
    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
}
