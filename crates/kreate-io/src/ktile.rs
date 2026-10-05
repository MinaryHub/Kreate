//! `.ktile` — an image's resident tiles, laid out so that loading is a cast.
//!
//! The same argument as [`crate::kmesh`], applied to pixels: the layout on disk
//! is the layout in memory, every tile is 16-byte aligned, and opening a file is
//! a memory map plus a copy per tile rather than a decode.
//!
//! # Why the archive does not just hold a PNG
//!
//! It is the obvious thing and it is wrong for three reasons, each of which
//! would have to be undone later.
//!
//! **A PNG is dense.** §3's sixth decision made images sparse — a tile is
//! individually present or absent, and "upload only what the camera sees" and
//! "load only the mip levels in use" are that one mechanism. Encoding to PNG
//! throws the residency away on every save: a 16K texture with four tiles
//! painted comes back as a gigabyte of mostly-nothing.
//!
//! **A PNG is a decode.** The point of storing tiles is that a tile is already
//! the thing a sampler and an undo record address. Round-tripping through a
//! codec puts a parse between the file and the buffer, and does it for the
//! whole image when four tiles changed.
//!
//! **A PNG cannot hold what the datablock holds.** `Rgba16F` is what an HDRI
//! is, and the range above one is the entire reason it exists — §2.1's working
//! space has no upper bound. A format that clips at one would quietly turn a
//! sky into a flat white ceiling.
//!
//! So a `.png` on disk is an image's *source* (see [`Source::File`]), and this
//! is how the archive carries pixels that exist nowhere else.
//!
//! # What the format is, in order
//!
//! ```text
//! Header   48 bytes, fixed, versioned
//! Index    24 bytes per resident tile: where it is and where its bytes are
//! Payload  the tiles, each padded to a 16-byte boundary
//! ```
//!
//! Only *resident* tiles are written, and the index states which. An image with
//! four tiles painted writes four tiles, whatever its declared size.
//!
//! # A blank resident tile is still written
//!
//! A tile that is resident and all zeroes costs its full 64 KB here, and
//! skipping it would save that. It is not skipped, because "resident and blank"
//! and "not resident" are different states of the datablock — they read the
//! same and they cost differently, and a save that collapsed one into the other
//! would make a round trip stop being an identity. A format whose round trip is
//! *nearly* an identity is one nobody can write a test for.
//!
//! # Reading untrusted files
//!
//! Every offset and length is checked against the actual file size before any
//! of it is used, and the tile count is checked against what the index can
//! physically occupy *before* anything is allocated from it. That is the lesson
//! [`crate::kmesh`] records the hard way: a flipped bit in a count asked for a
//! 25 GB allocation, and an allocation failure aborts the process — there is no
//! error to report and nothing to catch.

use bytemuck::{Pod, Zeroable};
use kreate_scene::image::{Image, PixelFormat, Tile, TILE};
use std::sync::Arc;

/// File magic. Eight bytes so the header stays 16-byte aligned throughout.
pub const MAGIC: [u8; 8] = *b"KTILE\0\0\0";

pub const VERSION_MAJOR: u32 = 1;
pub const VERSION_MINOR: u32 = 0;

/// Everything in the payload is aligned to this, for the same reason
/// [`crate::kmesh`] uses it: one rule is easier to keep than one per format.
const ALIGN: usize = 16;

#[derive(Debug, thiserror::Error)]
pub enum KtileError {
    #[error("not a ktile file")]
    BadMagic,
    #[error("image needs a reader of major version {needed}, this is {VERSION_MAJOR}")]
    TooNew { needed: u32 },
    #[error("truncated: expected at least {expected} bytes, found {found}")]
    Truncated { expected: usize, found: usize },
    #[error("unknown pixel format {0}")]
    UnknownFormat(u32),
    /// The tile edge is a constant of the datablock, not a free parameter.
    ///
    /// Refused rather than resampled: a file written when a tile was 128 pixels
    /// cannot be read into an image whose tiles are 256 without moving every
    /// pixel, and doing that silently would make the constant unchangeable in
    /// practice while pretending otherwise.
    #[error("the file's tiles are {found} pixels, this build's are {expected}")]
    TileSize { found: u32, expected: u32 },
    #[error("tile {index} runs past the end of the file")]
    TileOutOfBounds { index: usize },
    #[error("tile {index} is not {ALIGN}-byte aligned, so it cannot be cast in place")]
    Misaligned { index: usize },
    #[error("tile {index} holds {found} bytes, not the {expected} a tile of this format is")]
    TileLength {
        index: usize,
        found: u64,
        expected: usize,
    },
    #[error("tile {index} is at ({x}, {y}), outside an image of {across}×{down} tiles")]
    TileOutsideImage {
        index: usize,
        x: u32,
        y: u32,
        across: u32,
        down: u32,
    },
    #[error("an image of {width}×{height} cannot be addressed or allocated")]
    ImplausibleSize { width: u32, height: u32 },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    version_major: u32,
    version_minor: u32,
    /// The oldest major version that can still make sense of this file.
    min_reader_major: u32,
    /// [`PixelFormat`] as a small integer.
    format: u32,
    width: u32,
    height: u32,
    /// [`TILE`] as it was when this was written. See [`KtileError::TileSize`].
    tile_edge: u32,
    /// How many tiles the index holds.
    tile_count: u32,
    /// Padding to a round 48 bytes, which keeps the index that follows
    /// naturally aligned and leaves room for a later minor version.
    _reserved: [u32; 2],
}

const HEADER_SIZE: usize = std::mem::size_of::<Header>();
const _: () = assert!(HEADER_SIZE == 48);

/// One row of the tile index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct IndexEntry {
    tile_x: u32,
    tile_y: u32,
    /// From the start of the file. Always a multiple of [`ALIGN`].
    offset: u64,
    /// In bytes.
    ///
    /// Derivable from the format for every tile this build writes, and stored
    /// anyway: it is what lets a reader *skip* a tile it does not understand,
    /// which is the rule that makes a minor version addable at all. Per-tile
    /// compression is the obvious one, and without a length here it could not
    /// be added without a major bump.
    length: u64,
}

const INDEX_SIZE: usize = std::mem::size_of::<IndexEntry>();
const _: () = assert!(INDEX_SIZE == 24);

/// A tile is a multiple of [`ALIGN`] whatever its format, and this is the proof.
///
/// `TILE × TILE` is 16,384, so a tile of any whole number of bytes per pixel is
/// a multiple of sixteen by construction. **The padding between tiles in
/// [`write`] therefore never runs**, and the sweep found it: a break that
/// removed it changed nothing.
///
/// It stays rather than being deleted, for the reason [`IndexEntry::length`]
/// exists — a per-tile compression is the obvious minor version, and a
/// compressed tile is not a multiple of anything. Said here with a proof beside
/// it rather than left as a maybe, because a guard nobody can tell is dead is
/// worse than one that says so.
const _: () = assert!(((TILE * TILE) as usize).is_multiple_of(ALIGN));

fn format_code(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Rgba8 => 0,
        PixelFormat::Rgba16F => 1,
        PixelFormat::R8 => 2,
    }
}

fn format_of(code: u32) -> Option<PixelFormat> {
    Some(match code {
        0 => PixelFormat::Rgba8,
        1 => PixelFormat::Rgba16F,
        2 => PixelFormat::R8,
        _ => return None,
    })
}

/// How many bytes one tile of `format` occupies.
fn tile_bytes(format: PixelFormat) -> usize {
    (TILE * TILE) as usize * format.bytes_per_pixel()
}

/// Serialises an image's resident tiles.
///
/// Deterministic: tiles are written in row-major order, padding is zeroed, and
/// nothing carries a timestamp — so two saves of an unchanged image produce the
/// same bytes and version control stores one copy.
pub fn write(image: &Image) -> Vec<u8> {
    let (across, down) = image.tile_counts();
    let stride = tile_bytes(image.format);

    // Row-major, which is both the order the tiles are stored in and the order
    // a person reading a hex dump would guess.
    let resident: Vec<(u32, u32)> = (0..down)
        .flat_map(|y| (0..across).map(move |x| (x, y)))
        .filter(|(x, y)| image.tile(*x, *y).is_some())
        .collect();

    let index_start = HEADER_SIZE;
    let payload_start = pad_to(index_start + resident.len() * INDEX_SIZE);

    let mut out = Vec::with_capacity(payload_start + resident.len() * pad_to(stride));
    out.extend_from_slice(bytemuck::bytes_of(&Header {
        magic: MAGIC,
        version_major: VERSION_MAJOR,
        version_minor: VERSION_MINOR,
        min_reader_major: VERSION_MAJOR,
        format: format_code(image.format),
        width: image.width(),
        height: image.height(),
        tile_edge: TILE,
        tile_count: resident.len() as u32,
        _reserved: [0; 2],
    }));

    let mut offset = payload_start;
    for (x, y) in &resident {
        out.extend_from_slice(bytemuck::bytes_of(&IndexEntry {
            tile_x: *x,
            tile_y: *y,
            offset: offset as u64,
            length: stride as u64,
        }));
        offset = pad_to(offset + stride);
    }

    while out.len() < payload_start {
        out.push(0);
    }
    for (x, y) in &resident {
        let tile = image.tile(*x, *y).expect("listed because it is resident");
        out.extend_from_slice(tile.bytes());
        // Unreachable today — see the assertion beside `INDEX_SIZE` — and kept
        // for the compressed tile that `IndexEntry::length` is there for.
        while !out.len().is_multiple_of(ALIGN) {
            out.push(0);
        }
    }
    out
}

/// Reads an image's geometry and resident tiles.
///
/// The name is passed in because it is not here: an image's name, colour space
/// and source belong to the scene, which is where a reader has already got them
/// from. Writing them twice would be two places to disagree.
pub fn read(bytes: &[u8], name: &str) -> Result<Image, KtileError> {
    if bytes.len() < HEADER_SIZE {
        return Err(KtileError::Truncated {
            expected: HEADER_SIZE,
            found: bytes.len(),
        });
    }
    let header: Header = bytemuck::pod_read_unaligned(&bytes[..HEADER_SIZE]);
    if header.magic != MAGIC {
        return Err(KtileError::BadMagic);
    }
    if header.min_reader_major > VERSION_MAJOR {
        return Err(KtileError::TooNew {
            needed: header.min_reader_major,
        });
    }
    if header.tile_edge != TILE {
        return Err(KtileError::TileSize {
            found: header.tile_edge,
            expected: TILE,
        });
    }
    let format = format_of(header.format).ok_or(KtileError::UnknownFormat(header.format))?;

    // The tile count before the index is read: the file has to physically
    // contain the rows it claims.
    let claimed = header.tile_count as usize;
    let index_end = HEADER_SIZE
        .checked_add(claimed.saturating_mul(INDEX_SIZE))
        .ok_or(KtileError::Truncated {
            expected: usize::MAX,
            found: bytes.len(),
        })?;
    if bytes.len() < index_end {
        return Err(KtileError::Truncated {
            expected: index_end,
            found: bytes.len(),
        });
    }

    // A declared size costs a pointer per tile before any pixels arrive, so a
    // size out of a file is asked for rather than taken — `Image::try_new` is
    // there for exactly this caller, and the alternative to it is the allocator
    // ending the process with nothing to report.
    let mut image = Image::try_new(name, header.width, header.height, format).ok_or(
        KtileError::ImplausibleSize {
            width: header.width,
            height: header.height,
        },
    )?;
    let stride = tile_bytes(format);
    let (across, down) = image.tile_counts();

    for i in 0..claimed {
        let start = HEADER_SIZE + i * INDEX_SIZE;
        let entry: IndexEntry = bytemuck::pod_read_unaligned(&bytes[start..start + INDEX_SIZE]);
        let offset = entry.offset as usize;
        let length = entry.length as usize;
        // Two checks, and the first does nothing on a 64-bit machine: there
        // `offset` is the whole of `entry.offset`, so anything past the end is
        // caught by the second. It is here for a 32-bit one — wasm32 is on the
        // roadmap — where the cast truncates and a `u64` offset of four
        // gigabytes plus one lands at one. The same shape as the bounds that
        // exist for the shader in `kreate-math`'s AgX: a line that does nothing
        // where the tests run and everything where they do not.
        if entry.offset > bytes.len() as u64
            || offset
                .checked_add(length)
                .is_none_or(|end| end > bytes.len())
        {
            return Err(KtileError::TileOutOfBounds { index: i });
        }
        if !offset.is_multiple_of(ALIGN) {
            return Err(KtileError::Misaligned { index: i });
        }
        if length != stride {
            return Err(KtileError::TileLength {
                index: i,
                found: entry.length,
                expected: stride,
            });
        }
        if entry.tile_x >= across || entry.tile_y >= down {
            return Err(KtileError::TileOutsideImage {
                index: i,
                x: entry.tile_x,
                y: entry.tile_y,
                across,
                down,
            });
        }
        let mut tile = Tile::blank(format);
        tile.bytes_mut()
            .copy_from_slice(&bytes[offset..offset + length]);
        image.set_tile(entry.tile_x, entry.tile_y, Some(Arc::new(tile)));
    }
    Ok(image)
}

fn pad_to(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An image with a few tiles written, each recognisably its own.
    fn patchy() -> Image {
        let mut image = Image::new("Patchy", 500, 300, PixelFormat::Rgba8);
        for (n, (x, y)) in [(0u32, 0u32), (3, 1), (1, 2)].iter().enumerate() {
            let tile = image.tile_mut(*x, *y).expect("in range");
            tile.bytes_mut()[0] = n as u8 + 1;
            // The last byte too, so a length that is short by anything at all
            // loses something the test can see.
            let end = tile.bytes().len() - 1;
            tile.bytes_mut()[end] = 0xAB;
        }
        image
    }

    #[test]
    fn only_the_resident_tiles_are_written() {
        // The whole reason this is not a PNG. A 500×300 image is 4×3 tiles;
        // three of the twelve hold anything, and the file is the size of three.
        let image = patchy();
        assert_eq!(image.tile_count(), 12);
        assert_eq!(image.resident_tiles(), 3);
        let bytes = write(&image);
        let stride = tile_bytes(PixelFormat::Rgba8);
        assert!(
            bytes.len() < 4 * stride,
            "three tiles were written as {} bytes, which is {} tiles' worth",
            bytes.len(),
            bytes.len() / stride
        );
    }

    #[test]
    fn a_patchy_image_comes_back_exactly() {
        let image = patchy();
        let back = read(&write(&image), "Patchy").expect("what we wrote reads");
        assert_eq!(back.width(), image.width());
        assert_eq!(back.height(), image.height());
        assert_eq!(back.format, image.format);
        assert_eq!(back.resident_tiles(), image.resident_tiles());
        let (across, down) = image.tile_counts();
        for y in 0..down {
            for x in 0..across {
                assert_eq!(
                    image.tile(x, y).map(|t| t.bytes()),
                    back.tile(x, y).map(|t| t.bytes()),
                    "tile ({x}, {y}) differs"
                );
            }
        }
    }

    #[test]
    fn a_resident_blank_tile_is_still_there_afterwards() {
        // "Resident and blank" and "not resident" read the same and cost
        // differently, so collapsing one into the other on save would make the
        // round trip stop being an identity — and a format whose round trip is
        // *nearly* an identity is one nobody can write a test for.
        let mut image = Image::new("Blank", 128, 128, PixelFormat::Rgba8);
        image.tile_mut(0, 0).expect("in range");
        assert_eq!(image.resident_tiles(), 1);
        assert!(image
            .tile(0, 0)
            .expect("resident")
            .bytes()
            .iter()
            .all(|b| *b == 0));

        let back = read(&write(&image), "Blank").expect("reads");
        assert_eq!(
            back.resident_tiles(),
            1,
            "a blank tile was dropped, so an image's memory changed across a save"
        );
    }

    #[test]
    fn the_same_image_writes_the_same_bytes() {
        // What lets version control store one copy of an unchanged file, and
        // what makes a golden-file test mean anything.
        assert_eq!(write(&patchy()), write(&patchy()));
    }

    #[test]
    fn every_tile_lands_on_an_alignment_boundary() {
        // The reason the format exists: a tile that is not aligned cannot be
        // cast in place, which is the whole of what STORED buys.
        let image = patchy();
        let bytes = write(&image);
        for i in 0..3 {
            let start = HEADER_SIZE + i * INDEX_SIZE;
            let entry: IndexEntry = bytemuck::pod_read_unaligned(&bytes[start..start + INDEX_SIZE]);
            assert!(
                (entry.offset as usize).is_multiple_of(ALIGN),
                "tile {i} is at {}",
                entry.offset
            );
        }
    }

    #[test]
    fn an_hdr_image_keeps_its_range() {
        // `Rgba16F` is the reason a PNG could not do this job: §2.1's working
        // space has no upper bound, and a sky is mostly above one.
        let mut image = Image::new("Sky", 128, 128, PixelFormat::Rgba16F);
        let tile = image.tile_mut(0, 0).expect("in range");
        // 800.0 as a half float, which no eight-bit format can hold.
        let bright = half_bits(800.0);
        tile.bytes_mut()[0..2].copy_from_slice(&bright.to_le_bytes());
        let stride = tile_bytes(PixelFormat::Rgba16F);
        assert_eq!(tile.bytes().len(), stride);

        let back = read(&write(&image), "Sky").expect("reads");
        let got = back.tile(0, 0).expect("resident").bytes();
        assert_eq!(
            u16::from_le_bytes([got[0], got[1]]),
            bright,
            "the bright pixel came back as something else"
        );
    }

    /// The IEEE half-precision bits of a value, for the test above.
    fn half_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mantissa = ((bits >> 13) & 0x3ff) as u16;
        sign | ((exponent as u16) << 10) | mantissa
    }

    #[test]
    fn a_file_that_is_not_one_is_refused_rather_than_read() {
        assert!(matches!(read(b"", "x"), Err(KtileError::Truncated { .. })));
        assert!(matches!(read(&[0u8; 64], "x"), Err(KtileError::BadMagic)));
    }

    #[test]
    fn a_file_is_read_wherever_its_bytes_start() {
        // Inside a zip, or after anything else in a buffer, the bytes need not
        // start where a header's fields would like them to. Reading must not
        // care: an alignment assumption here was a panic on a real file.
        let bytes = write(&patchy());
        let mut shifted = vec![0u8; bytes.len() + 1];
        shifted[1..].copy_from_slice(&bytes);
        assert!(read(&shifted[1..], "x").is_ok());
    }

    #[test]
    fn a_corrupt_count_does_not_reach_an_allocation() {
        // The lesson `kmesh` records: a flipped bit in a count asked for 25 GB,
        // and an allocation failure aborts the process — there is no error to
        // report and nothing to catch. So the count is checked against what the
        // file physically contains, before anything is made from it.
        let mut bytes = write(&patchy());
        let header: &mut Header = bytemuck::from_bytes_mut(&mut bytes[..HEADER_SIZE]);
        header.tile_count = u32::MAX;
        assert!(matches!(
            read(&bytes, "x"),
            Err(KtileError::Truncated { .. })
        ));

        // And the declared size, which costs a pointer per tile even with no
        // pixels at all.
        let mut bytes = write(&patchy());
        let header: &mut Header = bytemuck::from_bytes_mut(&mut bytes[..HEADER_SIZE]);
        header.width = 1 << 31;
        header.height = 1 << 31;
        assert!(matches!(
            read(&bytes, "x"),
            Err(KtileError::ImplausibleSize { .. })
        ));
    }

    #[test]
    fn a_tile_pointing_outside_the_file_is_refused() {
        let mut bytes = write(&patchy());
        let entry: &mut IndexEntry =
            bytemuck::from_bytes_mut(&mut bytes[HEADER_SIZE..HEADER_SIZE + INDEX_SIZE]);
        entry.offset = u64::MAX - 16;
        assert!(matches!(
            read(&bytes, "x"),
            Err(KtileError::TileOutOfBounds { .. })
        ));
    }

    #[test]
    fn a_tile_claiming_a_position_the_image_does_not_have_is_refused() {
        // A 500×300 image is 4×3 tiles, so (9, 9) is not one of them — and
        // `set_tile` would quietly answer `false` rather than say so.
        let mut bytes = write(&patchy());
        let entry: &mut IndexEntry =
            bytemuck::from_bytes_mut(&mut bytes[HEADER_SIZE..HEADER_SIZE + INDEX_SIZE]);
        entry.tile_x = 9;
        entry.tile_y = 9;
        assert!(matches!(
            read(&bytes, "x"),
            Err(KtileError::TileOutsideImage { .. })
        ));
    }

    #[test]
    fn a_tile_of_the_wrong_length_is_refused_rather_than_padded() {
        // Reading it anyway would fill the rest with whatever follows in the
        // file, which is the classic way a corrupt texture becomes a view of
        // unrelated memory.
        let mut bytes = write(&patchy());
        let entry: &mut IndexEntry =
            bytemuck::from_bytes_mut(&mut bytes[HEADER_SIZE..HEADER_SIZE + INDEX_SIZE]);
        entry.length -= 4;
        assert!(matches!(
            read(&bytes, "x"),
            Err(KtileError::TileLength { .. })
        ));
    }

    #[test]
    fn a_file_whose_tiles_are_a_different_size_is_refused_rather_than_resampled() {
        // The tile edge is a constant of the datablock, not a parameter.
        // Resampling silently would make it unchangeable in practice while
        // pretending otherwise.
        let mut bytes = write(&patchy());
        let header: &mut Header = bytemuck::from_bytes_mut(&mut bytes[..HEADER_SIZE]);
        header.tile_edge = TILE * 2;
        assert!(matches!(
            read(&bytes, "x"),
            Err(KtileError::TileSize { .. })
        ));
    }

    #[test]
    fn an_image_with_nothing_resident_is_still_a_valid_file() {
        // A declared 16K texture nobody has painted. It has to survive a save,
        // because the declaration is the authored thing — the pixels arrive
        // later.
        let image = Image::new("Declared", 16384, 16384, PixelFormat::Rgba8);
        let bytes = write(&image);
        assert_eq!(bytes.len(), HEADER_SIZE, "an empty image wrote pixels");
        let back = read(&bytes, "Declared").expect("reads");
        assert_eq!(back.width(), 16384);
        assert_eq!(back.resident_tiles(), 0);
        assert_eq!(back.tile_count(), 128 * 128);
    }

    #[test]
    fn a_tile_that_is_not_aligned_is_refused_rather_than_read() {
        // The reader's half of the alignment rule. The writer's half has its own
        // test, and the two are not the same claim: a file written by somebody
        // else can put a tile anywhere, and the whole of what STORED buys is
        // that a tile can be cast where it lies.
        //
        // This reader copies rather than casts, so it would happily read a
        // misaligned tile — which is exactly why the refusal needs a test. The
        // mapping reader the format exists for does not exist yet, and a file
        // it cannot use must not be written now and accepted now.
        let mut bytes = write(&patchy());
        let entry: &mut IndexEntry =
            bytemuck::from_bytes_mut(&mut bytes[HEADER_SIZE..HEADER_SIZE + INDEX_SIZE]);
        entry.offset -= 8;
        assert!(matches!(
            read(&bytes, "x"),
            Err(KtileError::Misaligned { .. })
        ));
    }
}
