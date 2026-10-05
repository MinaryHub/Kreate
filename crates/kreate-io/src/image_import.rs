//! Reading an image file into the tiled datablock.
//!
//! The producer `images/` was missing. An archive layout with nothing that can
//! put an image in a document is wiring with one end unattached, and this is
//! the other end.
//!
//! # What the colour space argument is for, and what it is not
//!
//! §2.1's fourth rule: **an imported image's colour space is declared, never
//! guessed.** A PNG's bytes say nothing about whether they are an albedo
//! (sRGB-encoded) or a roughness (not encoded at all), and a file format that
//! could say it mostly does not. Guessing is how a roughness map ends up run
//! through a transfer function and looking like a material that needs tuning.
//!
//! So the caller declares it, and the declaration is *about the bytes*. A float
//! file has no transfer function to undo, so for one the declaration does not
//! apply and the image comes back [`ColorSpace::Raw`] whatever was asked for.
//! That is not the loader overruling the caller: it is that the question does
//! not arise.
//!
//! # Nothing is decoded here
//!
//! An eight-bit image is stored as the bytes the file held, and
//! [`ColorSpace::decode_byte`] is applied where it is sampled. Decoding on the
//! way in would mean widening every texture to float — four times the memory
//! for the common case — to undo a function the GPU applies for free.
//!
//! # Which decoders are on
//!
//! PNG and JPEG because that is what a texture arrives as, and Radiance HDR
//! because that is what an environment arrives as. Nothing else, deliberately:
//! every additional decoder is parsing surface for a file somebody was handed,
//! and this crate's whole job is reading files from elsewhere.

use half::f16;
use kreate_scene::image::{ColorSpace, Image, PixelFormat, Source, Tile, TILE};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("this is not an image this build can read: {0}")]
    UnknownFormat(String),
    #[error("the image could not be decoded: {0}")]
    Decode(String),
    /// A width or height the datablock cannot address or allocate.
    ///
    /// A decoder will happily report a header's dimensions, and an image is a
    /// pointer per tile before any pixels arrive — so this is the same guard
    /// [`crate::ktile`] needs, for the same reason.
    #[error("an image of {width}×{height} cannot be addressed or allocated")]
    ImplausibleSize { width: u32, height: u32 },
}

/// Decodes an image file.
///
/// `space` says what the file's *byte* values mean; see the module docs. It is
/// not consulted for a float file, which has no transfer function to undo.
pub fn load_image(bytes: &[u8], name: &str, space: ColorSpace) -> Result<Image, ImageError> {
    // From the bytes and not from the extension. A file's name is what somebody
    // typed and its magic is what it is, and the two disagree often enough that
    // trusting the name means refusing files that are fine.
    let format =
        image::guess_format(bytes).map_err(|e| ImageError::UnknownFormat(e.to_string()))?;
    let decoded = image::load_from_memory_with_format(bytes, format)
        .map_err(|e| ImageError::Decode(e.to_string()))?;

    // A float source keeps its range, and that is the entire reason an
    // environment is worth importing at all: §2.1's working space has no upper
    // bound, and a sky squeezed into eight bits is a flat white ceiling.
    let float = matches!(
        decoded,
        image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_)
    );

    if float {
        let source = decoded.to_rgba32f();
        let (width, height) = source.dimensions();
        let mut out = declare(name, width, height, PixelFormat::Rgba16F)?;
        // Half rather than full float: an environment is sampled, not computed
        // with, and half covers the range a camera can see many times over at
        // half the memory. Sixteen megabytes for a 2K environment instead of
        // thirty-two.
        fill(&mut out, width, height, |x, y, into| {
            let p = source.get_pixel(x, y).0;
            for (c, channel) in p.iter().enumerate() {
                let bits = f16::from_f32(*channel).to_bits().to_le_bytes();
                into[c * 2..c * 2 + 2].copy_from_slice(&bits);
            }
        });
        // Not `space`. A float file's numbers are already what a shader should
        // sample, so there is no transfer function to declare.
        out.color_space = ColorSpace::Raw;
        return Ok(out);
    }

    let source = decoded.to_rgba8();
    let (width, height) = source.dimensions();
    let mut out = declare(name, width, height, PixelFormat::Rgba8)?;
    fill(&mut out, width, height, |x, y, into| {
        into.copy_from_slice(&source.get_pixel(x, y).0);
    });
    out.color_space = space;
    Ok(out)
}

/// The same, remembering where it came from.
///
/// The path is the authored thing: a user who edits `sky.hdr` outside and
/// reopens expects the new sky, and without it there is nothing to reload from.
pub fn load_image_file(path: &std::path::Path, space: ColorSpace) -> Result<Image, ImageError> {
    let bytes = std::fs::read(path).map_err(|e| ImageError::Decode(e.to_string()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Image".to_string());
    let mut image = load_image(&bytes, &name, space)?;
    image.source = Source::File(path.to_path_buf());
    Ok(image)
}

/// An image datablock from eight-bit RGBA rows, as a file carried them.
///
/// For pixels that arrive already decoded from a container — a glTF's embedded
/// PNG, which the glTF reader has unpacked — rather than as a file of their
/// own. `space` says what the bytes mean, for the reason [`load_image`]'s
/// does. Short pixels leave the rest of the image at zero rather than failing:
/// a truncated buffer is a damaged file, and half a texture beats no model.
pub fn image_from_rgba8(
    name: &str,
    width: u32,
    height: u32,
    pixels: &[u8],
    space: ColorSpace,
) -> Result<Image, ImageError> {
    let mut out = declare(name, width, height, PixelFormat::Rgba8)?;
    fill(&mut out, width, height, |x, y, into| {
        let at = (y as usize * width as usize + x as usize) * 4;
        if let Some(p) = pixels.get(at..at + 4) {
            into.copy_from_slice(p);
        }
    });
    out.color_space = space;
    Ok(out)
}

/// An image's pixels as eight-bit RGBA rows, top row first.
///
/// The inverse of [`image_from_rgba8`], and what a writer or an upload wants.
/// Eight-bit images come out as stored; a single channel is replicated and
/// opaque; a float image is encoded for `space` — sRGB for a colour, clamped
/// numbers for data — and loses its range above one, which is the caller's to
/// note. Tiles that are not resident read as transparent black.
pub fn rgba8_of(image: &Image, space: ColorSpace) -> Vec<u8> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut out = vec![0u8; width * height * 4];
    let (across, down) = image.tile_counts();
    let stride = image.format.bytes_per_pixel();
    let tile = TILE as usize;
    for ty in 0..down {
        for tx in 0..across {
            let Some(held) = image.tile(tx, ty) else {
                continue;
            };
            let bytes = held.bytes();
            for row in 0..tile {
                let y = ty as usize * tile + row;
                if y >= height {
                    break;
                }
                for column in 0..tile {
                    let x = tx as usize * tile + column;
                    if x >= width {
                        break;
                    }
                    let at = (row * tile + column) * stride;
                    let to = (y * width + x) * 4;
                    out[to..to + 4].copy_from_slice(&texel(
                        image.format,
                        space,
                        &bytes[at..at + stride],
                    ));
                }
            }
        }
    }
    out
}

/// One texel as four bytes. See [`rgba8_of`].
fn texel(format: PixelFormat, space: ColorSpace, bytes: &[u8]) -> [u8; 4] {
    match format {
        PixelFormat::Rgba8 => [bytes[0], bytes[1], bytes[2], bytes[3]],
        PixelFormat::R8 => [bytes[0], bytes[0], bytes[0], 255],
        PixelFormat::Rgba16F => {
            let channel = |i: usize| {
                let v = f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])
                    .to_f32()
                    .clamp(0.0, 1.0);
                let e = match space {
                    ColorSpace::Srgb => kreate_math::linear_to_srgb(v),
                    ColorSpace::Raw => v,
                };
                (e * 255.0).round() as u8
            };
            let alpha = f16::from_le_bytes([bytes[6], bytes[7]])
                .to_f32()
                .clamp(0.0, 1.0);
            [
                channel(0),
                channel(1),
                channel(2),
                (alpha * 255.0).round() as u8,
            ]
        }
    }
}

/// An empty image of the decoded size, or an error if that size is absurd.
fn declare(name: &str, width: u32, height: u32, format: PixelFormat) -> Result<Image, ImageError> {
    Image::try_new(name, width, height, format).ok_or(ImageError::ImplausibleSize { width, height })
}

/// Walks the image's pixels into tiles.
///
/// Tile by tile rather than row by row, so each tile is touched once and
/// allocated once. The edge tiles keep the blank beyond the image's logical
/// size: tiles are always square and always full, which is what keeps the
/// special case out of every sampler (see [`kreate_scene::image`]).
fn fill(image: &mut Image, width: u32, height: u32, mut pixel: impl FnMut(u32, u32, &mut [u8])) {
    let (across, down) = image.tile_counts();
    let stride = image.format.bytes_per_pixel();
    for ty in 0..down {
        for tx in 0..across {
            let mut tile = Tile::blank(image.format);
            let bytes = tile.bytes_mut();
            for row in 0..TILE {
                let y = ty * TILE + row;
                if y >= height {
                    break;
                }
                for column in 0..TILE {
                    let x = tx * TILE + column;
                    if x >= width {
                        break;
                    }
                    let at = (row * TILE + column) as usize * stride;
                    pixel(x, y, &mut bytes[at..at + stride]);
                }
            }
            image.set_tile(tx, ty, Some(Arc::new(tile)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny PNG, encoded here so the test carries its own fixture.
    ///
    /// Written rather than checked in as bytes: a literal blob is a fixture
    /// nobody can change, and the thing being checked is where the pixels land,
    /// which needs them to be recognisable.
    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        let mut pixels = image::RgbaImage::new(width, height);
        for (x, y, p) in pixels.enumerate_pixels_mut() {
            // Recognisable: the red channel is the column and the green the
            // row, so a pixel that lands in the wrong tile says where it came
            // from.
            *p = image::Rgba([(x % 256) as u8, (y % 256) as u8, 7, 255]);
        }
        image::DynamicImage::ImageRgba8(pixels)
            .write_to(&mut buffer, image::ImageFormat::Png)
            .expect("png encodes");
        buffer.into_inner()
    }

    /// Reads one pixel back out of the tiles.
    fn pixel_at(image: &Image, x: u32, y: u32) -> Vec<u8> {
        let stride = image.format.bytes_per_pixel();
        let tile = image
            .tile(x / TILE, y / TILE)
            .expect("the tile is resident");
        let at = ((y % TILE) * TILE + (x % TILE)) as usize * stride;
        tile.bytes()[at..at + stride].to_vec()
    }

    #[test]
    fn every_pixel_lands_in_the_tile_it_belongs_to() {
        // The one thing this module actually does, and the one that goes wrong
        // silently: an off-by-one in the tile walk shifts a texture by a row
        // and looks like a slightly soft image rather than a bug.
        //
        // 300×200 is deliberately not a multiple of 128 — the ragged edge is
        // where a walk written for whole tiles breaks.
        let image = load_image(&png(300, 200), "Test", ColorSpace::Srgb).expect("decodes");
        assert_eq!(image.width(), 300);
        assert_eq!(image.height(), 200);
        assert_eq!(image.tile_counts(), (3, 2));
        for (x, y) in [(0, 0), (127, 127), (128, 128), (299, 199), (200, 5)] {
            assert_eq!(
                pixel_at(&image, x, y),
                vec![(x % 256) as u8, (y % 256) as u8, 7, 255],
                "the pixel at ({x}, {y}) is not the one that belongs there"
            );
        }
    }

    #[test]
    fn a_tile_past_the_edge_keeps_the_blank() {
        // Tiles are always square and always full, and what is beyond the
        // image's logical size is the blank. A walk that wrapped instead would
        // put column 0 there, which is a texture that repeats a sliver of
        // itself along its right edge.
        let image = load_image(&png(300, 200), "Test", ColorSpace::Srgb).expect("decodes");
        // 300 wide is two whole tiles and 44 pixels, so x = 44.. of tile 2 is
        // past the end.
        let tile = image.tile(2, 0).expect("resident");
        // Row 0, column 44 of the tile — which is x = 300 of the image, one
        // past its right edge.
        let at = 44 * 4;
        assert_eq!(&tile.bytes()[at..at + 4], &[0, 0, 0, 0]);
        // And the pixel just inside it is the real one.
        assert_eq!(pixel_at(&image, 299, 0), vec![(299 % 256) as u8, 0, 7, 255]);
    }

    #[test]
    fn eight_bit_pixels_are_not_decoded_on_the_way_in() {
        // The declaration is carried, not applied. Decoding here would widen
        // every texture to float — four times the memory for the common case —
        // to undo a function the GPU applies for free.
        let image = load_image(&png(8, 8), "Test", ColorSpace::Srgb).expect("decodes");
        assert_eq!(image.color_space, ColorSpace::Srgb);
        assert_eq!(image.format, PixelFormat::Rgba8);
        // 7 in the blue channel, exactly as written. Decoded, it would be about
        // 0.002 and could not be a byte at all.
        assert_eq!(pixel_at(&image, 0, 0)[2], 7);
    }

    #[test]
    fn what_the_caller_declares_is_what_the_image_carries() {
        // §2.1's rule. A roughness map and an albedo are the same bytes and a
        // different space, and nothing in the file says which.
        let bytes = png(8, 8);
        let albedo = load_image(&bytes, "Albedo", ColorSpace::Srgb).expect("decodes");
        let roughness = load_image(&bytes, "Roughness", ColorSpace::Raw).expect("decodes");
        assert_eq!(albedo.color_space, ColorSpace::Srgb);
        assert_eq!(roughness.color_space, ColorSpace::Raw);
        // And the bytes are the same, which is what makes the space the only
        // thing telling them apart.
        assert_eq!(pixel_at(&albedo, 3, 3), pixel_at(&roughness, 3, 3));
    }

    /// A Radiance HDR holding one very bright pixel.
    fn hdr(value: f32) -> Vec<u8> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        let mut pixels = image::Rgb32FImage::new(4, 4);
        for p in pixels.pixels_mut() {
            *p = image::Rgb([value, value * 0.5, 0.25]);
        }
        image::codecs::hdr::HdrEncoder::new(&mut buffer)
            .encode(
                pixels
                    .as_raw()
                    .as_slice()
                    .chunks(3)
                    .map(|c| image::Rgb([c[0], c[1], c[2]]))
                    .collect::<Vec<_>>()
                    .as_slice(),
                4,
                4,
            )
            .expect("hdr encodes");
        buffer.into_inner()
    }

    #[test]
    fn an_environment_keeps_the_range_that_is_the_reason_to_have_one() {
        // The whole argument for `Rgba16F`. A sky is mostly above one, and
        // eight bits would make a sun and a wall the same white — which is
        // §2.1's clipping complaint arriving one step earlier, at the import.
        let image = load_image(&hdr(600.0), "Sky", ColorSpace::Srgb).expect("decodes");
        assert_eq!(image.format, PixelFormat::Rgba16F);
        let p = pixel_at(&image, 1, 1);
        let red = f16::from_le_bytes([p[0], p[1]]).to_f32();
        // Radiance is a shared-exponent format, so the value comes back close
        // rather than exact — what matters is that it is nowhere near one.
        assert!(
            red > 500.0,
            "a 600-unit sky came back as {red}, which is a clipped sky"
        );
    }

    #[test]
    fn a_float_file_is_linear_whatever_the_caller_declares() {
        // Not the loader overruling the caller: a float file has no transfer
        // function to undo, so the question the declaration answers does not
        // arise. Asserted for both answers so it cannot be an accident of the
        // default.
        for declared in [ColorSpace::Srgb, ColorSpace::Raw] {
            let image = load_image(&hdr(2.0), "Sky", declared).expect("decodes");
            assert_eq!(image.color_space, ColorSpace::Raw);
        }
    }

    #[test]
    fn something_that_is_not_an_image_is_refused_rather_than_guessed_at() {
        assert!(matches!(
            load_image(b"not an image at all", "x", ColorSpace::Srgb),
            Err(ImageError::UnknownFormat(_))
        ));
        // And a file whose magic is right and whose body is not.
        let mut broken = png(8, 8);
        let tail = broken.len() - 20;
        broken.truncate(tail);
        assert!(load_image(&broken, "x", ColorSpace::Srgb).is_err());
    }

    #[test]
    fn the_bottom_edge_keeps_the_blank_too() {
        // The right edge had a test and the bottom did not, and the walk has a
        // separate bound for each — so a break that clamped the row instead of
        // stopping repeated the last row of the image down the rest of the tile
        // and nothing noticed. A texture that repeats a sliver of itself along
        // one edge is exactly the kind of wrong that reads as "slightly soft".
        let image = load_image(&png(300, 200), "Test", ColorSpace::Srgb).expect("decodes");
        // 200 tall is one whole tile and 72 rows, so row 72.. of tile row 1 is
        // past the end.
        let tile = image.tile(0, 1).expect("resident");
        let at = (72 * TILE) as usize * 4;
        assert_eq!(&tile.bytes()[at..at + 4], &[0, 0, 0, 0]);
        // And the row just inside it is the real one.
        assert_eq!(pixel_at(&image, 0, 199), vec![0, 199, 7, 255]);
    }

    #[test]
    fn a_size_no_image_could_have_is_answered_rather_than_reached() {
        // Called directly, because it cannot be reached through `load_image`:
        // the decoder applies its own limits and refuses an absurd header
        // before this sees it. That makes the guard a second line of defence
        // behind somebody else's first, and the reason to keep it is that the
        // first belongs to a dependency — a decoder that raises its limits, or
        // a format added later that has none, arrives here.
        assert!(matches!(
            declare("Absurd", 1 << 31, 1 << 31, PixelFormat::Rgba8),
            Err(ImageError::ImplausibleSize { .. })
        ));
    }

    #[test]
    fn a_file_loaded_from_disk_remembers_where_it_came_from() {
        // The path is the authored thing, and the only way to check it is
        // through a real file — which is why nothing else here does.
        let path = std::env::temp_dir().join(format!(
            "kreate-image-import-{}-{:?}.png",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, png(16, 16)).expect("writes");
        let image = load_image_file(&path, ColorSpace::Raw).expect("loads");
        let _ = std::fs::remove_file(&path);

        match &image.source {
            Source::File(remembered) => assert_eq!(remembered, &path),
            other => panic!("it came back as {other:?}"),
        }
        // And the name comes from the file rather than being invented.
        assert!(image.name.starts_with("kreate-image-import-"));
        assert_eq!(image.color_space, ColorSpace::Raw);
    }
}
