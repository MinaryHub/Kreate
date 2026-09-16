//! Handing the world's image to the renderer.
//!
//! The seam §0 draws: `kreate-render` never sees a `Document`, so an
//! environment reaches it as a size and a block of bytes, the way a mesh
//! reaches it as buffers. The joining is the application's, and so is the
//! decoding — which is the part with a decision in it.
//!
//! # Everything arrives linear, whatever it was stored as
//!
//! A sky loaded from an HDR file is float and already linear. One loaded from a
//! PNG is eight bits and usually sRGB-encoded, and §2.1's rule is that the
//! encoding is *declared* on the datablock rather than guessed. So this reads
//! the declaration and applies it, and what leaves is `Rgba16Float` in the
//! working space either way.
//!
//! Decoding here rather than in the shader is the same trade [`kreate_io`]
//! makes going the other way: an image is stored as the bytes the file held and
//! decoded where it is used. The environment is used once, when the cube is
//! built, so the decode happens once for the whole image instead of once per
//! sample for the life of the scene.
//!
//! # A tile that is not resident reads as black
//!
//! §3's sixth decision made images sparse, and a half-loaded one is a real
//! state. What is missing comes out as nothing rather than as whatever was in
//! memory — half a sky, which is better than a dialog.

use kreate_scene::image::{Image, PixelFormat, TILE};
use kreate_scene::Document;

/// Which environment the renderer is currently holding.
///
/// Compared rather than the image itself, so that a frame does not rebuild six
/// faces and their mips to discover that nothing changed. The identity is in
/// it, so picking a different sky rebuilds; so are the dimensions and how much
/// of the image has arrived, so a sky that finishes loading rebuilds too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvironmentKey {
    uid: u64,
    width: u32,
    height: u32,
    resident: usize,
}

/// What the active world's background is, as a key.
///
/// `None` is a flat world, which is most of them.
pub fn key_of(doc: &Document) -> Option<EnvironmentKey> {
    let image = background_image(doc)?;
    Some(EnvironmentKey {
        uid: image.uid.raw(),
        width: image.width(),
        height: image.height(),
        resident: image.resident_tiles(),
    })
}

/// The active world's background image, if it has one.
pub fn background_image(doc: &Document) -> Option<&Image> {
    let world = doc.world(doc.active_world()?)?;
    match world.background {
        kreate_scene::Background::Image(id) => doc.image(id),
        kreate_scene::Background::Flat(_) => None,
    }
}

/// The image as a row-major `Rgba16Float` block, in the working space.
///
/// Row-major because that is what a texture upload wants and what the tiles
/// are not: the whole of this function is walking the tiling back out.
pub fn pixels_of(image: &Image) -> Vec<u8> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut out = vec![0u8; width * height * 8];
    let (across, down) = image.tile_counts();
    let stride = image.format.bytes_per_pixel();

    for ty in 0..down {
        for tx in 0..across {
            // Not resident, and that is not an error: the rows stay zero, which
            // is black, which is what a missing part of a sky should be.
            let Some(tile) = image.tile(tx, ty) else {
                continue;
            };
            let bytes = tile.bytes();
            for row in 0..TILE {
                let y = (ty * TILE + row) as usize;
                if y >= height {
                    break;
                }
                for column in 0..TILE {
                    let x = (tx * TILE + column) as usize;
                    if x >= width {
                        break;
                    }
                    let at = (row * TILE + column) as usize * stride;
                    let rgb = read(image, &bytes[at..at + stride]);
                    let to = (y * width + x) * 8;
                    for (c, value) in rgb.iter().enumerate() {
                        out[to + c * 2..to + c * 2 + 2]
                            .copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
                    }
                    // Opaque. An environment has no transparency to carry, and
                    // a zero here would make a reader that respects alpha see
                    // nothing at all.
                    out[to + 6..to + 8]
                        .copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                }
            }
        }
    }
    out
}

/// One texel, decoded into the working space.
fn read(image: &Image, texel: &[u8]) -> [f32; 3] {
    match image.format {
        // Already linear and already float. §2.1's working space has no upper
        // bound and this is the format that respects it — a sky is mostly above
        // one, and the whole reason an HDRI is not a PNG.
        PixelFormat::Rgba16F => {
            let channel = |i: usize| {
                half::f16::from_le_bytes([texel[i * 2], texel[i * 2 + 1]]).to_f32()
            };
            [channel(0), channel(1), channel(2)]
        }
        // Eight bits, decoded by what the datablock says they mean. Guessing
        // here is how §2.1's two colour bugs happened.
        PixelFormat::Rgba8 => [
            image.color_space.decode_byte(texel[0]),
            image.color_space.decode_byte(texel[1]),
            image.color_space.decode_byte(texel[2]),
        ],
        PixelFormat::R8 => {
            let v = image.color_space.decode_byte(texel[0]);
            [v, v, v]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::image::ColorSpace;

    fn at(pixels: &[u8], width: u32, x: u32, y: u32) -> [f32; 3] {
        let base = ((y as usize * width as usize) + x as usize) * 8;
        let channel =
            |i: usize| half::f16::from_le_bytes([pixels[base + i * 2], pixels[base + i * 2 + 1]]).to_f32();
        [channel(0), channel(1), channel(2)]
    }

    #[test]
    fn a_float_sky_keeps_the_range_that_is_the_point_of_it() {
        // Untouched: a float file is already in the working space, and running
        // it through a transfer function would flatten a sun into a wall.
        let mut image = Image::new("Sky", 8, 4, PixelFormat::Rgba16F);
        image.color_space = ColorSpace::Raw;
        let tile = image.tile_mut(0, 0).expect("in range");
        let bright = half::f16::from_f32(600.0).to_le_bytes();
        tile.bytes_mut()[0..2].copy_from_slice(&bright);

        let out = pixels_of(&image);
        assert!(
            (at(&out, 8, 0, 0)[0] - 600.0).abs() < 1.0,
            "a 600-unit sky came out at {}",
            at(&out, 8, 0, 0)[0]
        );
    }

    #[test]
    fn an_srgb_sky_is_decoded_and_a_raw_one_is_not() {
        // §2.1's rule, and the decision it protects. The same byte means two
        // different amounts of light depending on what the datablock says it
        // is, and nothing in the pixels says which.
        let make = |space| {
            let mut image = Image::new("Sky", 8, 4, PixelFormat::Rgba8);
            image.color_space = space;
            let tile = image.tile_mut(0, 0).expect("in range");
            tile.bytes_mut()[0] = 128;
            pixels_of(&image)
        };
        let encoded = at(&make(ColorSpace::Srgb), 8, 0, 0)[0];
        let raw = at(&make(ColorSpace::Raw), 8, 0, 0)[0];
        assert!(
            (raw - 128.0 / 255.0).abs() < 1e-3,
            "a raw 128 came out at {raw}"
        );
        // Mid grey encoded is about a fifth of the light, not half of it. The
        // gap is the whole reason the declaration exists.
        assert!(
            (encoded - 0.2158).abs() < 1e-3,
            "an sRGB 128 came out at {encoded}"
        );
    }

    #[test]
    fn every_pixel_comes_out_where_it_went_in() {
        // Walking the tiling back out, which is the whole of this function and
        // the part that goes wrong silently: a row lost here rotates the sky by
        // a fraction of a degree, and nothing about a picture says so.
        //
        // 300×140 is deliberately not a multiple of the tile size.
        let (width, height) = (300u32, 140u32);
        let mut image = Image::new("Sky", width, height, PixelFormat::Rgba16F);
        image.color_space = ColorSpace::Raw;
        let (across, down) = image.tile_counts();
        for ty in 0..down {
            for tx in 0..across {
                let tile = image.tile_mut(tx, ty).expect("in range");
                for row in 0..TILE {
                    for column in 0..TILE {
                        let x = tx * TILE + column;
                        let y = ty * TILE + row;
                        if x >= width || y >= height {
                            continue;
                        }
                        // Recognisable, and distinct for every pixel of an
                        // image this size.
                        let value = half::f16::from_f32((y * 1000 + x) as f32).to_le_bytes();
                        let at = (row * TILE + column) as usize * 8;
                        tile.bytes_mut()[at..at + 2].copy_from_slice(&value);
                    }
                }
            }
        }

        let out = pixels_of(&image);
        assert_eq!(out.len(), (width * height) as usize * 8);
        for (x, y) in [(0, 0), (127, 127), (128, 128), (299, 139), (200, 5)] {
            let wanted = half::f16::from_f32((y * 1000 + x) as f32).to_f32();
            assert_eq!(
                at(&out, width, x, y)[0],
                wanted,
                "the pixel at ({x}, {y}) is not the one that belongs there"
            );
        }
    }

    #[test]
    fn a_tile_that_never_arrived_is_black_rather_than_whatever_was_there() {
        // A half-loaded image is a real state, not an error. Half a sky beats a
        // dialog, and it beats a sky with somebody else's memory in it.
        let mut image = Image::new("Sky", 300, 140, PixelFormat::Rgba16F);
        image.color_space = ColorSpace::Raw;
        let tile = image.tile_mut(0, 0).expect("in range");
        tile.bytes_mut()[0..2].copy_from_slice(&half::f16::from_f32(5.0).to_le_bytes());
        assert_eq!(image.resident_tiles(), 1);

        let out = pixels_of(&image);
        assert_eq!(at(&out, 300, 0, 0)[0], 5.0);
        // Anywhere in a tile that is not resident.
        assert_eq!(at(&out, 300, 200, 130), [0.0; 3]);
    }

    #[test]
    fn the_key_changes_when_the_sky_does_and_not_when_it_does_not() {
        // What stops a frame rebuilding six faces and their mips to find out
        // that nothing happened — and, the other way, what makes a sky that
        // finishes loading actually appear.
        let mut doc = Document::new();
        // Big enough to have more than one tile: at 64×32 the whole image is
        // one tile and "a tile arrived" is not a state it can be in, so the
        // first version of this fixture asked a question with no answer.
        let mut sky = Image::new("Sky", 300, 140, PixelFormat::Rgba16F);
        sky.tile_mut(0, 0).expect("in range");
        assert!(sky.tile_count() > 1);
        let id = doc.add_image(sky);
        let world = doc.add_world(kreate_scene::World {
            background: kreate_scene::Background::Image(id),
            ..kreate_scene::World::new("Sky")
        });
        doc.set_active_world(Some(world));

        let first = key_of(&doc).expect("an image background");
        assert_eq!(first, key_of(&doc).expect("still there"));

        // A tile arriving is a different sky.
        doc.image_mut(id)
            .expect("the image")
            .tile_mut(1, 0)
            .expect("in range");
        assert_ne!(first, key_of(&doc).expect("still there"));

        // And a flat world has no key at all, which is what turns the
        // environment back off.
        let flat = doc.add_world(kreate_scene::World::new("Grey"));
        doc.set_active_world(Some(flat));
        assert_eq!(key_of(&doc), None);
    }

    #[test]
    fn two_skies_of_the_same_size_are_not_the_same_sky() {
        // The identity in the key, and the reason a fixture with one image
        // cannot see it: a key that dropped the identity would still change
        // when the *dimensions* did, and one image is never compared with
        // another. Two images of the same size and the same residency differ in
        // nothing else — which is exactly the case a user hits when they try a
        // second HDRI from the same pack.
        let mut doc = Document::new();
        let mut first = Image::new("Dawn", 300, 140, PixelFormat::Rgba16F);
        first.tile_mut(0, 0).expect("in range");
        let mut second = Image::new("Dusk", 300, 140, PixelFormat::Rgba16F);
        second.tile_mut(0, 0).expect("in range");
        let a = doc.add_image(first);
        let b = doc.add_image(second);

        let world = doc.add_world(kreate_scene::World {
            background: kreate_scene::Background::Image(a),
            ..kreate_scene::World::new("Sky")
        });
        doc.set_active_world(Some(world));
        let with_a = key_of(&doc).expect("an image background");

        doc.world_mut(world).expect("the world").background =
            kreate_scene::Background::Image(b);
        let with_b = key_of(&doc).expect("an image background");
        assert_ne!(
            with_a, with_b,
            "swapping the sky for another of the same size changed nothing"
        );
    }
}
