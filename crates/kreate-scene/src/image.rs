//! Images, stored in tiles.
//!
//! §3's sixth "decide now", and the reason it is one: the tiling has to be
//! there before the first sampler and the first undo path are written, because
//! afterwards it is a change to all of them.
//!
//! # A brush stroke is the whole argument
//!
//! One dab on a 4K texture touches a few thousand pixels. Held flat, undoing it
//! means keeping a copy of the image — sixty-four megabytes a step, so ten
//! strokes is most of a gigabyte and a painting session is not possible. Held in
//! tiles, the same dab copies the tiles it landed on and shares the rest, which
//! is [`Arc::make_mut`] doing for an image exactly what it already does for a
//! mesh's attribute arrays.
//!
//! That is not an optimisation to add later. A flat image has one buffer and
//! every sampler indexes it directly; introducing tiles afterwards changes the
//! addressing in every one of them, and the undo path with it.
//!
//! # Why 128
//!
//! A tile of RGBA bytes at 128 is 64 KB, at 256 it is 256 KB. Undo granularity
//! wants the smaller — a dab should cost 64 KB and not a quarter of a megabyte
//! — and per-tile overhead wants the larger. A 4K image is 32×32 tiles at 128,
//! which is a thousand `Arc`s: enough that the overhead is measurable and
//! nowhere near enough that it matters against 64 MB of pixels.
//!
//! # Tiles are always square and always full
//!
//! An image whose width is not a multiple of 128 still has whole tiles, and the
//! image's logical size clips what a reader may ask for. The alternative —
//! partial tiles at the right and bottom edges — puts a special case in every
//! sampler, every blit and every mip generator, forever, to save a strip of
//! memory: a 4097-pixel image pays for one extra tile column, which is 0.8% of
//! it.
//!
//! # Sparse residency falls out of it
//!
//! Not built here, but worth saying why the shape is right: a tile is
//! individually present or absent, so "upload only the tiles the camera can
//! see" and "load only the mip levels in use" are the same mechanism with a
//! different predicate. A flat image can only be all there or not there.

use std::sync::Arc;

use kreate_reflect::Uid;

/// The edge of a tile, in pixels. See the module docs for why 128.
pub const TILE: u32 = 128;

/// What the numbers in an image mean.
///
/// Declared at import and never guessed. §2.1's rule applied to pixels: a
/// texture whose space nobody wrote down is a texture two systems will disagree
/// about, and the disagreement looks like a material that is slightly wrong
/// rather than a bug.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// sRGB-encoded colour: an albedo, an emission, a painted layer.
    ///
    /// Decoded on the way in, so what a shader samples is linear. The default
    /// because a PNG or a JPEG from anywhere is this unless someone says
    /// otherwise.
    #[default]
    Srgb,
    /// Numbers that are not colour: a normal map, a roughness, a metalness, a
    /// displacement, a mask.
    ///
    /// Passed through untouched. Decoding one of these is the classic texture
    /// bug — a roughness map run through a transfer function is smooth where it
    /// should be rough, and it looks like a material that needs tuning.
    Raw,
}

impl ColorSpace {
    pub fn id(self) -> &'static str {
        match self {
            ColorSpace::Srgb => "srgb",
            ColorSpace::Raw => "raw",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "srgb" => ColorSpace::Srgb,
            "raw" => ColorSpace::Raw,
            _ => return None,
        })
    }

    /// What a channel of this kind of image is, given its stored byte.
    ///
    /// The one place the two spaces differ in behaviour, so it is the one place
    /// worth having a function for.
    pub fn decode_byte(self, v: u8) -> f32 {
        let unit = f32::from(v) / 255.0;
        match self {
            ColorSpace::Srgb => kreate_math::srgb_to_linear(unit),
            ColorSpace::Raw => unit,
        }
    }
}

/// How many bytes a pixel takes, and what they are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PixelFormat {
    /// Eight bits a channel, four channels. What a PNG usually is.
    #[default]
    Rgba8,
    /// Sixteen-bit float a channel, four channels. What an HDRI wants.
    Rgba16F,
    /// One channel of eight bits: a mask, a height, a roughness.
    R8,
}

impl PixelFormat {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::Rgba8 => 4,
            PixelFormat::Rgba16F => 8,
            PixelFormat::R8 => 1,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            PixelFormat::Rgba8 => "rgba8",
            PixelFormat::Rgba16F => "rgba16f",
            PixelFormat::R8 => "r8",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "rgba8" => PixelFormat::Rgba8,
            "rgba16f" => PixelFormat::Rgba16F,
            "r8" => PixelFormat::R8,
            _ => return None,
        })
    }
}

/// Where an image's pixels came from.
///
/// Decides what a save has to write. A file on disk writes its path; anything
/// generated or painted has to write its pixels, because nothing else knows
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Loaded from a path. The pixels are reproducible, so a save writes the
    /// path.
    File(std::path::PathBuf),
    /// Made by the program — a noise, a gradient, a flat colour.
    Generated,
    /// Painted in. The pixels exist nowhere else and a save must write them.
    Painted,
}

/// One tile's pixels, in the image's format.
///
/// Always `TILE × TILE`, so its length is a function of the format alone and
/// every reader can address it the same way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tile {
    pixels: Vec<u8>,
}

impl Tile {
    /// A tile of zeroes.
    ///
    /// Which is transparent black for `Rgba8` and zero for a mask, and is the
    /// right blank in both cases: a painted image starts empty, and a
    /// half-loaded one should read as nothing rather than as noise.
    pub fn blank(format: PixelFormat) -> Self {
        Self {
            pixels: vec![0; (TILE * TILE) as usize * format.bytes_per_pixel()],
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.pixels
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }
}

/// An image datablock.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub name: String,
    pub linkage: crate::document::Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,

    /// The image's size in pixels. Not a multiple of [`TILE`] in general.
    width: u32,
    height: u32,
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    pub source: Source,

    /// Row-major, `tiles_across × tiles_down`.
    ///
    /// One `Arc` each, which is the whole point: a brush stroke replaces the
    /// handles it touched and the rest of the image is still the same
    /// allocation. `None` for a tile that is not resident — see the module docs
    /// on sparse residency — and a read of one yields the blank, so a
    /// half-loaded image draws as empty rather than as whatever was in memory.
    tiles: Vec<Option<Arc<Tile>>>,
}

impl Image {
    /// An empty image of a given size.
    ///
    /// No tiles are allocated. An image nobody has written to costs its
    /// dimensions and a vector of `None`, which is what makes a 16K texture
    /// cheap to *declare* — the pixels arrive when something loads or paints
    /// them.
    pub fn new(name: impl Into<String>, width: u32, height: u32, format: PixelFormat) -> Self {
        let name = name.into();
        Self::try_new(name.clone(), width, height, format).unwrap_or_else(|| {
            panic!("an image of {width}×{height} named {name} cannot be addressed or allocated")
        })
    }

    /// The same, for a caller whose size came from somewhere it does not
    /// control.
    ///
    /// `None` rather than a panic, and the distinction is the whole reason this
    /// exists: a size written in Rust that is too big is a bug and should
    /// abort, and a size read out of a *file* that is too big is an ordinary
    /// corrupt file that has to be reported.
    ///
    /// Two ways it can be too big, and they are different:
    ///
    /// - **The grid cannot be indexed.** A tile is addressed as
    ///   `tile_y * across + tile_x` in `u32`, so a grid with more than
    ///   [`u32::MAX`] tiles is one this type cannot reach the far end of. That
    ///   is a fact about the datablock, not a policy.
    /// - **The grid cannot be allocated.** A declared image costs a pointer per
    ///   tile whether or not any pixels arrive, so a file claiming a four
    ///   billion tile image is asking for tens of gigabytes of nothing.
    ///   [`Vec::try_reserve`] turns that into an answer instead of an abort —
    ///   and an abort really is the alternative, because a failed allocation
    ///   ends the process with nothing to catch.
    ///
    /// No arbitrary maximum is picked, deliberately. Every number that could go
    /// here would be a guess about what a document holds, and the two limits
    /// above are the real ones.
    pub fn try_new(
        name: impl Into<String>,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Option<Self> {
        let across = tiles_for(width) as u64;
        let down = tiles_for(height) as u64;
        // No overflow to check for: each side is at most `u32::MAX / TILE`,
        // under 2^25, so the product is under 2^50. A `checked_mul` here was a
        // guard that could not fire, which is worse than the multiplication it
        // was guarding — it reads as though the case exists.
        let count = across * down;
        // Above this the grid cannot be *indexed*, which is the type's own
        // limit: a tile is reached as `tile_y * across + tile_x` in `u32`.
        //
        // On any machine that exists the reservation below fails first — four
        // billion tiles is sixty-eight gigabytes of pointers before a single
        // pixel — so this is the invariant written down rather than a branch
        // anything reaches. Written down anyway, because the reservation
        // succeeding on some future machine is the case where it matters and
        // there would be nothing to read.
        if count > u32::MAX as u64 {
            return None;
        }
        let count = usize::try_from(count).ok()?;
        let mut tiles: Vec<Option<Arc<Tile>>> = Vec::new();
        tiles.try_reserve_exact(count).ok()?;
        tiles.resize(count, None);
        Some(Self {
            name: name.into(),
            linkage: crate::document::Linkage::Local,
            uid: Uid::NONE,
            width,
            height,
            format,
            color_space: ColorSpace::default(),
            source: Source::Generated,
            tiles,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// How many tiles across and down.
    pub fn tile_counts(&self) -> (u32, u32) {
        (tiles_for(self.width), tiles_for(self.height))
    }

    /// How many tiles this image is divided into, resident or not.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// How many tiles actually hold pixels.
    pub fn resident_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_some()).count()
    }

    fn index_of(&self, tile_x: u32, tile_y: u32) -> Option<usize> {
        let (across, down) = self.tile_counts();
        if tile_x >= across || tile_y >= down {
            return None;
        }
        Some((tile_y * across + tile_x) as usize)
    }

    /// One tile's pixels, or `None` when it is not resident.
    pub fn tile(&self, tile_x: u32, tile_y: u32) -> Option<&Tile> {
        let index = self.index_of(tile_x, tile_y)?;
        self.tiles[index].as_deref()
    }

    /// One tile, ready to be written to, allocating it if it was not resident.
    ///
    /// This is where copy-on-write happens: [`Arc::make_mut`] copies the tile
    /// only if something else is holding it — an undo record, or a snapshot —
    /// and otherwise writes in place. A stroke that touches four tiles of a 4K
    /// image copies four of a thousand.
    pub fn tile_mut(&mut self, tile_x: u32, tile_y: u32) -> Option<&mut Tile> {
        let index = self.index_of(tile_x, tile_y)?;
        let format = self.format;
        let slot = self.tiles.get_mut(index)?;
        Some(Arc::make_mut(
            slot.get_or_insert_with(|| Arc::new(Tile::blank(format))),
        ))
    }

    /// The tile handle, for a caller that wants to hold it — an undo record.
    ///
    /// Cloning the `Arc` is what makes an undo step cheap: the record and the
    /// image share the pixels until one of them is written to.
    pub fn tile_handle(&self, tile_x: u32, tile_y: u32) -> Option<Arc<Tile>> {
        let index = self.index_of(tile_x, tile_y)?;
        self.tiles[index].clone()
    }

    /// Puts a tile back, for an undo restoring one.
    ///
    /// `None` restores "was not resident", which is a real state a stroke can
    /// have created a tile out of.
    pub fn set_tile(&mut self, tile_x: u32, tile_y: u32, tile: Option<Arc<Tile>>) -> bool {
        match self.index_of(tile_x, tile_y) {
            Some(index) => {
                self.tiles[index] = tile;
                true
            }
            None => false,
        }
    }

    /// Which tiles a rectangle of pixels touches.
    ///
    /// The rectangle is clipped to the image first, so a brush that runs off
    /// the edge does not ask for a tile that is not there. Inclusive of both
    /// corners, because a brush's footprint is a set of pixels and the last one
    /// is in it.
    pub fn tiles_covering(&self, x: u32, y: u32, w: u32, h: u32) -> Vec<(u32, u32)> {
        if w == 0 || h == 0 || x >= self.width || y >= self.height {
            return Vec::new();
        }
        let right = (x + w - 1).min(self.width - 1);
        let bottom = (y + h - 1).min(self.height - 1);
        let mut out = Vec::new();
        for ty in (y / TILE)..=(bottom / TILE) {
            for tx in (x / TILE)..=(right / TILE) {
                out.push((tx, ty));
            }
        }
        out
    }

    /// What this image occupies, counting only the tiles that are resident.
    ///
    /// The number a memory budget wants. Declaring a 16K texture costs a
    /// vector of `None`; the pixels are charged when they arrive.
    pub fn memory_bytes(&self) -> usize {
        let per_tile = (TILE * TILE) as usize * self.format.bytes_per_pixel();
        self.resident_tiles() * per_tile
            + self.tiles.len() * std::mem::size_of::<Option<Arc<Tile>>>()
    }

    /// What the same image would cost held flat, for comparison.
    ///
    /// Here so that the tiling's overhead is answerable rather than assumed:
    /// tiles cost a little more for a full image and enormously less for a
    /// sparse one, and both halves of that are worth being able to state.
    pub fn flat_bytes(&self) -> usize {
        self.width as usize * self.height as usize * self.format.bytes_per_pixel()
    }
}

/// How many tiles cover `pixels` along one axis.
fn tiles_for(pixels: u32) -> u32 {
    pixels.div_ceil(TILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declared_image_costs_almost_nothing_until_something_writes_to_it() {
        // What makes a 16K texture cheap to *declare*. The pixels arrive when
        // they are loaded or painted, and until then the image is its
        // dimensions and a vector of absences.
        let image = Image::new("Big", 16384, 16384, PixelFormat::Rgba8);
        assert_eq!(image.resident_tiles(), 0);
        assert_eq!(image.tile_count(), 128 * 128);
        assert!(
            image.memory_bytes() < 200_000,
            "an empty 16K image holds {} bytes",
            image.memory_bytes()
        );
        // Where flat it would be a gigabyte.
        assert_eq!(image.flat_bytes(), 16384 * 16384 * 4);
    }

    #[test]
    fn a_brush_stroke_copies_the_tiles_it_touched_and_no_others() {
        // The whole argument for tiling, measured. A dab on a 4K texture
        // touches a few tiles; held flat, undoing it would mean keeping
        // sixty-four megabytes.
        let mut image = Image::new("Albedo", 4096, 4096, PixelFormat::Rgba8);
        // Fill it, as a loaded texture would be.
        let (across, down) = image.tile_counts();
        for ty in 0..down {
            for tx in 0..across {
                image.tile_mut(tx, ty).expect("in range").bytes_mut()[0] = 1;
            }
        }
        assert_eq!(image.resident_tiles(), (across * down) as usize);

        // An undo record takes handles to what the stroke is about to touch.
        // Placed inside one tile: 520..=559 is all in tile four, since the
        // boundaries are at multiples of 128.
        let touched = image.tiles_covering(520, 520, 40, 40);
        assert_eq!(touched, vec![(4, 4)], "a 40px dab here is one tile");
        // And a dab of the same size across a boundary is four — which is the
        // honest worst case for a small brush and still four tiles out of a
        // thousand.
        assert_eq!(image.tiles_covering(500, 500, 40, 40).len(), 4);
        let kept: Vec<_> = touched
            .iter()
            .map(|(x, y)| (*x, *y, image.tile_handle(*x, *y)))
            .collect();

        // The stroke writes. Only the held tile can have been copied.
        for (x, y, _) in &kept {
            image.tile_mut(*x, *y).expect("in range").bytes_mut()[0] = 2;
        }

        // The record still has the old pixels, and the rest of the image was
        // never touched.
        let (x, y, before) = &kept[0];
        assert_eq!(before.as_ref().expect("resident").bytes()[0], 1);
        assert_eq!(image.tile(*x, *y).expect("resident").bytes()[0], 2);

        // What the undo record costs: one tile, not the image.
        let one_tile = (TILE * TILE) as usize * 4;
        assert_eq!(one_tile, 65_536);
        assert!(
            one_tile * 1000 < image.flat_bytes(),
            "a thousand strokes should still be less than one flat copy"
        );
    }

    #[test]
    fn writing_a_tile_twice_does_not_copy_it_twice() {
        // `Arc::make_mut` copies only while something else holds the tile. A
        // stroke that goes over the same tile a hundred times must not allocate
        // a hundred tiles, which is what a naive "always clone before writing"
        // would do.
        let mut image = Image::new("Paint", 512, 512, PixelFormat::Rgba8);
        image.tile_mut(0, 0).expect("in range");
        let first = image.tile(0, 0).expect("resident").bytes().as_ptr();
        for i in 0..100u8 {
            image.tile_mut(0, 0).expect("in range").bytes_mut()[0] = i;
        }
        let after = image.tile(0, 0).expect("resident").bytes().as_ptr();
        assert_eq!(first, after, "the tile was reallocated while unshared");
    }

    #[test]
    fn a_held_tile_is_copied_exactly_once_however_many_times_it_is_written() {
        let mut image = Image::new("Paint", 512, 512, PixelFormat::Rgba8);
        image.tile_mut(0, 0).expect("in range");
        let held = image.tile_handle(0, 0).expect("resident");

        image.tile_mut(0, 0).expect("in range").bytes_mut()[0] = 1;
        let after_first = image.tile(0, 0).expect("resident").bytes().as_ptr();
        image.tile_mut(0, 0).expect("in range").bytes_mut()[0] = 2;
        let after_second = image.tile(0, 0).expect("resident").bytes().as_ptr();

        assert_eq!(after_first, after_second, "copied a second time");
        assert_eq!(held.bytes()[0], 0, "the record's copy was written through");
    }

    #[test]
    fn an_undo_can_put_a_tile_back_including_putting_it_back_to_absent() {
        // A stroke on an empty image *creates* a tile, and undoing that has to
        // remove it again rather than leave a blank one behind. A blank tile is
        // not the same as no tile: one is 64 KB and one is nothing, and a
        // painting session that accumulated the difference would grow without
        // anybody painting.
        let mut image = Image::new("Paint", 512, 512, PixelFormat::Rgba8);
        assert_eq!(image.resident_tiles(), 0);
        let before = image.tile_handle(0, 0);
        assert!(before.is_none());

        image.tile_mut(0, 0).expect("in range").bytes_mut()[0] = 7;
        assert_eq!(image.resident_tiles(), 1);

        assert!(image.set_tile(0, 0, before));
        assert_eq!(image.resident_tiles(), 0, "an empty tile was left behind");
    }

    #[test]
    fn a_tile_that_is_not_resident_reads_as_nothing_rather_than_as_noise() {
        // A half-loaded image draws as empty. The alternative — uninitialised
        // memory, or whatever the allocator had — is a texture that looks
        // corrupt for one frame and is impossible to attribute.
        let image = Image::new("Loading", 512, 512, PixelFormat::Rgba8);
        assert!(image.tile(0, 0).is_none());
        assert_eq!(
            Tile::blank(PixelFormat::Rgba8).bytes().iter().sum::<u8>(),
            0
        );
    }

    #[test]
    fn tiles_are_always_full_even_at_a_ragged_edge() {
        // The special case that is not there. A partial tile at the right and
        // bottom edges would put a branch in every sampler, every blit and
        // every mip generator, to save 0.8% of a 4097-pixel image.
        let image = Image::new("Ragged", 4097, 130, PixelFormat::Rgba8);
        assert_eq!(image.tile_counts(), (33, 2));
        assert_eq!(image.tile_count(), 66);

        let mut image = image;
        let full = (TILE * TILE) as usize * 4;
        assert_eq!(
            image.tile_mut(32, 1).expect("in range").bytes().len(),
            full,
            "the corner tile is a different size from the rest"
        );
        assert_eq!(image.tile_mut(0, 0).expect("in range").bytes().len(), full);
    }

    #[test]
    fn a_rectangle_names_the_tiles_it_crosses() {
        let image = Image::new("Grid", 512, 512, PixelFormat::Rgba8);
        // Wholly inside one tile.
        assert_eq!(image.tiles_covering(10, 10, 20, 20), vec![(0, 0)]);
        // Straddling a boundary in x only.
        assert_eq!(image.tiles_covering(120, 10, 20, 10), vec![(0, 0), (1, 0)]);
        // Straddling both.
        let both = image.tiles_covering(120, 120, 20, 20);
        assert_eq!(both, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn a_rectangle_running_off_the_edge_is_clipped_rather_than_out_of_range() {
        // A brush near a border. Asking for a tile that does not exist would be
        // a panic in the middle of a stroke.
        let image = Image::new("Small", 200, 200, PixelFormat::Rgba8);
        assert_eq!(image.tile_counts(), (2, 2));
        let covered = image.tiles_covering(190, 190, 1000, 1000);
        assert_eq!(covered, vec![(1, 1)]);
        for (x, y) in covered {
            assert!(x < 2 && y < 2);
        }

        // And entirely outside is nothing at all, not the nearest tile.
        assert!(image.tiles_covering(500, 500, 10, 10).is_empty());
        assert!(image.tiles_covering(0, 0, 0, 0).is_empty());
    }

    #[test]
    fn the_last_pixel_of_a_rectangle_is_in_it() {
        // Off by one here means a brush leaves an unpainted seam along the
        // right and bottom of its footprint, one tile wide, and only sometimes.
        let image = Image::new("Grid", 512, 512, PixelFormat::Rgba8);
        // A rectangle ending exactly at the tile boundary stays in tile zero:
        // pixels 0..=127.
        assert_eq!(image.tiles_covering(0, 0, TILE, 1), vec![(0, 0)]);
        // One more pixel reaches the next tile.
        assert_eq!(
            image.tiles_covering(0, 0, TILE + 1, 1),
            vec![(0, 0), (1, 0)]
        );
    }

    #[test]
    fn a_colour_texture_is_decoded_and_a_data_texture_is_not() {
        // §2.1's rule applied to pixels, and the classic texture bug: a
        // roughness map run through a transfer function is smooth where it
        // should be rough — which looks like a material that needs tuning, so
        // people tune it.
        let mid = 128u8;
        let colour = ColorSpace::Srgb.decode_byte(mid);
        let data = ColorSpace::Raw.decode_byte(mid);
        assert!((data - 128.0 / 255.0).abs() < 1e-6, "raw was altered");
        assert!(
            (colour - 0.2158).abs() < 0.001,
            "sRGB 128 should decode to about 0.216, got {colour}"
        );
        assert!(
            (colour - data).abs() > 0.1,
            "the two spaces gave the same answer, so the distinction does nothing"
        );
    }

    #[test]
    fn a_colour_space_and_a_format_round_trip_through_their_text_forms() {
        // The file holds the text.
        for space in [ColorSpace::Srgb, ColorSpace::Raw] {
            assert_eq!(ColorSpace::from_id(space.id()), Some(space));
        }
        assert_eq!(ColorSpace::from_id("linear"), None);
        for format in [PixelFormat::Rgba8, PixelFormat::Rgba16F, PixelFormat::R8] {
            assert_eq!(PixelFormat::from_id(format.id()), Some(format));
        }
        assert_eq!(PixelFormat::from_id("bgra8"), None);
    }

    #[test]
    fn the_default_colour_space_is_the_one_a_png_from_anywhere_is() {
        // Which is sRGB. Guessing wrong in that direction makes an albedo
        // slightly too bright; guessing wrong the other way makes every
        // ordinary texture visibly washed out.
        assert_eq!(ColorSpace::default(), ColorSpace::Srgb);
    }

    #[test]
    fn tiling_costs_a_little_for_a_full_image_and_saves_a_lot_for_a_sparse_one() {
        // Both halves, because the overhead is worth being able to state rather
        // than assume.
        let mut full = Image::new("Full", 1024, 1024, PixelFormat::Rgba8);
        let (across, down) = full.tile_counts();
        for ty in 0..down {
            for tx in 0..across {
                full.tile_mut(tx, ty);
            }
        }
        let overhead = full.memory_bytes() as f64 / full.flat_bytes() as f64;
        assert!(
            (1.0..1.01).contains(&overhead),
            "tiling a full image cost {overhead:.4} of flat"
        );

        let mut sparse = Image::new("Sparse", 1024, 1024, PixelFormat::Rgba8);
        sparse.tile_mut(0, 0);
        assert!(
            sparse.memory_bytes() * 30 < sparse.flat_bytes(),
            "one tile of a 1K image holds {} against {} flat",
            sparse.memory_bytes(),
            sparse.flat_bytes()
        );
    }

    #[test]
    fn a_size_too_large_to_address_is_answered_rather_than_reached() {
        // Found by the archive's reader, which takes a size out of a file: an
        // image whose tile grid overflows a `u32` was a multiplication that
        // panicked in debug and silently made the wrong-sized vector in
        // release. The grid is indexed as `tile_y * across + tile_x` in `u32`,
        // so this is the type's own limit and not a taste.
        assert!(Image::try_new("Absurd", u32::MAX, u32::MAX, PixelFormat::Rgba8).is_none());
        // And one just inside it is fine, which is what makes the limit a limit
        // rather than a refusal to try: 16K square is 128×128 tiles.
        let ok = Image::try_new("Big", 16384, 16384, PixelFormat::Rgba8).expect("addressable");
        assert_eq!(ok.tile_count(), 128 * 128);
    }

    #[test]
    fn a_grid_too_large_to_allocate_is_answered_rather_than_aborting() {
        // A declared image costs a pointer per tile before any pixels arrive,
        // so a hostile size asks for tens of gigabytes of nothing. A failed
        // allocation ends the process — there is no error to report and nothing
        // to catch — so the reservation is tried rather than made.
        //
        // 2^31 pixels square is 2^24 × 2^24 tiles, which is inside `u32::MAX`
        // pixels per axis and far outside any machine's memory.
        assert!(Image::try_new("Absurd", 1 << 31, 1 << 31, PixelFormat::Rgba8).is_none());
    }
}
