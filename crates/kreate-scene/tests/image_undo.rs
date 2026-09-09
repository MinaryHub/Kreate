//! What a brush stroke costs to undo.
//!
//! The tiling and the undo record are two halves of one decision, and neither
//! is worth much alone: tiles with a whole-image undo record still cost 64 MB a
//! step, and a tile-shaped record over a flat image cannot exist. So the claim
//! these tests make is the *joint* one — a stroke's undo record is proportional
//! to what the brush touched and not to the texture.
//!
//! Measured rather than asserted from the shape of the code, because "the undo
//! stack got big" is exactly the kind of regression that is invisible until a
//! painting session runs out of memory.

use kreate_scene::document::Document;
use kreate_scene::image::{Image, PixelFormat, Tile, TILE};
use kreate_scene::undo::UndoPayload;
use std::sync::Arc;

/// A 4K texture with every tile resident, as a loaded one would be.
fn loaded_texture(doc: &mut Document) -> kreate_scene::document::ImageId {
    let id = doc.add_image(Image::new("Albedo", 4096, 4096, PixelFormat::Rgba8));
    let image = doc.image_mut(id).expect("just added");
    let (across, down) = image.tile_counts();
    for ty in 0..down {
        for tx in 0..across {
            // A value that is not the blank, so a restore can be told from a
            // fresh tile.
            image.tile_mut(tx, ty).expect("in range").bytes_mut()[0] = 200;
        }
    }
    id
}

/// The record a stroke over `tiles` would leave, taken before it writes.
fn record(
    doc: &Document,
    id: kreate_scene::document::ImageId,
    tiles: &[(u32, u32)],
) -> UndoPayload {
    let image = doc.image(id).expect("an image");
    UndoPayload::ImageTileDelta {
        image: id,
        tiles: tiles.to_vec(),
        was: tiles
            .iter()
            .map(|(x, y)| image.tile_handle(*x, *y))
            .collect(),
    }
}

#[test]
fn a_strokes_undo_record_is_the_size_of_the_brush_and_not_of_the_texture() {
    // The whole point of §3's sixth and eleventh decisions together. One dab on
    // a 4K albedo: held flat, the record is the image.
    let mut doc = Document::new();
    let id = loaded_texture(&mut doc);
    let flat = doc.image(id).expect("an image").flat_bytes();
    assert_eq!(flat, 4096 * 4096 * 4, "the fixture is not 64 MB");

    let touched = doc
        .image(id)
        .expect("an image")
        .tiles_covering(520, 520, 40, 40);
    assert_eq!(touched.len(), 1);
    let payload = record(&doc, id, &touched);

    // Write, which is what makes the record's tiles uniquely held.
    {
        let image = doc.image_mut(id).expect("an image");
        for (x, y) in &touched {
            image.tile_mut(*x, *y).expect("in range").bytes_mut()[0] = 1;
        }
    }

    let cost = payload.bytes();
    assert!(
        cost < 70_000,
        "one dab's record holds {cost} bytes, against {flat} for the image"
    );
    assert!(
        cost * 900 < flat,
        "nine hundred dabs should still cost less than one flat copy: {cost} each"
    );
}

#[test]
fn the_record_holds_the_pixels_that_were_there() {
    // Cheap is no use if it is wrong. The restore has to put back what the
    // stroke covered, and the tiles it did not touch have to be untouched.
    let mut doc = Document::new();
    let id = loaded_texture(&mut doc);
    let touched = vec![(4u32, 4u32)];
    let payload = record(&doc, id, &touched);

    {
        let image = doc.image_mut(id).expect("an image");
        image.tile_mut(4, 4).expect("in range").bytes_mut()[0] = 1;
        assert_eq!(image.tile(4, 4).expect("resident").bytes()[0], 1);
    }

    payload.restore(&mut doc);
    let image = doc.image(id).expect("an image");
    assert_eq!(
        image.tile(4, 4).expect("resident").bytes()[0],
        200,
        "the stroke was not undone"
    );
    assert_eq!(
        image.tile(0, 0).expect("resident").bytes()[0],
        200,
        "a tile nobody touched changed"
    );
}

#[test]
fn undoing_the_first_dab_on_an_empty_texture_leaves_no_tile_behind() {
    // A blank tile is 64 KB and no tile is nothing. A painting session that
    // accumulated the difference would grow while being undone, which is the
    // opposite of what an undo is for.
    let mut doc = Document::new();
    let id = doc.add_image(Image::new("Paint", 1024, 1024, PixelFormat::Rgba8));
    assert_eq!(doc.image(id).expect("an image").resident_tiles(), 0);

    let touched = vec![(2u32, 3u32)];
    let payload = record(&doc, id, &touched);
    doc.image_mut(id)
        .expect("an image")
        .tile_mut(2, 3)
        .expect("in range")
        .bytes_mut()[0] = 9;
    assert_eq!(doc.image(id).expect("an image").resident_tiles(), 1);

    payload.restore(&mut doc);
    assert_eq!(
        doc.image(id).expect("an image").resident_tiles(),
        0,
        "the undo left an empty tile behind"
    );
}

#[test]
fn a_record_of_a_shared_tile_is_not_charged_twice() {
    // The same argument `Snapshot` makes about a mesh's arrays. A record whose
    // tile the image still holds is sharing it, and charging both would have
    // the memory budget say a painting session uses twice what it does — which
    // would make the budget useless in the one place it matters.
    let mut doc = Document::new();
    let id = loaded_texture(&mut doc);
    let payload = record(&doc, id, &[(1, 1)]);

    // Nothing has been written, so the image and the record share the tile.
    assert!(
        payload.bytes() < 1_000,
        "a shared tile was charged {} bytes",
        payload.bytes()
    );

    // Writing makes the record's copy its own, and now it is charged.
    doc.image_mut(id)
        .expect("an image")
        .tile_mut(1, 1)
        .expect("in range")
        .bytes_mut()[0] = 1;
    let one_tile = (TILE * TILE) as usize * 4;
    assert!(
        payload.bytes() >= one_tile,
        "an owned tile was charged {} bytes, not {one_tile}",
        payload.bytes()
    );
}

#[test]
fn a_stroke_across_a_shared_texture_does_not_copy_the_other_users_pixels() {
    // Two materials on one texture is the ordinary case for datablock
    // indirection. Painting on it has to affect both — that is what sharing
    // means — and must not duplicate the image to do it.
    let mut doc = Document::new();
    let id = loaded_texture(&mut doc);
    let before = doc.image(id).expect("an image").resident_tiles();

    let held = doc.image(id).expect("an image").tile_handle(0, 0);
    doc.image_mut(id)
        .expect("an image")
        .tile_mut(0, 0)
        .expect("in range")
        .bytes_mut()[0] = 1;

    assert_eq!(doc.image(id).expect("an image").resident_tiles(), before);
    assert_eq!(
        held.expect("resident").bytes()[0],
        200,
        "the held copy was written through"
    );
    assert_eq!(doc.image_count(), 1, "the image was duplicated");
}

#[test]
fn a_record_for_an_image_that_has_gone_is_a_no_op_rather_than_a_panic() {
    // Undo runs against a document other things have changed. A record naming
    // an image that is no longer there must do nothing, not bring the process
    // down halfway through a restore that has already put half the scene back.
    //
    // The image goes away the way it actually can: a snapshot from before it
    // was created is restored over the top. There is no `remove_image`, and an
    // earlier version of this test simply asserted against an image that was
    // still present — which passed with the guard replaced by an `expect`. A
    // falsification sweep caught that.
    let mut doc = Document::new();
    let before_any_image = UndoPayload::Snapshot(Box::new(doc.clone()));

    let id = doc.add_image(Image::new("Doomed", 256, 256, PixelFormat::Rgba8));
    let payload = UndoPayload::ImageTileDelta {
        image: id,
        tiles: vec![(0, 0)],
        was: vec![Some(Arc::new(Tile::blank(PixelFormat::Rgba8)))],
    };

    before_any_image.restore(&mut doc);
    assert_eq!(doc.image_count(), 0, "the fixture still has the image");

    // The record now names nothing. Restoring it must be quiet.
    payload.restore(&mut doc);
    assert_eq!(doc.image_count(), 0);
}

#[test]
fn a_tile_coordinate_outside_the_image_is_ignored() {
    // What a record written before the image was resized looks like. Writing
    // past the end of the tile vector would be a panic in the middle of an
    // undo, and clamping it would put someone else's pixels back.
    let mut doc = Document::new();
    let id = doc.add_image(Image::new("Small", 256, 256, PixelFormat::Rgba8));
    assert_eq!(doc.image(id).expect("an image").tile_counts(), (2, 2));

    let payload = UndoPayload::ImageTileDelta {
        image: id,
        tiles: vec![(0, 0), (99, 99)],
        was: vec![Some(Arc::new(Tile::blank(PixelFormat::Rgba8))), None],
    };
    payload.restore(&mut doc);
    // The in-range one landed; the out-of-range one did nothing.
    assert_eq!(doc.image(id).expect("an image").resident_tiles(), 1);
    assert!(doc.image(id).expect("an image").tile(0, 0).is_some());
}

#[test]
fn an_image_gets_an_identity_and_keeps_it_when_painted() {
    // A texture's identity is what a material's reference and a paint stroke's
    // undo record both point at, so writing pixels must not change it.
    let mut doc = Document::new();
    let id = doc.add_image(Image::new("Albedo", 256, 256, PixelFormat::Rgba8));
    let uid = doc.image_uid(id).expect("an identity");
    assert!(!uid.is_none());
    assert_eq!(doc.image_by_uid(uid), Some(id));

    doc.image_mut(id)
        .expect("an image")
        .tile_mut(0, 0)
        .expect("in range")
        .bytes_mut()[0] = 1;
    assert_eq!(
        doc.image_uid(id),
        Some(uid),
        "painting changed the identity"
    );
}
