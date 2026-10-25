//! Lights and the world, through a save and a load.
//!
//! The claim under test is not "the fields come back" but the two things that
//! would be expensive to get wrong:
//!
//! - **The unit is the stored quantity.** A file holding a number in a unit of
//!   our own would need a conversion at every boundary, and every round trip
//!   would accumulate rounding with nobody able to say which side was right.
//! - **Identity survives.** A curve, a driver or a constraint pointing at a
//!   light has to still be pointing at it after a save, which is what makes
//!   `restore_light` rather than `add_light` the loader's call.

use kreate_io::container;
use kreate_io::container::Manifest;
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{AreaShape, Background, Document, Light, LightKind, World};

/// Writes a document and reads it back.
fn round_trip(doc: &Document) -> Document {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(std::io::Cursor::new(bytes))
        .expect("opens")
        .document
}

fn only_light(doc: &Document) -> &Light {
    let (_, light) = doc.lights().next().expect("a light datablock");
    light
}

#[test]
fn every_kind_comes_back_as_the_kind_it_was() {
    // Four kinds and four different parameter sets, flattened into one record.
    // A kind whose parameters were written under the wrong names would come
    // back as a default of the right kind — the failure that looks like
    // nothing at all until somebody asks why the shadows went hard.
    for kind in LightKind::ALL {
        let mut doc = Document::new();
        let id = doc.add_light(Light::of_kind("Lamp", kind));
        doc.add_object(Object::new("Lamp", ObjectData::Light(id)));

        let back = round_trip(&doc);
        assert_eq!(only_light(&back).kind, kind, "{}", kind.id());
    }
}

#[test]
fn every_parameter_of_every_kind_survives_with_nothing_at_its_default() {
    // Deliberately nothing at its default, and that is the whole design of
    // this test. `LightKind::ALL` is four lights each sitting on its own
    // defaults, so a field written into the wrong slot comes back *from the
    // fallback* with the value it should have had, and a round-trip test built
    // on it cannot tell a preserved field from a dropped one.
    //
    // A falsification sweep proved that: writing a spot's cone angle into the
    // sun's field survived every other test in this file, because a missing
    // `cone_angle` falls back to 45° and `LightKind::LAMP`'s cone *is* 45°.
    let kinds = [
        LightKind::Point { radius: 0.234 },
        LightKind::Sun {
            angular_diameter: 0.0731,
        },
        LightKind::Spot {
            radius: 0.171,
            cone_angle: 0.9137,
            blend: 0.372,
        },
        LightKind::Area {
            shape: AreaShape::Ellipse {
                width: 1.317,
                height: 0.482,
            },
            spread: 0.771,
        },
    ];
    for kind in kinds {
        let mut doc = Document::new();
        let mut light = Light::of_kind("Lamp", kind);
        // The rest of the light off its defaults too, for the same reason.
        light.intensity = 137.5;
        light.exposure = 0.75;
        light.color = kreate_math::SceneRgb::new(0.93, 0.41, 0.17);
        light.cast_shadow = false;
        light.affect_specular = false;
        light.max_bounces = 3;
        let id = doc.add_light(light.clone());
        doc.add_object(Object::new("Lamp", ObjectData::Light(id)));

        let back = round_trip(&doc);
        let got = only_light(&back);
        assert_eq!(got.kind, kind, "{} lost a parameter", kind.id());
        assert_eq!(got.intensity, 137.5);
        assert_eq!(got.exposure, 0.75);
        assert_eq!(got.color.to_array(), light.color.to_array());
        assert!(!got.cast_shadow);
        assert!(got.affect_diffuse);
        assert!(!got.affect_specular);
        assert_eq!(got.max_bounces, 3);
    }
}

#[test]
fn a_spots_cone_and_a_suns_disc_are_not_confused() {
    // Both are one angle in radians on a light that has exactly one, so a
    // record that wrote either into a single shared field would round trip
    // both of them and still be wrong.
    //
    // Both angles are off their defaults, so a record that wrote either into
    // the other's field cannot be rescued by a fallback.
    let spot_kind = LightKind::Spot {
        radius: 0.09,
        cone_angle: 1.234,
        blend: 0.41,
    };
    let sun_kind = LightKind::Sun {
        angular_diameter: 0.0567,
    };
    let mut doc = Document::new();
    let spot = doc.add_light(Light::of_kind("Spot", spot_kind));
    let sun = doc.add_light(Light::of_kind("Sun", sun_kind));
    doc.add_object(Object::new("Spot", ObjectData::Light(spot)));
    doc.add_object(Object::new("Sun", ObjectData::Light(sun)));

    let back = round_trip(&doc);
    let mut kinds: Vec<LightKind> = back.lights().map(|(_, l)| l.kind).collect();
    // Sorted by the text id, so "spot" comes before "sun".
    kinds.sort_by_key(|k| k.id());
    assert_eq!(kinds, vec![spot_kind, sun_kind]);
}

#[test]
fn the_intensity_is_stored_in_the_kinds_own_unit() {
    // 105,000 lux is direct summer sun. The number is meaningful only with its
    // unit, and the unit comes from the kind — which is why the file holds
    // both and converts neither.
    let mut doc = Document::new();
    let mut sun = Light::of_kind("Sun", LightKind::SUN);
    sun.intensity = 105_000.0;
    sun.exposure = -1.5;
    let id = doc.add_light(sun);
    doc.add_object(Object::new("Sun", ObjectData::Light(id)));

    let back = round_trip(&doc);
    let light = only_light(&back);
    assert_eq!(light.intensity, 105_000.0);
    assert_eq!(light.exposure, -1.5);
    assert_eq!(light.kind.intensity_unit(), kreate_scene::Intensity::Lux);
    // Which is the point: the two numbers still mean what they meant.
    assert_eq!(
        light.illuminance_at(50.0),
        Some(105_000.0 * 2f64.powf(-1.5))
    );
}

#[test]
fn the_colour_crosses_the_file_linear_and_unchanged() {
    // Linear because that is what the document holds and what glTF holds.
    // Encoding here and decoding there would put a transfer function in the
    // middle of a round trip, which is exactly the shape of the two colour
    // bugs this discipline was introduced after.
    let mut doc = Document::new();
    let mut light = Light::of_kind("Warm", LightKind::BULB);
    light.color = kreate_math::SceneRgb::new(1.0, 0.5, 0.25);
    let id = doc.add_light(light);
    doc.add_object(Object::new("Warm", ObjectData::Light(id)));

    let back = round_trip(&doc);
    assert_eq!(only_light(&back).color.to_array(), [1.0, 0.5, 0.25]);
}

#[test]
fn a_lights_identity_survives_a_save() {
    // Because a curve pointing at it has to. Asserted against a real
    // identity, not just against equality: two sides that both assigned
    // nothing would compare equal and prove nothing.
    let mut doc = Document::new();
    let id = doc.add_light(Light::of_kind("Lamp", LightKind::BULB));
    doc.add_object(Object::new("Lamp", ObjectData::Light(id)));
    let before = doc.light_uid(id).expect("an identity");
    assert!(!before.is_none(), "the writer had nothing to write");

    let back = round_trip(&doc);
    let after = only_light(&back).uid;
    assert!(!after.is_none(), "the loader issued nothing");
    assert_eq!(after, before);
    // And the identity resolves, which is what a property path does with it.
    assert!(back.light_by_uid(after).is_some());
}

#[test]
fn identities_with_gaps_in_them_come_back_as_they_were() {
    // A re-issuing loader passes a test that saves three lights and gets three
    // back, because the numbers come out in the same order. It fails this one:
    // the lights that were deleted took their numbers with them.
    let mut doc = Document::new();
    let a = doc.add_light(Light::of_kind("A", LightKind::BULB));
    let _b = doc.add_light(Light::of_kind("B", LightKind::BULB));
    let _c = doc.add_light(Light::of_kind("C", LightKind::BULB));
    let d = doc.add_light(Light::of_kind("D", LightKind::SUN));
    // Only the first and the last are pointed at, so only they are kept.
    let mut sparse = Document::new();
    let first = sparse.restore_light(
        Light::of_kind("A", LightKind::BULB),
        doc.light_uid(a).unwrap(),
    );
    let last = sparse.restore_light(
        Light::of_kind("D", LightKind::SUN),
        doc.light_uid(d).unwrap(),
    );
    sparse.add_object(Object::new("A", ObjectData::Light(first)));
    sparse.add_object(Object::new("D", ObjectData::Light(last)));

    let expected = [
        sparse.light_uid(first).unwrap().raw(),
        sparse.light_uid(last).unwrap().raw(),
    ];
    assert_eq!(expected, [1, 4], "the fixture has no gap in it");

    let back = round_trip(&sparse);
    let mut uids: Vec<u64> = back.lights().map(|(_, l)| l.uid.raw()).collect();
    uids.sort_unstable();
    assert_eq!(uids, expected.to_vec());
}

#[test]
fn a_new_light_after_a_load_cannot_collide_with_a_loaded_one() {
    // The counter has to be advanced past every identity the file used,
    // including the ones it skipped. A loader counting what it loaded would
    // hand the next light number 3 while the file already used 4.
    let mut doc = Document::new();
    let sparse = doc.restore_light(
        Light::of_kind("D", LightKind::SUN),
        kreate_scene::Uid::from_raw(9),
    );
    doc.add_object(Object::new("D", ObjectData::Light(sparse)));

    let mut back = round_trip(&doc);
    let fresh = back.add_light(Light::of_kind("New", LightKind::BULB));
    assert!(back.light_uid(fresh).unwrap().raw() > 9);
}

#[test]
fn an_object_comes_back_pointing_at_its_own_light() {
    // The link, which is a within-file index and not an identity. Two lights
    // and two objects is the smallest case that can be got the wrong way
    // round.
    let mut doc = Document::new();
    let sun = doc.add_light(Light::of_kind("Sun", LightKind::SUN));
    let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
    doc.add_object(Object::new("Outdoor", ObjectData::Light(sun)));
    doc.add_object(Object::new("Indoor", ObjectData::Light(bulb)));

    let back = round_trip(&doc);
    for (_, object) in back.objects() {
        let ObjectData::Light(id) = object.data else {
            panic!("{} came back as something else", object.name)
        };
        let light = back.light(id).expect("its datablock");
        assert_eq!(
            light.name,
            if object.name == "Outdoor" {
                "Sun"
            } else {
                "Bulb"
            }
        );
    }
}

#[test]
fn two_objects_sharing_one_bulb_still_share_it_after_a_load() {
    // The whole reason a light is a datablock. A loader that wrote the light
    // into each object would give the pair two bulbs, and the next edit to one
    // of them would only move half the scene.
    let mut doc = Document::new();
    let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
    doc.add_object(Object::new("Left", ObjectData::Light(bulb)));
    doc.add_object(Object::new("Right", ObjectData::Light(bulb)));

    let back = round_trip(&doc);
    assert_eq!(back.light_count(), 1, "the bulb was copied");
    let ids: Vec<_> = back
        .objects()
        .filter_map(|(_, o)| match o.data {
            ObjectData::Light(id) => Some(id),
            _ => None,
        })
        .collect();
    assert_eq!(ids.len(), 2);
    assert_eq!(ids[0], ids[1]);
}

#[test]
fn the_switches_a_lighting_artist_reaches_for_survive() {
    // The cheats, and they are not optional: a rim light that must not blow
    // out a highlight is an ordinary request. A round trip that dropped them
    // would silently turn a finished shot back into a physical one.
    let mut doc = Document::new();
    let mut rim = Light::of_kind("Rim", LightKind::PANEL);
    rim.cast_shadow = false;
    rim.affect_diffuse = false;
    rim.max_bounces = 2;
    let id = doc.add_light(rim);
    doc.add_object(Object::new("Rim", ObjectData::Light(id)));

    let back = round_trip(&doc);
    let light = only_light(&back);
    assert!(!light.cast_shadow);
    assert!(!light.affect_diffuse);
    assert!(light.affect_specular);
    assert_eq!(light.max_bounces, 2);
}

#[test]
fn an_area_lights_shape_and_size_come_back() {
    // Which is its brightness: nits times area is candela, so a panel that
    // came back the wrong size came back the wrong brightness.
    let shapes = [
        AreaShape::Rect {
            width: 2.0,
            height: 0.5,
        },
        AreaShape::Disc { radius: 0.75 },
        AreaShape::Ellipse {
            width: 3.0,
            height: 1.0,
        },
    ];
    for shape in shapes {
        let mut doc = Document::new();
        let id = doc.add_light(Light::of_kind(
            "Panel",
            LightKind::Area { shape, spread: 1.2 },
        ));
        doc.add_object(Object::new("Panel", ObjectData::Light(id)));

        let back = round_trip(&doc);
        let LightKind::Area { shape: got, spread } = only_light(&back).kind else {
            panic!("not an area light")
        };
        assert_eq!(got, shape);
        assert!((spread - 1.2).abs() < 1e-12);
        // And so the brightness along the normal is unchanged.
        assert!((got.area_m2() - shape.area_m2()).abs() < 1e-12);
    }
}

#[test]
fn a_light_kind_this_build_has_never_heard_of_opens_as_a_light() {
    // A scene from a later version is worth opening with one lamp of the wrong
    // shape. Refusing the file would make a forwards-incompatible addition
    // into a scene nobody can recover, which is exactly what the format's
    // unknown-entry passthrough exists to prevent elsewhere.
    let mut doc = Document::new();
    let id = doc.add_light(Light::of_kind("Mystery", LightKind::BULB));
    doc.add_object(Object::new("Mystery", ObjectData::Light(id)));
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");

    let tampered = retag_the_only_light(&bytes, "hyperbolic");
    let back = container::read(std::io::Cursor::new(tampered))
        .expect("a later version's light must not fail the load")
        .document;
    let light = only_light(&back);
    assert_eq!(light.name, "Mystery");
    assert_eq!(light.intensity, 1000.0, "the brightness was lost");
    assert_eq!(light.kind, LightKind::BULB, "it did not fall back");
}

#[test]
fn the_world_is_saved_and_the_scene_still_points_at_it() {
    let mut doc = Document::new();
    let studio = doc.add_world(World {
        background: Background::Flat(kreate_math::SceneRgb::new(0.2, 0.3, 0.5)),
        strength: 2.5,
        rotation: 1.25,
        ..World::new("Studio")
    });
    doc.add_world(World::new("Night"));
    doc.set_active_world(Some(studio));

    let back = round_trip(&doc);
    assert_eq!(back.world_count(), 2);
    let active = back.active_world().expect("a world is lighting the scene");
    let world = back.world(active).expect("its datablock");
    assert_eq!(world.name, "Studio");
    assert_eq!(world.strength, 2.5);
    assert_eq!(world.rotation, 1.25);
    assert_eq!(
        world.flat_radiance().map(|c| c.to_array()),
        Some([0.5, 0.75, 1.25])
    );
}

#[test]
fn a_scene_with_no_world_comes_back_with_none() {
    // Rather than with a default one invented on the way in. A scene lit by
    // nothing but its own lights is a real thing to want, and a loader that
    // supplied a grey background would add light to every file that had none.
    let mut doc = Document::new();
    doc.add_world(World::new("Unused"));
    assert_eq!(doc.active_world(), None);

    let back = round_trip(&doc);
    assert_eq!(back.world_count(), 1, "the unused world was dropped");
    assert_eq!(back.active_world(), None);
}

/// A world lit by an environment, with pixels in it.
fn sky_document() -> (Document, kreate_scene::document::ImageId) {
    let mut doc = Document::new();
    let mut sky = kreate_scene::Image::new("sky", 64, 32, kreate_scene::PixelFormat::Rgba16F);
    sky.color_space = kreate_scene::ColorSpace::Raw;
    // One recognisable value, so "the image came back" is a claim about pixels
    // and not only about a name.
    sky.tile_mut(0, 0).expect("in range").bytes_mut()[0] = 42;
    let image = doc.add_image(sky);
    let world = doc.add_world(World {
        background: Background::Image(image),
        strength: 0.8,
        ..World::new("Sky")
    });
    doc.set_active_world(Some(world));
    (doc, image)
}

#[test]
fn an_hdri_background_finds_its_image_again() {
    // The record has stored the background's *identity* rather than an index
    // since minor 2, against a day when the archive would carry images. This is
    // that day: the pointer survived three minor versions of not resolving, and
    // resolves now with no change to the format.
    let (doc, image) = sky_document();
    let uid = doc.image_uid(image).expect("an identity").raw();

    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let back = container::read(std::io::Cursor::new(bytes.clone()))
        .expect("read")
        .document;
    let world = back
        .world(back.active_world().expect("active"))
        .expect("world");
    let Background::Image(id) = world.background else {
        panic!("the background came back as {:?}", world.background);
    };
    assert_eq!(world.strength, 0.8, "the rest of the world was lost");
    let image = back.image(id).expect("and the image it names");
    assert_eq!(image.name, "sky");
    assert_eq!(
        image.tile(0, 0).map(|t| t.bytes()[0]),
        Some(42),
        "the background resolved to an image with the wrong pixels"
    );

    // And by identity rather than by position, which is what makes it survive a
    // save that reorders the arena.
    let scene = scene_json(&bytes);
    assert_eq!(
        scene["worlds"][0]["image_uid"].as_u64(),
        Some(uid),
        "the background's identity was not written"
    );
}

#[test]
fn a_background_naming_an_image_that_is_not_there_lands_on_a_colour() {
    // The fallback, which used to be the ordinary case and is now an error
    // case: a file whose image was removed, or one from a build that wrote an
    // image this one dropped. It comes back flat rather than pointing at
    // nothing, because a background that is `Image(missing)` would make every
    // consumer carry a branch for it.
    //
    // Constructed by editing the identity in the file, which is the only way to
    // reach the branch now that images are written — and that is the point:
    // the branch has no other test, and an untested fallback is one that
    // quietly stops working.
    let (doc, _) = sky_document();
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let broken = replace_in_scene(&bytes, "\"image_uid\":1", "\"image_uid\":9999");

    let back = container::read(std::io::Cursor::new(broken))
        .expect("read")
        .document;
    let world = back
        .world(back.active_world().expect("active"))
        .expect("world");
    assert!(
        matches!(world.background, Background::Flat(_)),
        "an identity nothing answers to came back as a pointer"
    );
    // Grey and not black: a missing environment should read as a missing
    // environment rather than as a broken file.
    let Background::Flat(color) = world.background else {
        unreachable!()
    };
    assert!(color.r > 0.0, "the fallback is black");
    assert_eq!(world.strength, 0.8, "the rest of the world was lost");
}

#[test]
fn an_out_of_range_active_world_opens_the_file() {
    // Same judgement as an unknown projection: a scene from a later version,
    // or a hand-edited one, is worth opening lit by its own lights.
    let mut doc = Document::new();
    let world = doc.add_world(World::new("W"));
    doc.set_active_world(Some(world));
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");

    let tampered = replace_in_scene(&bytes, "\"active_world\":0", "\"active_world\":7");
    let back = container::read(std::io::Cursor::new(tampered))
        .expect("a bad index must not fail the load")
        .document;
    assert_eq!(back.active_world(), None);
    assert_eq!(back.world_count(), 1);
}

// --- reaching into the archive --------------------------------------------
//
// These tests need to write a file this build cannot produce — a light of a
// kind from the future, an index out of range — so they edit the scene JSON
// inside the archive. Reading it back out is also how the pointer test proves
// something was written rather than merely not lost.

fn scene_bytes(archive: &[u8]) -> String {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive.to_vec())).expect("archive");
    let mut file = zip.by_name("scene.json").expect("scene.json");
    let mut text = String::new();
    std::io::Read::read_to_string(&mut file, &mut text).expect("utf-8");
    text
}

fn scene_json(archive: &[u8]) -> serde_json::Value {
    serde_json::from_str(&scene_bytes(archive)).expect("json")
}

/// Rewrites one archive with an edited `scene.json`.
fn replace_in_scene(archive: &[u8], from: &str, to: &str) -> Vec<u8> {
    let text = scene_bytes(archive);
    let compact = serde_json::to_string(&serde_json::from_str::<serde_json::Value>(&text).unwrap())
        .expect("compact");
    assert!(compact.contains(from), "{from} is not in the scene");
    rewrite_scene(archive, &compact.replace(from, to))
}

fn retag_the_only_light(archive: &[u8], kind: &str) -> Vec<u8> {
    let mut scene = scene_json(archive);
    scene["lights"][0]["kind"] = serde_json::Value::String(kind.to_string());
    rewrite_scene(archive, &serde_json::to_string(&scene).expect("json"))
}

fn rewrite_scene(archive: &[u8], scene: &str) -> Vec<u8> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive.to_vec())).expect("archive");
    let mut out = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).expect("entry");
        let name = entry.name().expect("a named entry").into_owned();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut bytes).expect("read");
        out.start_file(&name, options).expect("start");
        let body = if name == "scene.json" {
            scene.as_bytes()
        } else {
            &bytes
        };
        std::io::Write::write_all(&mut out, body).expect("write");
    }
    out.finish().expect("finish").into_inner()
}
