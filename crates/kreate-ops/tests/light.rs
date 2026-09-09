//! A light is a datablock, and its brightness is in a real unit.
//!
//! §3's fourth and tenth "decide now", checked from both directions: the
//! operator that creates one, and what the document then holds. Two claims are
//! being tested and neither is "the struct stores its fields":
//!
//! - The object is *where* the light is and the datablock is *what* it is, so
//!   two objects can share one bulb and editing it edits both.
//! - The stored intensity is in the unit a datasheet quotes, which is what
//!   makes a glTF round trip lossless and a light meter's reading checkable.

use kreate_ops::operator::{Context, Mode};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Document, Intensity, Light, LightKind};

fn setup() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (
        registry,
        History::new(),
        Document::new(),
        Context { mode: Mode::Object },
    )
}

fn run(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    op: &str,
    pairs: &[(&'static str, PropertyValue)],
) -> kreate_ops::OpResult {
    let mut params = PropertyMap::new();
    for (k, v) in pairs {
        params.set(k, v.clone());
    }
    history
        .dispatch(registry, doc, ctx, op, &params)
        .expect("dispatch")
}

/// The light datablock the one light object in a document refers to.
fn only_light(doc: &Document) -> &Light {
    let id = doc
        .objects()
        .filter_map(|(_, o)| match o.data {
            ObjectData::Light(id) => Some(id),
            _ => None,
        })
        .next()
        .expect("a light object");
    doc.light(id).expect("its datablock")
}

#[test]
fn adding_a_light_makes_an_object_and_a_datablock() {
    // The distinction is the point: `ObjectData::Light` was a bare tag, which
    // said an object was a light and nothing about the light.
    let (r, mut h, mut d, c) = setup();
    run(&r, &mut h, &mut d, &c, "object.add_light", &[]);

    assert_eq!(d.objects().count(), 1);
    assert_eq!(d.light_count(), 1);
    let light = only_light(&d);
    assert_eq!(light.kind, LightKind::BULB);
    assert_eq!(light.intensity, 1000.0);
    assert!(!light.uid.is_none(), "the datablock has no identity");
}

#[test]
fn each_kind_arrives_at_its_own_brightness_in_its_own_unit() {
    // The reason `intensity` defaults to zero rather than to a number. There is
    // no single number: 1000 is a bright room in candela and a dull overcast
    // afternoon in lux, so a shared default would make three of the four kinds
    // wrong and the sun wrong in the way that is hardest to notice — flat
    // outdoor light that looks like a lighting choice.
    let expected = [
        (0, Intensity::Candela, 1000.0),
        (1, Intensity::Lux, 20_000.0),
        (2, Intensity::Candela, 1000.0),
        (3, Intensity::Nit, 1000.0),
    ];
    for (index, unit, intensity) in expected {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_light",
            &[("kind", PropertyValue::Enum(index))],
        );
        let light = only_light(&d);
        assert_eq!(light.kind.intensity_unit(), unit, "kind {index}");
        assert_eq!(light.intensity, intensity, "kind {index}");
    }
}

#[test]
fn a_given_intensity_is_stored_as_given() {
    // Because it is already in the unit the file holds. A tool that converted
    // here would accumulate rounding on every round trip and leave nobody able
    // to say which side was authoritative.
    let (r, mut h, mut d, c) = setup();
    run(
        &r,
        &mut h,
        &mut d,
        &c,
        "object.add_light",
        &[
            ("kind", PropertyValue::Enum(1)),
            ("intensity", PropertyValue::Float(105_000.0)),
        ],
    );
    let light = only_light(&d);
    assert_eq!(light.intensity, 105_000.0);
    assert_eq!(light.effective_intensity(), 105_000.0);
}

#[test]
fn a_stale_keymap_naming_a_kind_this_build_lacks_still_adds_a_light() {
    // Two mechanisms in a row, and it is worth being precise about which one
    // does the work here: the *property map* clamps an out-of-range enum into
    // the variant list before an operator sees it, so by the time `kind`
    // reaches the operator it is already 3. This test covers the dispatch
    // path — a keymap or a script naming a kind from a later version — and
    // `an_out_of_range_kind_from_rust_is_clamped_too` covers the other path,
    // where nothing has clamped anything yet.
    //
    // A falsification sweep found this: breaking the operator's own clamp did
    // not fail this test, because this test never reaches it.
    let (r, mut h, mut d, c) = setup();
    run(
        &r,
        &mut h,
        &mut d,
        &c,
        "object.add_light",
        &[("kind", PropertyValue::Enum(99))],
    );
    assert_eq!(d.light_count(), 1);
    assert_eq!(only_light(&d).kind, LightKind::PANEL);
}

#[test]
fn an_out_of_range_kind_from_rust_is_clamped_too() {
    // The typed path: `Operator::execute` with a struct built in Rust, which is
    // how a Rust caller reaches an operator and the one route where nothing has
    // clamped anything. A panic here would be a panic inside somebody's tool;
    // the last kind is the worst honest answer.
    //
    // Reached directly rather than through `dispatch`, because `dispatch` goes
    // through a `PropertyMap` and that is exactly the clamp being stepped
    // around.
    use kreate_ops::operator::Operator as _;

    let (_r, _h, mut d, c) = setup();
    let params = kreate_ops::ops::object::AddLightParams {
        kind: 99,
        ..Default::default()
    };
    {
        let mut edit = kreate_scene::DocMut::new(&mut d);
        kreate_ops::ops::object::AddLight::execute(&params, &mut edit, &c)
            .expect("an out-of-range kind must not panic");
    }

    assert_eq!(d.light_count(), 1);
    assert_eq!(
        d.lights().next().expect("the light").1.kind,
        LightKind::PANEL
    );
}

#[test]
fn the_colour_is_decoded_out_of_the_pickers_space() {
    // The colour work's rule, applied at the one boundary it exists for: the
    // picker holds sRGB and the document holds the working space. A tint that
    // went in encoded would make every mid-grey lamp about twice as bright as
    // it looks — which is precisely one of the two colour bugs this discipline
    // was introduced after.
    let (r, mut h, mut d, c) = setup();
    run(
        &r,
        &mut h,
        &mut d,
        &c,
        "object.add_light",
        &[("color", PropertyValue::Vec3([0.5, 0.5, 0.5]))],
    );
    let light = only_light(&d);
    let expected = kreate_math::srgb_to_linear(0.5);
    assert!(
        (light.color.r - expected).abs() < 1e-6,
        "{} is not {expected}",
        light.color.r
    );
    assert!(expected < 0.25, "the transfer function did nothing");
}

#[test]
fn white_survives_the_decode_exactly() {
    // Both ends of the curve are fixed points, and the default light is white.
    // A conversion that was even slightly off at one would tint every light a
    // scene starts with.
    let (r, mut h, mut d, c) = setup();
    run(&r, &mut h, &mut d, &c, "object.add_light", &[]);
    let color = only_light(&d).color;
    assert_eq!(color.to_array(), [1.0, 1.0, 1.0]);
}

#[test]
fn the_object_carries_the_placement_and_the_datablock_does_not() {
    // Which is what lets two objects share one bulb, and what makes a sun's
    // direction a *rotation* rather than a vector in the light. A light with
    // its own position would need its transform kept in step with the object's,
    // and nothing keeps two copies of a transform in step.
    let (r, mut h, mut d, c) = setup();
    run(
        &r,
        &mut h,
        &mut d,
        &c,
        "object.add_light",
        &[("location", PropertyValue::Vec3([1.0, 2.0, 3.0]))],
    );
    let (_, object) = d.objects().next().expect("the light object");
    assert_eq!(object.transform.translation.to_array(), [1.0, 2.0, 3.0]);
}

#[test]
fn two_objects_can_share_one_bulb() {
    // §2.2's whole argument for a datablock. A hundred linked copies of a lamp
    // share one setting, so changing the studio's colour is one edit.
    let mut doc = Document::new();
    let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
    doc.add_object(Object::new("Left", ObjectData::Light(bulb)));
    doc.add_object(Object::new("Right", ObjectData::Light(bulb)));

    doc.light_mut(bulb).expect("the bulb").intensity = 250.0;

    assert_eq!(doc.light_count(), 1, "the bulb was copied");
    for (_, object) in doc.objects() {
        let ObjectData::Light(id) = object.data else {
            panic!("not a light")
        };
        assert_eq!(doc.light(id).expect("its datablock").intensity, 250.0);
    }
}

#[test]
fn a_light_is_undone_as_one_step() {
    // Both halves of it. An undo that took back the object and left the
    // datablock would leave a light nothing points at in every file from then
    // on, and the leak would only show as a growing file.
    let (r, mut h, mut d, c) = setup();
    run(&r, &mut h, &mut d, &c, "object.add_light", &[]);
    assert_eq!((d.objects().count(), d.light_count()), (1, 1));

    assert!(h.undo(&mut d), "the creation should undo");
    assert_eq!((d.objects().count(), d.light_count()), (0, 0));

    assert!(
        h.redo(&r, &mut d, &c).expect("redo"),
        "the creation should redo"
    );
    assert_eq!((d.objects().count(), d.light_count()), (1, 1));
}

#[test]
fn a_redone_light_keeps_the_identity_it_had() {
    // Because a curve or a driver pointing at it has to still be pointing at
    // it. The counter travels with the snapshot, which is what makes this true
    // rather than lucky — a counter outside the snapshot hands the redone
    // light a fresh id and orphans its animation.
    let (r, mut h, mut d, c) = setup();
    run(&r, &mut h, &mut d, &c, "object.add_light", &[]);
    let before = only_light(&d).uid;

    assert!(h.undo(&mut d), "the creation should undo");
    assert!(
        h.redo(&r, &mut d, &c).expect("redo"),
        "the creation should redo"
    );
    assert_eq!(only_light(&d).uid, before);
}

#[test]
fn the_scene_reports_every_light_with_the_transform_that_places_it() {
    // The join a renderer needs, and the reason the document does it: a
    // caller walking objects itself reads the datablock once and gets the
    // linked-copy case wrong, and a caller walking the light arena has no
    // transform at all.
    let mut doc = Document::new();
    let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
    let mut left = Object::new("Left", ObjectData::Light(bulb));
    left.transform.translation = kreate_math::DVec3::new(-2.0, 0.0, 0.0);
    doc.add_object(left);
    let mut right = Object::new("Right", ObjectData::Light(bulb));
    right.transform.translation = kreate_math::DVec3::new(2.0, 0.0, 0.0);
    doc.add_object(right);
    doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(1.0));

    let placed: Vec<f64> = doc
        .lit_by()
        .map(|(_, light, world)| {
            assert_eq!(light.intensity, 1000.0);
            world.translation.x
        })
        .collect();
    assert_eq!(placed.len(), 2, "the mesh or a copy went missing");
    assert!(placed.contains(&-2.0) && placed.contains(&2.0));
}

#[test]
fn a_lights_placement_follows_its_parent() {
    // A light parented to a rig moves with the rig, and a renderer reading the
    // object's *local* transform would leave every rigged lamp at the origin.
    let mut doc = Document::new();
    let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
    let rig = doc.add_object(Object::new("Rig", ObjectData::Empty));
    let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(bulb)));
    // Parented first and moved afterwards, because `set_parent` deliberately
    // keeps the object where it was — parenting a lamp must not teleport it.
    doc.set_parent(lamp, Some(rig)).expect("parent");
    doc.object_mut(rig).expect("the rig").transform.translation =
        kreate_math::DVec3::new(0.0, 0.0, 10.0);

    let (_, _, world) = doc.lit_by().next().expect("the lamp");
    assert_eq!(world.translation.z, 10.0);
}

#[test]
fn switching_an_object_off_switches_its_light_off() {
    // Anything else makes the eye icon a lie, and "why is this still lighting
    // the scene" is not a question a user should have to ask about a hidden
    // object.
    let mut doc = Document::new();
    let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
    let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(bulb)));
    assert_eq!(doc.lit_by().count(), 1);

    doc.object_mut(lamp).expect("the lamp").visible = false;
    assert_eq!(doc.lit_by().count(), 0);
}

#[test]
fn a_light_whose_datablock_is_gone_lights_nothing_rather_than_panicking() {
    // Reachable through undo: a snapshot from before the light existed, put
    // back under an object that was added afterwards. Skipping it is the only
    // answer that leaves the file openable.
    let mut doc = Document::new();
    let bulb = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));

    // The same slot, in a document that never had it filled — which is what
    // undoing to a snapshot from before the light was created leaves behind.
    let mut earlier = Document::new();
    earlier.add_object(Object::new("Lamp", ObjectData::Light(bulb)));
    assert_eq!(earlier.light_count(), 0);
    assert_eq!(earlier.lit_by().count(), 0);

    // And the intact document is unaffected, so the guard is not just
    // returning nothing for everybody.
    doc.add_object(Object::new("Lamp", ObjectData::Light(bulb)));
    assert_eq!(doc.lit_by().count(), 1);
}

#[test]
fn the_world_is_a_datablock_and_the_scene_points_at_one() {
    // A pointer, so that swapping a lighting setup is one assignment and the
    // setup that was there is still in the file — which is what a studio
    // library is.
    let mut doc = Document::new();
    assert_eq!(doc.active_world(), None, "a scene starts with no world");

    let studio = doc.add_world(kreate_scene::World::new("Studio"));
    let night = doc.add_world(kreate_scene::World::new("Night"));
    assert!(doc.set_active_world(Some(studio)));
    assert_eq!(doc.active_world(), Some(studio));

    assert!(doc.set_active_world(Some(night)));
    assert_eq!(doc.world_count(), 2, "swapping a world deleted one");
    assert!(doc.set_active_world(None), "a scene may have no world");
}

#[test]
fn a_world_that_is_not_in_the_document_is_refused() {
    // An active pointer to nothing would make every consumer of the background
    // carry a "and if it is missing" branch for a state no honest path
    // produces.
    let mut doc = Document::new();
    let stranger = kreate_scene::Document::new().add_world(kreate_scene::World::new("Elsewhere"));
    assert!(!doc.set_active_world(Some(stranger)));
    assert_eq!(doc.active_world(), None);
}

#[test]
fn a_light_and_a_world_have_separate_identity_counters() {
    // One counter per kind, so the numbers stay small and a person reading a
    // scene file can match a curve to its target by eye. `DatablockKind` is
    // what keeps light 1 and world 1 apart.
    let mut doc = Document::new();
    let light = doc.add_light(Light::of_kind("A", LightKind::BULB));
    let world = doc.add_world(kreate_scene::World::new("W"));
    assert_eq!(doc.light_uid(light), doc.world_uid(world));
    assert_eq!(doc.light_by_uid(doc.light_uid(light).unwrap()), Some(light));
    assert_eq!(doc.world_by_uid(doc.world_uid(world).unwrap()), Some(world));
}
