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

/// Adds a light of `kind` and leaves it active.
fn a_light(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    kind: u32,
) {
    run(
        registry,
        history,
        doc,
        ctx,
        "object.add_light",
        &[("kind", PropertyValue::Enum(kind))],
    );
}

#[test]
fn showing_a_light_and_sending_it_straight_back_changes_nothing() {
    // The panel's whole contract in one line. An absolute operator needs the
    // current values to start from, and if reading them and writing them back
    // is not an identity then opening a panel edits the light — which shows up
    // as a lamp that drifts every time it is looked at.
    //
    // Every kind, because the kinds do not share their numbers: a spot's cone
    // read off a sun would be a default arriving as though it were the light's.
    let (registry, mut history, mut doc, ctx) = setup();
    for kind in 0..4u32 {
        let mut doc = Document::new();
        let mut history = History::new();
        a_light(&registry, &mut history, &mut doc, &ctx, kind);
        let before = only_light(&doc).clone();

        let map = kreate_reflect::OpParams::to_map(
            &kreate_ops::ops::object::light_params(&before),
        );
        history
            .dispatch(&registry, &mut doc, &ctx, "light.set_properties", &map)
            .expect("dispatch");

        let after = only_light(&doc);
        assert_eq!(&before, after, "kind {kind} drifted when shown and stored");
    }
    let _ = (&mut history, &mut doc);
}

#[test]
fn the_panel_edits_the_datablock_and_every_copy_sees_it() {
    // §2.2's argument, from the user's side: a hundred linked copies of a lamp
    // share one bulb, so turning this one down turns all of them down. That is
    // the intent and not a bug, and it is the reason the datablock is not a
    // field on the object.
    let (registry, mut history, mut doc, ctx) = setup();
    a_light(&registry, &mut history, &mut doc, &ctx, 0);
    let id = doc
        .objects()
        .filter_map(|(_, o)| match o.data {
            ObjectData::Light(id) => Some(id),
            _ => None,
        })
        .next()
        .expect("a light");
    // A second object on the same datablock.
    let twin = doc.add_object(Object::new("Twin", ObjectData::Light(id)));

    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "light.set_properties",
        &[("intensity", PropertyValue::Float(42.0))],
    );
    let ObjectData::Light(twins) = doc.object(twin).expect("the twin").data else {
        panic!("the twin stopped being a light");
    };
    assert_eq!(doc.light(twins).expect("its bulb").intensity, 42.0);
}

#[test]
fn the_panel_does_not_offer_itself_for_a_cube() {
    // `poll`, which is what stops a Data tab appearing over a mesh with a
    // light's fields in it. Without it the operator would have to decide what
    // setting the light of a cube means, and there is no answer.
    let (registry, mut history, mut doc, ctx) = setup();
    let info = registry.get("light.set_properties").expect("registered");
    assert!(!info.poll(&ctx, &doc), "offered with nothing selected");

    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "object.add_primitive",
        &[],
    );
    assert!(!info.poll(&ctx, &doc), "offered for a mesh");

    a_light(&registry, &mut history, &mut doc, &ctx, 0);
    assert!(info.poll(&ctx, &doc), "not offered for a light");
}

#[test]
fn turning_a_light_all_the_way_down_leaves_it_down() {
    // The one place this operator deliberately disagrees with the add
    // operator: there, zero intensity means "the kind's own default", because
    // nobody adds a lamp of no brightness on purpose. Here zero means zero,
    // because turning one down to nothing is something people do all the time
    // and a panel that snapped it back would be unusable.
    let (registry, mut history, mut doc, ctx) = setup();
    a_light(&registry, &mut history, &mut doc, &ctx, 0);
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "light.set_properties",
        &[("intensity", PropertyValue::Float(0.0))],
    );
    assert_eq!(only_light(&doc).intensity, 0.0);
}

#[test]
fn changing_the_kind_keeps_the_numbers_that_kind_has() {
    // A point light's radius is a spot's radius, so switching between them
    // keeps it. A sun's angular diameter is nobody else's, so switching to a
    // sun brings its own.
    let (registry, mut history, mut doc, ctx) = setup();
    a_light(&registry, &mut history, &mut doc, &ctx, 0);
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "light.set_properties",
        &[
            ("kind", PropertyValue::Enum(2)),
            ("radius", PropertyValue::Float(0.25)),
            ("cone_angle", PropertyValue::Float(1.0)),
            ("blend", PropertyValue::Float(0.5)),
        ],
    );
    match only_light(&doc).kind {
        LightKind::Spot {
            radius,
            cone_angle,
            blend,
        } => {
            assert!((radius - 0.25).abs() < 1e-9);
            assert!((cone_angle - 1.0).abs() < 1e-9);
            assert!((blend - 0.5).abs() < 1e-9);
        }
        other => panic!("asked for a spot and got {other:?}"),
    }
}

#[test]
fn a_slider_drag_collapses_into_one_undo_step() {
    // A drag across a slider changes the value on every frame it moves, and one
    // undo step per frame makes undo useless in exactly the place people reach
    // for it most.
    //
    // The mechanism already exists and this is the rule for using it: the first
    // change dispatches, and every change that continues the same gesture goes
    // through `adjust_last`, which restores the step and re-runs the operator
    // in its place. Stated here, in the operator's own tests, because it is the
    // panel's obligation and a panel that forgets it looks fine until somebody
    // presses ctrl-Z.
    let (registry, mut history, mut doc, ctx) = setup();
    a_light(&registry, &mut history, &mut doc, &ctx, 0);
    let before = only_light(&doc).intensity;

    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "light.set_properties",
        &[("intensity", PropertyValue::Float(100.0))],
    );
    for v in [200.0f64, 300.0] {
        let mut params = PropertyMap::new();
        params.set("intensity", PropertyValue::Float(v));
        history
            .adjust_last(&registry, &mut doc, &ctx, &params)
            .expect("adjust");
    }
    assert_eq!(only_light(&doc).intensity, 300.0);

    assert!(history.undo(&mut doc), "nothing to undo");
    assert_eq!(
        only_light(&doc).intensity,
        before,
        "the drag left more than one step behind"
    );
}

#[test]
fn every_datablock_panel_shows_what_it_would_store() {
    // The contract every one of these panels rests on, and the one that is
    // invisible when it breaks: an absolute operator needs the current values
    // to start from, so reading them and writing them straight back has to be
    // an identity. If it is not, opening a panel edits the thing it is
    // describing — which shows up as a value that creeps every time somebody
    // looks at it.
    //
    // One test over all of them rather than one each, because it is one rule.
    let (registry, mut history, mut doc, ctx) = setup();

    // A world, a material and a camera, each reachable from its own panel.
    let world = doc.add_world(kreate_scene::World::new("World"));
    doc.set_active_world(Some(world));
    run(&registry, &mut history, &mut doc, &ctx, "object.add_primitive", &[]);
    let material = doc.add_material(kreate_scene::Material {
        name: "Paint".into(),
        base_color: kreate_math::SceneRgb::new(0.2, 0.55, 0.9),
        metallic: 0.3,
        roughness: 0.7,
        alpha: 0.9,
        ..Default::default()
    });
    let object = doc.selection.active.expect("the cube");
    if let Some(mesh) = doc.object(object).and_then(|o| o.mesh()) {
        doc.set_mesh_slots(mesh, vec![Some(material)]);
    }

    // The world.
    let before = doc.world(world).expect("a world").clone();
    let map = kreate_reflect::OpParams::to_map(&kreate_ops::ops::object::world_params(&before));
    history
        .dispatch(&registry, &mut doc, &ctx, "world.set_properties", &map)
        .expect("dispatch");
    let after = doc.world(world).expect("still there");
    assert_eq!(after.strength, before.strength);
    assert_eq!(after.rotation, before.rotation);
    // The colour to a tolerance and the rest exactly, because a colour goes
    // through the picker's space and back. A value that never came *from* a
    // picker — a default written as 0.05 — is not the image of any exact
    // swatch, so it lands a few parts in ten million away. A light's survives
    // exactly, and only because its colour was made by the same conversion in
    // the first place.
    let (kreate_scene::Background::Flat(a), kreate_scene::Background::Flat(b)) =
        (&after.background, &before.background)
    else {
        panic!("the background stopped being a colour");
    };
    for (x, y) in a.to_array().iter().zip(b.to_array().iter()) {
        assert!((x - y).abs() < 1e-6, "the world's colour drifted");
    }

    // The material.
    let before = doc.material(material).expect("a material").clone();
    let map =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::object::material_params(&before));
    history
        .dispatch(&registry, &mut doc, &ctx, "material.set_properties", &map)
        .expect("dispatch");
    let after = doc.material(material).expect("still there");
    assert!(
        (after.metallic - before.metallic).abs() < 1e-6
            && (after.roughness - before.roughness).abs() < 1e-6
            && (after.alpha - before.alpha).abs() < 1e-6,
        "the material drifted"
    );
    for (a, b) in after
        .base_color
        .to_array()
        .iter()
        .zip(before.base_color.to_array().iter())
    {
        assert!((a - b).abs() < 1e-5, "the material's colour drifted");
    }

    // The camera.
    run(&registry, &mut history, &mut doc, &ctx, "object.add_camera", &[]);
    let id = doc
        .objects()
        .filter_map(|(_, o)| match o.data {
            kreate_scene::document::ObjectData::Camera(id) => Some(id),
            _ => None,
        })
        .next()
        .expect("a camera");
    let before = doc.camera(id).expect("a camera").clone();
    let map =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::object::camera_params(&before));
    history
        .dispatch(&registry, &mut doc, &ctx, "camera.set_properties", &map)
        .expect("dispatch");
    let after = doc.camera(id).expect("still there");
    assert_eq!(after.focal_length_mm, before.focal_length_mm);
    assert_eq!(after.auto_exposure, before.auto_exposure);
    assert_eq!(after.f_stop, before.f_stop, "a pinhole stopped being one");
    assert_eq!(after.iso, before.iso);
}

#[test]
fn a_pinhole_is_off_and_not_a_very_small_aperture() {
    // `f_stop` is an `Option` because "off" and "f/1000" want different code
    // paths: one skips the depth-of-field sampling entirely. A panel has one
    // number to offer, so zero is how off is spelled — and it has to survive
    // the trip.
    let (registry, mut history, mut doc, ctx) = setup();
    run(&registry, &mut history, &mut doc, &ctx, "object.add_camera", &[]);
    let id = doc
        .objects()
        .filter_map(|(_, o)| match o.data {
            kreate_scene::document::ObjectData::Camera(id) => Some(id),
            _ => None,
        })
        .next()
        .expect("a camera");

    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "camera.set_properties",
        &[("f_stop", PropertyValue::Float(2.8))],
    );
    assert_eq!(doc.camera(id).expect("a camera").f_stop, Some(2.8));

    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "camera.set_properties",
        &[("f_stop", PropertyValue::Float(0.0))],
    );
    assert_eq!(doc.camera(id).expect("a camera").f_stop, None);
}

#[test]
fn a_colour_typed_into_the_world_panel_does_not_discard_an_image() {
    // The panel offers a colour because a colour is what a picker gives, and an
    // image background has none. Writing one anyway would throw away an HDRI
    // somebody loaded, which is not what typing in a field means — changing
    // *which* kind of background it is belongs to the control that picks the
    // image.
    let (registry, mut history, mut doc, ctx) = setup();
    let mut world = kreate_scene::World::new("World");
    // An image that is not in the document. What matters here is that the
    // background stays the *kind* it is, and an id nothing resolves is the
    // cheapest way to say "an image was chosen".
    let image = doc.add_image(kreate_scene::Image::new("HDRI", 4, 4, kreate_scene::PixelFormat::Rgba8));
    world.background = kreate_scene::Background::Image(image);
    let id = doc.add_world(world);
    doc.set_active_world(Some(id));

    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "world.set_properties",
        &[
            ("color", PropertyValue::Vec3([1.0, 0.0, 0.0])),
            ("strength", PropertyValue::Float(2.0)),
        ],
    );
    let world = doc.world(id).expect("a world");
    assert!(
        matches!(world.background, kreate_scene::Background::Image(_)),
        "the image was replaced by a colour"
    );
    // And the numbers that *do* apply to an image still went through.
    assert_eq!(world.strength, 2.0);
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
