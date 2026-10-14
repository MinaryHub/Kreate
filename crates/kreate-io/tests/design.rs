//! A design's settings through the file and back.

use kreate_io::container::{self, Manifest};
use kreate_scene::design::LengthUnit;
use kreate_scene::Document;
use std::io::Cursor;

fn round_trip(doc: &Document) -> Document {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(Cursor::new(bytes)).expect("opens").document
}

#[test]
fn the_length_unit_is_saved() {
    for unit in LengthUnit::ALL {
        let mut doc = Document::new();
        doc.design.length_unit = unit;
        assert_eq!(round_trip(&doc).design.length_unit, unit);
    }
}

#[test]
fn default_design_settings_write_nothing() {
    // The default costs no bytes, which is what keeps the golden files still.
    let bytes = container::write(&Document::new(), &Manifest::default(), &[]).expect("writes");
    assert!(!String::from_utf8_lossy(&bytes).contains("\"design\""));
}

#[test]
fn measurements_are_saved_where_they_were_taken() {
    use kreate_math::DVec3;
    use kreate_scene::design::Measurement;
    let mut doc = Document::new();
    doc.design.measurements = vec![
        Measurement {
            points: vec![DVec3::new(0.5, 1.0, -2.0), DVec3::new(0.0, 0.25, 3.0)],
        },
        Measurement {
            points: vec![DVec3::X, DVec3::ZERO, DVec3::new(0.0, 1.5, 0.0)],
        },
    ];
    assert_eq!(
        round_trip(&doc).design.measurements,
        doc.design.measurements
    );
}

#[test]
fn the_work_plane_is_saved_with_the_cursor() {
    let mut doc = Document::new();
    doc.cursor_rotation = kreate_math::DQuat::from_rotation_y(0.7);
    let back = round_trip(&doc);
    assert!(back.cursor_rotation.abs_diff_eq(doc.cursor_rotation, 1e-12));
    let bytes = container::write(&Document::new(), &Manifest::default(), &[]).expect("writes");
    assert!(!String::from_utf8_lossy(&bytes).contains("cursor_rotation"));
}

#[test]
fn the_density_is_saved() {
    let mut doc = Document::new();
    doc.design.density = 1240.0;
    assert_eq!(round_trip(&doc).design.density, 1240.0);
}

#[test]
fn the_section_view_is_saved() {
    use kreate_scene::design::Section;
    let mut doc = Document::new();
    let section = Section {
        axis: 3,
        offset: -0.25,
        flip: true,
        cap: false,
    };
    doc.design.section = Some(section);
    assert_eq!(round_trip(&doc).design.section, Some(section));
}

#[test]
fn sketch_constraints_are_saved_with_their_object() {
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::sketch::Constraint;
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::plane(2.0));
    let id = doc.add_object(Object::new("Sketch", ObjectData::Mesh(mesh)));
    let constraints = vec![
        Constraint::Distance {
            a: 0,
            b: 1,
            value: 0.04,
        },
        Constraint::Horizontal { a: 1, b: 2 },
        Constraint::Fixed {
            a: 3,
            at: [0.5, -0.25],
        },
        Constraint::Angle {
            a: 0,
            b: 1,
            c: 2,
            d: 3,
            value: 0.7,
        },
        Constraint::Equal {
            a: 0,
            b: 1,
            c: 2,
            d: 3,
        },
    ];
    doc.object_mut(id).unwrap().sketch = constraints.clone();
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Sketch").unwrap();
    assert_eq!(o.sketch, constraints);
}

/// A sketch's own lines, its circle and tangent, and a body's features —
/// an extrude cut and a revolve joined, naming the sketch by identity — come
/// back as they went, and the body is built from them again.
#[test]
fn sketch_lines_and_a_bodys_features_are_saved() {
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::feature::{Feature, Operation, Shape};
    use kreate_scene::sketch::Constraint;
    let mut doc = Document::new();
    let square = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mesh = doc.add_mesh(kreate_mesh::MeshData::from_faces(
        square
            .iter()
            .map(|p| kreate_math::Vec3::new(p[0], p[1], 0.0))
            .collect(),
        &[],
    ));
    let mut sketch = Object::new("Sketch", ObjectData::Mesh(mesh));
    sketch.sketch_lines = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
    sketch.sketch = vec![
        Constraint::Coincident { a: 0, b: 1 },
        Constraint::Circle {
            centre: 0,
            ring: vec![1, 2, 3],
        },
        Constraint::Tangent {
            a: 0,
            b: 1,
            centre: 2,
            on: 3,
        },
    ];
    let id = doc.add_object(sketch);
    let uid = doc.object_uid(id).unwrap();
    let body_mesh = doc.add_mesh(kreate_mesh::MeshData::empty());
    let mut body = Object::new("Body", ObjectData::Mesh(body_mesh));
    body.features = vec![
        Feature {
            shape: Shape::Extrude {
                distance: 0.25,
                symmetric: true,
            },
            sketch: uid,
            seeds: vec![[0.5, 0.5]],
            operation: Operation::Join,
            order: 1,
        },
        Feature {
            shape: Shape::Revolve {
                axis: [0, 3],
                angle: 1.5,
            },
            sketch: uid,
            seeds: vec![[0.25, 0.75]],
            operation: Operation::Cut,
            order: 2,
        },
    ];
    let original = body.features.clone();
    doc.add_object(body);
    let back = round_trip(&doc);
    let (_, s) = back.objects().find(|(_, o)| o.name == "Sketch").unwrap();
    assert_eq!(s.sketch_lines, vec![[0, 1], [1, 2], [2, 3], [3, 0]]);
    assert_eq!(s.sketch.len(), 3);
    let (b, o) = back.objects().find(|(_, o)| o.name == "Body").unwrap();
    assert_eq!(o.features, original);
    // And it still names the sketch: the body is built.
    let mut cache = kreate_scene::modifier::ModifierCache::new();
    let built = cache
        .evaluated(&back, b, &kreate_scene::modifier::EvalCtx::still())
        .unwrap();
    assert!(built.num_faces() > 0);
}

/// A path array — the path by its identity, every setting off its default —
/// comes back as it went, and still lays its copies along the path.
#[test]
fn a_path_array_is_saved_with_its_path_and_settings() {
    use kreate_mesh::modifiers::PathArrayParams;
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::modifier::{ModifierInstance, ModifierKind, PathArrayModifier};
    let mut doc = Document::new();
    let line = doc.add_mesh(kreate_mesh::MeshData::from_faces(
        vec![
            kreate_math::Vec3::ZERO,
            kreate_math::Vec3::new(3.0, 0.0, 0.0),
        ],
        &[],
    ));
    let mut path = Object::new("Path", ObjectData::Mesh(line));
    path.sketch_lines = vec![[0, 1]];
    let path = doc.add_object(path);
    let uid = doc.object_uid(path).unwrap();
    let post = doc.add_mesh_object("Post", kreate_mesh::primitives::cube(0.1));
    let modifier = PathArrayModifier {
        path: uid,
        params: PathArrayParams {
            count: 4,
            spacing: 0.75,
            start: 0.25,
            offset: kreate_math::Vec3::new(0.0, 0.5, 0.125),
            align: false,
        },
    };
    doc.object_mut(post)
        .unwrap()
        .modifiers
        .push(ModifierInstance::new(ModifierKind::PathArray(modifier)));
    let back = round_trip(&doc);
    let (p, o) = back.objects().find(|(_, o)| o.name == "Post").unwrap();
    assert_eq!(o.modifiers[0].kind, ModifierKind::PathArray(modifier));
    let mut cache = kreate_scene::modifier::ModifierCache::new();
    let built = cache
        .evaluated(&back, p, &kreate_scene::modifier::EvalCtx::still())
        .unwrap();
    assert_eq!(built.num_faces(), 24, "four posts");
}
