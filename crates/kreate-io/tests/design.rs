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
    assert_eq!(round_trip(&doc).design.measurements, doc.design.measurements);
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
        Constraint::Distance { a: 0, b: 1, value: 0.04 },
        Constraint::Horizontal { a: 1, b: 2 },
        Constraint::Fixed { a: 3, at: [0.5, -0.25] },
        Constraint::Angle { a: 0, b: 1, c: 2, d: 3, value: 0.7 },
        Constraint::Equal { a: 0, b: 1, c: 2, d: 3 },
    ];
    doc.object_mut(id).unwrap().sketch = constraints.clone();
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Sketch").unwrap();
    assert_eq!(o.sketch, constraints);
}
