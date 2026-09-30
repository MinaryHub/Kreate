//! The 3D cursor through the file and back.

use kreate_io::container::{self, Manifest};
use kreate_math::DVec3;
use kreate_scene::Document;
use std::io::Cursor;

fn round_trip(doc: &Document) -> Document {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(Cursor::new(bytes)).expect("opens").document
}

#[test]
fn the_cursor_is_saved_where_it_was() {
    let mut doc = Document::new();
    doc.cursor = DVec3::new(1.5, -2.25, 0.125);
    assert_eq!(round_trip(&doc).cursor, doc.cursor);
}

#[test]
fn a_cursor_at_the_origin_writes_nothing() {
    // The default costs no bytes, which is what keeps the golden files still.
    let doc = Document::new();
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let text = String::from_utf8_lossy(&bytes);
    assert!(!text.contains("\"cursor\""));
    assert_eq!(round_trip(&doc).cursor, DVec3::ZERO);
}

#[test]
fn collections_are_saved_with_what_is_in_them_and_whether_hidden() {
    use kreate_scene::document::{Collection, Object, ObjectData};
    let mut doc = Document::new();
    let a = doc.add_object(Object::new("A", ObjectData::Empty));
    let _b = doc.add_object(Object::new("B", ObjectData::Empty));
    let set = doc.add_collection(Collection { name: "Set".into(), hide: true, ..Default::default() });
    doc.move_to_collection(a, Some(set));
    let back = round_trip(&doc);
    let id = back.find_collection("Set").expect("the collection came back");
    let c = back.collection(id).unwrap();
    assert!(c.hide);
    assert_eq!(c.objects.len(), 1);
    assert_eq!(back.object(c.objects[0]).unwrap().name, "A");
}
