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
