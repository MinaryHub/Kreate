//! A document fragment as text, for the clipboard and for a paste's history.
//!
//! **The same bytes a file is.** A copied set of objects is a whole small
//! document — `Document::extract_objects` — and it goes through the same
//! container writer a save does. Two reasons. Everything that survives a save
//! survives a copy, including the parts added next year, without a second
//! serialiser to keep in step. And a paste recorded in the history carries its
//! own bytes, so replaying it a week later, with the copied objects long
//! deleted, produces what it produced the first time.
//!
//! Hex rather than base64: the parameter system carries text, a fragment is a
//! few kilobytes, and one more dependency is not worth halving that.

use std::io::Cursor;

use kreate_scene::Document;

use crate::container::{self, ContainerError, Manifest};

/// A fragment as text.
pub fn encode(fragment: &Document) -> Result<String, ContainerError> {
    let bytes = container::write(fragment, &Manifest::default(), &[])?;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    Ok(out)
}

/// The fragment the text was.
///
/// `None` for text that is not a fragment — anything a script or another
/// application might put on the clipboard — rather than an error, because
/// "that was not ours" is the normal case for a paste and not a fault.
pub fn decode(text: &str) -> Option<Document> {
    let text = text.trim();
    if text.is_empty() || !text.len().is_multiple_of(2) {
        return None;
    }
    let mut bytes = Vec::with_capacity(text.len() / 2);
    for i in (0..text.len()).step_by(2) {
        bytes.push(u8::from_str_radix(text.get(i..i + 2)?, 16).ok()?);
    }
    container::read(Cursor::new(bytes)).ok().map(|loaded| loaded.document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::document::{Object, ObjectData};

    #[test]
    fn a_fragment_comes_back_as_it_went() {
        let mut doc = Document::new();
        let a = doc.add_mesh_object("A", kreate_mesh::primitives::cube(2.0));
        let light = doc.add_light(kreate_scene::Light::of_kind(
            "Bulb",
            kreate_scene::LightKind::BULB,
        ));
        let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(light)));
        let fragment = doc.extract_objects(&[a, lamp]);

        let text = encode(&fragment).expect("encodes");
        let back = decode(&text).expect("decodes");
        assert_eq!(back.object_count(), 2);
        assert_eq!(back.mesh_count(), 1);
        assert_eq!(back.light_count(), 1);
        let names: Vec<String> = back.objects().map(|(_, o)| o.name.clone()).collect();
        assert_eq!(names, vec!["A", "Lamp"]);
    }

    #[test]
    fn text_that_is_not_a_fragment_is_nothing_rather_than_an_error() {
        assert!(decode("").is_none());
        assert!(decode("hello").is_none());
        assert!(decode("zz").is_none());
        assert!(decode("504b0304").is_none(), "a zip header alone is not a document");
    }

    #[test]
    fn the_text_is_plain_hex() {
        // Something a parameter map, a log line and another process can all
        // carry without escaping.
        let doc = Document::new();
        let text = encode(&doc).expect("encodes");
        assert!(text.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!text.is_empty());
    }
}
