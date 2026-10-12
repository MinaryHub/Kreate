//! A shared session's patches, as bytes.
//!
//! A patch is a [`Document`] (see `kreate_scene::document::Changes`), so its
//! body is the same `.kreate` container a file is — meshes as `.kmesh`, images
//! as tiles — behind a short JSON header naming what changed.
//!
//! ```text
//! u32 header length (big endian) · header JSON · container
//! ```

use std::io::Cursor;

use kreate_scene::{Changes, Document};
use serde::{Deserialize, Serialize};

use crate::container::{self, ContainerError, Manifest};

#[derive(Serialize, Deserialize, Default)]
struct Header {
    changed: Vec<(String, u64)>,
    removed: Vec<u64>,
    settings: bool,
}

/// What can go wrong reading a patch.
#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    #[error("the patch is cut short")]
    Short,
    #[error("the patch header is not readable: {0}")]
    Header(#[from] serde_json::Error),
    #[error(transparent)]
    Container(#[from] ContainerError),
}

/// The datablocks `changes` names, from `doc`, as bytes.
pub fn encode(doc: &Document, changes: &Changes) -> Result<Vec<u8>, ContainerError> {
    let header = Header {
        changed: changes
            .changed
            .iter()
            .map(|(k, u)| (k.id().to_string(), u.raw()))
            .collect(),
        removed: changes.removed.iter().map(|u| u.raw()).collect(),
        settings: changes.settings,
    };
    let header = serde_json::to_vec(&header).expect("a header always serialises");
    let body = container::write(&doc.patch(changes), &Manifest::default(), &[])?;
    let mut out = Vec::with_capacity(4 + header.len() + body.len());
    out.extend_from_slice(&(header.len() as u32).to_be_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&body);
    Ok(out)
}

/// The whole of `doc`, as a patch: what a joiner starts from.
pub fn encode_all(doc: &Document) -> Result<Vec<u8>, ContainerError> {
    encode(doc, &everything(doc))
}

/// Every datablock `doc` holds, as changed, with its settings.
pub fn everything(doc: &Document) -> Changes {
    let mut all = doc.changes_since(&Document::new());
    all.settings = true;
    all
}

/// A patch back from bytes: the patch document and what it changes.
pub fn decode(bytes: &[u8]) -> Result<(Document, Changes), SyncError> {
    let len =
        u32::from_be_bytes(bytes.get(..4).ok_or(SyncError::Short)?.try_into().unwrap()) as usize;
    let header: Header = serde_json::from_slice(bytes.get(4..4 + len).ok_or(SyncError::Short)?)?;
    let body = &bytes[4 + len..];
    let loaded = container::read(Cursor::new(body))?;
    let changes = Changes {
        changed: header
            .changed
            .iter()
            .filter_map(|(k, u)| {
                Some((
                    kreate_scene::DatablockKind::from_id(k)?,
                    kreate_scene::Uid::from_raw(*u),
                ))
            })
            .collect(),
        removed: header
            .removed
            .iter()
            .map(|u| kreate_scene::Uid::from_raw(*u))
            .collect(),
        settings: header.settings,
    };
    Ok((loaded.document, changes))
}

/// Reads a patch and takes it into `doc`. Returns the objects it touched.
pub fn apply(doc: &mut Document, bytes: &[u8]) -> Result<Vec<kreate_scene::ObjectId>, SyncError> {
    let (patch, changes) = decode(bytes)?;
    Ok(doc.apply_patch(&patch, &changes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::{Image, Material, PixelFormat};

    #[test]
    fn a_scene_travels_whole_and_then_by_the_edit() {
        let mut a = Document::new();
        let cube = a.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let image = a.add_image(Image::new("Bricks", 8, 8, PixelFormat::Rgba8));
        let material = a.add_material(Material {
            name: "Brick".into(),
            base_color_image: Some(image),
            ..Default::default()
        });
        let mesh = a.object(cube).unwrap().mesh().unwrap();
        a.set_mesh_slots(mesh, vec![Some(material)]);
        a.render.samples = 7;

        let mut b = Document::new();
        apply(&mut b, &encode_all(&a).unwrap()).unwrap();
        assert_eq!(b.object_count(), 1);
        assert_eq!(b.render.samples, 7);
        let (_, cube_b) = b.objects().next().unwrap();
        assert_eq!(cube_b.uid, a.object(cube).unwrap().uid);
        let slot = b.mesh_slots(cube_b.mesh().unwrap())[0].unwrap();
        let image_b = b.material(slot).unwrap().base_color_image.unwrap();
        assert_eq!(b.image(image_b).unwrap().width(), 8);

        let base = a.clone();
        a.object_mut(cube).unwrap().transform.translation.x = 4.0;
        let changes = a.changes_since(&base);
        let bytes = encode(&a, &changes).unwrap();
        apply(&mut b, &bytes).unwrap();
        let (_, cube_b) = b.objects().next().unwrap();
        assert_eq!(cube_b.transform.translation.x, 4.0);
        assert_eq!(b.mesh(cube_b.mesh().unwrap()).unwrap().num_faces(), 6);
        let image_b = b.material(slot).unwrap().base_color_image.unwrap();
        assert_eq!(b.image(image_b).unwrap().width(), 8, "not the stand-in");
    }
}
