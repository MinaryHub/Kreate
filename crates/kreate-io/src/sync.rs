//! A shared session's patches, as bytes.
//!
//! A patch is a [`Document`] (see `kreate_scene::document::Changes`), so its
//! body is the same `.kreate` container a file is — meshes as `.kmesh`, images
//! as tiles — behind a short JSON header naming what changed.
//!
//! ```text
//! u32 header length (big endian) · header JSON · container
//! ```

use std::io::{Read, Write};
use std::sync::Arc;

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_scene::{Changes, DatablockKind, Document, Uid};
use serde::{Deserialize, Serialize};

use crate::container::{self, ContainerError, Manifest};

/// Magic prefix for zlib-compressed sync payloads (`"KSZ1"`).
const COMPRESSED_MAGIC: &[u8; 4] = b"KSZ1";

/// Safety ceiling on decompressed sync payload size (1 GiB, matching `wire::MAX_FRAME`).
const MAX_UNPACKED: u64 = 1 << 30;

#[derive(Serialize, Deserialize, Default)]
struct Header {
    changed: Vec<(String, u64)>,
    removed: Vec<u64>,
    settings: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pos_patches: Vec<PosPatchMeta>,
}

/// Metadata for a mesh whose topology, material slots, and non-position
/// attributes match the agreed base, allowing its vertex positions to travel
/// without re-serializing connectivity or UV layers.
#[derive(Serialize, Deserialize)]
struct PosPatchMeta {
    uid: u64,
    num_verts: u32,
    /// `true`: payload holds `count` `(u32 index, [f32; 3])` 16-byte records.
    /// `false`: payload holds `num_verts` `[f32; 3]` 12-byte records.
    sparse: bool,
    count: u32,
}

/// What can go wrong reading a patch.
#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    #[error("the patch is cut short")]
    Short,
    #[error("the patch header is not readable: {0}")]
    Header(#[from] serde_json::Error),
    #[error("compressed patch is damaged: {0}")]
    Compressed(String),
    #[error(transparent)]
    Container(#[from] ContainerError),
}

fn wrap_compressed(raw: &[u8]) -> Vec<u8> {
    let mut enc = ZlibEncoder::new(Vec::with_capacity(raw.len() / 2 + 64), Compression::fast());
    if enc.write_all(raw).is_ok() {
        if let Ok(compressed) = enc.finish() {
            if compressed.len() + 4 < raw.len() {
                let mut framed = Vec::with_capacity(4 + compressed.len());
                framed.extend_from_slice(COMPRESSED_MAGIC);
                framed.extend_from_slice(&compressed);
                return framed;
            }
        }
    }
    raw.to_vec()
}

fn unwrap_compressed(bytes: &[u8]) -> Result<std::borrow::Cow<'_, [u8]>, SyncError> {
    if let Some(body) = bytes.strip_prefix(COMPRESSED_MAGIC) {
        let mut out = Vec::new();
        let mut dec = ZlibDecoder::new(body).take(MAX_UNPACKED + 1);
        dec.read_to_end(&mut out)
            .map_err(|e| SyncError::Compressed(e.to_string()))?;
        if out.len() as u64 > MAX_UNPACKED {
            return Err(SyncError::Compressed("patch exceeds maximum size".into()));
        }
        Ok(std::borrow::Cow::Owned(out))
    } else {
        Ok(std::borrow::Cow::Borrowed(bytes))
    }
}

/// The datablocks `changes` names, from `doc`, as bytes.
pub fn encode(doc: &Document, changes: &Changes) -> Result<Vec<u8>, ContainerError> {
    encode_inner(None, doc, changes)
}

/// Encodes `changes` from `doc` against an agreed `base` document, emitting
/// compact vertex-position deltas whenever a mesh changed only in its vertex
/// coordinates (e.g. sculpting or edit-mode transforms).
pub fn encode_since(
    base: &Document,
    doc: &Document,
    changes: &Changes,
) -> Result<Vec<u8>, ContainerError> {
    encode_inner(Some(base), doc, changes)
}

fn encode_inner(
    base: Option<&Document>,
    doc: &Document,
    changes: &Changes,
) -> Result<Vec<u8>, ContainerError> {
    let mut container_changes = changes.clone();
    let mut pos_patches = Vec::new();
    let mut pos_blob = Vec::new();

    if let Some(base) = base {
        let slot_uids = |d: &Document, m: kreate_scene::MeshId| -> Vec<Option<Uid>> {
            d.mesh_slots(m)
                .iter()
                .map(|s| s.and_then(|mat| d.material_uid(mat)))
                .collect()
        };
        for &(kind, uid) in &changes.changed {
            if kind != DatablockKind::Mesh {
                continue;
            }
            let (Some(base_id), Some(cur_id)) = (base.mesh_by_uid(uid), doc.mesh_by_uid(uid))
            else {
                continue;
            };
            if slot_uids(base, base_id) != slot_uids(doc, cur_id) {
                continue;
            }
            let (Some(base_mesh), Some(cur_mesh)) = (base.mesh(base_id), doc.mesh(cur_id)) else {
                continue;
            };
            if !base_mesh.only_positions_differ(cur_mesh) {
                continue;
            }
            let old_pos = base_mesh.positions();
            let new_pos = cur_mesh.positions();
            if old_pos.len() != new_pos.len() {
                continue;
            }
            let num_verts = cur_mesh.num_verts();
            let mut moved: Vec<(u32, [f32; 3])> = Vec::new();
            for (i, (a, b)) in old_pos.iter().zip(new_pos).enumerate() {
                if a.x.to_bits() != b.x.to_bits()
                    || a.y.to_bits() != b.y.to_bits()
                    || a.z.to_bits() != b.z.to_bits()
                {
                    moved.push((i as u32, b.to_array()));
                }
            }
            container_changes
                .changed
                .remove(&(DatablockKind::Mesh, uid));
            if moved.len() * 16 < new_pos.len() * 12 / 2 {
                pos_patches.push(PosPatchMeta {
                    uid: uid.raw(),
                    num_verts,
                    sparse: true,
                    count: moved.len() as u32,
                });
                for (idx, xyz) in moved {
                    pos_blob.extend_from_slice(&idx.to_le_bytes());
                    for c in xyz {
                        pos_blob.extend_from_slice(&c.to_le_bytes());
                    }
                }
            } else {
                pos_patches.push(PosPatchMeta {
                    uid: uid.raw(),
                    num_verts,
                    sparse: false,
                    count: num_verts,
                });
                for p in new_pos {
                    for c in p.to_array() {
                        pos_blob.extend_from_slice(&c.to_le_bytes());
                    }
                }
            }
        }
    }

    let header = Header {
        changed: changes
            .changed
            .iter()
            .map(|(k, u)| (k.id().to_string(), u.raw()))
            .collect(),
        removed: changes.removed.iter().map(|u| u.raw()).collect(),
        settings: changes.settings,
        pos_patches,
    };
    let header_bytes = serde_json::to_vec(&header).expect("a header always serialises");
    let body = container::write(&doc.patch(&container_changes), &Manifest::default(), &[])?;
    let has_pos = !pos_blob.is_empty();
    let mut raw = Vec::with_capacity(8 + header_bytes.len() + pos_blob.len() + body.len());
    raw.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
    raw.extend_from_slice(&header_bytes);
    if has_pos {
        raw.extend_from_slice(&(pos_blob.len() as u32).to_be_bytes());
        raw.extend_from_slice(&pos_blob);
    }
    raw.extend_from_slice(&body);
    Ok(wrap_compressed(&raw))
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
    decode_inner(None, bytes)
}

/// Decodes a patch against `existing` so any position-only mesh deltas in the
/// patch reuse `existing`'s connectivity, UV, and material-slot `Arc`s directly.
pub fn decode_for(existing: &Document, bytes: &[u8]) -> Result<(Document, Changes), SyncError> {
    decode_inner(Some(existing), bytes)
}

fn decode_inner(
    existing: Option<&Document>,
    bytes: &[u8],
) -> Result<(Document, Changes), SyncError> {
    let unpacked = unwrap_compressed(bytes)?;
    let raw = unpacked.as_ref();
    let len =
        u32::from_be_bytes(raw.get(..4).ok_or(SyncError::Short)?.try_into().unwrap()) as usize;
    let header: Header = serde_json::from_slice(raw.get(4..4 + len).ok_or(SyncError::Short)?)?;
    let mut cursor = 4 + len;

    let pos_slice = if !header.pos_patches.is_empty() {
        let plen = u32::from_be_bytes(
            raw.get(cursor..cursor + 4)
                .ok_or(SyncError::Short)?
                .try_into()
                .unwrap(),
        ) as usize;
        cursor += 4;
        let slice = raw.get(cursor..cursor + plen).ok_or(SyncError::Short)?;
        cursor += plen;
        slice
    } else {
        &[][..]
    };

    let body = raw.get(cursor..).ok_or(SyncError::Short)?;
    let mut doc = container::read_slice(body)?.document;

    if let Some(existing) = existing {
        let mut at = 0usize;
        for meta in &header.pos_patches {
            let record_bytes = if meta.sparse {
                (meta.count as usize)
                    .checked_mul(16)
                    .ok_or(SyncError::Short)?
            } else {
                (meta.count as usize)
                    .checked_mul(12)
                    .ok_or(SyncError::Short)?
            };
            let chunk = pos_slice
                .get(at..at + record_bytes)
                .ok_or(SyncError::Short)?;
            at += record_bytes;

            let uid = Uid::from_raw(meta.uid);
            let Some(existing_id) = existing.mesh_by_uid(uid) else {
                continue;
            };
            let Some(base_arc) = existing.mesh_arc(existing_id) else {
                continue;
            };
            if base_arc.num_verts() != meta.num_verts {
                continue;
            }
            let mut positions = base_arc.positions().to_vec();
            if meta.sparse {
                for rec in chunk.as_chunks::<16>().0 {
                    let idx = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as usize;
                    let x = f32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
                    let y = f32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
                    let z = f32::from_le_bytes([rec[12], rec[13], rec[14], rec[15]]);
                    if let Some(slot) = positions.get_mut(idx) {
                        *slot = glam::Vec3::new(x, y, z);
                    }
                }
            } else {
                for (slot, rec) in positions.iter_mut().zip(chunk.as_chunks::<12>().0) {
                    let x = f32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
                    let y = f32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
                    let z = f32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
                    *slot = glam::Vec3::new(x, y, z);
                }
            }
            let mut updated = (**base_arc).clone();
            updated.attributes_mut().insert(
                names::POSITION,
                Domain::Point,
                AttrData::F32x3(Arc::new(positions)),
            );

            // Preserve material slot references via stand-in materials when not already in `doc`.
            let mut patch_slots = Vec::new();
            for slot_mat in existing.mesh_slots(existing_id) {
                let mapped = slot_mat.and_then(|mid| {
                    let muid = existing.material_uid(mid)?;
                    doc.material_by_uid(muid).or_else(|| {
                        Some(doc.restore_material(kreate_scene::Material::default(), muid))
                    })
                });
                patch_slots.push(mapped);
            }
            let patch_mesh_id = match doc.mesh_by_uid(uid) {
                Some(id) => {
                    if let Some(m) = doc.mesh_mut(id) {
                        *m = updated;
                    }
                    id
                }
                None => doc.restore_mesh(updated, uid),
            };
            doc.set_mesh_slots(patch_mesh_id, patch_slots);
        }
    }

    let changes = Changes {
        changed: header
            .changed
            .iter()
            .filter_map(|(k, u)| Some((DatablockKind::from_id(k)?, Uid::from_raw(*u))))
            .collect(),
        removed: header.removed.iter().map(|u| Uid::from_raw(*u)).collect(),
        settings: header.settings,
    };
    Ok((doc, changes))
}

/// Reads a patch and takes it into `doc`. Returns the objects it touched.
pub fn apply(doc: &mut Document, bytes: &[u8]) -> Result<Vec<kreate_scene::ObjectId>, SyncError> {
    let (patch, changes) = decode_for(doc, bytes)?;
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

    #[test]
    fn sculpt_dab_travels_as_tiny_sparse_delta_and_keeps_receiver_topology_shared() {
        let mut a = Document::new();
        let sphere = a.add_mesh_object("Sculpt", kreate_mesh::primitives::uv_sphere(1.0, 64, 32));
        let mesh_a = a.object(sphere).unwrap().mesh().unwrap();

        let mut b = Document::new();
        apply(&mut b, &encode_all(&a).unwrap()).unwrap();
        let mesh_b = b.objects().next().unwrap().1.mesh().unwrap();
        let b_before = b.mesh_arc(mesh_b).unwrap().clone();

        let base = a.clone();
        {
            let m = a.mesh_mut(mesh_a).unwrap();
            let mut pos = m.positions().to_vec();
            pos[0].z += 0.25;
            pos[5].x -= 0.125;
            m.attributes_mut().insert(
                names::POSITION,
                Domain::Point,
                AttrData::F32x3(Arc::new(pos)),
            );
        }
        let changes = a.changes_since(&base);
        let full_bytes = encode(&a, &changes).unwrap();
        let delta_bytes = encode_since(&base, &a, &changes).unwrap();
        assert!(
            delta_bytes.len() * 5 < full_bytes.len(),
            "sparse position delta ({} B) should be far smaller than full mesh patch ({} B)",
            delta_bytes.len(),
            full_bytes.len()
        );

        apply(&mut b, &delta_bytes).unwrap();
        let b_after = b.mesh_arc(mesh_b).unwrap();
        assert_eq!(
            b_after.positions()[0],
            a.mesh(mesh_a).unwrap().positions()[0]
        );
        assert_eq!(
            b_after.positions()[5],
            a.mesh(mesh_a).unwrap().positions()[5]
        );
        assert!(
            b_before.only_positions_differ(b_after),
            "receiver MeshData must preserve topology & UV Arc pointers so GPU repositions in place"
        );
    }
}
