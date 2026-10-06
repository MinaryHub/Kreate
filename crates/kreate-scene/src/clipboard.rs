//! Taking objects out of a document and putting them into another.
//!
//! The two halves of copy and paste. **A fragment is a whole document** — the
//! chosen objects with every datablock they reach: the mesh and its material
//! slots, those materials' images, the light, the camera. That is what lets the
//! fragment go through the same writer the file does, and what lets a paste be
//! replayed from its own bytes long after the objects it came from are gone.
//!
//! Every datablock is *re-added* on the way in, through the same `add_*` the
//! rest of the application uses, so each arrives with a fresh identity. A
//! pasted object is a new object: pasting twice gives two, and neither is the
//! one that was copied.

use std::collections::HashMap;

use crate::document::{ActionId, Document, ImageId, MaterialId, Object, ObjectData, ObjectId};
use kreate_math::DTransform;

impl Document {
    /// A document holding copies of `ids` and everything they reach.
    ///
    /// A parent that is also being taken keeps the relationship; one that is
    /// not has its world placement baked into the copy, so a child copied
    /// without its parent lands where it was rather than where its local
    /// transform alone would put it. Ids that name nothing are skipped.
    pub fn extract_objects(&self, ids: &[ObjectId]) -> Document {
        let mut out = Document::new();
        let taken: Vec<ObjectId> = ids
            .iter()
            .copied()
            .filter(|id| self.object(*id).is_some())
            .collect();
        let mut new_ids: HashMap<ObjectId, ObjectId> = HashMap::new();
        let mut materials: HashMap<MaterialId, MaterialId> = HashMap::new();
        let mut images: HashMap<ImageId, ImageId> = HashMap::new();
        let mut actions: HashMap<ActionId, ActionId> = HashMap::new();

        for id in &taken {
            let Some(object) = self.object(*id) else {
                continue;
            };
            let mut copy = object.clone();
            copy.data = match &object.data {
                ObjectData::Mesh(mesh) => {
                    let Some(data) = self.mesh(*mesh) else {
                        continue;
                    };
                    let new_mesh = out.add_mesh(data.clone());
                    let slots: Vec<Option<MaterialId>> = self
                        .mesh_slots(*mesh)
                        .iter()
                        .map(|slot| {
                            slot.map(|m| {
                                self.copy_material(
                                    m,
                                    &mut out,
                                    &mut materials,
                                    &mut images,
                                    &mut actions,
                                )
                            })
                        })
                        .collect();
                    out.set_mesh_slots(new_mesh, slots);
                    ObjectData::Mesh(new_mesh)
                }
                ObjectData::Light(light) => match self.light(*light) {
                    Some(l) => {
                        let mut l = l.clone();
                        l.action = l
                            .action
                            .and_then(|a| self.copy_action(a, &mut out, &mut actions));
                        ObjectData::Light(out.add_light(l))
                    }
                    None => continue,
                },
                ObjectData::Camera(camera) => match self.camera(*camera) {
                    Some(c) => ObjectData::Camera(out.add_camera(c.clone())),
                    None => continue,
                },
                ObjectData::Armature(armature) => match self.armature(*armature) {
                    Some(a) => ObjectData::Armature(out.add_armature(a.clone())),
                    None => continue,
                },
                ObjectData::Empty => ObjectData::Empty,
            };
            copy.material_overrides = object
                .material_overrides
                .iter()
                .map(|slot| {
                    slot.map(|m| {
                        self.copy_material(m, &mut out, &mut materials, &mut images, &mut actions)
                    })
                })
                .collect();
            // The action comes along, once however many copies play it, so
            // two pasted objects sharing a walk cycle still share one.
            copy.action = object
                .action
                .and_then(|a| self.copy_action(a, &mut out, &mut actions));
            copy.nla = object
                .nla
                .iter()
                .filter_map(|s| {
                    let action = self.copy_action(s.action, &mut out, &mut actions)?;
                    Some(crate::nla::NlaStrip { action, ..s.clone() })
                })
                .collect();
            // Parents are settled below, once every copy exists. Until then
            // the copy stands alone at its world placement.
            copy.parent = None;
            copy.parent_inverse = kreate_math::DAffine3::IDENTITY;
            copy.transform = DTransform::from_affine(self.world_transform(*id));
            let new_id = out.add_object(copy);
            new_ids.insert(*id, new_id);
        }

        // A parent that came along keeps its children: the child's local
        // transform and parent inverse are restored from the source, since the
        // pair is the same pair.
        for id in &taken {
            let Some(object) = self.object(*id) else {
                continue;
            };
            let (Some(parent), Some(new_id)) = (object.parent, new_ids.get(id)) else {
                continue;
            };
            let Some(new_parent) = new_ids.get(&parent) else {
                continue;
            };
            if let Some(copy) = out.object_mut(*new_id) {
                copy.transform = object.transform;
                copy.parent_inverse = object.parent_inverse;
                copy.parent = Some(*new_parent);
            }
        }
        out
    }

    /// Adds every object of `other` to this document, and returns the new
    /// objects in `other`'s order.
    ///
    /// Each datablock is re-added and so gets a fresh identity; parents inside
    /// `other` are kept as parents inside the result. Names take the `.001`
    /// suffix a duplicate takes, because that is what they are to this
    /// document.
    pub fn merge_from(&mut self, other: &Document) -> Vec<ObjectId> {
        let mut materials: HashMap<MaterialId, MaterialId> = HashMap::new();
        let mut images: HashMap<ImageId, ImageId> = HashMap::new();
        let mut new_ids: HashMap<ObjectId, ObjectId> = HashMap::new();
        let mut actions: HashMap<ActionId, ActionId> = HashMap::new();
        let mut added = Vec::new();

        let source_ids: Vec<ObjectId> = other.objects().map(|(id, _)| id).collect();
        for id in &source_ids {
            let Some(object) = other.object(*id) else {
                continue;
            };
            let mut copy = object.clone();
            copy.name = format!("{}.001", object.name);
            copy.data = match &object.data {
                ObjectData::Mesh(mesh) => {
                    let Some(data) = other.mesh(*mesh) else {
                        continue;
                    };
                    let new_mesh = self.add_mesh(data.clone());
                    let slots: Vec<Option<MaterialId>> = other
                        .mesh_slots(*mesh)
                        .iter()
                        .map(|slot| {
                            slot.map(|m| {
                                other.copy_material(
                                    m,
                                    self,
                                    &mut materials,
                                    &mut images,
                                    &mut actions,
                                )
                            })
                        })
                        .collect();
                    self.set_mesh_slots(new_mesh, slots);
                    ObjectData::Mesh(new_mesh)
                }
                ObjectData::Light(light) => match other.light(*light) {
                    Some(l) => {
                        let mut l = l.clone();
                        l.action = l
                            .action
                            .and_then(|a| other.copy_action(a, self, &mut actions));
                        ObjectData::Light(self.add_light(l))
                    }
                    None => continue,
                },
                ObjectData::Camera(camera) => match other.camera(*camera) {
                    Some(c) => ObjectData::Camera(self.add_camera(c.clone())),
                    None => continue,
                },
                ObjectData::Armature(armature) => match other.armature(*armature) {
                    Some(a) => ObjectData::Armature(self.add_armature(a.clone())),
                    None => continue,
                },
                ObjectData::Empty => ObjectData::Empty,
            };
            copy.material_overrides = object
                .material_overrides
                .iter()
                .map(|slot| {
                    slot.map(|m| {
                        other.copy_material(m, self, &mut materials, &mut images, &mut actions)
                    })
                })
                .collect();
            copy.action = object
                .action
                .and_then(|a| other.copy_action(a, self, &mut actions));
            copy.nla = object
                .nla
                .iter()
                .filter_map(|s| {
                    let action = other.copy_action(s.action, self, &mut actions)?;
                    Some(crate::nla::NlaStrip { action, ..s.clone() })
                })
                .collect();
            copy.parent = None;
            let new_id = self.add_object(copy);
            new_ids.insert(*id, new_id);
            added.push(new_id);
        }
        for id in &source_ids {
            let Some(object) = other.object(*id) else {
                continue;
            };
            let (Some(parent), Some(new_id)) = (object.parent, new_ids.get(id)) else {
                continue;
            };
            if let Some(new_parent) = new_ids.get(&parent) {
                if let Some(copy) = self.object_mut(*new_id) {
                    copy.parent = Some(*new_parent);
                }
            }
        }
        added
    }

    /// Copies one of this document's materials into `into`, once.
    ///
    /// Through the maps, so a material shared by two slots is one material in
    /// the result — the same sharing the source had — and so is its image.
    fn copy_material(
        &self,
        id: MaterialId,
        into: &mut Document,
        materials: &mut HashMap<MaterialId, MaterialId>,
        images: &mut HashMap<ImageId, ImageId>,
        actions: &mut HashMap<ActionId, ActionId>,
    ) -> MaterialId {
        if let Some(done) = materials.get(&id) {
            return *done;
        }
        let mut material = self.material(id).cloned().unwrap_or_default();
        // Every map, through the one image table: a file used as both the
        // colour and the roughness is one image in the result, as it was here.
        let mut carry = |image: ImageId| -> Option<ImageId> {
            if let Some(done) = images.get(&image) {
                return Some(*done);
            }
            let copied = into.add_image(self.image(image)?.clone());
            images.insert(image, copied);
            Some(copied)
        };
        for map in crate::document::MaterialMap::ALL {
            let slot = material.map_mut(map);
            *slot = slot.and_then(&mut carry);
        }
        // And the graph's own image nodes, through the same table.
        if let Some(graph) = &mut material.graph {
            for node in &mut graph.nodes {
                if let crate::shader::NodeKind::Image { image } = &mut node.kind {
                    *image = image.and_then(&mut carry);
                }
            }
        }
        // Its action, through the same table, so a material and the lamp
        // beside it sharing one blink still share it in the paste.
        material.action = material
            .action
            .and_then(|a| self.copy_action(a, into, actions));
        let new_id = into.add_material(material);
        materials.insert(id, new_id);
        new_id
    }

    /// Copies one of this document's actions into `into`, once.
    fn copy_action(
        &self,
        id: ActionId,
        into: &mut Document,
        actions: &mut HashMap<ActionId, ActionId>,
    ) -> Option<ActionId> {
        if let Some(done) = actions.get(&id) {
            return Some(*done);
        }
        let made = into.add_action(self.action(id)?.clone());
        actions.insert(id, made);
        Some(made)
    }
}

// An `Object` clone keeps the source's identity in its `uid` field; `add_object`
// overwrites it, which is what makes a paste a new object. Stated here so the
// reader of `extract_objects` does not have to go and check.
#[allow(dead_code)]
fn _uid_is_reissued_on_add(_: &Object) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Image, Light, LightKind, Material, PixelFormat};
    use kreate_math::DVec3;

    fn cube(doc: &mut Document, name: &str) -> ObjectId {
        doc.add_mesh_object(name, kreate_mesh::primitives::cube(2.0))
    }

    #[test]
    fn a_fragment_holds_the_objects_and_what_they_reach() {
        let mut doc = Document::new();
        let a = cube(&mut doc, "A");
        let _b = cube(&mut doc, "B");
        let image = doc.add_image(Image::new("Bricks", 4, 4, PixelFormat::Rgba8));
        let material = doc.add_material(Material {
            name: "Brick".into(),
            base_color_image: Some(image),
            ..Default::default()
        });
        let mesh = doc.object(a).unwrap().mesh().unwrap();
        doc.set_mesh_slots(mesh, vec![Some(material)]);

        let fragment = doc.extract_objects(&[a]);
        assert_eq!(fragment.object_count(), 1, "only A was asked for");
        assert_eq!(fragment.mesh_count(), 1);
        assert_eq!(fragment.material_count(), 1, "A's material came along");
        assert_eq!(fragment.image_count(), 1, "and the material's image");
        let (_, copied) = fragment.objects().next().unwrap();
        assert_eq!(copied.name, "A");
        let copied_mesh = copied.mesh().unwrap();
        let slot = fragment.mesh_slots(copied_mesh)[0].expect("the slot is filled");
        assert_eq!(fragment.material(slot).unwrap().name, "Brick");
        assert!(fragment.material(slot).unwrap().base_color_image.is_some());
    }

    #[test]
    fn pasting_makes_new_objects_and_leaves_the_originals_alone() {
        let mut doc = Document::new();
        let a = cube(&mut doc, "A");
        let fragment = doc.extract_objects(&[a]);
        let before = doc.object_count();
        let pasted = doc.merge_from(&fragment);
        assert_eq!(pasted.len(), 1);
        assert_eq!(doc.object_count(), before + 1);
        assert_ne!(pasted[0], a);
        assert_ne!(
            doc.object_uid(pasted[0]),
            doc.object_uid(a),
            "a paste is a new identity"
        );
        assert_eq!(doc.object(pasted[0]).unwrap().name, "A.001");
        assert_eq!(
            doc.object(a).unwrap().name,
            "A",
            "the original kept its name"
        );
        // And its own mesh: editing the paste must not edit the original.
        assert_ne!(
            doc.object(pasted[0]).unwrap().mesh(),
            doc.object(a).unwrap().mesh()
        );
    }

    #[test]
    fn a_shared_material_is_shared_in_the_paste_too() {
        let mut doc = Document::new();
        let a = cube(&mut doc, "A");
        let b = cube(&mut doc, "B");
        let material = doc.add_material(Material::default());
        for id in [a, b] {
            let mesh = doc.object(id).unwrap().mesh().unwrap();
            doc.set_mesh_slots(mesh, vec![Some(material)]);
        }
        let fragment = doc.extract_objects(&[a, b]);
        assert_eq!(
            fragment.material_count(),
            1,
            "one material, not one per object"
        );
        let pasted = doc.merge_from(&fragment);
        let m0 = doc.material_id_at(pasted[0], 0);
        let m1 = doc.material_id_at(pasted[1], 0);
        assert_eq!(m0, m1, "the two pastes share as the two originals did");
        assert_ne!(m0, Some(material), "and it is not the original's");
    }

    #[test]
    fn a_child_copied_with_its_parent_keeps_it_and_one_copied_alone_keeps_its_place() {
        let mut doc = Document::new();
        let parent = cube(&mut doc, "Parent");
        doc.object_mut(parent).unwrap().transform.translation = DVec3::new(5.0, 0.0, 0.0);
        let child = cube(&mut doc, "Child");
        doc.object_mut(child).unwrap().transform.translation = DVec3::new(1.0, 0.0, 0.0);
        doc.set_parent(child, Some(parent)).unwrap();
        let world_before = doc.world_transform(child).translation;

        // Both: the pair survives.
        let both = doc.extract_objects(&[parent, child]);
        let pasted = doc.merge_from(&both);
        let new_child = pasted[1];
        assert_eq!(doc.object(new_child).unwrap().parent, Some(pasted[0]));
        assert!((doc.world_transform(new_child).translation - world_before).length() < 1e-9);

        // Alone: no parent, but where it was.
        let alone = doc.extract_objects(&[child]);
        let pasted = doc.merge_from(&alone);
        assert_eq!(doc.object(pasted[0]).unwrap().parent, None);
        assert!((doc.world_transform(pasted[0]).translation - world_before).length() < 1e-9);
    }

    #[test]
    fn a_lamp_and_a_camera_paste_with_their_datablocks() {
        let mut doc = Document::new();
        let light = doc.add_light(Light::of_kind("Bulb", LightKind::BULB));
        let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(light)));
        let cam = doc.add_camera(crate::Camera::new("Cam"));
        let camera = doc.add_object(Object::new("Camera", ObjectData::Camera(cam)));
        let fragment = doc.extract_objects(&[lamp, camera]);
        let pasted = doc.merge_from(&fragment);
        assert!(matches!(doc.object(pasted[0]).unwrap().data, ObjectData::Light(l) if l != light));
        assert!(matches!(doc.object(pasted[1]).unwrap().data, ObjectData::Camera(c) if c != cam));
    }

    #[test]
    fn ids_that_name_nothing_are_skipped() {
        // A stale id: one that named an object this document has since
        // deleted. (An id from a *different* document is not a test of this —
        // arena ids are slot and generation, and two fresh documents hand out
        // the same first id.)
        let mut doc = Document::new();
        let a = cube(&mut doc, "A");
        let gone = cube(&mut doc, "Gone");
        doc.remove_object(gone);
        assert!(doc.object(gone).is_none(), "the fixture's id must be stale");
        let fragment = doc.extract_objects(&[gone, a]);
        assert_eq!(fragment.object_count(), 1);
    }
}
