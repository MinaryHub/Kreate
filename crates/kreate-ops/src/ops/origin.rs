//! The 3D cursor, and where an object's origin sits relative to its geometry.
//!
//! # The cursor is the document's
//!
//! Blender keeps the 3D cursor in the scene, and so does this: it is saved
//! with the file, it is where the next object appears, and a rotation can turn
//! about it. Setting it is an operator, so a script that places things at the
//! cursor replays with the cursor where it was.
//!
//! # Moving an origin moves nothing on screen
//!
//! An object's origin is where its transform puts the mesh's zero. Setting it
//! somewhere else is two changes that cancel: the vertices shift one way in
//! the mesh, the object shifts the other way in its parent. Applying a
//! transform is the same bargain the other way round — the transform goes
//! into the vertices and comes out of the object. In both, what hangs from
//! the object (its children) is held where it was, and so is every other
//! object using the same mesh.

use kreate_math::{DAffine3, DQuat, DTransform, DVec3};
use kreate_scene::document::ObjectId;
use kreate_scene::{DocMut, Document, MeshId};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<CursorSet>();
    registry.register::<OriginSet>();
    registry.register::<TransformApply>();
}

// --- the cursor -----------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct CursorSetParams {
    /// Where the cursor goes, in the world.
    #[param(unit = "length")]
    pub location: [f64; 3],
    /// Turn it too: the work plane. Off, its turn is left as it is.
    pub orient: bool,
    /// How it is turned, as Euler XYZ, when `orient` is on.
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
}

/// Puts the 3D cursor somewhere — `Shift+Right-click`, and the snap menu's
/// "Cursor to Selected" and "Cursor to World Origin".
pub struct CursorSet;

impl Operator for CursorSet {
    type Params = CursorSetParams;
    fn id() -> &'static str {
        "scene.cursor_set"
    }
    fn label() -> &'static str {
        "Set 3D Cursor"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let at = DVec3::from_array(params.location);
        let turn = params
            .orient
            .then(|| kreate_scene::animation::from_euler(DVec3::from_array(params.rotation)));
        let d = doc.document();
        let same_turn = turn.is_none_or(|q| q.abs_diff_eq(d.cursor_rotation, 1e-12));
        if !at.is_finite() || turn.is_some_and(|q| !q.is_finite()) || (at == d.cursor && same_turn)
        {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        d.cursor = at;
        if let Some(q) = turn {
            d.cursor_rotation = q.normalize();
        }
        Ok(OpResult::Finished)
    }
}

// --- shared ---------------------------------------------------------------------

/// Only in object mode: in edit mode the origin is not what the user is
/// working on, and Blender offers neither there.
fn objects_mode(ctx: &Context, doc: &Document) -> bool {
    ctx.mode.acts_on() == crate::mode::ActsOn::Objects && !doc.selection.is_empty()
}

/// The selected objects that are meshes, one per mesh: the first selected user
/// of each, which is the one the new origin is measured against.
fn selected_meshes(doc: &Document) -> Vec<(ObjectId, MeshId)> {
    let mut out: Vec<(ObjectId, MeshId)> = Vec::new();
    for id in doc
        .selection
        .active
        .into_iter()
        .chain(doc.selection.objects.iter().copied())
    {
        let Some(mesh) = doc.object(id).and_then(|o| o.mesh()) else {
            continue;
        };
        if !out.iter().any(|(_, m)| *m == mesh) {
            out.push((id, mesh));
        }
    }
    out
}

/// Replaces `mesh`'s vertices `v` with `change(v)`, and every object using it
/// with a transform that puts the result where the old vertices were, whose
/// children stay put too.
///
/// `keep` is what the object's transform becomes, given the old one: for an
/// origin move, the same transform shifted; for an apply, the transform with
/// the applied parts taken out. The vertex change is then whatever makes the
/// two agree, `keep⁻¹ · old`, worked out per object — and it has to be the
/// same for every user of the mesh, which the callers see to.
fn rebase(
    doc: &mut Document,
    mesh: MeshId,
    change: DAffine3,
    keep: impl Fn(&DTransform) -> DTransform,
) {
    if let Some(m) = doc.mesh_mut(mesh) {
        for p in m.positions_mut() {
            *p = change.transform_point3(p.as_dvec3()).as_vec3();
        }
    }
    let users: Vec<ObjectId> = doc
        .objects()
        .filter(|(_, o)| o.mesh() == Some(mesh))
        .map(|(id, _)| id)
        .collect();
    for id in users {
        let Some(old) = doc.object(id).map(|o| o.transform) else {
            continue;
        };
        let new = keep(&old);
        // The children hang from the object's frame, which just moved by
        // `new⁻¹ · old` — the same `change` — so their link takes it back.
        let children: Vec<ObjectId> = doc
            .objects()
            .filter(|(_, o)| o.parent == Some(id))
            .map(|(c, _)| c)
            .collect();
        for c in children {
            if let Some(o) = doc.object_mut(c) {
                o.parent_inverse = change * o.parent_inverse;
            }
        }
        if let Some(o) = doc.object_mut(id) {
            o.transform = new;
        }
    }
}

// --- origin ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct OriginSetParams {
    /// Where the origin goes: to the middle of the geometry, to the 3D cursor,
    /// or the geometry to where the origin is.
    #[param(variants("origin_to_geometry", "origin_to_cursor", "geometry_to_origin"))]
    pub kind: u32,
}

/// Blender's Set Origin: moves the selected meshes' origins without moving
/// what is drawn, or moves the geometry onto the origin.
pub struct OriginSet;

impl Operator for OriginSet {
    type Params = OriginSetParams;
    fn id() -> &'static str {
        "object.origin_set"
    }
    fn label() -> &'static str {
        "Set Origin"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        objects_mode(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = selected_meshes(doc.document());
        let mut changed = false;
        for (object, mesh) in targets {
            let d = doc.document();
            let Some(data) = d.mesh(mesh) else { continue };
            if data.num_verts() == 0 {
                continue;
            }
            // Where the new origin is, in the mesh's own space. The middle is
            // the median of the vertices, Blender's default for this.
            let middle = data
                .positions()
                .iter()
                .fold(DVec3::ZERO, |s, p| s + p.as_dvec3())
                / data.num_verts() as f64;
            let target = match params.kind {
                1 => {
                    let to_local = d.world_transform(object).inverse();
                    if !to_local.is_finite() {
                        continue;
                    }
                    to_local.transform_point3(d.cursor)
                }
                _ => middle,
            };
            if target.length() < 1e-12 {
                continue;
            }
            let change = DAffine3::from_translation(-target);
            let d = doc.document_mut();
            if params.kind == 2 {
                // The geometry comes to the origin: the objects stay, and so
                // the drawing moves. Nothing to compensate.
                if let Some(m) = d.mesh_mut(mesh) {
                    for p in m.positions_mut() {
                        *p = (p.as_dvec3() - target).as_vec3();
                    }
                }
            } else {
                rebase(d, mesh, change, |t| {
                    // The object moves by where the old zero now is, as its
                    // own rotation and scale carry it.
                    let mut moved = *t;
                    moved.translation += t.rotation * (t.scale * target);
                    moved
                });
            }
            changed = true;
        }
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

// --- apply ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct TransformApplyParams {
    pub location: bool,
    pub rotation: bool,
    pub scale: bool,
}

impl Default for TransformApplyParams {
    fn default() -> Self {
        Self {
            location: false,
            rotation: true,
            scale: true,
        }
    }
}

/// Blender's `Ctrl+A`: bakes the chosen parts of the selected meshes'
/// transforms into their vertices and resets them, leaving everything where
/// it was on screen.
///
/// Refused for a mesh another object also uses: the other object's transform
/// would have to take the change back, and that is a different transform for
/// each of them — Blender refuses the same case.
pub struct TransformApply;

impl Operator for TransformApply {
    type Params = TransformApplyParams;
    fn id() -> &'static str {
        "object.transform_apply"
    }
    fn label() -> &'static str {
        "Apply Transform"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        objects_mode(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = selected_meshes(doc.document());
        let mut changed = false;
        let mut shared = 0;
        for (object, mesh) in targets {
            let d = doc.document();
            if d.mesh_users(mesh) > 1 {
                shared += 1;
                continue;
            }
            let Some(old) = d.object(object).map(|o| o.transform) else {
                continue;
            };
            let keep = DTransform {
                translation: if params.location {
                    DVec3::ZERO
                } else {
                    old.translation
                },
                rotation: if params.rotation {
                    DQuat::IDENTITY
                } else {
                    old.rotation
                },
                scale: if params.scale { DVec3::ONE } else { old.scale },
            };
            if keep == old {
                continue;
            }
            let inverse = keep.to_affine().inverse();
            if !inverse.is_finite() {
                continue;
            }
            // What the vertices take on so that `keep` draws them where `old`
            // did. Rotation kept with a scale applied, or the other way round
            // with a scale that is not the same on every axis, bakes a shear
            // into the mesh — the only way the picture can stay the same.
            let change = inverse * old.to_affine();
            rebase(doc.document_mut(), mesh, change, |_| keep);
            changed = true;
        }
        if !changed && shared > 0 {
            return Err(OpError::Failed(
                "a mesh used by more than one object cannot have its transform applied".into(),
            ));
        }
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::OperatorRegistry;
    use crate::{PropertyMap, PropertyValue};
    use kreate_scene::document::{Object, ObjectData};

    fn run(
        doc: &mut Document,
        op: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> Result<OpResult, OpError> {
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        let ctx = Context::default();
        let mut h = History::new();
        h.dispatch(&r, doc, &ctx, op, &p).map(|_| {
            if h.entries().is_empty() {
                OpResult::Cancelled
            } else {
                OpResult::Finished
            }
        })
    }

    /// A 2-unit cube off-centre in its own mesh (its middle at (1, 0, 0)),
    /// placed at (0, 0, 3), turned and stretched, with a child hanging off it.
    fn scene() -> (Document, ObjectId, ObjectId) {
        let mut doc = Document::new();
        let mut mesh = kreate_mesh::primitives::cube(2.0);
        for p in mesh.positions_mut() {
            p.x += 1.0;
        }
        let mesh = doc.add_mesh(mesh);
        let mut cube = Object::new("Cube", ObjectData::Mesh(mesh));
        cube.transform = DTransform {
            translation: DVec3::new(0.0, 0.0, 3.0),
            rotation: DQuat::from_rotation_z(0.7),
            scale: DVec3::new(2.0, 1.0, 1.5),
        };
        let cube = doc.add_object(cube);
        let mut child = Object::new("Child", ObjectData::Empty);
        child.transform.translation = DVec3::new(5.0, 1.0, 0.0);
        let child = doc.add_object(child);
        doc.set_parent(child, Some(cube)).unwrap();
        doc.selection.add(cube);
        (doc, cube, child)
    }

    /// Every vertex of the cube, in the world.
    fn drawn(doc: &Document, cube: ObjectId) -> Vec<DVec3> {
        let world = doc.world_transform(cube);
        let mesh = doc.mesh(doc.object(cube).unwrap().mesh().unwrap()).unwrap();
        mesh.positions()
            .iter()
            .map(|p| world.transform_point3(p.as_dvec3()))
            .collect()
    }

    fn same(a: &[DVec3], b: &[DVec3]) -> bool {
        a.iter().zip(b).all(|(p, q)| (*p - *q).length() < 1e-5)
    }

    #[test]
    fn the_cursor_is_set_and_setting_it_where_it_is_does_nothing() {
        let mut doc = Document::new();
        let at = [1.0, -2.0, 0.5];
        assert_eq!(
            run(
                &mut doc,
                "scene.cursor_set",
                &[("location", PropertyValue::Vec3(at))]
            )
            .unwrap(),
            OpResult::Finished
        );
        assert_eq!(doc.cursor, DVec3::from_array(at));
        assert_eq!(
            run(
                &mut doc,
                "scene.cursor_set",
                &[("location", PropertyValue::Vec3(at))]
            )
            .unwrap(),
            OpResult::Cancelled
        );
    }

    #[test]
    fn origin_to_geometry_moves_the_origin_and_nothing_on_screen() {
        let (mut doc, cube, child) = scene();
        let before = drawn(&doc, cube);
        let child_before = doc.world_transform(child).translation;
        run(
            &mut doc,
            "object.origin_set",
            &[("kind", PropertyValue::Enum(0))],
        )
        .unwrap();
        assert!(same(&before, &drawn(&doc, cube)), "the cube moved");
        assert!(
            (doc.world_transform(child).translation - child_before).length() < 1e-6,
            "the child moved"
        );
        // The origin is now the middle of what is drawn.
        let middle = before.iter().fold(DVec3::ZERO, |s, p| s + *p) / before.len() as f64;
        assert!((doc.object(cube).unwrap().transform.translation - middle).length() < 1e-6);
    }

    #[test]
    fn origin_to_cursor_puts_the_origin_on_the_cursor() {
        let (mut doc, cube, _) = scene();
        doc.cursor = DVec3::new(-1.0, 2.0, 0.5);
        let before = drawn(&doc, cube);
        run(
            &mut doc,
            "object.origin_set",
            &[("kind", PropertyValue::Enum(1))],
        )
        .unwrap();
        assert!(same(&before, &drawn(&doc, cube)));
        assert!((doc.world_transform(cube).translation - doc.cursor).length() < 1e-6);
    }

    #[test]
    fn geometry_to_origin_moves_the_geometry_instead() {
        let (mut doc, cube, _) = scene();
        let at = doc.object(cube).unwrap().transform.translation;
        run(
            &mut doc,
            "object.origin_set",
            &[("kind", PropertyValue::Enum(2))],
        )
        .unwrap();
        let now = drawn(&doc, cube);
        let middle = now.iter().fold(DVec3::ZERO, |s, p| s + *p) / now.len() as f64;
        assert!(
            (middle - at).length() < 1e-6,
            "the geometry did not come to the origin"
        );
    }

    #[test]
    fn applying_rotation_and_scale_bakes_them_in_and_keeps_the_picture() {
        let (mut doc, cube, child) = scene();
        let before = drawn(&doc, cube);
        let child_before = doc.world_transform(child);
        run(&mut doc, "object.transform_apply", &[]).unwrap();
        let t = doc.object(cube).unwrap().transform;
        assert_eq!((t.rotation, t.scale), (DQuat::IDENTITY, DVec3::ONE));
        assert_eq!(
            t.translation,
            DVec3::new(0.0, 0.0, 3.0),
            "location was not asked for"
        );
        assert!(same(&before, &drawn(&doc, cube)));
        assert!(
            doc.world_transform(child).abs_diff_eq(child_before, 1e-6),
            "the child moved"
        );
        // All of it, location too.
        run(
            &mut doc,
            "object.transform_apply",
            &[("location", PropertyValue::Bool(true))],
        )
        .unwrap();
        assert_eq!(doc.object(cube).unwrap().transform, DTransform::IDENTITY);
        assert!(same(&before, &drawn(&doc, cube)));
    }

    #[test]
    fn a_shared_mesh_cannot_have_its_transform_applied() {
        let (mut doc, cube, _) = scene();
        let mesh = doc.object(cube).unwrap().mesh().unwrap();
        doc.add_object(Object::new("Twin", ObjectData::Mesh(mesh)));
        assert!(run(&mut doc, "object.transform_apply", &[]).is_err());
        // Moving the origin is fine: every user is compensated.
        let before = drawn(&doc, cube);
        run(
            &mut doc,
            "object.origin_set",
            &[("kind", PropertyValue::Enum(0))],
        )
        .unwrap();
        assert!(same(&before, &drawn(&doc, cube)));
    }
}
