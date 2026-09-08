//! The decisions a GPU-previewed drag makes, away from the app that makes them.
//!
//! `App` has no test harness, so anything in `main.rs` is unguarded by
//! construction. What is here is everything about the preview that can be
//! decided from data: which gesture the shader should run, which weight each
//! render vertex carries, and — the part that is easy to get wrong — when the
//! preview has to be abandoned because the buffers it was written against are
//! gone.
//!
//! # What the preview replaces
//!
//! Today a drag frame writes the document, which produces a new `MeshData`,
//! which `gpu_sync` recognises as positions-only and repositions — and
//! `update_positions` calls `RenderMeshCpu::extract` to find out what to write.
//! That extraction is 83 ms at two hundred thousand vertices against a budget
//! of one, and almost all of it is work a move does not change.
//!
//! So while a preview is live the document is not written at all. The operator
//! runs once, when the pointer comes up, and that single run is what the undo
//! stack, the adjust panel and a script replay see. Which is what §1.7 asked
//! for in the first place: a modal transaction is uncommitted until it is
//! confirmed.
//!
//! # Why the gesture is rebuilt from params rather than from the modal state
//!
//! [`gesture_for`] takes the same `(op_id, PropertyMap)` pair that would have
//! been handed to `execute`. Not the `modal::Gesture` — that has axis
//! constraints, typed input, snapping and precision mode in it, and every one
//! of those already resolves into the params. Reading the params means the
//! preview shows what the operator would do, including the things the operator
//! learns about only through them, and there is no second interpretation of
//! `XX` or of a typed `2` to keep in step.

use glam::Vec3;
use kreate_reflect::PropertyMap;
use kreate_render::Gesture;

use crate::gpu_sync::SyncPlan;

/// The gesture the shader should run for an operator call, if it is one the
/// preview can express.
///
/// `None` for anything else, and the caller then drives the drag the old way.
/// That is a real fallback rather than a formality: extrude, inset, bevel and
/// the rest are modal too, and they change topology, which no amount of
/// deforming a frozen buffer will do.
pub fn gesture_for(op_id: &str, params: &PropertyMap, pivot: Vec3) -> Option<Gesture> {
    match op_id {
        "mesh.translate" => {
            let [x, y, z] = params.get("translation")?.as_vec3()?;
            // The pivot is not used by a translation, and is deliberately not
            // passed on: a gesture that carried one would invite the reader to
            // wonder whether a move happens about something.
            Some(Gesture::moved(Vec3::new(x as f32, y as f32, z as f32)))
        }
        "mesh.rotate" => {
            let angle = params.get("angle")?.as_float()? as f32;
            let axis = match params.get("axis")?.as_int()? {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            };
            Some(Gesture::turned(
                glam::Quat::from_axis_angle(axis, angle),
                pivot,
            ))
        }
        "mesh.scale" => {
            let [x, y, z] = params.get("scale")?.as_vec3()?;
            Some(Gesture::scaled(
                Vec3::new(x as f32, y as f32, z as f32),
                pivot,
            ))
        }
        _ => None,
    }
}

/// Whether an operator is one [`gesture_for`] can express, without needing the
/// params to find out.
///
/// What `begin_gesture` asks before paying for a `Deform`.
pub fn is_previewable(op_id: &str) -> bool {
    matches!(op_id, "mesh.translate" | "mesh.rotate" | "mesh.scale")
}

/// Spreads a weight-per-mesh-vertex falloff over the render vertices.
///
/// The two are not the same list. A mesh vertex splits into one render vertex
/// per distinct corner normal, so a cube's corner is three of them — and the
/// weight is a property of the mesh vertex while the buffer the shader writes
/// is indexed by render vertex. Every copy of a split vertex gets the same
/// weight, which is what keeps the copies together: they are the same point,
/// and a drag that moved them apart would tear the mesh open along its own
/// shading seams.
///
/// `render_to_vert` is [`kreate_render::RenderMeshCpu::render_to_vert`] and
/// `mesh_verts` the document mesh's vertex count. The table rather than the
/// whole render mesh, because a live preview holds whatever this asks for until
/// the pointer comes up — and a `RenderMeshCpu` is about a hundred bytes a
/// vertex, of which this needs four.
pub fn spread_weights(
    render_to_vert: &[u32],
    mesh_verts: usize,
    indices: &[u32],
    weights: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(indices.len(), weights.len());
    // Densified per mesh vertex first. A render vertex's mesh vertex is not
    // ascending in render-vertex order — `extract` walks vertices but emits
    // several render vertices for some of them — so a cursor over `indices`
    // cannot follow it directly.
    let mut dense = vec![0.0f32; mesh_verts];
    for (v, w) in indices.iter().zip(weights) {
        if let Some(slot) = dense.get_mut(*v as usize) {
            *slot = *w;
        }
    }
    render_to_vert
        .iter()
        .map(|v| dense.get(*v as usize).copied().unwrap_or(0.0))
        .collect()
}

/// Whether a live preview survives a GPU sync.
///
/// This is the check that stops a preview writing into buffers nothing draws.
/// A [`kreate_render::Deform`]'s bind groups name the `GpuMesh`'s own position
/// and normal buffers — that is the point of it, since writing anywhere else
/// would mean a copy — so a sync that drops that mesh or re-uploads it leaves
/// the preview deforming an orphan. wgpu keeps the allocation alive behind the
/// bind group, so nothing crashes: the mesh on screen simply stops following
/// the pointer, which is worse.
///
/// The rule is the whole plan and not the previewed slot. While a preview runs
/// the document is not written, so `plan_sync` finds nothing to do and its
/// caller returns early — which means any plan with work in it is some *other*
/// change to the scene, arriving during a drag. There is no useful way to tell
/// which of those touch the previewed mesh: a reflag adopts a new handle and
/// the preview was matched by pointer identity, an upload can land in any slot,
/// and a drop reorders the rest. So the answer is to put the mesh back where
/// the document says it is and let the drag finish on the CPU path.
pub fn preview_survives(plan: &SyncPlan, cached: usize) -> bool {
    !plan.needs_work(cached)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_reflect::PropertyValue;

    fn params(pairs: &[(&'static str, PropertyValue)]) -> PropertyMap {
        let mut map = PropertyMap::default();
        for (k, v) in pairs {
            map.set(k, v.clone());
        }
        map
    }

    #[test]
    fn a_move_becomes_a_translation_and_nothing_else() {
        let g = gesture_for(
            "mesh.translate",
            &params(&[("translation", PropertyValue::Vec3([1.0, -2.0, 0.5]))]),
            // A pivot is offered and must be ignored: a move is not about
            // anything, and a shader that rotated or scaled about this point
            // would be applying the identity to it — until someone made the
            // rotation non-identity by accident.
            Vec3::new(9.0, 9.0, 9.0),
        )
        .expect("a translation is previewable");
        assert_eq!(g.translation, [1.0, -2.0, 0.5, 0.0]);
        assert_eq!(g.rotation, Gesture::IDENTITY.rotation);
        assert_eq!(g.scale, Gesture::IDENTITY.scale);
        assert_eq!(g.pivot, [0.0; 4], "a move happens about nothing");
    }

    #[test]
    fn a_rotation_turns_about_the_selection_median() {
        let pivot = Vec3::new(0.5, -1.5, 2.0);
        for (axis, expected) in [(0u32, Vec3::X), (1, Vec3::Y), (2, Vec3::Z)] {
            let g = gesture_for(
                "mesh.rotate",
                &params(&[
                    ("angle", PropertyValue::Float(0.75)),
                    ("axis", PropertyValue::Enum(axis)),
                ]),
                pivot,
            )
            .expect("a rotation is previewable");
            let q = glam::Quat::from_array(g.rotation);
            let want = glam::Quat::from_axis_angle(expected, 0.75);
            assert!(
                q.abs_diff_eq(want, 1e-6),
                "axis {axis} gave {q:?}, wanted {want:?}"
            );
            assert_eq!(g.pivot, [pivot.x, pivot.y, pivot.z, 0.0]);
            assert_eq!(g.translation, [0.0; 4]);
            assert_eq!(g.scale, Gesture::IDENTITY.scale);
        }
    }

    #[test]
    fn the_three_axes_are_actually_different() {
        // Guards the test above: if `axis` were ignored, every branch would
        // agree with itself and the loop would pass.
        let mut seen = Vec::new();
        for axis in 0..3u32 {
            let g = gesture_for(
                "mesh.rotate",
                &params(&[
                    ("angle", PropertyValue::Float(0.75)),
                    ("axis", PropertyValue::Enum(axis)),
                ]),
                Vec3::ZERO,
            )
            .unwrap();
            seen.push(g.rotation);
        }
        assert_ne!(seen[0], seen[1]);
        assert_ne!(seen[1], seen[2]);
        assert_ne!(seen[0], seen[2]);
    }

    #[test]
    fn a_scale_is_per_axis_about_the_median() {
        let pivot = Vec3::new(1.0, 2.0, 3.0);
        let g = gesture_for(
            "mesh.scale",
            &params(&[("scale", PropertyValue::Vec3([2.0, 1.0, 0.25]))]),
            pivot,
        )
        .expect("a scale is previewable");
        // Per axis, not one factor: an axis-constrained scale sends `[f, 1, 1]`
        // and a preview that took one component would scale uniformly.
        assert_eq!(g.scale, [2.0, 1.0, 0.25, 0.0]);
        assert_eq!(g.pivot, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(g.rotation, Gesture::IDENTITY.rotation);
        assert_eq!(g.translation, [0.0; 4]);
    }

    #[test]
    fn an_operator_that_changes_topology_is_not_previewable() {
        // The fallback has to exist and has to be taken. Extrude, inset and
        // bevel are modal drags too, and no amount of deforming a frozen
        // buffer will add a face to it — a preview that tried would show the
        // old topology moving and then jump when the operator ran.
        for op in [
            "mesh.extrude",
            "mesh.inset",
            "mesh.bevel",
            "object.translate",
            "",
        ] {
            assert!(!is_previewable(op), "{op} was claimed previewable");
            assert!(
                gesture_for(op, &params(&[]), Vec3::ZERO).is_none(),
                "{op} produced a gesture"
            );
        }
        for op in ["mesh.translate", "mesh.rotate", "mesh.scale"] {
            assert!(is_previewable(op), "{op} should be previewable");
        }
    }

    #[test]
    fn a_previewable_operator_with_the_wrong_params_is_refused() {
        // `is_previewable` answers from the id alone, so the params are checked
        // where they are read. A missing one has to be a `None` rather than a
        // default: previewing a rotation as zero degrees while the operator
        // turns the mesh is the exact failure this whole path is meant to
        // avoid.
        assert!(gesture_for("mesh.translate", &params(&[]), Vec3::ZERO).is_none());
        assert!(gesture_for(
            "mesh.rotate",
            &params(&[("angle", PropertyValue::Float(1.0))]),
            Vec3::ZERO
        )
        .is_none());
        assert!(gesture_for(
            "mesh.scale",
            &params(&[("scale", PropertyValue::Float(2.0))]),
            Vec3::ZERO
        )
        .is_none());
        // A scale is three numbers and one number is not three of them. Worth
        // stating because `as_float` does accept an `Int` and `as_int` an
        // `Enum` — the accessors are deliberately lenient where a script could
        // reasonably write either — and `as_vec3` deliberately is not.
        assert!(gesture_for(
            "mesh.rotate",
            &params(&[
                ("angle", PropertyValue::Vec3([1.0, 0.0, 0.0])),
                ("axis", PropertyValue::Enum(0)),
            ]),
            Vec3::ZERO
        )
        .is_none());
    }

    fn extract(mesh: &kreate_mesh::MeshData) -> kreate_render::RenderMeshCpu {
        kreate_render::RenderMeshCpu::extract(mesh)
    }

    #[test]
    fn every_copy_of_a_split_vertex_gets_the_same_weight() {
        // A cube's corner is three render vertices at one point. If they took
        // different weights the drag would pull them apart, and since they are
        // the same point the mesh would open along its own shading seams — a
        // failure that looks like the mesh exploding and reads like a topology
        // bug.
        let cube = kreate_mesh::primitives::cube(2.0);
        let cpu = extract(&cube);
        assert_eq!(cpu.positions.len(), 24, "a cube splits three ways");

        // One mesh vertex at full weight, one at a half, the rest at nothing.
        let spread = spread_weights(
            &cpu.render_to_vert,
            cube.num_verts() as usize,
            &[1, 5],
            &[1.0, 0.5],
        );
        assert_eq!(spread.len(), cpu.positions.len());
        for (r, v) in cpu.render_to_vert.iter().enumerate() {
            let expected = match *v {
                1 => 1.0,
                5 => 0.5,
                _ => 0.0,
            };
            assert_eq!(
                spread[r], expected,
                "render vertex {r} came from mesh vertex {v}"
            );
        }
        // And the copies really were several, or this proves nothing.
        let copies = cpu.render_to_vert.iter().filter(|v| **v == 1).count();
        assert_eq!(copies, 3, "mesh vertex 1 should have three render copies");
    }

    #[test]
    fn a_weight_for_a_vertex_that_is_not_there_is_ignored() {
        // The falloff is computed from the document mesh and the render mesh is
        // derived from it, so they agree — but a stale weight list arriving here
        // must not index past the end of anything. Dropped rather than clamped:
        // clamping would give some innocent vertex someone else's weight.
        let grid = kreate_mesh::primitives::grid(2.0, 2, 2);
        let cpu = extract(&grid);
        let spread = spread_weights(
            &cpu.render_to_vert,
            grid.num_verts() as usize,
            &[0, 9_999],
            &[1.0, 1.0],
        );
        assert_eq!(spread.len(), cpu.positions.len());
        assert_eq!(spread.iter().filter(|w| **w > 0.0).count(), 1);
    }

    #[test]
    fn an_empty_falloff_moves_nothing() {
        let grid = kreate_mesh::primitives::grid(2.0, 2, 2);
        let cpu = extract(&grid);
        let spread = spread_weights(&cpu.render_to_vert, grid.num_verts() as usize, &[], &[]);
        assert_eq!(spread.len(), cpu.positions.len());
        assert!(spread.iter().all(|w| *w == 0.0));
    }

    /// A document of `count` cubes, and what a GPU cache holding all of them
    /// would look like.
    fn scene(
        count: usize,
    ) -> (
        kreate_scene::Document,
        Vec<std::sync::Arc<kreate_mesh::MeshData>>,
    ) {
        use kreate_scene::document::{Object, ObjectData};
        let mut doc = kreate_scene::Document::new();
        for i in 0..count {
            let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0 + i as f32));
            doc.add_object(Object::new(format!("Cube{i}"), ObjectData::Mesh(mesh)));
        }
        let mut cache = kreate_scene::modifier::ModifierCache::new();
        let held = cache
            .evaluate_scene(&doc)
            .iter()
            .map(|(_, m)| m.clone())
            .collect();
        (doc, held)
    }

    fn evaluate(
        doc: &kreate_scene::Document,
    ) -> Vec<(
        kreate_scene::document::ObjectId,
        std::sync::Arc<kreate_mesh::MeshData>,
    )> {
        kreate_scene::modifier::ModifierCache::new().evaluate_scene(doc)
    }

    #[test]
    fn a_preview_survives_a_frame_that_changes_nothing() {
        // The whole of a drag, and by far the common case: the document is not
        // written, so every frame's plan is empty and the preview carries on.
        let (doc, held) = scene(2);
        let plan = crate::gpu_sync::plan_sync(&held, &evaluate(&doc));
        assert!(
            preview_survives(&plan, held.len()),
            "an unchanged scene should not disturb a drag"
        );
    }

    #[test]
    fn a_preview_is_given_up_when_the_scene_changes_under_it() {
        // Something other than the drag changed the scene — a script, an
        // autosave restore, a modifier finishing. Whatever it was, buffers are
        // about to be rebuilt or dropped, and a preview whose bind groups name
        // the old ones would quietly stop following the pointer.
        use kreate_scene::document::{Object, ObjectData};

        // A second object arrived.
        let (mut doc, held) = scene(1);
        let mesh = doc.add_mesh(kreate_mesh::primitives::grid(2.0, 2, 2));
        doc.add_object(Object::new("Grid".to_string(), ObjectData::Mesh(mesh)));
        let grew = crate::gpu_sync::plan_sync(&held, &evaluate(&doc));
        assert!(
            !preview_survives(&grew, held.len()),
            "an upload rebuilds buffers"
        );

        // And an object went away.
        let (doc, held) = scene(2);
        let id = doc.objects().next().expect("an object").0;
        let mut fewer = doc;
        fewer.remove_object(id);
        let emptied = crate::gpu_sync::plan_sync(&held, &evaluate(&fewer));
        assert!(
            !preview_survives(&emptied, held.len()),
            "a dropped slot has no buffers"
        );
    }

    #[test]
    fn a_selection_change_mid_drag_also_ends_the_preview() {
        // Conservative on purpose, and worth its own test because it is the one
        // that looks safe: a reflag keeps the buffers and only rewrites the flag
        // arrays, so a preview could in principle carry on through it. But the
        // slot *adopts a new handle*, and the preview was matched by pointer
        // identity — so it would be looking for a mesh nothing holds any more.
        // Ending it is one CPU-path drag; getting it wrong is a mesh that stops
        // moving.
        let (mut doc, held) = scene(1);
        let id = doc.meshes().next().expect("a mesh").0;
        doc.mesh_mut(id).expect("a mesh").attributes_mut().insert(
            kreate_mesh::attribute::names::SELECT_FACE,
            kreate_mesh::attribute::Domain::Face,
            kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(vec![
                true, false, false, false, false, false,
            ])),
        );
        let plan = crate::gpu_sync::plan_sync(&held, &evaluate(&doc));
        assert_eq!(plan.reflags.len(), 1, "this should be a selection change");
        assert!(!preview_survives(&plan, held.len()));
    }
}
