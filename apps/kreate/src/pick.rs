//! Ray queries against the scene, and the BVH cache behind them.
//!
//! The CPU side of picking. The plan keeps this deliberately separate from the
//! GPU id buffer: the id buffer answers "what did the user click, as they see
//! it" with occlusion for free, and this answers "where exactly does this ray
//! meet the surface" with barycentric coordinates and full precision. Snapping,
//! knife projection, loop-cut placement, measurement and the camera's auto-depth
//! pivot all need the second question, and none of them can be answered by the
//! first.
//!
//! # Against the evaluated mesh
//!
//! Queries run against the geometry the user can see, which is the *evaluated*
//! mesh rather than the datablock. Anything else would let the camera orbit
//! about a pivot inside a surface that is on screen, and would snap to a cage
//! the user cannot see. The consequence is that a hit names a face of the
//! evaluated mesh, which need not exist in the document — see [`ScenePick`].
//!
//! # Invalidation
//!
//! The same rule as the GPU cache: hold the `Arc<MeshData>` and compare with
//! [`Arc::ptr_eq`]. Free, exact, and impossible to forget. A cache keyed on an
//! id alone would watch the mesh change in place behind an unchanged key and
//! quietly answer from stale geometry — which here means a snap to a surface
//! that is no longer there.

use std::sync::Arc;

use kreate_math::{Bvh, DVec3, Ray, Vec3};
use kreate_mesh::MeshData;
use kreate_scene::document::ObjectId;
use kreate_scene::Document;

/// One mesh's tree, plus what it was built from.
#[derive(Clone)]
struct Entry {
    mesh: Arc<MeshData>,
    triangles: Vec<[u32; 3]>,
    /// Which face each triangle came from, so a hit can name an n-gon rather
    /// than the triangle nobody asked about.
    tri_to_face: Vec<u32>,
    bvh: Bvh,
}

impl Entry {
    /// Nobody but this cache holds the mesh any more, so its tree is free to
    /// be refitted for another. Two live meshes sharing connectivity — two
    /// objects with different shape-key values — each keep their own, rather
    /// than one tree being refitted back and forth between them every query.
    fn replaceable(&self) -> bool {
        Arc::strong_count(&self.mesh) == 1
    }
}

/// Lazily built BVHs, one per distinct evaluated mesh.
#[derive(Default)]
pub struct PickCache {
    entries: Vec<Entry>,
}

/// Where a ray met the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScenePick {
    pub object: ObjectId,
    /// The face of the **evaluated** mesh, not of the object's datablock.
    ///
    /// With a modifier stack in play the two are different, and there is no
    /// mapping back to the cage yet — a face index from here must not be used to
    /// select or edit a document face. Callers that only want the surface point
    /// (auto-depth, measurement, snapping) are unaffected.
    pub face: u32,
    /// The hit, in world space.
    pub point: DVec3,
    /// Distance along the world-space ray.
    pub distance: f64,
}

impl PickCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many trees are cached. Exposed so a test can show that a query does
    /// not rebuild what it already has.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drops trees for meshes the scene no longer contains.
    ///
    /// Called when a document changes shape rather than on every query: an
    /// abandoned tree costs memory, not correctness, and scanning for them on
    /// the hot path would cost both.
    pub fn retain_live(&mut self, scene: &[(ObjectId, Arc<MeshData>)]) {
        // A tree whose mesh has only moved its vertices is kept for the mesh
        // that replaced it: `entry` refits it rather than building anew.
        self.entries.retain(|e| {
            scene.iter().any(|(_, m)| {
                Arc::ptr_eq(m, &e.mesh) || (e.replaceable() && m.same_topology(&e.mesh))
            })
        });
    }

    /// The tree for a mesh, building it if it is not already held.
    ///
    /// There is no "rebuild": an edited mesh is a different handle, so a changed
    /// mesh is simply one the cache has never seen. The stale entry is left for
    /// [`PickCache::retain_live`] to collect.
    fn entry(&mut self, mesh: &Arc<MeshData>) -> &Entry {
        if let Some(i) = self.entries.iter().position(|e| Arc::ptr_eq(&e.mesh, mesh)) {
            return &self.entries[i];
        }
        // The same connectivity with its vertices moved — a sculpt stroke, a
        // drag — refits the tree in a linear pass instead of rebuilding it,
        // which on a large mesh is the difference between a stroke that keeps
        // up with the pointer and one that does not. The triangles are kept:
        // the faces are the same faces.
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.replaceable() && e.mesh.same_topology(mesh))
        {
            let e = &mut self.entries[i];
            e.bvh.refit(mesh.positions(), &e.triangles);
            e.mesh = mesh.clone();
            return &self.entries[i];
        }
        // The same, but the mesh it was built for is still held — the undo
        // snapshot of a stroke's first frame holds the mesh as it was — so
        // the tree is copied and the copy refitted: a copy is a few
        // milliseconds, a build on a dense mesh a pause the hand feels at the
        // start of every stroke.
        if let Some(i) = self.entries.iter().position(|e| e.mesh.same_topology(mesh)) {
            let mut e = self.entries[i].clone();
            e.bvh.refit(mesh.positions(), &e.triangles);
            e.mesh = mesh.clone();
            self.entries.push(e);
            return self.entries.last().expect("just pushed");
        }

        let tri = mesh.triangulate();
        // The tessellator emits a flat index stream because that is what a GPU
        // index buffer wants; the tree wants triangles. Regrouped here rather
        // than changing the tessellator, which has the hotter caller.
        let (triangles, _) = tri.indices.as_chunks::<3>();
        let triangles = triangles.to_vec();
        let bvh = Bvh::build(mesh.positions(), &triangles);
        self.entries.push(Entry {
            mesh: mesh.clone(),
            triangles,
            tri_to_face: tri.tri_to_face,
            bvh,
        });
        self.entries.last().expect("just pushed")
    }

    /// The nearest surface a world-space ray meets.
    ///
    /// `scene` is the evaluated geometry of the objects that can be hit; it is
    /// already filtered for visibility, so this does not consult the document
    /// for it. The document is here for the world transforms.
    ///
    /// The ray is taken into each object's local space rather than the geometry
    /// being taken into the world: object-local coordinates are `f32` and small,
    /// which is the whole reason positions are stored that way, and transforming
    /// a million vertices per query to avoid transforming one ray would undo it.
    pub fn raycast(
        &mut self,
        doc: &Document,
        scene: &[(ObjectId, Arc<MeshData>)],
        origin: DVec3,
        direction: DVec3,
        max_distance: f64,
    ) -> Option<ScenePick> {
        let mut best: Option<ScenePick> = None;
        for (object, mesh) in scene {
            let world = doc.world_transform(*object);
            let inverse = world.inverse();
            // A mirrored or squashed transform changes the ray's length, so the
            // hit distance comes back in local units. Converting it to world
            // units needs the scale along the ray, measured here rather than
            // assumed to be one.
            let local_origin = inverse.transform_point3(origin);
            let local_dir = inverse.transform_vector3(direction);
            let scale = local_dir.length();
            if scale < 1e-12 {
                continue;
            }

            let entry = self.entry(mesh);
            let ray = Ray::new(local_origin.as_vec3(), (local_dir / scale).as_vec3());
            let limit = (max_distance * scale) as f32;
            let Some(hit) =
                entry
                    .bvh
                    .raycast(entry.mesh.positions(), &entry.triangles, &ray, limit)
            else {
                continue;
            };

            let distance = hit.t as f64 / scale;
            if best.is_some_and(|b| b.distance <= distance) {
                continue;
            }
            best = Some(ScenePick {
                object: *object,
                face: entry
                    .tri_to_face
                    .get(hit.primitive as usize)
                    .copied()
                    .unwrap_or(0),
                point: world.transform_point3(hit.point.as_dvec3()),
                distance,
            });
        }
        best
    }

    /// The nearest surface under a normalised device coordinate.
    ///
    /// `ndc` is `-1..1` in both axes with `+y` up, which is what a viewport
    /// rectangle converts a cursor position into.
    pub fn raycast_ndc(
        &mut self,
        doc: &Document,
        scene: &[(ObjectId, Arc<MeshData>)],
        camera: &kreate_render::Camera,
        ndc: glam::Vec2,
        aspect: f64,
    ) -> Option<ScenePick> {
        let direction: Vec3 = camera.ray_direction(ndc, aspect);
        self.raycast(
            doc,
            scene,
            camera.eye(),
            direction.as_dvec3(),
            f64::INFINITY,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::document::{MeshId, Object, ObjectData};
    use kreate_scene::modifier::{ModifierInstance, ModifierKind};
    use kreate_scene::ModifierCache;

    fn doc_with_cube() -> (Document, ObjectId, MeshId) {
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let object = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        (doc, object, mesh)
    }

    /// The evaluated scene, as the application would hand it over.
    fn scene(doc: &Document) -> Vec<(ObjectId, Arc<MeshData>)> {
        ModifierCache::new().evaluate_scene(doc, &kreate_scene::modifier::EvalCtx::still())
    }

    /// Straight down the -Z axis from above.
    fn from_above(
        cache: &mut PickCache,
        doc: &Document,
        scene: &[(ObjectId, Arc<MeshData>)],
        height: f64,
    ) -> Option<ScenePick> {
        cache.raycast(
            doc,
            scene,
            DVec3::new(0.0, 0.0, height),
            -DVec3::Z,
            f64::INFINITY,
        )
    }

    #[test]
    fn a_ray_finds_the_near_face_of_a_cube() {
        let (doc, object, _) = doc_with_cube();
        let mut cache = PickCache::new();
        let hit = from_above(&mut cache, &doc, &scene(&doc), 10.0).expect("down onto the lid");

        assert_eq!(hit.object, object);
        assert!((hit.distance - 9.0).abs() < 1e-4, "hit at {}", hit.distance);
        assert!((hit.point.z - 1.0).abs() < 1e-4, "the top, not the bottom");
    }

    /// A sculpt stroke moves vertices and nothing else: the tree is refitted
    /// for the new mesh, not rebuilt beside the old one, and the ray meets
    /// the surface where it now is.
    #[test]
    fn a_mesh_whose_vertices_moved_refits_its_tree() {
        let (mut doc, _, mesh) = doc_with_cube();
        let mut cache = PickCache::new();
        let before = scene(&doc);
        from_above(&mut cache, &doc, &before, 10.0).expect("the lid");
        drop(before);
        for p in doc.mesh_mut(mesh).unwrap().positions_mut().iter_mut() {
            if p.z > 0.0 {
                p.z += 2.0;
            }
        }
        let after = scene(&doc);
        cache.retain_live(&after);
        let hit = from_above(&mut cache, &doc, &after, 10.0).expect("the raised lid");
        assert!((hit.point.z - 3.0).abs() < 1e-4, "{}", hit.point.z);
        assert_eq!(cache.entries.len(), 1, "refitted, not built anew");
    }

    /// A stroke's first frame: the mesh as it was is still held (by the undo
    /// snapshot), so its tree stays, and the moved mesh gets a refitted copy
    /// of it — right for both, and no build.
    #[test]
    fn a_moved_mesh_whose_old_self_is_held_gets_a_copy() {
        let (mut doc, _, mesh) = doc_with_cube();
        let mut cache = PickCache::new();
        let before = scene(&doc);
        from_above(&mut cache, &doc, &before, 10.0).expect("the lid");
        for p in doc.mesh_mut(mesh).unwrap().positions_mut().iter_mut() {
            if p.z > 0.0 {
                p.z += 2.0;
            }
        }
        let after = scene(&doc);
        let hit = from_above(&mut cache, &doc, &after, 10.0).expect("the raised lid");
        assert!((hit.point.z - 3.0).abs() < 1e-4, "{}", hit.point.z);
        let hit = from_above(&mut cache, &doc, &before, 10.0).expect("the old lid");
        assert!((hit.point.z - 1.0).abs() < 1e-4, "{}", hit.point.z);
        assert_eq!(cache.entries.len(), 2, "one tree for each mesh");
    }

    #[test]
    fn a_ray_that_misses_returns_nothing() {
        let (doc, _, _) = doc_with_cube();
        let mut cache = PickCache::new();
        assert_eq!(
            cache.raycast(
                &doc,
                &scene(&doc),
                DVec3::new(50.0, 0.0, 10.0),
                -DVec3::Z,
                f64::INFINITY
            ),
            None
        );
    }

    #[test]
    fn the_hit_names_the_face_rather_than_the_triangle() {
        // A cube's lid is one quad and two triangles. Reporting the triangle
        // would make every caller — snap, knife, face select — do the mapping
        // itself, and get it wrong somewhere.
        let (doc, _, _) = doc_with_cube();
        let scene = scene(&doc);
        let mut cache = PickCache::new();
        let a = cache
            .raycast(
                &doc,
                &scene,
                DVec3::new(-0.4, -0.4, 5.0),
                -DVec3::Z,
                f64::INFINITY,
            )
            .unwrap();
        let b = cache
            .raycast(
                &doc,
                &scene,
                DVec3::new(0.4, 0.4, 5.0),
                -DVec3::Z,
                f64::INFINITY,
            )
            .unwrap();
        assert_eq!(a.face, b.face, "both halves of the lid are the same n-gon");
    }

    #[test]
    fn the_nearest_object_wins() {
        let (mut doc, _, _) = doc_with_cube();
        let far = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let far_object = doc.add_object(Object::new("Far", ObjectData::Mesh(far)));
        doc.object_mut(far_object).unwrap().transform.translation.z = 20.0;

        let scene = scene(&doc);
        let mut cache = PickCache::new();
        let above = from_above(&mut cache, &doc, &scene, 40.0).unwrap();
        assert_eq!(above.object, far_object, "the high cube is nearer");

        let below = cache
            .raycast(
                &doc,
                &scene,
                DVec3::new(0.0, 0.0, -40.0),
                DVec3::Z,
                f64::INFINITY,
            )
            .unwrap();
        assert_ne!(below.object, far_object);
    }

    #[test]
    fn a_transformed_object_is_hit_where_it_is_drawn() {
        // The ray goes into local space, so the transform has to be inverted
        // correctly — including its scale, which changes the distance the tree
        // reports.
        let (mut doc, object, _) = doc_with_cube();
        {
            let o = doc.object_mut(object).unwrap();
            o.transform.translation = DVec3::new(5.0, 0.0, 0.0);
            o.transform.scale = DVec3::splat(2.0);
        }
        let mut cache = PickCache::new();
        let hit = cache
            .raycast(
                &doc,
                &scene(&doc),
                DVec3::new(5.0, 0.0, 10.0),
                -DVec3::Z,
                f64::INFINITY,
            )
            .expect("the cube moved and grew, and the ray follows");

        assert!(
            (hit.point.z - 2.0).abs() < 1e-4,
            "the scaled lid is at z = 2"
        );
        assert!(
            (hit.distance - 8.0).abs() < 1e-4,
            "distance must be in world units, not local ones: {}",
            hit.distance
        );
    }

    #[test]
    fn an_invisible_object_is_not_hit() {
        // Visibility is settled when the scene is evaluated, which is also what
        // makes it impossible for the picker and the renderer to disagree about
        // what is on screen.
        let (mut doc, object, _) = doc_with_cube();
        doc.object_mut(object).unwrap().visible = false;
        let mut cache = PickCache::new();
        assert_eq!(from_above(&mut cache, &doc, &scene(&doc), 10.0), None);
    }

    #[test]
    fn a_modified_object_is_hit_where_it_is_drawn() {
        // The reason queries run against the evaluated mesh: the mirrored half is
        // on screen, so the camera has to be able to orbit about it and a snap
        // has to land on it. Against the datablock this ray would miss.
        let (mut doc, object, _) = doc_with_cube();
        {
            let o = doc.object_mut(object).unwrap();
            o.transform.translation = DVec3::new(4.0, 0.0, 0.0);
            o.modifiers.push(ModifierInstance::new(ModifierKind::Array(
                kreate_mesh::modifiers::ArrayParams {
                    count: 2,
                    ..Default::default()
                },
            )));
        }
        let mut cache = PickCache::new();
        let hit = cache
            .raycast(
                &doc,
                &scene(&doc),
                DVec3::new(6.0, 0.0, 10.0),
                -DVec3::Z,
                f64::INFINITY,
            )
            .expect("the second copy of the cube is at x = 6");
        assert_eq!(hit.object, object);
        assert!((hit.point.z - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_tree_is_built_once_and_reused() {
        let (doc, _, _) = doc_with_cube();
        let scene = scene(&doc);
        let mut cache = PickCache::new();
        for _ in 0..10 {
            from_above(&mut cache, &doc, &scene, 10.0);
        }
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn instances_share_one_tree() {
        let (mut doc, object, _) = doc_with_cube();
        for _ in 0..4 {
            doc.duplicate_linked(object).unwrap();
        }
        let scene = scene(&doc);
        assert_eq!(scene.len(), 5);

        let mut cache = PickCache::new();
        from_above(&mut cache, &doc, &scene, 10.0);
        assert_eq!(cache.len(), 1, "one tree for five instances");
    }

    #[test]
    fn editing_the_mesh_invalidates_the_tree() {
        // The failure this guards is silent: a cache that answers from the old
        // geometry makes a snap land on a surface that is no longer there.
        let (mut doc, _, mesh) = doc_with_cube();
        let mut cache = PickCache::new();
        let before = from_above(&mut cache, &doc, &scene(&doc), 10.0).unwrap();

        for p in doc.mesh_mut(mesh).unwrap().positions_mut() {
            p.z += 5.0;
        }
        let scene = scene(&doc);
        let after = from_above(&mut cache, &doc, &scene, 20.0).expect("the cube moved up");

        assert!((before.point.z - 1.0).abs() < 1e-4);
        assert!(
            (after.point.z - 6.0).abs() < 1e-4,
            "answered from stale geometry: {}",
            after.point.z
        );

        cache.retain_live(&scene);
        assert_eq!(cache.len(), 1, "the stale tree is collected");
    }

    #[test]
    fn trees_for_meshes_that_left_the_scene_are_dropped_on_request() {
        let (mut doc, object, _) = doc_with_cube();
        let mut cache = PickCache::new();
        from_above(&mut cache, &doc, &scene(&doc), 10.0);
        assert_eq!(cache.len(), 1);

        doc.remove_object(object);
        cache.retain_live(&scene(&doc));
        assert!(cache.is_empty());
    }

    #[test]
    fn an_empty_document_answers_nothing() {
        let doc = Document::new();
        let mut cache = PickCache::new();
        assert_eq!(
            cache.raycast(&doc, &[], DVec3::ZERO, DVec3::Z, f64::INFINITY),
            None
        );
        assert!(cache.is_empty());
    }
}
