//! Edit mode across several objects at once.
//!
//! Phase 3's last unimplemented item, and the reason it is worth having: a
//! modeller selects a body and its two arms and drags a shared seam. Doing that
//! one object at a time means three drags that have to agree with each other,
//! and they will not.
//!
//! What makes it one operation rather than three is the **pivot**: the median of
//! every selected vertex in every mesh. Three objects rotated about their own
//! separate medians each turn about their own centre, and the model comes apart
//! at the seams — which is not a rougher version of what the user asked for, it
//! is a different operation.
//!
//! The other thing these tests are about is a shared datablock. Two linked
//! duplicates are one `MeshData`, and visiting it once per object applies the
//! transform twice.

use kreate_math::glam::{DQuat, DVec3};
use kreate_math::DTransform;
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_mesh::MeshData;
use kreate_ops::operator::{Context, Mode};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{MeshId, Object, ObjectData, ObjectId};
use kreate_scene::Document;

fn registry() -> OperatorRegistry {
    let mut r = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut r);
    r
}

/// A single quad in the XY plane, one unit across, cornered at the origin.
fn quad() -> MeshData {
    let p = |x: f32, y: f32| glam::Vec3::new(x, y, 0.0);
    MeshData::from_faces(
        vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)],
        &[vec![0, 1, 2, 3]],
    )
}

struct Scene {
    doc: Document,
    ctx: Context,
    objects: Vec<ObjectId>,
    meshes: Vec<MeshId>,
}

impl Scene {
    /// `n` objects, each with its own quad, at `x = 0, 4, 8, …`.
    ///
    /// Spaced out so the joint median is somewhere none of them is, which is
    /// what makes a rotation about it distinguishable from a rotation about any
    /// one of them.
    fn new(n: usize) -> Self {
        let mut doc = Document::new();
        let mut objects = Vec::new();
        let mut meshes = Vec::new();
        for i in 0..n {
            let mesh = doc.add_mesh(quad());
            let id = doc.add_object(Object::new(format!("Quad{i}"), ObjectData::Mesh(mesh)));
            doc.object_mut(id).expect("object").transform = DTransform {
                translation: DVec3::new(4.0 * i as f64, 0.0, 0.0),
                ..DTransform::default()
            };
            objects.push(id);
            meshes.push(mesh);
        }
        Self {
            doc,
            ctx: Context {
                mode: Mode::EditMesh,
            },
            objects,
            meshes,
        }
    }

    /// Two objects sharing one mesh datablock — linked duplicates.
    fn linked() -> Self {
        let mut doc = Document::new();
        let mesh = doc.add_mesh(quad());
        let mut objects = Vec::new();
        for i in 0..2 {
            let id = doc.add_object(Object::new(format!("Linked{i}"), ObjectData::Mesh(mesh)));
            doc.object_mut(id).expect("object").transform = DTransform {
                translation: DVec3::new(4.0 * i as f64, 0.0, 0.0),
                ..DTransform::default()
            };
            objects.push(id);
        }
        Self {
            doc,
            ctx: Context {
                mode: Mode::EditMesh,
            },
            objects,
            meshes: vec![mesh],
        }
    }

    fn select_objects(&mut self, which: &[usize]) {
        for i in which {
            self.doc.selection.add(self.objects[*i]);
        }
    }

    /// Selects whole faces of one object's mesh.
    fn select_faces(&mut self, object: usize, faces: &[usize]) {
        let mesh_id = self
            .doc
            .object(self.objects[object])
            .expect("object")
            .mesh()
            .expect("a mesh");
        let len = self.doc.mesh(mesh_id).expect("mesh").num_faces() as usize;
        let mut flags = vec![false; len];
        for f in faces {
            flags[*f] = true;
        }
        let mesh = self.doc.mesh_mut(mesh_id).expect("mesh");
        mesh.ensure_selection_layers();
        mesh.attributes_mut().insert(
            names::SELECT_FACE,
            Domain::Face,
            AttrData::Bool(flags.into()),
        );
    }

    fn select_verts(&mut self, object: usize, verts: &[usize]) {
        let mesh_id = self
            .doc
            .object(self.objects[object])
            .expect("object")
            .mesh()
            .expect("a mesh");
        let len = self.doc.mesh(mesh_id).expect("mesh").num_verts() as usize;
        let mut flags = vec![false; len];
        for v in verts {
            flags[*v] = true;
        }
        let mesh = self.doc.mesh_mut(mesh_id).expect("mesh");
        mesh.ensure_selection_layers();
        mesh.attributes_mut().insert(
            names::SELECT_VERT,
            Domain::Point,
            AttrData::Bool(flags.into()),
        );
    }

    /// One object's vertices, in the world.
    fn world(&self, object: usize) -> Vec<DVec3> {
        let id = self.objects[object];
        let to_world = self.doc.world_transform(id);
        let mesh_id = self.doc.object(id).expect("object").mesh().expect("a mesh");
        self.doc
            .mesh(mesh_id)
            .expect("mesh")
            .positions()
            .iter()
            .map(|p| to_world.transform_point3(p.as_dvec3()))
            .collect()
    }

    /// Which elements of one object's mesh are selected, in a domain.
    fn selected(&self, object: usize, layer: &str) -> Vec<usize> {
        let mesh_id = self
            .doc
            .object(self.objects[object])
            .expect("object")
            .mesh()
            .expect("a mesh");
        let mesh = self.doc.mesh(mesh_id).expect("mesh");
        match mesh.attributes().get_bool(layer) {
            Some(flags) => flags
                .iter()
                .enumerate()
                .filter(|(_, on)| **on)
                .map(|(i, _)| i)
                .collect(),
            None => Vec::new(),
        }
    }

    /// The `object` parameter `mesh.select_pick` wants: the arena slot plus one.
    fn key(&self, object: usize) -> i64 {
        use kreate_mesh::arena::ArenaIndex as _;
        self.objects[object].index() as i64 + 1
    }

    /// How many faces one object's mesh has.
    fn faces(&self, object: usize) -> usize {
        let mesh_id = self
            .doc
            .object(self.objects[object])
            .expect("object")
            .mesh()
            .expect("a mesh");
        self.doc.mesh(mesh_id).expect("mesh").num_faces() as usize
    }

    /// Whether every face of one object's mesh is shaded smooth.
    fn all_smooth(&self, object: usize) -> bool {
        let mesh_id = self
            .doc
            .object(self.objects[object])
            .expect("object")
            .mesh()
            .expect("a mesh");
        let mesh = self.doc.mesh(mesh_id).expect("mesh");
        match mesh.attributes().get_bool(names::SHADE_SMOOTH) {
            Some(flags) => !flags.is_empty() && flags.iter().all(|on| *on),
            None => false,
        }
    }

    fn run(&mut self, op: &str, pairs: &[(&'static str, PropertyValue)]) -> kreate_ops::OpResult {
        let registry = registry();
        let mut history = History::new();
        let mut params = PropertyMap::new();
        for (k, v) in pairs {
            params.set(k, v.clone());
        }
        history
            .dispatch(&registry, &mut self.doc, &self.ctx, op, &params)
            .expect("dispatch")
    }
}

fn assert_close(label: &str, got: &[DVec3], want: &[DVec3]) {
    assert_eq!(got.len(), want.len(), "{label}: vertex count");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (*a - *b).length() < 1e-6,
            "{label}: vertex {i} is at {a:?}, wanted {b:?}"
        );
    }
}

#[test]
fn a_move_reaches_every_selected_object() {
    // The plainest form of the thing: two objects in edit mode, a face selected
    // in each, one drag. Both move, by the same world offset.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);
    let before = [scene.world(0), scene.world(1)];

    let offset = DVec3::new(0.5, -0.25, 2.0);
    assert_eq!(
        scene.run(
            "mesh.translate",
            &[(
                "translation",
                PropertyValue::Vec3([offset.x, offset.y, offset.z])
            )],
        ),
        kreate_ops::OpResult::Finished
    );

    for (object, was) in before.iter().enumerate() {
        let want: Vec<DVec3> = was.iter().map(|p| *p + offset).collect();
        assert_close(&format!("object {object}"), &scene.world(object), &want);
    }
}

#[test]
fn an_object_with_nothing_selected_stays_where_it_is() {
    // And does not cancel the operation. Reaching for a tool with three objects
    // in edit mode and elements selected in one of them is an ordinary thing to
    // do, and refusing it would make multi-object edit mode a mode you have to
    // leave in order to work.
    let mut scene = Scene::new(3);
    scene.select_objects(&[0, 1, 2]);
    scene.select_faces(1, &[0]);
    let before = [scene.world(0), scene.world(1), scene.world(2)];

    let offset = DVec3::new(1.0, 0.0, 0.0);
    assert_eq!(
        scene.run(
            "mesh.translate",
            &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
        ),
        kreate_ops::OpResult::Finished
    );

    assert_close("the untouched first", &scene.world(0), &before[0]);
    assert_close("the untouched third", &scene.world(2), &before[2]);
    let want: Vec<DVec3> = before[1].iter().map(|p| *p + offset).collect();
    assert_close("the selected one", &scene.world(1), &want);
}

#[test]
fn nothing_selected_anywhere_is_cancelled_rather_than_recorded() {
    // A gesture that does nothing must not leave a history entry that undoes to
    // the same mesh.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    let before = [scene.world(0), scene.world(1)];

    assert_eq!(
        scene.run(
            "mesh.translate",
            &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
        ),
        kreate_ops::OpResult::Cancelled
    );
    assert_close("first", &scene.world(0), &before[0]);
    assert_close("second", &scene.world(1), &before[1]);
}

#[test]
fn a_rotation_turns_about_the_joint_median_not_each_object_s_own() {
    // The claim that makes this one operation rather than several. The two
    // quads are four units apart, so a rotation about the point between them
    // *swaps their neighbourhoods*, while a rotation about each one's own
    // median leaves each roughly where it was. The two answers are nothing like
    // each other, which is the point.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);
    let before = [scene.world(0), scene.world(1)];

    // The joint median, worked out here so the test states it rather than
    // trusting the implementation's arithmetic.
    let all: Vec<DVec3> = before[0].iter().chain(before[1].iter()).copied().collect();
    let pivot = all.iter().copied().fold(DVec3::ZERO, |a, b| a + b) / all.len() as f64;

    let angle = std::f64::consts::PI;
    assert_eq!(
        scene.run(
            "mesh.rotate",
            &[
                ("angle", PropertyValue::Float(angle)),
                ("axis", PropertyValue::Enum(2)),
            ],
        ),
        kreate_ops::OpResult::Finished
    );

    let turn = DQuat::from_rotation_z(angle);
    for (object, was) in before.iter().enumerate() {
        let want: Vec<DVec3> = was.iter().map(|p| pivot + turn * (*p - pivot)).collect();
        assert_close(&format!("object {object}"), &scene.world(object), &want);
    }

    // And the two really did travel: a half turn about a point between them
    // sends each across to the other's side.
    let travelled = (scene.world(0)[0] - before[0][0]).length();
    assert!(
        travelled > 3.0,
        "object 0 moved only {travelled}; the pivot cannot have been the joint one"
    );
}

#[test]
fn the_joint_median_is_weighted_by_how_many_vertices_each_object_contributes() {
    // A mean of means is not a mean. One object with four vertices selected and
    // another with one puts the joint median four fifths of the way towards the
    // first — not half way, which is what averaging the two medians would give.
    //
    // Checked through a rotation, because the pivot is only observable through
    // one.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]); // all four
    scene.select_verts(1, &[0]); // one
    let before = [scene.world(0), scene.world(1)];

    let mut selected: Vec<DVec3> = before[0].to_vec();
    selected.push(before[1][0]);
    let pivot = selected.iter().copied().fold(DVec3::ZERO, |a, b| a + b) / 5.0;
    // The unweighted answer, which this test exists to rule out.
    let naive =
        (before[0].iter().copied().fold(DVec3::ZERO, |a, b| a + b) / 4.0 + before[1][0]) / 2.0;
    assert!(
        (pivot - naive).length() > 0.3,
        "the fixture cannot tell the two pivots apart"
    );

    let angle = std::f64::consts::FRAC_PI_2;
    scene.run(
        "mesh.rotate",
        &[
            ("angle", PropertyValue::Float(angle)),
            ("axis", PropertyValue::Enum(2)),
        ],
    );

    let turn = DQuat::from_rotation_z(angle);
    let want: Vec<DVec3> = before[0]
        .iter()
        .map(|p| pivot + turn * (*p - pivot))
        .collect();
    assert_close("the four-vertex object", &scene.world(0), &want);
}

#[test]
fn a_shared_datablock_is_transformed_once() {
    // Two linked duplicates are one `MeshData`. Visiting it once per object
    // applies the transform twice: a one-unit drag moves the vertices two
    // units, which reads as the tool being twice as sensitive on linked
    // objects — and only on linked objects, which is the sort of thing nobody
    // reproduces.
    let mut scene = Scene::linked();
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    assert_eq!(scene.meshes.len(), 1, "the fixture shares one datablock");

    let local_before = scene
        .doc
        .mesh(scene.meshes[0])
        .expect("mesh")
        .positions()
        .to_vec();

    scene.run(
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
    );

    let local_after = scene
        .doc
        .mesh(scene.meshes[0])
        .expect("mesh")
        .positions()
        .to_vec();
    for (i, (a, b)) in local_after.iter().zip(&local_before).enumerate() {
        let moved = (*a - *b).length();
        assert!(
            (moved - 1.0).abs() < 1e-5,
            "vertex {i} moved {moved}, not one unit — the datablock was visited twice"
        );
    }
}

#[test]
fn a_shared_datablock_is_listed_once_whichever_object_is_active() {
    // The order matters as well as the count: `edit_targets` puts the active
    // object first, so a shared mesh is edited in *its* space. Arbitrary but
    // not ambiguous — one set of vertices cannot live in two spaces at once,
    // and this is the object the user would name.
    let mut scene = Scene::linked();
    scene.select_objects(&[0, 1]);
    for (active, expected) in [(0usize, 0usize), (1, 1)] {
        scene.doc.selection.active = Some(scene.objects[active]);
        let targets = kreate_ops::ops::mesh::edit_targets(&scene.doc);
        assert_eq!(targets.len(), 1, "one datablock, one target");
        assert_eq!(
            targets[0].object, scene.objects[expected],
            "the active object should own the target"
        );
        assert_eq!(
            kreate_ops::ops::mesh::target_mesh(&scene.doc),
            Some(targets[0].mesh),
            "`target_mesh` and `edit_targets` disagree about which comes first"
        );
    }
}

#[test]
fn one_vertex_in_each_of_two_objects_can_still_be_rotated() {
    // The refusal that had to be rewritten to compose. A single vertex is its
    // own median, so rotating it is exactly the identity and the operator
    // declines — correctly. But one vertex in each of *two* objects is two
    // places with a point between them, and turning them about it is a real
    // operation. A per-mesh count of distinct places cannot see that; comparing
    // the two meshes' places can.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_verts(0, &[0]);
    scene.select_verts(1, &[0]);

    assert_eq!(
        scene.run(
            "mesh.rotate",
            &[
                ("angle", PropertyValue::Float(std::f64::consts::PI)),
                ("axis", PropertyValue::Enum(2)),
            ],
        ),
        kreate_ops::OpResult::Finished,
        "two vertices in two objects is not a single place"
    );
}

#[test]
fn one_vertex_in_one_object_is_still_refused() {
    // The other half of the same claim, so the test above is not passing
    // because the refusal was simply deleted.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_verts(0, &[0]);

    assert_eq!(
        scene.run(
            "mesh.rotate",
            &[
                ("angle", PropertyValue::Float(std::f64::consts::PI)),
                ("axis", PropertyValue::Enum(2)),
            ],
        ),
        kreate_ops::OpResult::Cancelled
    );
    assert_eq!(
        scene.run("mesh.scale", &[("scale", PropertyValue::Vec3([2.0; 3]))]),
        kreate_ops::OpResult::Cancelled
    );
}

#[test]
fn the_same_vertex_of_two_linked_objects_is_one_place() {
    // The awkward corner of the rule above. Two linked duplicates share their
    // vertices, so selecting "one vertex in each" selects *one* vertex — and
    // since the datablock is listed once, there is one place and the rotation
    // is correctly refused. A rule written per object rather than per datablock
    // would have found two objects, called it two places, and recorded a
    // history entry for the identity.
    let mut scene = Scene::linked();
    scene.select_objects(&[0, 1]);
    scene.select_verts(0, &[0]);

    assert_eq!(
        scene.run(
            "mesh.rotate",
            &[
                ("angle", PropertyValue::Float(std::f64::consts::PI)),
                ("axis", PropertyValue::Enum(2)),
            ],
        ),
        kreate_ops::OpResult::Cancelled
    );
}

#[test]
fn an_object_that_is_not_selected_is_not_in_edit_mode() {
    // Edit mode is over the *selection*, so an object nobody selected does not
    // move however its mesh is flagged. Which matters because selection layers
    // survive leaving edit mode — a mesh remembers what was selected in it, and
    // a rule that walked every mesh in the document would drag geometry the
    // user cannot even see the cage of.
    let mut scene = Scene::new(3);
    scene.select_objects(&[0]);
    scene.select_faces(0, &[0]);
    scene.select_faces(2, &[0]);
    let before = scene.world(2);

    scene.run(
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
    );

    assert_close("the unselected object", &scene.world(2), &before);
    assert_eq!(
        kreate_ops::ops::mesh::edit_targets(&scene.doc).len(),
        1,
        "only the selected object is in edit mode"
    );
}

#[test]
fn each_object_takes_the_transform_in_its_own_space() {
    // Every object has its own matrix, and the drag is one world vector. A loop
    // that reused the first object's transform for the rest is the obvious way
    // to write this and is wrong the moment two objects are oriented
    // differently.
    let mut scene = Scene::new(2);
    scene
        .doc
        .object_mut(scene.objects[1])
        .expect("object")
        .transform = DTransform {
        translation: DVec3::new(4.0, 0.0, 0.0),
        rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
        scale: DVec3::new(2.0, 0.5, 1.0),
    };
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);
    let before = [scene.world(0), scene.world(1)];

    let offset = DVec3::new(0.75, -0.5, 1.25);
    scene.run(
        "mesh.translate",
        &[(
            "translation",
            PropertyValue::Vec3([offset.x, offset.y, offset.z]),
        )],
    );

    // Both travel by the same *world* offset, whatever their own orientation.
    for (object, was) in before.iter().enumerate() {
        let want: Vec<DVec3> = was.iter().map(|p| *p + offset).collect();
        assert_close(&format!("object {object}"), &scene.world(object), &want);
    }
}

#[test]
fn a_flattened_object_is_skipped_and_the_others_still_move() {
    // An object with a zero in its scale has no way back from world space. It
    // was refused when there was one mesh, and refusing the whole operation now
    // would let one degenerate object in a selection of ten stop the drag —
    // which is worse than the thing the refusal protects against.
    let mut scene = Scene::new(2);
    scene
        .doc
        .object_mut(scene.objects[0])
        .expect("object")
        .transform = DTransform {
        scale: DVec3::new(1.0, 0.0, 1.0),
        ..DTransform::default()
    };
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);

    let flat_before = scene
        .doc
        .mesh(scene.meshes[0])
        .expect("mesh")
        .positions()
        .to_vec();
    let before = scene.world(1);

    assert_eq!(
        scene.run(
            "mesh.translate",
            &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
        ),
        kreate_ops::OpResult::Finished
    );

    let flat_after = scene
        .doc
        .mesh(scene.meshes[0])
        .expect("mesh")
        .positions()
        .to_vec();
    assert_eq!(flat_after, flat_before, "the flattened mesh was written");
    assert!(
        flat_after.iter().all(|p| p.is_finite()),
        "the flattened mesh has NaN in it"
    );
    let want: Vec<DVec3> = before.iter().map(|p| *p + DVec3::X).collect();
    assert_close("the ordinary object", &scene.world(1), &want);
}

#[test]
fn the_proportional_falloff_does_not_reach_across_objects() {
    // It reaches whatever is within the radius of a selected vertex, and
    // "within the radius" is a world distance — so two objects close enough
    // *do* pull on each other, and two far apart do not. Both halves are
    // stated, because a falloff computed per mesh against only that mesh's own
    // selection is the easy mistake, and it is invisible until someone drags a
    // seam between two touching objects and the far side does not follow.
    for (gap, expect_pull) in [(0.5f64, true), (40.0, false)] {
        let mut scene = Scene::new(2);
        scene
            .doc
            .object_mut(scene.objects[1])
            .expect("object")
            .transform = DTransform {
            translation: DVec3::new(1.0 + gap, 0.0, 0.0),
            ..DTransform::default()
        };
        scene.select_objects(&[0, 1]);
        // Selected only in the first object.
        scene.select_faces(0, &[0]);
        let before = scene.world(1);

        scene.run(
            "mesh.translate",
            &[
                ("translation", PropertyValue::Vec3([0.0, 0.0, 1.0])),
                ("proportional_size", PropertyValue::Float(1.0)),
            ],
        );

        let after = scene.world(1);
        let pulled = before
            .iter()
            .zip(&after)
            .any(|(a, b)| (*a - *b).length() > 1e-6);
        assert_eq!(
            pulled,
            expect_pull,
            "with a gap of {gap} the second object should {} have been pulled",
            if expect_pull { "" } else { "not" }
        );
    }
}

// --- selecting across the meshes -------------------------------------------
//
// Without these the transform work above is unreachable: a click could only
// ever land in the active object, so a second one would never have anything
// selected and would never move.

#[test]
fn a_click_can_select_in_a_mesh_that_is_not_the_active_one() {
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.doc.selection.active = Some(scene.objects[0]);

    let key = scene.key(1);
    assert_eq!(
        scene.run(
            "mesh.select_pick",
            &[
                ("element", PropertyValue::Int(0)),
                ("mode", PropertyValue::Enum(2)),
                ("extend", PropertyValue::Bool(false)),
                ("object", PropertyValue::Int(key)),
            ],
        ),
        kreate_ops::OpResult::Finished
    );
    assert_eq!(
        scene.selected(1, names::SELECT_FACE),
        vec![0],
        "the click did not land in the object it named"
    );
    assert!(
        scene.selected(0, names::SELECT_FACE).is_empty(),
        "it landed in the active object instead"
    );
}

#[test]
fn a_click_with_no_object_named_still_means_the_active_one() {
    // Every caller before edit mode could hold more than one mesh omitted the
    // object, and a script writing `mesh.select_pick(element=0, mode=2)` means
    // "the one I am editing". Zero has to keep meaning that.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.doc.selection.active = Some(scene.objects[1]);

    scene.run(
        "mesh.select_pick",
        &[
            ("element", PropertyValue::Int(0)),
            ("mode", PropertyValue::Enum(2)),
            ("extend", PropertyValue::Bool(false)),
        ],
    );
    assert_eq!(scene.selected(1, names::SELECT_FACE), vec![0]);
    assert!(scene.selected(0, names::SELECT_FACE).is_empty());
}

#[test]
fn a_plain_click_clears_the_selection_in_every_mesh() {
    // "The selection" is every mesh's. A click that only cleared the mesh it
    // landed in would leave the other object's elements selected and invisible
    // to the user's intent — and the next drag would move them.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);

    let key = scene.key(1);
    scene.run(
        "mesh.select_pick",
        &[
            ("element", PropertyValue::Int(1)),
            ("mode", PropertyValue::Enum(0)),
            ("extend", PropertyValue::Bool(false)),
            ("object", PropertyValue::Int(key)),
        ],
    );
    assert!(
        scene.selected(0, names::SELECT_FACE).is_empty(),
        "the other mesh kept its selection through a plain click"
    );
    assert!(
        scene.selected(0, names::SELECT_VERT).is_empty(),
        "and its vertices too"
    );
    assert_eq!(scene.selected(1, names::SELECT_VERT), vec![1]);
}

#[test]
fn a_shift_click_leaves_the_other_mesh_alone() {
    // The other half: extending adds to what is there, wherever it is.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);

    let key = scene.key(1);
    scene.run(
        "mesh.select_pick",
        &[
            ("element", PropertyValue::Int(0)),
            ("mode", PropertyValue::Enum(2)),
            ("extend", PropertyValue::Bool(true)),
            ("object", PropertyValue::Int(key)),
        ],
    );
    assert_eq!(scene.selected(0, names::SELECT_FACE), vec![0]);
    assert_eq!(scene.selected(1, names::SELECT_FACE), vec![0]);
}

#[test]
fn a_click_on_an_object_that_is_not_in_edit_mode_selects_nothing() {
    // The id buffer covers the whole scene, so a click in edit mode lands on
    // objects whose cage is not drawn. Selecting elements the user cannot see
    // is worse than a click that does nothing — and it would put them in the
    // next transform.
    let mut scene = Scene::new(3);
    scene.select_objects(&[0]);
    scene.doc.selection.active = Some(scene.objects[0]);

    let outsider = scene.key(2);
    assert_eq!(
        scene.run(
            "mesh.select_pick",
            &[
                ("element", PropertyValue::Int(0)),
                ("mode", PropertyValue::Enum(2)),
                ("extend", PropertyValue::Bool(false)),
                ("object", PropertyValue::Int(outsider)),
            ],
        ),
        kreate_ops::OpResult::Cancelled
    );
    assert!(scene.selected(2, names::SELECT_FACE).is_empty());
    assert!(scene.selected(0, names::SELECT_FACE).is_empty());
}

#[test]
fn select_all_means_all_of_every_mesh() {
    let mut scene = Scene::new(3);
    scene.select_objects(&[0, 1]);

    scene.run(
        "mesh.select",
        &[
            ("action", PropertyValue::Enum(0)),
            ("mode", PropertyValue::Enum(0)),
        ],
    );
    for object in 0..2 {
        assert_eq!(
            scene.selected(object, names::SELECT_VERT).len(),
            4,
            "object {object} was not fully selected"
        );
    }
    // And not the one that is not in edit mode.
    assert!(scene.selected(2, names::SELECT_VERT).is_empty());
}

#[test]
fn deselecting_everything_reaches_every_mesh() {
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);

    assert_eq!(
        scene.run("mesh.select_nothing", &[]),
        kreate_ops::OpResult::Finished
    );
    for object in 0..2 {
        assert!(
            scene.selected(object, names::SELECT_FACE).is_empty(),
            "object {object} kept its selection"
        );
    }
    // And twice is not two history entries.
    assert_eq!(
        scene.run("mesh.select_nothing", &[]),
        kreate_ops::OpResult::Cancelled
    );
}

#[test]
fn selecting_in_two_objects_then_dragging_moves_both() {
    // The whole thing, end to end, through the operators a click and a drag
    // actually dispatch — because the two halves being right separately is not
    // the same as them fitting together.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.doc.selection.active = Some(scene.objects[0]);

    let first = scene.key(0);
    scene.run(
        "mesh.select_pick",
        &[
            ("element", PropertyValue::Int(0)),
            ("mode", PropertyValue::Enum(2)),
            ("extend", PropertyValue::Bool(false)),
            ("object", PropertyValue::Int(first)),
        ],
    );
    let second = scene.key(1);
    scene.run(
        "mesh.select_pick",
        &[
            ("element", PropertyValue::Int(0)),
            ("mode", PropertyValue::Enum(2)),
            ("extend", PropertyValue::Bool(true)),
            ("object", PropertyValue::Int(second)),
        ],
    );

    let before = [scene.world(0), scene.world(1)];
    let offset = DVec3::new(0.0, 0.0, 3.0);
    assert_eq!(
        scene.run(
            "mesh.translate",
            &[("translation", PropertyValue::Vec3([0.0, 0.0, 3.0]))],
        ),
        kreate_ops::OpResult::Finished
    );
    for (object, was) in before.iter().enumerate() {
        let want: Vec<DVec3> = was.iter().map(|p| *p + offset).collect();
        assert_close(&format!("object {object}"), &scene.world(object), &want);
    }
}

// --- topology across the meshes --------------------------------------------
//
// The other half of what a modeller means by having several objects in edit
// mode: extruding, insetting and deleting act on all of them, on whatever each
// one has selected.

#[test]
fn an_extrude_reaches_every_selected_object() {
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);
    let before = [scene.faces(0), scene.faces(1)];

    assert_eq!(
        scene.run(
            "mesh.extrude_region",
            &[("offset", PropertyValue::Float(0.5))],
        ),
        kreate_ops::OpResult::Finished
    );
    for (object, was) in before.iter().enumerate() {
        assert!(
            scene.faces(object) > *was,
            "object {object} did not grow: {} faces, was {was}",
            scene.faces(object)
        );
    }
}

#[test]
fn an_extrude_leaves_a_mesh_with_nothing_selected_alone() {
    // What makes it safe to run the loop over the whole set: a mesh with no
    // faces selected returns early inside the closure, so it is not merely
    // unchanged — it is untouched, and its arrays stay shared.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    let untouched = scene.faces(1);

    assert_eq!(
        scene.run(
            "mesh.extrude_region",
            &[("offset", PropertyValue::Float(0.5))],
        ),
        kreate_ops::OpResult::Finished
    );
    assert_eq!(scene.faces(1), untouched, "the other mesh grew");
}

#[test]
fn an_extrude_with_nothing_selected_anywhere_is_cancelled() {
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    assert_eq!(
        scene.run(
            "mesh.extrude_region",
            &[("offset", PropertyValue::Float(0.5))],
        ),
        kreate_ops::OpResult::Cancelled
    );
}

#[test]
fn a_delete_reaches_every_selected_object() {
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);

    assert_eq!(
        scene.run("mesh.delete", &[("mode", PropertyValue::Enum(2))]),
        kreate_ops::OpResult::Finished
    );
    for object in 0..2 {
        assert_eq!(scene.faces(object), 0, "object {object} kept its face");
    }
}

#[test]
fn an_inset_reaches_every_selected_object() {
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.select_faces(0, &[0]);
    scene.select_faces(1, &[0]);
    let before = [scene.faces(0), scene.faces(1)];

    assert_eq!(
        scene.run("mesh.inset", &[("thickness", PropertyValue::Float(0.2))]),
        kreate_ops::OpResult::Finished
    );
    for (object, was) in before.iter().enumerate() {
        assert!(scene.faces(object) > *was, "object {object} was not inset");
    }
}

#[test]
fn shading_with_nothing_selected_anywhere_reaches_every_mesh() {
    // The convenience these operators have: with nothing selected they act on
    // the whole mesh, because shading a model flat is something a user does to
    // something they have just imported.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);

    assert_eq!(
        scene.run("mesh.shade", &[("smooth", PropertyValue::Bool(true))]),
        kreate_ops::OpResult::Finished
    );
    for object in 0..2 {
        assert!(scene.all_smooth(object), "object {object} was not shaded");
    }
}

#[test]
fn shading_a_selection_in_one_object_leaves_the_others_alone() {
    // The half that the convenience above would have got wrong. Asked per
    // mesh, "nothing selected here" is true of every object the user did *not*
    // select in — so selecting two faces in one object would have re-shaded
    // every face of every other object in edit mode, which is the opposite of
    // what selecting something means.
    let mut scene = Scene::new(3);
    scene.select_objects(&[0, 1, 2]);
    scene.select_faces(0, &[0]);

    assert_eq!(
        scene.run("mesh.shade", &[("smooth", PropertyValue::Bool(true))]),
        kreate_ops::OpResult::Finished
    );
    assert!(
        !scene.all_smooth(1),
        "object 1 was shaded although nothing in it was selected"
    );
    assert!(!scene.all_smooth(2), "and so was object 2");
}

#[test]
fn a_loop_cut_names_one_edge_and_so_stays_in_one_mesh() {
    // The exception to the loop, and the reason it is an exception: a loop cut
    // names the edge the pointer was over, and an index is in one mesh's
    // numbering. Handing it to another mesh names a different edge — so this
    // one keeps the target and the others are left alone.
    let mut scene = Scene::new(2);
    scene.select_objects(&[0, 1]);
    scene.doc.selection.active = Some(scene.objects[0]);
    let before = [scene.faces(0), scene.faces(1)];

    scene.run(
        "mesh.loop_cut",
        &[
            ("edge", PropertyValue::Int(0)),
            ("cuts", PropertyValue::Int(1)),
        ],
    );
    assert_eq!(
        scene.faces(1),
        before[1],
        "the loop cut reached a mesh whose numbering it does not speak"
    );
    let _ = before[0];
}

#[test]
fn every_mesh_in_edit_mode_gets_its_own_session() {
    // What stops the second object costing a decode and an encode of its whole
    // mesh on every operator — the cost `EditSession` exists to avoid, which a
    // single session would have put back for every object after the first.
    let (doc, meshes) = {
        let mut scene = Scene::new(3);
        scene.select_objects(&[0, 1, 2]);
        let meshes: Vec<_> = kreate_ops::ops::mesh::edit_targets(&scene.doc)
            .into_iter()
            .map(|t| t.mesh)
            .collect();
        (scene.doc, meshes)
    };
    assert_eq!(meshes.len(), 3);

    let mut sessions = kreate_ops::EditSessions::new();
    sessions.open_for(&doc, &meshes);
    assert_eq!(sessions.len(), 3);
    assert_eq!(sessions.meshes().collect::<Vec<_>>(), meshes);
}

#[test]
fn a_topology_edit_uses_the_session_of_the_mesh_it_is_editing() {
    // The mistake a set makes possible: baking the wrong session writes a mesh
    // nobody edited over itself, and the symptom is one object losing an edit
    // when a *different* one is extruded.
    //
    // Driven through `dispatch_in_session` with all three sessions open, so the
    // lookup is the thing under test rather than the fallback that converts for
    // itself.
    let mut scene = Scene::new(3);
    scene.select_objects(&[0, 1, 2]);
    scene.select_faces(1, &[0]);

    let meshes: Vec<_> = kreate_ops::ops::mesh::edit_targets(&scene.doc)
        .into_iter()
        .map(|t| t.mesh)
        .collect();
    let mut sessions = kreate_ops::EditSessions::new();
    sessions.open_for(&scene.doc, &meshes);

    let before = [scene.faces(0), scene.faces(1), scene.faces(2)];
    let registry = registry();
    let mut history = History::new();
    let mut params = PropertyMap::new();
    params.set("offset", PropertyValue::Float(0.5));
    history
        .dispatch_in_session(
            &registry,
            &mut scene.doc,
            Some(&mut sessions),
            &scene.ctx,
            "mesh.extrude_region",
            &params,
        )
        .expect("mesh.extrude_region");

    assert!(scene.faces(1) > before[1], "the selected mesh did not grow");
    assert_eq!(scene.faces(0), before[0], "another mesh was written");
    assert_eq!(scene.faces(2), before[2], "and so was the third");
}
