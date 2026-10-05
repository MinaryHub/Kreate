//! Snapping to geometry, and the exclusion that makes it possible.
//!
//! The arithmetic is resolvable without a window — a camera, a set of world
//! points and a cursor — so these are tests of the feel of the tool rather than
//! of the plumbing: does it catch what is under the cursor, does it let go when
//! nothing is near, and above all does it refuse to snap the selection to
//! itself.

use glam::{DAffine3, DVec3};
use kreate::snap::{self, Mode, SnapWorld};
use kreate_mesh::MeshData;
use kreate_render::Camera;

const VIEWPORT: (f32, f32) = (800.0, 600.0);

/// Looking along −Y at the origin, so world X is screen right and world Z is
/// screen up.
fn camera() -> Camera {
    Camera {
        pivot: DVec3::ZERO,
        distance: 8.0,
        rotation: kreate_math::DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
        ..Camera::default()
    }
}

/// Where a world point lands on screen, for aiming the cursor at it.
fn at_point(p: DVec3) -> (f32, f32) {
    snap::project(p, &camera(), VIEWPORT).expect("in front of the camera")
}

fn world_of(points: &[DVec3]) -> SnapWorld {
    SnapWorld {
        verts: points.to_vec(),
        ..SnapWorld::default()
    }
}

#[test]
fn a_vertex_under_the_cursor_is_caught() {
    let target = DVec3::new(1.0, 0.0, 0.5);
    let world = world_of(&[DVec3::new(-2.0, 0.0, 0.0), target]);
    let hit = snap::resolve(&world, Mode::Vertex, at_point(target), &camera(), VIEWPORT)
        .expect("the cursor is on it");
    assert!(
        (hit.point - target).length() < 1e-9,
        "caught {:?}",
        hit.point
    );
}

#[test]
fn nothing_is_caught_when_nothing_is_near() {
    // A snap that always finds something is a snap that never lets go: the
    // selection would jump to the least-distant vertex in the scene however far
    // away it was.
    let world = world_of(&[DVec3::new(-5.0, 0.0, 0.0)]);
    let far = at_point(DVec3::new(5.0, 0.0, 0.0));
    assert!(snap::resolve(&world, Mode::Vertex, far, &camera(), VIEWPORT).is_none());
}

#[test]
fn the_nearest_of_several_candidates_wins() {
    let near = DVec3::new(0.0, 0.0, 0.0);
    let also_close = DVec3::new(0.1, 0.0, 0.0);
    let world = world_of(&[also_close, near]);
    let hit = snap::resolve(&world, Mode::Vertex, at_point(near), &camera(), VIEWPORT)
        .expect("something is near");
    assert!((hit.point - near).length() < 1e-9, "caught {:?}", hit.point);
}

#[test]
fn increment_mode_resolves_nothing() {
    // It adjusts how far the drag went rather than choosing a destination, so it
    // must not return a point — one would override the offset entirely.
    let target = DVec3::ZERO;
    let world = world_of(&[target]);
    assert!(snap::resolve(
        &world,
        Mode::Increment,
        at_point(target),
        &camera(),
        VIEWPORT
    )
    .is_none());
}

#[test]
fn a_midpoint_is_caught_between_two_vertices() {
    let a = DVec3::new(-1.0, 0.0, 0.0);
    let b = DVec3::new(1.0, 0.0, 0.0);
    let world = SnapWorld {
        edges: vec![[a, b]],
        ..SnapWorld::default()
    };
    let middle = (a + b) * 0.5;
    let hit = snap::resolve(
        &world,
        Mode::EdgeMidpoint,
        at_point(middle),
        &camera(),
        VIEWPORT,
    )
    .expect("the cursor is on the middle");
    assert!(
        (hit.point - middle).length() < 1e-9,
        "caught {:?}",
        hit.point
    );

    // And the ends are not midpoints.
    assert!(snap::resolve(&world, Mode::EdgeMidpoint, at_point(a), &camera(), VIEWPORT).is_none());
}

#[test]
fn an_edge_catches_along_its_length_rather_than_at_its_ends() {
    // The distinction that makes edge snapping worth having: sliding towards a
    // long edge should catch the part of it beside the cursor. Catching an
    // endpoint instead would be vertex snapping with extra steps.
    let a = DVec3::new(-4.0, 0.0, 0.0);
    let b = DVec3::new(4.0, 0.0, 0.0);
    let world = SnapWorld {
        edges: vec![[a, b]],
        ..SnapWorld::default()
    };
    let aim = DVec3::new(2.5, 0.0, 0.0);
    let hit = snap::resolve(&world, Mode::Edge, at_point(aim), &camera(), VIEWPORT)
        .expect("the cursor is over the edge");
    assert!(
        (hit.point - aim).length() < 0.05,
        "caught {:?} rather than the point beside the cursor",
        hit.point
    );
}

#[test]
fn an_edge_snap_stays_on_the_edge_when_the_cursor_runs_past_it() {
    // Clamped, not extrapolated. A cursor beyond the end of an edge is nearest
    // to its endpoint, and a snap that ran off into space would put the
    // selection somewhere with no geometry at all.
    let a = DVec3::new(-1.0, 0.0, 0.0);
    let b = DVec3::new(1.0, 0.0, 0.0);
    let world = SnapWorld {
        edges: vec![[a, b]],
        ..SnapWorld::default()
    };
    // Just past `b`, close enough in screen space to still be within the radius.
    let past = at_point(DVec3::new(1.06, 0.0, 0.0));
    let hit = snap::resolve(&world, Mode::Edge, past, &camera(), VIEWPORT).expect("within reach");
    assert!(
        hit.point.x <= b.x + 1e-9,
        "ran off the end to {:?}",
        hit.point
    );
}

#[test]
fn a_face_catches_the_point_under_the_cursor() {
    // A face is a surface, not a point: what the user means is the place on it
    // that they are pointing at, which is a ray question rather than a distance
    // one.
    let world = SnapWorld {
        tris: vec![[
            DVec3::new(-4.0, 0.0, -4.0),
            DVec3::new(4.0, 0.0, -4.0),
            DVec3::new(0.0, 0.0, 4.0),
        ]],
        ..SnapWorld::default()
    };
    let aim = DVec3::new(0.5, 0.0, 0.5);
    let hit = snap::resolve(&world, Mode::Face, at_point(aim), &camera(), VIEWPORT)
        .expect("the cursor is over the face");
    assert!(
        (hit.point - aim).length() < 0.05,
        "caught {:?} rather than {aim:?}",
        hit.point
    );
}

#[test]
fn a_face_snap_takes_the_nearest_of_several() {
    // Two faces stacked along the view direction. The near one is what the user
    // can see and therefore what they are pointing at.
    let quad = |y: f64| {
        [
            DVec3::new(-2.0, y, -2.0),
            DVec3::new(2.0, y, -2.0),
            DVec3::new(0.0, y, 2.0),
        ]
    };
    // The camera is on −Y looking towards +Y, so the smaller y is nearer.
    let world = SnapWorld {
        tris: vec![quad(2.0), quad(-2.0)],
        ..SnapWorld::default()
    };
    let hit = snap::resolve(
        &world,
        Mode::Face,
        at_point(DVec3::new(0.0, 0.0, 0.0)),
        &camera(),
        VIEWPORT,
    )
    .expect("something is under the cursor");
    assert!(hit.point.y < 0.0, "caught the far face at {:?}", hit.point);
}

#[test]
fn a_face_behind_the_cursor_is_not_caught() {
    let world = SnapWorld {
        tris: vec![[
            DVec3::new(-2.0, -20.0, -2.0),
            DVec3::new(2.0, -20.0, -2.0),
            DVec3::new(0.0, -20.0, 2.0),
        ]],
        ..SnapWorld::default()
    };
    // A face twenty units behind the camera, which sits at y = -8.
    assert!(snap::resolve(&world, Mode::Face, (400.0, 300.0), &camera(), VIEWPORT).is_none());
}

// --- the exclusion ----------------------------------------------------------

fn quad_mesh() -> MeshData {
    MeshData::from_faces(
        vec![
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 1.0),
            glam::Vec3::new(0.0, 0.0, 1.0),
        ],
        &[vec![0, 1, 2, 3]],
    )
}

#[test]
fn moving_vertices_are_not_candidates() {
    // The rule everything rests on. A selection that can snap to itself locks
    // solid the instant the drag begins: the nearest vertex to a moving vertex
    // is itself, at zero distance, always.
    let mut world = SnapWorld::default();
    world.add_mesh(&quad_mesh(), &DAffine3::IDENTITY, &[0, 1]);
    assert_eq!(world.verts.len(), 2, "the moving pair should be excluded");
    assert!(world.verts.iter().all(|v| v.z > 0.5), "{:?}", world.verts);
}

#[test]
fn an_edge_with_a_moving_end_is_not_a_candidate() {
    // Half of it would move, so the point snapped to would not be where it
    // appeared to be by the time the drag ended.
    let mut mesh = quad_mesh();
    mesh.rebuild_edges();
    let mut world = SnapWorld::default();
    world.add_mesh(&mesh, &DAffine3::IDENTITY, &[0]);
    assert!(
        world.edges.iter().all(|e| {
            let moving = glam::Vec3::ZERO;
            let is_moving = |p: DVec3| (p.as_vec3() - moving).length() < 1e-6;
            !is_moving(e[0]) && !is_moving(e[1])
        }),
        "an edge touching the moving vertex survived: {:?}",
        world.edges
    );
    // The quad has four edges, two of which touch vertex 0.
    assert_eq!(world.edges.len(), 2);
}

#[test]
fn a_face_with_a_moving_corner_is_not_a_candidate() {
    let mut world = SnapWorld::default();
    world.add_mesh(&quad_mesh(), &DAffine3::IDENTITY, &[2]);
    assert!(world.tris.is_empty(), "{:?}", world.tris);
}

#[test]
fn a_mesh_with_nothing_moving_contributes_everything() {
    let mut mesh = quad_mesh();
    mesh.rebuild_edges();
    let mut world = SnapWorld::default();
    world.add_mesh(&mesh, &DAffine3::IDENTITY, &[]);
    assert_eq!(world.verts.len(), 4);
    assert_eq!(world.edges.len(), 4);
    // A quad fans into two triangles.
    assert_eq!(world.tris.len(), 2);
}

#[test]
fn candidates_are_in_world_space() {
    // The object's transform has to be applied, or a snap catches where the mesh
    // would be if it sat at the origin — which for any object that has been
    // moved is nowhere the user is looking.
    let shifted = DAffine3::from_translation(DVec3::new(10.0, 0.0, 0.0));
    let mut world = SnapWorld::default();
    world.add_mesh(&quad_mesh(), &shifted, &[]);
    assert!(
        world.verts.iter().all(|v| v.x >= 10.0 - 1e-9),
        "{:?}",
        world.verts
    );
}

#[test]
fn the_snap_modes_cycle_back_to_where_they_started() {
    let mut mode = Mode::Increment;
    let mut seen = vec![mode];
    for _ in 0..5 {
        mode = mode.next();
        seen.push(mode);
    }
    assert_eq!(seen[5], Mode::Face);
    assert_eq!(mode.next(), Mode::Increment, "the cycle should close");
    // And every mode is reachable exactly once.
    let mut sorted: Vec<&str> = seen.iter().map(|m| m.label()).collect();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 6);
}

#[test]
fn only_increment_mode_needs_no_geometry() {
    assert!(!Mode::Increment.needs_geometry());
    for mode in [Mode::Vertex, Mode::Edge, Mode::EdgeMidpoint, Mode::Face] {
        assert!(
            mode.needs_geometry(),
            "{} claims to need none",
            mode.label()
        );
    }
}

#[test]
fn projection_is_camera_relative() {
    // §4.10 builds the view projection with the camera at the origin, so a scene
    // far from the world origin does not jitter. A projection that forgot to
    // subtract the eye would be wrong by exactly the camera's position — which
    // is invisible when the camera happens to be at the origin, and wrong
    // everywhere else.
    let mut far = camera();
    far.pivot = DVec3::new(1000.0, 0.0, 0.0);
    let point = DVec3::new(1000.0, 0.0, 0.0);
    let px = snap::project(point, &far, VIEWPORT).expect("in view");
    assert!(
        (px.0 - VIEWPORT.0 * 0.5).abs() < 1.0 && (px.1 - VIEWPORT.1 * 0.5).abs() < 1.0,
        "the camera's own pivot did not project to the centre: {px:?}"
    );
}

/// Vertex, then edge, then face: the smaller thing is the more deliberate aim.
#[test]
fn element_snapping_takes_a_vertex_then_an_edge_then_a_face() {
    let a = DVec3::new(-1.0, 0.0, -1.0);
    let b = DVec3::new(1.0, 0.0, -1.0);
    let c = DVec3::new(0.0, 0.0, 1.0);
    let world = SnapWorld {
        verts: vec![a, b, c],
        edges: vec![[a, b], [b, c], [c, a]],
        tris: vec![[a, b, c]],
    };
    let hit =
        |p: DVec3| snap::resolve(&world, Mode::Element, at_point(p), &camera(), VIEWPORT).unwrap();

    let vertex = hit(c);
    assert_eq!(vertex.element, snap::Element::Vertex);
    assert!((vertex.point - c).length() < 1e-9);

    let edge = hit((a + b) * 0.5);
    assert_eq!(edge.element, snap::Element::Edge);
    assert!((edge.point - (a + b) * 0.5).length() < 1e-6);

    let middle = DVec3::new(0.0, 0.0, -0.3);
    let face = hit(middle);
    assert_eq!(face.element, snap::Element::Face);
    assert!((face.point - middle).length() < 1e-6, "{:?}", face.point);

    let far = at_point(DVec3::new(4.0, 0.0, 3.0));
    assert!(snap::resolve(&world, Mode::Element, far, &camera(), VIEWPORT).is_none());
}
