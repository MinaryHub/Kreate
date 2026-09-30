//! Viewport navigation, measured the way a user experiences it.
//!
//! No GPU: these are camera arithmetic and ray queries, and the thing worth
//! asserting is not "the pivot moved" but "the point the user was looking at
//! stayed where it was". A test of the internals would pass with the feature
//! subtly backwards.

use glam::Vec2;
use kreate::pick::PickCache;
use kreate_math::DVec3;
use kreate_render::Camera;
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::{Document, ModifierCache};

const ASPECT: f64 = 16.0 / 9.0;

/// The evaluated scene, as the application hands it to the picker.
///
/// Ray queries run against the geometry the user can see, so they take the
/// evaluated meshes rather than reading the document's datablocks.
fn evaluated(doc: &Document) -> Vec<(ObjectId, std::sync::Arc<kreate_mesh::MeshData>)> {
    ModifierCache::new().evaluate_scene(doc, &kreate_scene::modifier::EvalCtx::still())
}

/// A cube off to one side, which is the situation auto-depth exists for: the
/// thing being worked on is not at the origin the camera orbits about.
fn scene() -> (Document, ObjectId) {
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let object = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.object_mut(object).unwrap().transform.translation = DVec3::new(6.0, 0.0, 0.0);
    (doc, object)
}

/// Where a world point lands on screen, as normalised device coordinates.
///
/// `None` when it is behind the camera. The view matrix is camera-relative — the
/// camera sits at the origin and everything is rebased in `f64` first — so the
/// point has to be rebased the same way before it is projected.
fn project(camera: &Camera, point: DVec3) -> Option<Vec2> {
    let relative = (point - camera.eye()).as_vec3();
    let clip = camera.view_projection(ASPECT) * relative.extend(1.0);
    (clip.w > 1e-6).then(|| Vec2::new(clip.x / clip.w, clip.y / clip.w))
}

/// The NDC of a point, found by aiming the camera's own ray generator at it.
///
/// Rather than inverting the projection: this is the same function picking uses,
/// so if it disagreed with `project` the test would be measuring the disagreement
/// instead of the navigation.
fn ndc_of(camera: &Camera, point: DVec3) -> Vec2 {
    project(camera, point).expect("the point is in front of the camera")
}

#[test]
fn a_ray_through_the_cursor_finds_what_is_drawn_there() {
    // The bridge between the two halves: the camera's ray generator and the
    // BVH must agree, or auto-depth aims at the wrong thing.
    let (doc, _) = scene();
    let camera = Camera::default();
    let mut pick = PickCache::new();

    let centre = DVec3::new(6.0, 0.0, 0.0);
    let ndc = ndc_of(&camera, centre);
    let hit = pick
        .raycast_ndc(&doc, &evaluated(&doc), &camera, ndc, ASPECT)
        .expect("the cube is under that point on screen");

    // The hit is on the cube's surface, in front of its centre.
    assert!(
        (hit.point - centre).length() <= 1.8,
        "hit at {:?}, which is not on the cube",
        hit.point
    );
    let back = project(&camera, hit.point).unwrap();
    assert!(
        (back - ndc).length() < 1e-3,
        "the hit does not project back to where the ray was aimed"
    );
}

#[test]
fn auto_depth_holds_a_surface_the_camera_is_working_close_to() {
    // The case the feature is really for: the pivot is out at the scene's
    // centre, and the user is up against a surface much nearer than that.
    // Orbiting about the far pivot swings the near surface across the screen for
    // a small mouse movement — the drunk feeling. Orbiting about the surface's
    // own depth removes it.
    let camera = Camera::default();
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let object = doc.add_object(Object::new("Near", ObjectData::Mesh(mesh)));
    // Right in the middle of the view, at well under the orbit distance.
    let near = camera.eye() + camera.forward() * 3.0;
    doc.object_mut(object).unwrap().transform.translation = near;

    let mut pick = PickCache::new();
    let ndc = Vec2::ZERO;
    let hit = pick
        .raycast_ndc(&doc, &evaluated(&doc), &camera, ndc, ASPECT)
        .expect("the near cube is dead centre");
    assert!(
        hit.distance < camera.distance * 0.5,
        "the surface has to be much nearer than the pivot: {} against {}",
        hit.distance,
        camera.distance
    );

    let orbit = (0.25, 0.15);
    let mut without = camera;
    without.orbit(orbit.0, orbit.1);
    let drift_without = (project(&without, hit.point).unwrap() - ndc).length();

    let mut with = camera;
    with.set_pivot_depth(with.depth_of(hit.point).unwrap());
    with.orbit(orbit.0, orbit.1);
    let drift_with = (project(&with, hit.point).unwrap() - ndc).length();

    assert!(
        drift_without > 0.1,
        "the test needs a case that actually swings: {drift_without:.4}"
    );
    assert!(
        drift_with < 0.01,
        "the surface moved {drift_with:.4} of the screen, against {drift_without:.4} without"
    );
}

#[test]
fn auto_depth_at_least_halves_the_drift_off_centre() {
    // The honest limit. The pivot goes to the hit's *depth* along the view axis,
    // not to the hit point itself, so a target far off to one side still swings
    // some — it is on a larger circle about the new centre.
    //
    // Moving the pivot sideways to the exact hit would remove the rest, and is
    // rejected on purpose: it drags the eye with it, so the view would jump the
    // instant the button went down. A jump on press is worse than a drift during
    // a drag, because the user did not ask for anything yet.
    let (doc, _) = scene();
    let mut pick = PickCache::new();

    let base = Camera::default();
    let ndc = ndc_of(&base, DVec3::new(6.0, 0.0, 0.0));
    assert!(ndc.length() > 0.3, "this test wants an off-centre target");
    let hit = pick
        .raycast_ndc(&doc, &evaluated(&doc), &base, ndc, ASPECT)
        .unwrap();

    let orbit = (0.25, 0.15);

    let mut without = base;
    without.orbit(orbit.0, orbit.1);
    let drift_without = (project(&without, hit.point).unwrap() - ndc).length();

    let mut with = base;
    with.set_pivot_depth(with.depth_of(hit.point).unwrap());
    with.orbit(orbit.0, orbit.1);
    let drift_with = (project(&with, hit.point).unwrap() - ndc).length();

    assert!(
        drift_with < drift_without * 0.5,
        "auto-depth barely helped: {drift_with:.4} against {drift_without:.4}"
    );
}

#[test]
fn setting_the_pivot_does_not_move_the_picture() {
    // The half of auto-depth that must be invisible. If placing the pivot
    // shifted the view, the user would see a jump the instant they pressed the
    // button — which reads as a bug, not as a feature.
    let (doc, _) = scene();
    let mut pick = PickCache::new();
    let before = Camera::default();
    let ndc = ndc_of(&before, DVec3::new(6.0, 0.0, 0.0));
    let hit = pick
        .raycast_ndc(&doc, &evaluated(&doc), &before, ndc, ASPECT)
        .unwrap();

    let mut after = before;
    after.set_pivot_depth(after.depth_of(hit.point).unwrap());

    assert!((after.eye() - before.eye()).length() < 1e-9);
    for probe in [
        DVec3::new(6.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(-3.0, 4.0, 2.0),
    ] {
        let a = project(&before, probe);
        let b = project(&after, probe);
        match (a, b) {
            (Some(a), Some(b)) => assert!(
                (a - b).length() < 1e-5,
                "{probe:?} moved from {a:?} to {b:?}"
            ),
            (None, None) => {}
            _ => panic!("{probe:?} appeared or vanished"),
        }
    }
}

#[test]
fn a_miss_leaves_the_pivot_alone() {
    // Orbiting over empty space keeps turning about whatever it was turning
    // about before. Falling back to the origin would make the view jump
    // whenever the cursor strayed off the model — the opposite of the point.
    let (doc, _) = scene();
    let mut pick = PickCache::new();
    let camera = Camera::default();

    // A corner of the screen, well clear of the cube.
    let empty = Vec2::new(-0.95, -0.95);
    assert!(
        pick.raycast_ndc(&doc, &evaluated(&doc), &camera, empty, ASPECT)
            .is_none(),
        "the test needs a ray that misses"
    );
}

#[test]
fn zooming_toward_the_cursor_keeps_that_point_in_place() {
    // Zoom-to-cursor, measured the same way: the thing being pointed at should
    // stay under the pointer as the view closes in.
    let (doc, _) = scene();
    let mut pick = PickCache::new();
    let base = Camera::default();
    let ndc = ndc_of(&base, DVec3::new(6.0, 0.0, 0.0));
    let hit = pick
        .raycast_ndc(&doc, &evaluated(&doc), &base, ndc, ASPECT)
        .unwrap();
    let direction = base.ray_direction(ndc, ASPECT).as_dvec3();

    let mut plain = base;
    plain.dolly(4.0);
    let drift_plain = (project(&plain, hit.point).unwrap() - ndc).length();

    let mut aimed = base;
    aimed.dolly_towards(4.0, direction);
    let drift_aimed = (project(&aimed, hit.point).unwrap() - ndc).length();

    assert!(
        aimed.distance < base.distance,
        "the view has to actually close in"
    );
    assert!(
        drift_aimed < drift_plain,
        "zoom-to-cursor drifted {drift_aimed:.4} against a plain dolly's {drift_plain:.4}"
    );
}

#[test]
fn the_pick_cache_survives_an_edit_between_queries() {
    // Navigation asks the cache constantly and never tells it anything changed.
    // If invalidation depended on being told, auto-depth would aim at where the
    // model used to be — a failure that only shows up after an edit.
    let (mut doc, object) = scene();
    let mut pick = PickCache::new();
    let camera = Camera::default();
    let ndc = ndc_of(&camera, DVec3::new(6.0, 0.0, 0.0));
    let before = pick
        .raycast_ndc(&doc, &evaluated(&doc), &camera, ndc, ASPECT)
        .unwrap();

    let mesh = doc.object(object).unwrap().mesh().unwrap();
    for p in doc.mesh_mut(mesh).unwrap().positions_mut() {
        *p *= 0.25;
    }
    let scene = evaluated(&doc);
    let after = pick.raycast_ndc(&doc, &scene, &camera, ndc, ASPECT);

    // Shrinking the cube can also take it out from under that exact ray, which
    // is equally proof the cache noticed.
    if let Some(after) = after {
        assert!(
            (after.point - before.point).length() > 0.1,
            "the cache answered from the old geometry"
        );
    }

    // The edited mesh is a different handle, so the tree for the old one is left
    // behind rather than replaced. It has to actually go when swept, or the
    // cache grows by one mesh for every edit the user makes.
    assert_eq!(pick.len(), 2, "both the old tree and the new one");
    pick.retain_live(&scene);
    assert_eq!(pick.len(), 1, "the stale tree is collected");
}

// --- keymap integration -----------------------------------------------------

/// Ids the application handles itself rather than sending to the registry.
const APP_ACTIONS: &[&str] = &[
    // The 3D cursor: placed from the pointer and the selection, which the
    // application knows and the operator (`scene.cursor_set`) is handed.
    "view.cursor_place",
    "view.cursor_to_origin",
    "view.cursor_to_selected",
    "view.selection_to_cursor",
    // Shift+A and Ctrl+A: a named menu under the pointer.
    "app.menu",
    // C: the selection brush, which is the application's until a stroke ends.
    "view.circle_select",
    // Views of the scene, which are the application's and not the document's.
    "view.local_view",
    "view.camera",
    "view.frame_selected",
    "view.toggle_statistics",
    "view.toggle_face_orientation",
    // File commands are the application's, not the document's: there is no
    // undo step for "the document you had open is gone".
    "file.new",
    "file.open",
    "file.save",
    "file.save_as",
    "view.orbit",
    "view.pan",
    "view.dolly",
    "view.frame_all",
    "view.toggle_edit_mode",
    "view.select_mode",
    "view.toggle_auto_depth",
    "view.toggle_grid",
    "view.toggle_wireframe",
    "view.toggle_sidebar",
    "view.cycle_shading",
    "view.toggle_projection",
    // The two standing transform settings, which are application state rather
    // than document state: no undo step restores "snapping was off".
    "view.toggle_snap",
    "view.toggle_proportional",
    "view.cycle_snap_mode",
    "edit.undo",
    // Copy edits nothing; paste is given the clipboard's text by the
    // application before the `object.paste` operator runs.
    "edit.copy",
    "edit.paste",
    "edit.redo",
    "edit.adjust_last",
    "app.print_script",
    "app.search",
    "app.pie",
    "time.play",
    "time.set_frame",
    "time.step",
    "time.jump",
    "time.set_range",
    "time.toggle_autokey",
    "edit.copy_script",
    "file.run_script",
    "object.add_next_primitive",
];

#[test]
fn the_sidebar_key_belongs_to_the_viewport_and_not_to_every_panel() {
    // `N` opens the 3D viewport's sidebar, and the UV editor will want its own
    // `N` for its own. So the binding lives in `Context::Viewport`: in `Global`
    // it would answer over every panel, and the UV editor could then never have
    // it without a conflict.
    //
    // Written as a claim about *resolution* rather than about the table,
    // because that is what the user meets. A sweep moving the binding to
    // `Global` broke nothing until this existed.
    let keymap = kreate_input::presets::load(kreate_input::Preset::Kreate);
    let press = |context: kreate_input::Context| -> Option<String> {
        keymap
            .resolve(
                context,
                &kreate_input::keymap::Input::Key("N".into()),
                kreate_input::keymap::Modifiers::default(),
                kreate_input::keymap::EventKind::Press,
            )
            .map(|action| action.op_id.clone())
    };

    for context in [
        kreate_input::Context::ObjectMode,
        kreate_input::Context::EditMesh,
    ] {
        assert_eq!(
            press(context).as_deref(),
            Some("view.toggle_sidebar"),
            "`N` does not open the sidebar in {context:?}"
        );
    }
    assert_eq!(
        press(kreate_input::Context::UvEditor),
        None,
        "`N` is taken over the UV editor, which needs it for its own sidebar"
    );
}

#[test]
fn every_binding_names_something_that_exists() {
    // The failure a keymap makes easy: a typo in an operator id is a key that
    // silently does nothing, in a table nobody reads. Checked for every preset,
    // because a preset only used by other people is exactly where such a typo
    // survives.
    let mut registry = kreate_ops::OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);

    for preset in kreate_input::presets::ALL.iter().copied() {
        let keymap = kreate_input::presets::load(preset);
        for binding in keymap.bindings() {
            let id = &binding.action.op_id;
            let known = APP_ACTIONS.contains(&id.as_str()) || registry.get(id).is_some();
            assert!(
                known,
                "{preset:?} binding `{}` names `{id}`, which is neither an \
                 operator nor an action the application handles",
                binding.id.0
            );
        }
    }
}

#[test]
fn every_binding_parameter_is_one_its_operator_declares() {
    // The other half of the same failure: a parameter name that no longer
    // exists is silently ignored, so the key runs the operator with its
    // defaults and the user sees the wrong amount of something.
    let mut registry = kreate_ops::OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);

    for preset in kreate_input::presets::ALL.iter().copied() {
        let keymap = kreate_input::presets::load(preset);
        for binding in keymap.bindings() {
            let Some(info) = registry.get(&binding.action.op_id) else {
                continue;
            };
            let declared: Vec<&str> = info.schema.iter().map(|d| d.name).collect();
            for (name, _) in binding.action.params.iter() {
                assert!(
                    declared.contains(&name),
                    "{preset:?} binding `{}` sets `{name}`, which `{}` does not \
                     declare; it has {declared:?}",
                    binding.id.0,
                    binding.action.op_id
                );
            }
        }
    }
}

// --- navigation gizmo -------------------------------------------------------

#[test]
fn the_gizmo_points_the_same_way_the_scene_does() {
    // The failure mode this exists for: getting the camera rotation's inverse
    // the wrong way round mirrors every axis. The widget still looks like a
    // plausible axis widget — three coloured spokes at sensible angles — and is
    // wrong only once the model turns, at which point it turns the other way.
    //
    // Checked against the real projection rather than against a hand-computed
    // answer, so it also catches a screen-`y` sign error.
    use kreate_ui::nav_gizmo;

    let mut camera = Camera::default();
    for orbit in [(0.0, 0.0), (0.6, 0.2), (-1.4, -0.5), (2.6, 0.9)] {
        camera.orbit(orbit.0, orbit.1);
        let widget_centre = egui::Pos2::new(400.0, 100.0);
        let radius = 40.0;
        let balls = nav_gizmo::layout(camera.rotation, widget_centre, radius);

        let origin_ndc = project(&camera, camera.pivot).expect("the pivot is in view");
        for ball in &balls {
            let axis = ball.axis.direction();
            // A point a little way along the axis from the pivot, projected the
            // way the scene is.
            let probe = camera.pivot + axis * camera.distance * 0.2;
            let Some(probe_ndc) = project(&camera, probe) else {
                continue;
            };
            // NDC is not isotropic: `x` is divided by the aspect ratio, so a
            // direction read straight out of it is squashed horizontally. The
            // widget is square, so the comparison has to undo that first —
            // otherwise every axis appears to disagree by a few degrees and the
            // test looks like a real defect.
            let scene_dir = (probe_ndc - origin_ndc) * Vec2::new(ASPECT as f32, 1.0);
            if scene_dir.length() < 1e-3 {
                // The axis points nearly at or away from the camera, so it has
                // no screen direction to compare against.
                continue;
            }
            // NDC has +y up; the widget is in screen coordinates with +y down.
            let widget_dir = ball.centre - widget_centre;
            let widget_dir = Vec2::new(widget_dir.x, -widget_dir.y);
            if widget_dir.length() < 1e-3 {
                continue;
            }

            let agreement = scene_dir.normalize().dot(widget_dir.normalize());
            assert!(
                agreement > 0.98,
                "{:?} runs {scene_dir:?} in the scene but {widget_dir:?} on the \
                 widget (agreement {agreement:.3})",
                ball.axis
            );
        }
    }
}

#[test]
fn clicking_an_axis_ball_gives_that_axis_view() {
    // The click path end to end: layout, hit-test, and the camera it asks for.
    use kreate_ui::nav_gizmo;

    let camera = Camera::default();
    let widget_centre = egui::Pos2::new(400.0, 100.0);
    let radius = 40.0;
    let balls = nav_gizmo::layout(camera.rotation, widget_centre, radius);

    for ball in &balls {
        let Some(axis) = nav_gizmo::hit(&balls, ball.centre, nav_gizmo::ball_radius(radius)) else {
            panic!("{:?} is not clickable at its own centre", ball.axis);
        };
        // Overlapping balls resolve to the nearer one, which is the documented
        // rule; what must hold for every ball is that the axis it resolves to
        // produces a view along that axis.
        let snapped = camera.looking_along(axis);
        assert!(
            (snapped.forward() + axis.direction()).length() < 1e-9,
            "{axis:?} did not produce a view along itself"
        );
    }
}

#[test]
fn snapping_is_animated_and_lands_on_the_axis() {
    // Together with the motion layer: an axis snap is a timed move, not a jump.
    use kreate_render::CameraMotion;

    let mut current = Camera::default();
    let destination = current.looking_along(kreate_render::ViewAxis::PosX);
    let mut motion = CameraMotion::new(current);
    motion.animate_to(current, destination, kreate_render::DEFAULT_TRANSITION);

    let mut frames = 0;
    while motion.advance(&mut current, 1.0 / 60.0) {
        frames += 1;
        assert!(frames < 600, "the snap never finished");
    }
    assert!(frames > 5, "an axis snap must not teleport");
    assert!((current.forward() + DVec3::X).length() < 1e-9);
}

#[test]
fn every_menu_entry_runs_something_that_exists() {
    // The menu tree names what each entry runs as a string, and a typo there
    // is a menu entry — and a search result — that does nothing. Checked in
    // both modes, against the registry and the application's own actions.
    let mut registry = kreate_ops::OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let catalog = kreate::shell::load_catalog("en");
    for mode in kreate_ops::Mode::ALL {
        let menus = kreate::shell::searchable_menus(&catalog, mode, 0, true, true);
        for (path, entry) in kreate::shell::flatten(&menus) {
            let Some(id) = entry.runs() else { continue };
            assert!(
                APP_ACTIONS.contains(&id) || registry.get(id).is_some(),
                "{mode:?}: {path} / {} runs `{id}`, which nothing handles",
                entry.label
            );
        }
    }
}
