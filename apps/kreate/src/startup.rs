//! What a new document contains.

use kreate_math::{DMat3, DQuat, DVec3};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

/// What a new document contains.
///
/// Nothing to see: no geometry, no lamp, no camera — the viewport starts
/// empty, ready for whatever is added or imported. Only the world, which is
/// not an object and draws nothing: a dim grey sky, so that something added
/// and looked at in the rendered mode is lit rather than black. §2.1 is
/// emphatic that a scene's light comes from the scene, so the renderer does
/// not invent any.
/// The view a fresh start shows: the renderer's default angle, looked at
/// orthographically, as a part is drawn — no foreshortening to judge a size
/// through. `5` on the numpad switches to perspective.
pub fn starting_camera() -> kreate_render::Camera {
    kreate_render::Camera {
        projection: kreate_render::Projection::Orthographic,
        ..kreate_render::Camera::default()
    }
}

pub fn starting_scene(doc: &mut Document) {
    let world = doc.add_world(kreate_scene::World::new("World"));
    doc.set_active_world(Some(world));
    // Lengths shown and typed in millimetres, as a part is drawn. Only what is
    // shown: lengths are stored in metres whatever this says, and a file that
    // names no unit is still read as metres, so documents saved before this
    // open as they were.
    doc.design.length_unit = kreate_scene::design::LengthUnit::Millimetres;
}

/// A scene that renders: a 2 m cube at the origin, selected, a lamp above and
/// to one side, and a camera looking at it — what a new document held before
/// it started empty, and what the tests that render a whole scene draw.
///
/// Every piece goes through the same constructors the Add menu uses, so there
/// is one definition of what a lamp is and this cannot drift from what the
/// menu produces.
pub fn sample_scene(doc: &mut Document) {
    starting_scene(doc);
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let cube = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));

    // Above and to one side, which is where a key light goes and where it casts
    // a shadow the camera can see. Directly overhead puts the shadow under the
    // object, which is a lighting setup nobody chooses.
    let bulb = doc.add_light(kreate_scene::Light::of_kind(
        "Lamp",
        kreate_scene::LightKind::BULB,
    ));
    let mut lamp = Object::new("Lamp", ObjectData::Light(bulb));
    lamp.transform.translation = DVec3::new(3.0, -3.0, 5.0);
    // Turned to the cube: a bulb shines every way, but a lamp switched to a
    // spot or a sun keeps the aim, and its glyph points where it lights.
    lamp.transform.rotation = looking_at(lamp.transform.translation, DVec3::ZERO);
    doc.add_object(lamp);

    // The camera is the scene's, not the viewport's — it is what the exposure
    // is metered for and what a render is made through.
    let camera = doc.add_camera(kreate_scene::Camera::new("Camera"));
    let mut eye = Object::new("Camera", ObjectData::Camera(camera));
    eye.transform.translation = DVec3::new(7.0, -7.0, 5.0);
    eye.transform.rotation = looking_at(eye.transform.translation, DVec3::ZERO);
    doc.add_object(eye);

    doc.selection.add(cube);
}

/// The rotation that points an object's −Z at `target` from `from`, with its
/// +Y as near world up as it can be — Blender's convention for where a camera
/// and a lamp look.
pub fn looking_at(from: DVec3, target: DVec3) -> DQuat {
    let forward = (target - from).normalize_or_zero();
    if forward == DVec3::ZERO {
        return DQuat::IDENTITY;
    }
    // Straight up or down has no "right" from world up; any will do.
    let right = forward.cross(DVec3::Z).try_normalize().unwrap_or(DVec3::X);
    let up = right.cross(forward);
    DQuat::from_mat3(&DMat3::from_cols(right, up, -forward)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new document has no objects at all, only the world that lights
    /// what is added.
    #[test]
    fn a_new_document_is_empty_but_lit() {
        let mut doc = Document::new();
        starting_scene(&mut doc);
        assert_eq!(doc.objects().count(), 0, "a new document starts empty");
        assert!(doc.selection.is_empty());
        assert!(doc.active_world().is_some());
    }

    /// The sample scene's camera and lamp both point at its cube.
    #[test]
    fn the_sample_camera_and_lamp_look_at_the_cube() {
        let mut doc = Document::new();
        sample_scene(&mut doc);
        let mut aimed = 0;
        for (_, object) in doc.objects() {
            if !matches!(object.data, ObjectData::Camera(_) | ObjectData::Light(_)) {
                continue;
            }
            let t = &object.transform;
            let forward = t.rotation * DVec3::NEG_Z;
            let to_cube = (DVec3::ZERO - t.translation).normalize();
            assert!(
                forward.dot(to_cube) > 0.9999,
                "{} looks along {forward:?}",
                object.name
            );
            // Upright: the camera's up has a positive world Z.
            assert!(
                (t.rotation * DVec3::Y).z > 0.0,
                "{} is upside down",
                object.name
            );
            aimed += 1;
        }
        assert_eq!(aimed, 2);
    }
}
