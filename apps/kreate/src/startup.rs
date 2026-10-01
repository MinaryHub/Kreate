//! What a new document contains.

use kreate_math::{DMat3, DQuat, DVec3};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

/// What a new document contains.
///
/// A cube, a lamp, a camera and a world — which is to say, a scene that can be
/// rendered. That is the whole reason for each of them.
///
/// A cube alone was the starting point for a long time, and it meant that
/// pressing `Z` into the rendered mode on a new document gave a black screen.
/// Correct, and useless: nothing was lighting anything. The fix is not to make
/// the renderer invent light — §2.1 is emphatic that a scene's light comes from
/// the scene — but to start the user with a scene that has some.
///
/// Every piece goes through the same constructors the Add menu uses, so there
/// is one definition of what a lamp is and this cannot drift from what the menu
/// produces.
pub fn starting_scene(doc: &mut Document) {
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

    // A dim grey sky. Not for the look of it: without a world the only thing
    // lighting a face that the lamp does not reach is nothing, and an object
    // with pure black on one side reads as broken rather than as unlit.
    let world = doc.add_world(kreate_scene::World::new("World"));
    doc.set_active_world(Some(world));

    // The camera is the scene's, not the viewport's — it is what the exposure
    // is metered for and what a render will eventually be made through.
    let camera = doc.add_camera(kreate_scene::Camera::new("Camera"));
    let mut eye = Object::new("Camera", ObjectData::Camera(camera));
    eye.transform.translation = DVec3::new(7.0, -7.0, 5.0);
    // Looking at the cube, so the view through it (Numpad 0) and a render
    // show the scene rather than the floor beneath the camera.
    eye.transform.rotation = looking_at(eye.transform.translation, DVec3::ZERO);
    doc.add_object(eye);

    // The cube, because that is what somebody opening the application is going
    // to do something to.
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

    /// The camera and the lamp of a new document both point at the cube.
    #[test]
    fn the_camera_and_the_lamp_look_at_the_cube() {
        let mut doc = Document::new();
        starting_scene(&mut doc);
        let mut aimed = 0;
        for (_, object) in doc.objects() {
            if !matches!(object.data, ObjectData::Camera(_) | ObjectData::Light(_)) {
                continue;
            }
            let t = &object.transform;
            let forward = t.rotation * DVec3::NEG_Z;
            let to_cube = (DVec3::ZERO - t.translation).normalize();
            assert!(forward.dot(to_cube) > 0.9999, "{} looks along {forward:?}", object.name);
            // Upright: the camera's up has a positive world Z.
            assert!((t.rotation * DVec3::Y).z > 0.0, "{} is upside down", object.name);
            aimed += 1;
        }
        assert_eq!(aimed, 2);
    }
}
