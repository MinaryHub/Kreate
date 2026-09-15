//! What a new document contains.

use kreate_math::DVec3;
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
    doc.add_object(eye);

    // The cube, because that is what somebody opening the application is going
    // to do something to.
    doc.selection.add(cube);
}
