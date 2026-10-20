//! The scene camera, and the timeline's markers that cut between cameras.
//!
//! Blender's way, both: Set Active Camera picks the camera the scene is seen
//! and rendered through, and a marker with a camera bound switches to that
//! camera from its frame on — how one animation is shot from several places.
//! Which camera wins at a frame is the document's to say
//! ([`kreate_scene::Document::camera_at`]); these only set what it reads.

use kreate_scene::animation::Marker;
use kreate_scene::document::ObjectId;
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<SetSceneCamera>();
    registry.register::<AddMarker>();
    registry.register::<RemoveMarker>();
    registry.register::<BindCamera>();
}

/// The camera called `name`, or with no name the active object when it is a
/// camera.
fn camera_named(doc: &Document, name: &str) -> Option<ObjectId> {
    if name.is_empty() {
        return doc.selection.active.filter(|a| doc.is_camera(*a));
    }
    doc.objects()
        .find(|(id, o)| o.name == name && doc.is_camera(*id))
        .map(|(id, _)| id)
}

/// A frame as a marker holds it: whole.
fn whole(frame: f64) -> Option<i32> {
    frame
        .is_finite()
        .then(|| frame.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetSceneCameraParams {
    /// The camera's name; empty for the active object.
    pub camera: String,
}

/// Ctrl+Numpad 0: the scene is seen and rendered through this camera.
pub struct SetSceneCamera;

impl Operator for SetSceneCamera {
    type Params = SetSceneCameraParams;
    fn id() -> &'static str {
        "scene.camera_set"
    }
    fn label() -> &'static str {
        "Set Active Camera"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.objects().any(|(id, _)| doc.is_camera(id))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(camera) = camera_named(doc.document(), &params.camera) else {
            return Err(OpError::Failed(if params.camera.is_empty() {
                "the active object is not a camera".into()
            } else {
                format!("no camera called {:?}", params.camera)
            }));
        };
        if doc.document().scene_camera == Some(camera) {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().scene_camera = Some(camera);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddMarkerParams {
    /// Which frame. The application fills in the one the timeline is on.
    pub frame: f64,
    /// Its name; empty names it after its frame, as Blender does.
    pub name: String,
}

/// M: a marker on the frame.
pub struct AddMarker;

impl Operator for AddMarker {
    type Params = AddMarkerParams;
    fn id() -> &'static str {
        "marker.add"
    }
    fn label() -> &'static str {
        "Add Marker"
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(frame) = whole(params.frame) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document();
        let existing = d.markers.iter().find(|m| m.frame == frame);
        let name = if params.name.is_empty() {
            // i18n-exempt: Blender's marker name, a frame number
            existing.map_or_else(|| format!("F_{frame:02}"), |m| m.name.clone())
        } else {
            params.name.clone()
        };
        if existing.is_some_and(|m| m.name == name) {
            return Ok(OpResult::Cancelled);
        }
        let camera = existing.and_then(|m| m.camera);
        doc.document_mut().set_marker(Marker {
            frame,
            name,
            camera,
        });
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RemoveMarkerParams {
    /// The frame whose marker goes.
    pub frame: f64,
}

/// Takes away the marker on a frame, and with it the cut it made.
pub struct RemoveMarker;

impl Operator for RemoveMarker {
    type Params = RemoveMarkerParams;
    fn id() -> &'static str {
        "marker.remove"
    }
    fn label() -> &'static str {
        "Remove Marker"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.markers.is_empty()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(frame) = whole(params.frame) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(i) = doc.document().markers.iter().position(|m| m.frame == frame) else {
            return Ok(OpResult::Cancelled);
        };
        doc.document_mut().markers.remove(i);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct BindCameraParams {
    /// Which frame's marker; one is put there if there is none.
    pub frame: f64,
    /// The camera's name; empty for the active object.
    pub camera: String,
}

/// Ctrl+B in Blender's timeline: from this frame on, the scene is seen
/// through this camera.
pub struct BindCamera;

impl Operator for BindCamera {
    type Params = BindCameraParams;
    fn id() -> &'static str {
        "marker.camera_bind"
    }
    fn label() -> &'static str {
        "Bind Camera to Marker"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.objects().any(|(id, _)| doc.is_camera(id))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(frame) = whole(params.frame) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document();
        let Some(camera) = camera_named(d, &params.camera) else {
            return Err(OpError::Failed(if params.camera.is_empty() {
                "the active object is not a camera".into()
            } else {
                format!("no camera called {:?}", params.camera)
            }));
        };
        let existing = d.markers.iter().find(|m| m.frame == frame);
        if existing.is_some_and(|m| m.camera == Some(camera)) {
            return Ok(OpResult::Cancelled);
        }
        // A new marker is named after its camera: on the timeline the cut
        // reads as where it cuts to.
        let name = existing.map_or_else(
            || d.object(camera).map(|o| o.name.clone()).unwrap_or_default(),
            |m| m.name.clone(),
        );
        doc.document_mut().set_marker(Marker {
            frame,
            name,
            camera: Some(camera),
        });
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::OperatorRegistry;
    use kreate_reflect::{PropertyMap, PropertyValue as V};

    fn run(doc: &mut Document, op: &str, pairs: &[(&'static str, V)]) -> Result<(), OpError> {
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        History::new()
            .dispatch(&r, doc, &Context::default(), op, &p)
            .map(|_| ())
    }

    fn camera(doc: &mut Document, name: &str) -> ObjectId {
        let data = doc.add_camera(kreate_scene::Camera::new(name));
        doc.add_object(kreate_scene::document::Object::new(
            name,
            kreate_scene::document::ObjectData::Camera(data),
        ))
    }

    fn named(name: &str) -> (&'static str, V) {
        ("camera", V::Text(name.into()))
    }

    #[test]
    fn markers_cut_between_cameras_and_the_scene_camera_fills_in() {
        let mut doc = Document::new();
        let a = camera(&mut doc, "A");
        let b = camera(&mut doc, "B");
        assert_eq!(doc.camera_at(1.0), None, "nothing chosen: the caller picks");

        run(&mut doc, "scene.camera_set", &[named("B")]).unwrap();
        assert_eq!(doc.camera_at(50.0), Some(b));

        run(
            &mut doc,
            "marker.camera_bind",
            &[("frame", V::Float(10.0)), named("A")],
        )
        .unwrap();
        run(
            &mut doc,
            "marker.camera_bind",
            &[("frame", V::Float(40.0)), named("B")],
        )
        .unwrap();
        let names: Vec<&str> = doc.markers.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["A", "B"]);
        assert_eq!(
            doc.camera_at(1.0),
            Some(a),
            "before the first marker, its camera"
        );
        assert_eq!(doc.camera_at(10.0), Some(a));
        assert_eq!(
            doc.camera_at(39.9999999),
            Some(b),
            "a hair short of 40 is 40"
        );
        assert_eq!(doc.camera_at(39.5), Some(a));
        assert_eq!(doc.camera_at(100.0), Some(b));

        // A plain marker cuts to nothing; removing a bound one undoes its cut.
        run(&mut doc, "marker.add", &[("frame", V::Float(20.0))]).unwrap();
        assert_eq!(doc.markers[1].name, "F_20");
        assert_eq!(doc.camera_at(25.0), Some(a));
        run(&mut doc, "marker.remove", &[("frame", V::Float(40.0))]).unwrap();
        assert_eq!(doc.camera_at(100.0), Some(a));
        let frames: Vec<i32> = doc.markers.iter().map(|m| m.frame).collect();
        assert_eq!(frames, [10, 20]);
    }

    #[test]
    fn only_a_camera_can_be_the_scene_camera() {
        let mut doc = Document::new();
        camera(&mut doc, "Cam");
        assert!(run(&mut doc, "scene.camera_set", &[named("Nope")]).is_err());
        assert!(
            run(&mut doc, "scene.camera_set", &[]).is_err(),
            "nothing active"
        );
        assert_eq!(doc.scene_camera, None);
    }
}
