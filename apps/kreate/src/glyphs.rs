//! The markers drawn for objects that have no surface: lights and cameras.
//!
//! A lamp has a position and a direction and no mesh, so the solid pass has
//! nothing for it. Until something drew it, adding a light changed the
//! lighting and nothing else — in the studio shading, which does not use the
//! scene's lights, it changed nothing at all. A user who adds a light and sees
//! the viewport unchanged concludes the button is broken, and is right to.
//!
//! **Blender's glyphs**, because a modeller reads them without being told: a
//! ring for a point light with a dashed line dropped to the ground plane; the
//! ring with parallel rays for a sun; a cone for a spot; a square for an area
//! light; a pyramid for a camera, with a triangle on top saying which way is
//! up.
//!
//! **Built here, drawn by the renderer.** This turns the document into lines
//! and rings, the way `scenelights` turns it into `PlacedLight`s, and the
//! renderer draws them in its own pass against its depth — so a lamp behind a
//! wall is behind the wall. The interface painted these over the frame for a
//! while, and a lamp's ring in front of the wall it was behind is what that
//! looked like.

use kreate_math::DVec3;
use kreate_render::{palette, MarkerShape, OverlayLine, OverlayPoint};
use kreate_scene::document::ObjectData;
use kreate_scene::{Document, LightKind};

/// Radius of a lamp's ring on screen, in pixels.
const BODY_PX: f32 = 7.0;
/// Width of the glyph's lines, in pixels.
const LINE_PX: f32 = 1.2;

/// The lines and rings for every visible lamp and camera in `doc`.
///
/// Selected objects wear the accent, the rest the handle grey — the same two
/// colours a mesh's wireframe uses for the same two states, so a selected lamp
/// and a selected cube say "selected" the same way.
pub fn markers_for(doc: &Document) -> (Vec<OverlayLine>, Vec<OverlayPoint>) {
    markers_posing(doc, None)
}

/// The same, with `posing` the armature in pose mode, whose bones then say
/// which of *them* is selected rather than whether the object is.
pub fn markers_posing(
    doc: &Document,
    posing: Option<kreate_scene::ObjectId>,
) -> (Vec<OverlayLine>, Vec<OverlayPoint>) {
    let mut lines = Vec::new();
    let mut points = Vec::new();
    let accent = palette::display_rgba(palette::ACCENT);
    let mut dim = palette::display_rgba(palette::HANDLE);
    dim[3] = 0.85;

    for (id, object) in doc.objects() {
        if !object.visible {
            continue;
        }
        let color = if doc.selection.contains(id) { accent } else { dim };
        let transform = doc.world_transform(id);
        match &object.data {
            ObjectData::Light(light) => {
                if let Some(light) = doc.light(*light) {
                    lamp(&transform, light.kind, color, &mut lines, &mut points);
                }
            }
            ObjectData::Armature(_) => {
                armature_glyph(doc, id, posing == Some(id), color, dim, accent, &mut lines);
            }
            ObjectData::Camera(camera) => {
                if let Some(camera) = doc.camera(*camera) {
                    // The frame one unit ahead, from the lens: half the sensor
                    // over the focal length is the tangent of the half angle.
                    let f = camera.focal_length_mm.max(1.0);
                    let half = (
                        camera.sensor_width_mm * 0.5 / f,
                        camera.sensor_height_mm * 0.5 / f,
                    );
                    camera_glyph(&transform, half, color, &mut lines);
                }
            }
            _ => {}
        }
    }
    (lines, points)
}

/// Blender's octahedral bones: a diamond a tenth of the way up, joined to the
/// head below and the tail above.
///
/// In pose mode a bone wears the accent when it is selected and white when it
/// is the active one; otherwise every bone wears the object's colour.
fn armature_glyph(
    doc: &Document,
    object: kreate_scene::ObjectId,
    posing: bool,
    color: [f32; 4],
    dim: [f32; 4],
    accent: [f32; 4],
    lines: &mut Vec<OverlayLine>,
) {
    let (Some((_, armature)), Some(o)) = (doc.armature_of(object), doc.object(object)) else { return };
    let frames = doc.posed_bones(object);
    for (bone, frame) in armature.bones.iter().zip(&frames) {
        let tint = if !posing {
            color
        } else if o.pose.active.as_deref() == Some(bone.name.as_str()) && o.pose.is_selected(&bone.name) {
            [1.0, 1.0, 1.0, 1.0]
        } else if o.pose.is_selected(&bone.name) {
            accent
        } else {
            dim
        };
        let [head, a, b, c, d, tail] = kreate_scene::armature::octahedron(frame, bone.length());
        let ring = [a, b, c, d];
        for i in 0..4 {
            lines.push(line(head, ring[i], tint, false));
            lines.push(line(ring[i], ring[(i + 1) % 4], tint, false));
            lines.push(line(ring[i], tail, tint, false));
        }
    }
}

fn line(a: DVec3, b: DVec3, color: [f32; 4], dashed: bool) -> OverlayLine {
    OverlayLine {
        a,
        b,
        color,
        width_px: LINE_PX,
        dashed,
    }
}

/// A faint version of `color`, for the parts of a glyph that are not its body.
fn faint(color: [f32; 4]) -> [f32; 4] {
    [color[0], color[1], color[2], color[3] * 0.55]
}

fn lamp(
    transform: &kreate_math::DAffine3,
    kind: LightKind,
    color: [f32; 4],
    lines: &mut Vec<OverlayLine>,
    points: &mut Vec<OverlayPoint>,
) {
    let origin = transform.translation;
    // A light points down its own -Z: glTF's convention and Blender's.
    let forward = (transform.matrix3 * DVec3::NEG_Z).normalize_or_zero();

    // The body: a ring with a dot, or a square for an area light.
    points.push(OverlayPoint {
        position: origin,
        color,
        radius_px: BODY_PX,
        shape: match kind {
            LightKind::Area { .. } => MarkerShape::Square,
            _ => MarkerShape::Ring,
        },
        dot: !matches!(kind, LightKind::Area { .. }),
    });

    // The line to the ground: straight down to z = 0, dashed, which is how the
    // eye reads a height. Only when the light is off the ground.
    if origin.z.abs() > 1e-6 {
        let foot = DVec3::new(origin.x, origin.y, 0.0);
        lines.push(line(origin, foot, faint(color), true));
    }

    match kind {
        LightKind::Point { .. } => {}
        LightKind::Sun { .. } => {
            // Parallel rays: one through the body along the direction and two
            // beside it, which is what says "from everywhere at once".
            let side = perpendicular(forward);
            for k in -1..=1 {
                let offset = side * (k as f64 * 0.25);
                lines.push(line(
                    origin + offset,
                    origin + offset + forward * 1.2,
                    if k == 0 { color } else { faint(color) },
                    false,
                ));
            }
        }
        LightKind::Spot { cone_angle, .. } => {
            // The cone: a ring one and a half units along the direction, whose
            // radius is the cone's there, and four edges from the lamp to it.
            let length = 1.5;
            let radius = (cone_angle * 0.5).tan() * length;
            let (u, v) = basis(forward);
            let centre = origin + forward * length;
            let ring: Vec<DVec3> = (0..24)
                .map(|i| {
                    let t = i as f64 / 24.0 * std::f64::consts::TAU;
                    centre + (u * t.cos() + v * t.sin()) * radius
                })
                .collect();
            for i in 0..24 {
                lines.push(line(ring[i], ring[(i + 1) % 24], faint(color), false));
            }
            for i in [0usize, 6, 12, 18] {
                lines.push(line(origin, ring[i], faint(color), false));
            }
        }
        LightKind::Area { .. } => {
            lines.push(line(origin, origin + forward * 0.8, faint(color), false));
        }
    }
}

fn camera_glyph(
    transform: &kreate_math::DAffine3,
    half_extents: (f64, f64),
    color: [f32; 4],
    lines: &mut Vec<OverlayLine>,
) {
    let origin = transform.translation;
    let m = transform.matrix3;
    let forward = m * DVec3::NEG_Z;
    let right = m * DVec3::X;
    let up = m * DVec3::Y;
    let (hw, hh) = half_extents;
    let centre = origin + forward;
    let corners = [
        centre + right * hw + up * hh,
        centre - right * hw + up * hh,
        centre - right * hw - up * hh,
        centre + right * hw - up * hh,
    ];
    for i in 0..4 {
        lines.push(line(corners[i], corners[(i + 1) % 4], color, false));
        lines.push(line(origin, corners[i], faint(color), false));
    }
    // The up triangle, sitting on the top edge of the frame.
    let a = centre + up * hh + right * hw * 0.5;
    let b = centre + up * hh - right * hw * 0.5;
    let apex = centre + up * (hh * 1.7);
    lines.push(line(a, apex, color, false));
    lines.push(line(b, apex, color, false));
}

/// A unit vector perpendicular to `v`.
fn perpendicular(v: DVec3) -> DVec3 {
    let helper = if v.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    v.cross(helper).normalize_or_zero()
}

/// Two unit vectors perpendicular to `v` and to each other.
fn basis(v: DVec3) -> (DVec3, DVec3) {
    let u = perpendicular(v);
    (u, v.cross(u).normalize_or_zero())
}

#[cfg(test)]
mod armature_tests {
    use super::*;

    #[test]
    fn a_bone_is_twelve_lines_and_pose_mode_colours_by_bone() {
        let mut doc = Document::new();
        let a = doc.add_armature(kreate_scene::armature::Armature::new("Rig"));
        let rig = doc.add_object(kreate_scene::document::Object::new("Rig", ObjectData::Armature(a)));
        let (lines, _) = markers_for(&doc);
        assert_eq!(lines.len(), 12);
        // The tail is a unit up: the bone stands on its head.
        assert!(lines.iter().any(|l| (l.b - DVec3::Z).length() < 1e-9));

        doc.object_mut(rig).unwrap().pose.entry("Bone").selected = true;
        let accent = palette::display_rgba(palette::ACCENT);
        let (unposed, _) = markers_posing(&doc, None);
        assert!(unposed.iter().all(|l| l.color != accent), "an unselected armature looks selected");
        let (posed, _) = markers_posing(&doc, Some(rig));
        assert!(posed.iter().all(|l| l.color == accent), "a selected bone does not say so");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::document::Object;

    fn lamp_object(doc: &mut Document, kind: LightKind, at: DVec3) -> kreate_scene::ObjectId {
        let light = doc.add_light(kreate_scene::Light::of_kind("L", kind));
        let mut object = Object::new("L", ObjectData::Light(light));
        object.transform.translation = at;
        doc.add_object(object)
    }

    #[test]
    fn a_point_lamp_is_a_ring_and_a_line_to_the_ground() {
        let mut doc = Document::new();
        lamp_object(&mut doc, LightKind::BULB, DVec3::new(1.0, 2.0, 3.0));
        let (lines, points) = markers_for(&doc);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].position, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(points[0].shape, MarkerShape::Ring);
        assert_eq!(lines.len(), 1, "one line, to the ground");
        assert!(lines[0].dashed);
        assert_eq!(lines[0].b, DVec3::new(1.0, 2.0, 0.0), "the foot is straight below");
    }

    #[test]
    fn a_lamp_on_the_ground_has_no_line_to_it() {
        let mut doc = Document::new();
        lamp_object(&mut doc, LightKind::BULB, DVec3::new(1.0, 2.0, 0.0));
        let (lines, _) = markers_for(&doc);
        assert!(lines.is_empty());
    }

    #[test]
    fn a_hidden_lamp_draws_nothing() {
        let mut doc = Document::new();
        let id = lamp_object(&mut doc, LightKind::BULB, DVec3::new(0.0, 0.0, 2.0));
        doc.object_mut(id).unwrap().visible = false;
        let (lines, points) = markers_for(&doc);
        assert!(lines.is_empty() && points.is_empty());
    }

    #[test]
    fn a_selected_lamp_wears_the_accent() {
        let mut doc = Document::new();
        let id = lamp_object(&mut doc, LightKind::BULB, DVec3::new(0.0, 0.0, 2.0));
        let (_, unselected) = markers_for(&doc);
        doc.selection.add(id);
        let (_, selected) = markers_for(&doc);
        assert_eq!(selected[0].color, palette::display_rgba(palette::ACCENT));
        assert_ne!(selected[0].color, unselected[0].color);
    }

    #[test]
    fn each_kind_has_its_own_shape() {
        let mut doc = Document::new();
        lamp_object(&mut doc, LightKind::SUN, DVec3::new(0.0, 0.0, 2.0));
        let (sun_lines, _) = markers_for(&doc);
        let mut doc = Document::new();
        lamp_object(&mut doc, LightKind::LAMP, DVec3::new(0.0, 0.0, 2.0));
        let (spot_lines, _) = markers_for(&doc);
        let mut doc = Document::new();
        lamp_object(&mut doc, LightKind::PANEL, DVec3::new(0.0, 0.0, 2.0));
        let (_, area_points) = markers_for(&doc);
        // A sun has its rays, a spot its cone, an area light a square body.
        assert_eq!(sun_lines.len(), 1 + 3);
        assert_eq!(spot_lines.len(), 1 + 24 + 4);
        assert_eq!(area_points[0].shape, MarkerShape::Square);
    }

    #[test]
    fn a_camera_is_a_pyramid_with_its_apex_at_the_camera() {
        let mut doc = Document::new();
        let cam = doc.add_camera(kreate_scene::Camera::new("C"));
        let mut object = Object::new("C", ObjectData::Camera(cam));
        object.transform.translation = DVec3::new(0.0, -5.0, 1.0);
        doc.add_object(object);
        let (lines, points) = markers_for(&doc);
        assert!(points.is_empty(), "a camera has no ring");
        // Four frame edges, four to the apex, two for the up triangle.
        assert_eq!(lines.len(), 10);
        let from_apex = lines
            .iter()
            .filter(|l| l.a == DVec3::new(0.0, -5.0, 1.0))
            .count();
        assert_eq!(from_apex, 4);
    }
}
