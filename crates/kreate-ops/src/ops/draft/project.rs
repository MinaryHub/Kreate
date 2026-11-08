//! Geometry brought into a drawing from the parts around it, and kept tied
//! to them: Fusion's Project (a part's outline and edges seen square to the
//! drawing's plane) and Intersect (where the plane cuts the part).
//!
//! What comes in is lines and polylines on a PROJECTED layer, every point
//! held where it is, so a profile can be drawn against them and closed by
//! them. The drawing remembers which part made which entities
//! (`Drawing::projections`); `draft.project_update` makes them again from
//! the parts as they now are — the application runs it when a part a
//! drawing projects has changed.
//!
//! From the part as it is shown (its modifiers, a body's features), in its
//! meshes: a round hole projects as the many sides its mesh has. With the
//! exact kernel that will be a circle (design/brep-kernel.md, P4).

use std::collections::HashMap;

use kreate_geom2d::{Curve2D, DVec2};
use kreate_math::{DAffine3, Vec3};
use kreate_mesh::MeshData;
use kreate_scene::document::ObjectId;
use kreate_scene::drawing::{
    ColorRef, Constraint, Drawing, EntityId, LayerId, Part, Projection, ProjectionKind,
};
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::{DocMut, Document};

use super::{active_drawing, edit};
use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<DraftProject>();
    registry.register::<DraftProjectUpdate>();
}

/// The layer projected geometry goes on, made the first time.
const LAYER: &str = "PROJECTED";

/// Edges whose faces turn by more than this are drawn: creases.
const CREASE: f32 = 30.0 * std::f32::consts::PI / 180.0;

/// A part's mesh as shown, in the drawing's space: the drawing's plane is
/// z = 0.
fn mesh_in_drawing(doc: &Document, drawing: ObjectId, part: ObjectId) -> Option<MeshData> {
    let mesh = ModifierCache::new().evaluated(doc, part, &EvalCtx::still())?;
    let to: DAffine3 = doc.world_transform(drawing).inverse() * doc.world_transform(part);
    let positions: Vec<Vec3> = mesh
        .positions()
        .iter()
        .map(|p| to.transform_point3(p.as_dvec3()).as_vec3())
        .collect();
    let faces: Vec<Vec<u32>> = mesh.faces().map(|f| f.to_vec()).collect();
    let out = MeshData::from_faces(positions, &faces);
    (out.num_faces() > 0).then_some(out)
}

/// Points along a run with the ones on a straight stretch between their
/// neighbours taken out.
fn straighten(points: Vec<DVec2>, closed: bool) -> Vec<DVec2> {
    let mut pts = points;
    let scale = pts.iter().map(|p| p.length()).fold(0.0, f64::max).max(1e-9);
    loop {
        let n = pts.len();
        if n <= if closed { 3 } else { 2 } {
            break;
        }
        let inner = if closed { 0..n } else { 1..n - 1 };
        let drop = inner.into_iter().find(|&i| {
            let (a, b, c) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
            let (u, v) = (b - a, c - b);
            u.length() < 1e-9 * scale
                || (u.perp_dot(v).abs() <= 1e-7 * u.length() * v.length() && u.dot(v) > 0.0)
        });
        match drop {
            Some(i) => {
                pts.remove(i);
            }
            None => break,
        }
    }
    pts
}

/// The curves a projection of `part` makes in `drawing`.
pub fn projected_curves(
    doc: &Document,
    drawing: ObjectId,
    part: ObjectId,
    kind: ProjectionKind,
) -> Vec<Curve2D> {
    let Some(mesh) = mesh_in_drawing(doc, drawing, part) else {
        return Vec::new();
    };
    match kind {
        ProjectionKind::Intersect => {
            kreate_mesh::section::section_loops(&mesh, Vec3::ZERO, Vec3::Z)
                .into_iter()
                .map(|l| {
                    let pts = straighten(
                        l.iter()
                            .map(|p| DVec2::new(p.x as f64, p.y as f64))
                            .collect(),
                        true,
                    );
                    Curve2D::Polyline {
                        verts: pts.into_iter().map(|p| (p, 0.0)).collect(),
                        closed: true,
                    }
                })
                .filter(|c| matches!(c, Curve2D::Polyline { verts, .. } if verts.len() >= 3))
                .collect()
        }
        ProjectionKind::Outline => outline(&mesh),
    }
}

/// The edges a part shows seen square to the plane — its open edges, its
/// creases and its silhouette — laid onto it, each place once, straight
/// runs joined into one line.
fn outline(mesh: &MeshData) -> Vec<Curve2D> {
    let p = mesh.positions();
    let normal = |f: u32| {
        let ring = mesh.face_verts(f);
        let mut n = Vec3::ZERO;
        for i in 0..ring.len() {
            let (a, b) = (p[ring[i] as usize], p[ring[(i + 1) % ring.len()] as usize]);
            n += a.cross(b);
        }
        n.normalize_or_zero()
    };
    let mut faces_of: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            faces_of.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    let size = p
        .iter()
        .fold(0.0f32, |m, q| m.max(q.truncate().length()))
        .max(1e-6) as f64;
    let key = |q: DVec2| {
        (
            (q.x / size * 1e7).round() as i64,
            (q.y / size * 1e7).round() as i64,
        )
    };
    let mut seen: std::collections::HashSet<((i64, i64), (i64, i64))> = Default::default();
    let mut segments: Vec<[DVec2; 2]> = Vec::new();
    for (&(a, b), faces) in &faces_of {
        let shown = match faces.as_slice() {
            [_] => true,
            [f, g] => {
                let (n, m) = (normal(*f), normal(*g));
                n.dot(m).clamp(-1.0, 1.0).acos() > CREASE || (n.z > 1e-6) != (m.z > 1e-6)
            }
            _ => true,
        };
        if !shown {
            continue;
        }
        let (pa, pb) = (p[a as usize], p[b as usize]);
        let (qa, qb) = (
            DVec2::new(pa.x as f64, pa.y as f64),
            DVec2::new(pb.x as f64, pb.y as f64),
        );
        if qa.distance(qb) < size * 1e-7 {
            continue;
        }
        let (ka, kb) = (key(qa), key(qb));
        if !seen.insert((ka.min(kb), ka.max(kb))) {
            continue;
        }
        segments.push([qa, qb]);
    }
    join_straight(segments, &key)
}

/// Segments meeting end to end in a straight line — and only those two at
/// the point — joined into one line.
fn join_straight(
    mut segments: Vec<[DVec2; 2]>,
    key: &impl Fn(DVec2) -> (i64, i64),
) -> Vec<Curve2D> {
    loop {
        let mut at: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, s) in segments.iter().enumerate() {
            at.entry(key(s[0])).or_default().push(i);
            at.entry(key(s[1])).or_default().push(i);
        }
        let mut joined = false;
        for (_, ids) in at {
            let [i, j] = ids.as_slice() else {
                continue;
            };
            if i == j {
                continue;
            }
            let (s, t) = (segments[*i], segments[*j]);
            // Oriented so s ends where t starts.
            let shared = if key(s[1]) == key(t[0]) {
                (s[0], s[1], t[1])
            } else if key(s[1]) == key(t[1]) {
                (s[0], s[1], t[0])
            } else if key(s[0]) == key(t[0]) {
                (s[1], s[0], t[1])
            } else {
                (s[1], s[0], t[0])
            };
            let (a, m, b) = shared;
            let (u, v) = (m - a, b - m);
            if u.perp_dot(v).abs() <= 1e-7 * u.length() * v.length() && u.dot(v) > 0.0 {
                let (lo, hi) = ((*i).min(*j), (*i).max(*j));
                segments.remove(hi);
                segments.remove(lo);
                segments.push([a, b]);
                joined = true;
                break;
            }
        }
        if !joined {
            break;
        }
    }
    segments
        .into_iter()
        .map(|[a, b]| Curve2D::Line { a, b })
        .collect()
}

/// The PROJECTED layer, made the first time.
fn layer(d: &mut Drawing) -> LayerId {
    if let Some(l) = d.layers.iter().find(|l| l.name == LAYER) {
        return l.id;
    }
    let id = d.add_layer(LAYER);
    if let Some(l) = d.layers.iter_mut().find(|l| l.id == id) {
        // Fusion draws projected geometry purple.
        l.color = ColorRef::Index(6);
    }
    id
}

/// `curves` into the drawing on the PROJECTED layer, every point held where
/// it is. Their ids.
fn put(d: &mut Drawing, curves: Vec<Curve2D>) -> Vec<EntityId> {
    let on = layer(d);
    let before = d.current_layer;
    d.current_layer = on;
    let ids: Vec<EntityId> = curves.into_iter().map(|c| d.add_curve(c)).collect();
    d.current_layer = before;
    for id in &ids {
        for (r, p) in d.parts(*id) {
            if matches!(r.part, Part::Start | Part::End | Part::Vertex(_)) {
                d.constraints.push(Constraint::Fixed(r, p));
            }
        }
    }
    ids
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftProjectParams {
    /// The parts, by name, separated by commas; empty for the selected
    /// objects other than the drawing.
    pub objects: String,
    #[param(variants("outline", "intersect"))]
    pub kind: u32,
}

/// Brings parts into the active drawing — their outline and edges seen
/// square to its plane, or where its plane cuts them — held, and tied to
/// the parts so they follow them.
pub struct DraftProject;

impl Operator for DraftProject {
    type Params = DraftProjectParams;
    fn id() -> &'static str {
        "draft.project"
    }
    fn label() -> &'static str {
        "Project"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let drawing =
            active_drawing(d).ok_or_else(|| OpError::Failed("no drawing is being drawn".into()))?;
        let names: Vec<&str> = params
            .objects
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let parts: Vec<ObjectId> = if names.is_empty() {
            d.selection
                .objects
                .iter()
                .copied()
                .filter(|id| *id != drawing)
                .collect()
        } else {
            d.objects()
                .filter(|(_, o)| names.contains(&o.name.as_str()))
                .map(|(id, _)| id)
                .collect()
        };
        let parts: Vec<ObjectId> = parts
            .into_iter()
            .filter(|id| {
                d.object(*id)
                    .is_some_and(|o| o.mesh().is_some() && o.drawing.is_none())
            })
            .collect();
        if parts.is_empty() {
            return Err(OpError::Failed("select the parts to project".into()));
        }
        let kind = if params.kind == 1 {
            ProjectionKind::Intersect
        } else {
            ProjectionKind::Outline
        };
        let made: Vec<(kreate_scene::Uid, Vec<Curve2D>)> = parts
            .iter()
            .map(|id| {
                (
                    d.object_uid(*id).expect("an object"),
                    projected_curves(d, drawing, *id, kind),
                )
            })
            .filter(|(_, c)| !c.is_empty())
            .collect();
        if made.is_empty() {
            return Err(OpError::Failed(match kind {
                ProjectionKind::Intersect => "the drawing's plane does not cut them".into(),
                ProjectionKind::Outline => "they show nothing on the drawing's plane".into(),
            }));
        }
        edit(doc, |d| {
            for (source, curves) in made {
                let entities = put(d, curves);
                d.projections.push(Projection {
                    source,
                    kind,
                    entities,
                });
            }
            Ok(())
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DraftProjectUpdateParams {}

/// Makes the active drawing's projected geometry again from the parts as
/// they now are. A part gone takes its geometry with it. Nothing to do
/// when nothing moved.
pub struct DraftProjectUpdate;

impl Operator for DraftProjectUpdate {
    type Params = DraftProjectUpdateParams;
    fn id() -> &'static str {
        "draft.project_update"
    }
    fn label() -> &'static str {
        "Update Projections"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_drawing(doc)
            .and_then(|id| doc.object(id)?.drawing.as_deref())
            .is_some_and(|d| !d.projections.is_empty())
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(drawing) = active_drawing(d) else {
            return Ok(OpResult::Cancelled);
        };
        let current = d
            .object(drawing)
            .and_then(|o| o.drawing.as_deref())
            .cloned()
            .unwrap_or_default();
        let fresh: Vec<(usize, Vec<Curve2D>)> = current
            .projections
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let curves = d
                    .object_by_uid(p.source)
                    .map(|part| projected_curves(d, drawing, part, p.kind))
                    .unwrap_or_default();
                (i, curves)
            })
            .collect();
        let same = fresh.iter().all(|(i, curves)| {
            let old: Vec<Curve2D> = current.projections[*i]
                .entities
                .iter()
                .filter_map(|e| current.curve(*e).cloned())
                .collect();
            old == *curves
        });
        if same {
            return Ok(OpResult::Cancelled);
        }
        edit(doc, |d| {
            let old = std::mem::take(&mut d.projections);
            for (p, (_, curves)) in old.into_iter().zip(fresh) {
                d.remove(&p.entities);
                if curves.is_empty() {
                    continue;
                }
                let entities = put(d, curves);
                d.projections.push(Projection {
                    source: p.source,
                    kind: p.kind,
                    entities,
                });
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::{Mode, OperatorRegistry};
    use kreate_scene::drawing::EntityKind;

    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        doc: &mut Document,
        line: &str,
    ) -> Result<OpResult, OpError> {
        let call = crate::script::parse_call(r, line).unwrap_or_else(|e| panic!("{line}: {e}"));
        h.dispatch(
            r,
            doc,
            &Context { mode: Mode::Object },
            call.op_id,
            &call.params,
        )
    }

    fn drawing(doc: &Document) -> &Drawing {
        let id = active_drawing(doc).unwrap();
        doc.object(id).unwrap().drawing.as_deref().unwrap()
    }

    /// A 40 × 20 × 10 mm block standing on the ground, a drawing on the
    /// ground: projected, its outline is four lines held where they are;
    /// cut at 5 mm up, a closed rectangle. Moved, they follow it.
    #[test]
    fn a_block_projects_its_outline_and_is_cut_by_the_plane_and_followed() {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let mut cube = kreate_mesh::primitives::cube(1.0);
        for p in cube.positions_mut() {
            *p = *p * Vec3::new(0.04, 0.02, 0.01) + Vec3::new(0.02, 0.01, 0.005);
        }
        let block = doc.add_mesh_object("Block", cube);
        run(&r, &mut h, &mut doc, "draft.new()").unwrap();
        let sketch = active_drawing(&doc).unwrap();
        doc.selection.add(block);
        doc.selection.active = Some(sketch);
        run(&r, &mut h, &mut doc, "draft.project(objects=\"Block\")").unwrap();
        let d = drawing(&doc);
        assert_eq!(d.projections.len(), 1);
        let lines: Vec<Curve2D> = d.projections[0]
            .entities
            .iter()
            .filter_map(|e| d.curve(*e).cloned())
            .collect();
        assert_eq!(lines.len(), 4, "{lines:?}");
        assert_eq!(d.regions(1e-9).len(), 1);
        assert!((d.regions(1e-9)[0].area - 0.0008).abs() < 1e-9);
        assert!(
            d.constraints
                .iter()
                .filter(|c| matches!(c, Constraint::Fixed(..)))
                .count()
                >= 8
        );
        let on = d.layers.iter().find(|l| l.name == LAYER).unwrap().id;
        assert!(d
            .entities
            .iter()
            .filter(|e| matches!(e.kind, EntityKind::Curve(_)))
            .all(|e| e.layer == on));

        // A drawing 5 mm up cuts it in a rectangle.
        run(&r, &mut h, &mut doc, "draft.new()").unwrap();
        let cut = active_drawing(&doc).unwrap();
        doc.object_mut(cut).unwrap().transform.translation.z = 0.005;
        run(
            &r,
            &mut h,
            &mut doc,
            "draft.project(objects=\"Block\", kind=\"intersect\")",
        )
        .unwrap();
        let d = drawing(&doc);
        assert_eq!(d.regions(1e-9).len(), 1);
        assert!((d.regions(1e-9)[0].area - 0.0008).abs() < 1e-9);
        // Above the block: nothing to cut.
        doc.object_mut(cut).unwrap().transform.translation.z = 0.02;
        assert!(run(
            &r,
            &mut h,
            &mut doc,
            "draft.project(objects=\"Block\", kind=\"intersect\")"
        )
        .is_err());
        doc.object_mut(cut).unwrap().transform.translation.z = 0.005;

        // The block moved 10 mm along X: brought up to date, the cut moves.
        doc.object_mut(block).unwrap().transform.translation.x = 0.01;
        run(&r, &mut h, &mut doc, "draft.project_update()").unwrap();
        let d = drawing(&doc);
        let (lo, hi) = d.extents().unwrap();
        assert!(
            (lo.x - 0.01).abs() < 1e-9 && (hi.x - 0.05).abs() < 1e-9,
            "{lo} {hi}"
        );
        // Nothing moved since: nothing to do.
        assert_eq!(
            run(&r, &mut h, &mut doc, "draft.project_update()").unwrap(),
            OpResult::Cancelled
        );
    }
}
