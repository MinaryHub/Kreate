//! Edges to Sketch: the selected edges of the mesh being edited, laid into a
//! new sketch on the plane they lie in — an outline taken off a model to
//! extrude, cut or revolve from.
//!
//! The plane is the one the edges' ends fit best; ends off it are pressed
//! onto it. The sketch's origin is their middle and its Z faces away from the
//! object, so a positive extrude comes out of the model. Joined edges share
//! their ends in the sketch as in the mesh, so a closed loop of edges is a
//! closed region. The mesh is left as it was.

use kreate_math::{DMat3, DQuat, DVec2, DVec3, Vec3};
use kreate_mesh::MeshData;
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::sketch::Constraint;
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::ops::mesh::{can_edit, target_mesh, target_object};
use crate::OpParams;

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct EdgesToSketchParams {}

/// Lays the selected edges into a new sketch.
pub struct EdgesToSketch;

impl Operator for EdgesToSketch {
    type Params = EdgesToSketchParams;
    fn id() -> &'static str {
        "mesh.edges_to_sketch"
    }
    fn label() -> &'static str {
        "Edges to Sketch"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let (Some(object), Some(mesh_id)) = (target_object(d), target_mesh(d)) else {
            return Err(OpError::Failed("no mesh is being edited".into()));
        };
        let mesh = d.mesh(mesh_id).expect("the target's mesh");
        let selected = mesh
            .attributes()
            .get_bool(kreate_mesh::attribute::names::SELECT_EDGE);
        let edges: Vec<[u32; 2]> = mesh
            .edges()
            .iter()
            .enumerate()
            .filter(|(i, _)| selected.is_some_and(|s| s.get(*i).copied().unwrap_or(false)))
            .map(|(_, e)| *e)
            .collect();
        if edges.is_empty() {
            return Err(OpError::Failed(
                "select the edges to make a sketch of".into(),
            ));
        }
        let to_world = d.world_transform(object);
        let world = |v: u32| to_world.transform_point3(mesh.positions()[v as usize].as_dvec3());

        // The ends, once each, in the order the edges reach them.
        let mut ends: Vec<u32> = Vec::new();
        for e in &edges {
            for v in e {
                if !ends.contains(v) {
                    ends.push(*v);
                }
            }
        }
        let points: Vec<DVec3> = ends.iter().map(|v| world(*v)).collect();
        let centre = points.iter().copied().sum::<DVec3>() / points.len() as f64;
        let first = world(edges[0][1]) - world(edges[0][0]);
        let mut normal = fit_normal(&points, first);
        // Away from the object, so out of the model is the positive way.
        let middle = object_middle(mesh, &to_world);
        if (centre - middle).dot(normal) < -1e-12
            || ((centre - middle).dot(normal).abs() <= 1e-12 && normal.z < 0.0)
        {
            normal = -normal;
        }
        // The sketch's X along the first edge, laid in the plane.
        let mut x = first - normal * first.dot(normal);
        if x.length_squared() < 1e-24 {
            x = normal.any_orthonormal_vector();
        }
        let x = x.normalize();
        let y = normal.cross(x);

        // Point 0 is the sketch's origin, held, as a new sketch has it.
        let mut plane: Vec<DVec2> = vec![DVec2::ZERO];
        plane.extend(points.iter().map(|p| {
            let d = *p - centre;
            DVec2::new(d.dot(x), d.dot(y))
        }));
        let index = |v: u32| ends.iter().position(|e| *e == v).expect("an end") as u32 + 1;
        let lines: Vec<[u32; 2]> = edges.iter().map(|[a, b]| [index(*a), index(*b)]).collect();

        let name = {
            let source = &d.object(object).expect("the target").name;
            d.unique_object_name(&format!("{source} Sketch"))
        };
        let positions: Vec<Vec3> = plane
            .iter()
            .map(|p| Vec3::new(p.x as f32, p.y as f32, 0.0))
            .collect();
        let d = doc.document_mut();
        let sketch_mesh = d.add_mesh(MeshData::from_faces(positions, &[]));
        let mut sketch = Object::new(name, ObjectData::Mesh(sketch_mesh));
        sketch.transform.translation = centre;
        sketch.transform.rotation = DQuat::from_mat3(&DMat3::from_cols(x, y, normal));
        sketch.sketch.push(Constraint::Fixed {
            a: 0,
            at: [0.0, 0.0],
        });
        sketch.sketch_lines = lines;
        d.add_object(sketch);
        Ok(OpResult::Finished)
    }
}

/// The normal of the plane `points` lie nearest: the direction they spread
/// least in. Points in a row lie in many planes; the one taken holds the row
/// and the world's up, or its Y when the row is upright.
fn fit_normal(points: &[DVec3], along: DVec3) -> DVec3 {
    let centre = points.iter().copied().sum::<DVec3>() / points.len() as f64;
    let mut c = [[0.0f64; 3]; 3];
    for p in points {
        let d = (*p - centre).to_array();
        for (i, row) in c.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }
    let (values, vectors) = jacobi(c);
    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let spread = values[order[2]].max(1e-30);
    // A row: the second smallest spread is nothing too.
    if values[order[1]] / spread < 1e-10 {
        let dir = if along.length_squared() > 0.0 {
            along.normalize()
        } else {
            DVec3::X
        };
        let up = if dir.z.abs() > 0.99 {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let n = dir.cross(up).cross(dir);
        return n.normalize_or(DVec3::Z);
    }
    let v = vectors[order[0]];
    DVec3::from_array(v).normalize_or(DVec3::Z)
}

/// The middle of the object's mesh in the world: what "away from it" is
/// measured from.
fn object_middle(mesh: &MeshData, to_world: &kreate_math::DAffine3) -> DVec3 {
    let p = mesh.positions();
    if p.is_empty() {
        return to_world.translation;
    }
    let sum = p.iter().fold(DVec3::ZERO, |s, v| s + v.as_dvec3());
    to_world.transform_point3(sum / p.len() as f64)
}

/// The eigenvalues and eigenvectors of a symmetric 3 × 3 matrix, by Jacobi
/// rotations; each vector is a row of the second.
fn jacobi(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let (mut p, mut q, mut largest) = (0, 1, 0.0);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if a[i][j].abs() > largest {
                (p, q, largest) = (i, j, a[i][j].abs());
            }
        }
        if largest < 1e-18 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for row in &mut a {
            let (akp, akq) = (row[p], row[q]);
            row[p] = c * akp - s * akq;
            row[q] = s * akp + c * akq;
        }
        let (rp, rq) = (a[p], a[q]);
        for (k, (apk, aqk)) in rp.into_iter().zip(rq).enumerate() {
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for row in &mut v {
            let (vp, vq) = (row[p], row[q]);
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }
    // Columns of `v` are the vectors; handed back as rows.
    let vectors = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    ([a[0][0], a[1][1], a[2][2]], vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::Mode;
    use crate::{OperatorRegistry, PropertyMap};

    /// A 2-unit cube, its top face's four edges selected, in edit mode.
    fn cube_with_top_edges() -> (Document, kreate_scene::document::ObjectId) {
        let mut doc = Document::new();
        let id = doc.add_mesh_object("Box", kreate_mesh::primitives::cube(2.0));
        let mesh_id = doc.object(id).unwrap().mesh().unwrap();
        let m = doc.mesh_mut(mesh_id).unwrap();
        let top: Vec<bool> = m
            .edges()
            .iter()
            .map(|[a, b]| m.positions()[*a as usize].z > 0.5 && m.positions()[*b as usize].z > 0.5)
            .collect();
        m.attributes_mut().insert(
            kreate_mesh::attribute::names::SELECT_EDGE,
            kreate_mesh::attribute::Domain::Edge,
            kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(top)),
        );
        doc.selection.add(id);
        doc.selection.active = Some(id);
        (doc, id)
    }

    fn run(doc: &mut Document) -> Result<OpResult, OpError> {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        History::new().dispatch(
            &r,
            doc,
            &Context {
                mode: Mode::EditMesh,
            },
            "mesh.edges_to_sketch",
            &PropertyMap::new(),
        )
    }

    #[test]
    fn a_faces_edges_become_a_closed_sketch_on_its_plane_facing_out() {
        let (mut doc, _) = cube_with_top_edges();
        assert_eq!(run(&mut doc).unwrap(), OpResult::Finished);
        let (sketch, o) = doc
            .objects()
            .find(|(_, o)| o.name == "Box Sketch")
            .expect("a sketch");
        assert_eq!(o.sketch_lines.len(), 4);
        let pts = crate::ops::sketch_draw::points(&doc, sketch);
        assert_eq!(pts.len(), 5, "the origin and four corners: {pts:?}");
        // On the top face, facing up and out of the box.
        let world = doc.world_transform(sketch);
        assert!((world.translation.z - 1.0).abs() < 1e-6, "{world:?}");
        let up = world.transform_vector3(DVec3::Z);
        assert!((up - DVec3::Z).length() < 1e-6, "{up:?}");
        for p in &pts[1..] {
            let w = world.transform_point3(DVec3::new(p.x, p.y, 0.0));
            assert!(
                (w.z - 1.0).abs() < 1e-6 && (w.x.abs() - 1.0).abs() < 1e-6,
                "{w:?}"
            );
        }
        // A closed loop: one region to extrude.
        assert_eq!(kreate_scene::feature::sketch_regions(&doc, sketch).len(), 1);
    }

    #[test]
    fn nothing_selected_is_said() {
        let (mut doc, id) = cube_with_top_edges();
        let mesh_id = doc.object(id).unwrap().mesh().unwrap();
        doc.mesh_mut(mesh_id)
            .unwrap()
            .attributes_mut()
            .remove(kreate_mesh::attribute::names::SELECT_EDGE);
        assert!(run(&mut doc).is_err());
    }

    #[test]
    fn a_plane_is_fitted_and_a_row_of_points_still_makes_one() {
        let pts = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(1.0, 1.0, 1.0),
        ];
        let n = fit_normal(&pts, DVec3::X);
        let want = DVec3::new(-1.0, 0.0, 1.0).normalize();
        assert!(
            (n - want).length() < 1e-9 || (n + want).length() < 1e-9,
            "{n:?}"
        );
        let row = [DVec3::ZERO, DVec3::X, DVec3::X * 2.0];
        let n = fit_normal(&row, DVec3::X);
        assert!(n.dot(DVec3::X).abs() < 1e-9 && (n.length() - 1.0).abs() < 1e-9);
    }
}
