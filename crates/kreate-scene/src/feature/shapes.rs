//! The solids a sweep and a loft make, in the world: shared by the features
//! (`Shape::Sweep`, `Shape::Loft`) and by the Mechanical menu's one-off
//! operators, so the two make the same shape of the same sketches.

use kreate_math::{DVec3, Vec2, Vec3};
use kreate_mesh::MeshData;

use crate::document::ObjectId;
use crate::Document;

/// A closed region of a sketch, in its own plane: its outline and its holes.
pub type Region2 = (Vec<Vec2>, Vec<Vec<Vec2>>);

/// The largest region of a sketch: what a loft passes through, and what a
/// sweep carries when no region is picked. `None` for an object with no
/// closed region.
pub fn largest_region(doc: &Document, id: ObjectId) -> Option<Region2> {
    super::sketch_regions(doc, id)
        .into_iter()
        .max_by(|a, b| {
            a.area()
                .abs()
                .partial_cmp(&b.area().abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| (r.outline, r.holes))
}

/// The regions of a sketch a feature's seeds pick, outline and holes each.
pub fn picked_regions(doc: &Document, id: ObjectId, seeds: &[[f64; 2]]) -> Vec<Region2> {
    let all = super::sketch_regions(doc, id);
    super::picked(&all, seeds)
        .into_iter()
        .map(|i| (all[i].outline.clone(), all[i].holes.clone()))
        .collect()
}

/// An object's lines as one path, in the world: a sketch's lines, or a
/// mesh's edges — the longest chain through them, and whether it closes.
pub fn object_path(doc: &Document, id: ObjectId) -> Option<(Vec<DVec3>, bool)> {
    let o = doc.object(id)?;
    let mesh = doc.mesh(o.mesh()?)?;
    let mut edges: Vec<[u32; 2]> = mesh.edges().to_vec();
    edges.extend(o.sketch_lines.iter().copied());
    let (points, closed) =
        kreate_mesh::modifiers::path_array::path_chain(mesh.positions(), &edges)?;
    let world = doc.world_transform(id);
    let points: Vec<DVec3> = points
        .iter()
        .map(|p| world.transform_point3(p.as_dvec3()))
        .collect();
    (points.len() >= 2).then_some((points, closed))
}

/// `solid` with each of `holes` taken out of it.
fn cut_out(mut solid: MeshData, holes: Vec<MeshData>) -> Result<MeshData, String> {
    for hole in holes {
        solid = kreate_mesh::modifiers::boolean::try_boolean(
            &solid,
            &hole,
            kreate_mesh::modifiers::BooleanOp::Difference,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(solid)
}

/// Each profile carried along `path` (in the world) — as drawn about its
/// sketch's origin, square to the path at its start — its size going from
/// one to `scale_end`, the ends closed when `caps` (a closed path has none);
/// each profile's holes swept the same way and taken out. The solid in the
/// world, or why there is none.
pub fn sweep_world(
    path: &[DVec3],
    closed: bool,
    profiles: &[Region2],
    scale_end: f64,
    caps: bool,
) -> Result<MeshData, String> {
    let Some(&start) = path.first() else {
        return Err("the path has no points".into());
    };
    let local: Vec<Vec3> = path.iter().map(|p| (*p - start).as_vec3()).collect();
    let swept = |profile: &[Vec2]| {
        kreate_mesh::primitives::sweep(&kreate_mesh::primitives::Sweep {
            path: &local,
            profile,
            resolution: 1,
            closed,
            scale: (1.0, scale_end as f32),
            caps,
        })
    };
    let mut out = MeshData::empty();
    for (outline, holes) in profiles {
        let mesh = swept(outline);
        if mesh.num_faces() == 0 || !mesh.validate().is_empty() {
            return Err("the profile cannot be swept along this path".into());
        }
        // A profile with holes — a tube, a channel with slots — is hollow
        // all along, when its ends are closed for the cut to have a solid.
        let mesh = if holes.is_empty() || !caps && !closed {
            mesh
        } else {
            cut_out(mesh, holes.iter().map(|h| swept(h)).collect())?
        };
        out = if out.num_faces() == 0 {
            mesh
        } else {
            kreate_mesh::modifiers::boolean::boolean(
                &out,
                &mesh,
                kreate_mesh::modifiers::BooleanOp::Union,
            )
        };
    }
    for p in out.positions_mut() {
        *p = (p.as_dvec3() + start).as_vec3();
    }
    Ok(out)
}

/// A solid through outlines in the world, each with its holes, in order
/// through space — from an end, each next the nearest left — and the holes
/// lofted through the sections that each have as many and taken out. The
/// solid in the world, or why there is none.
pub fn loft_world(sections: Vec<(Vec<DVec3>, Vec<Vec<DVec3>>)>) -> Result<MeshData, String> {
    let (mut outlines, mut holes): (Vec<_>, Vec<_>) = sections.into_iter().unzip();
    if outlines.len() < 2 {
        return Err("a loft needs two or more closed outlines".into());
    }
    let centre = |s: &[DVec3]| s.iter().copied().sum::<DVec3>() / s.len().max(1) as f64;
    let mid = outlines.iter().map(|s| centre(s)).sum::<DVec3>() / outlines.len() as f64;
    let first = (0..outlines.len())
        .max_by(|a, b| {
            centre(&outlines[*a])
                .distance(mid)
                .total_cmp(&centre(&outlines[*b]).distance(mid))
        })
        .unwrap_or(0);
    let mut order = vec![outlines.swap_remove(first)];
    let mut order_holes = vec![holes.swap_remove(first)];
    while !outlines.is_empty() {
        let here = centre(order.last().expect("one"));
        let next = (0..outlines.len())
            .min_by(|a, b| {
                centre(&outlines[*a])
                    .distance(here)
                    .total_cmp(&centre(&outlines[*b]).distance(here))
            })
            .expect("one left");
        order.push(outlines.swap_remove(next));
        order_holes.push(holes.swap_remove(next));
    }
    let start = centre(&order[0]);
    let local: Vec<Vec<Vec3>> = order
        .iter()
        .map(|s| s.iter().map(|p| (*p - start).as_vec3()).collect())
        .collect();
    let mesh = kreate_mesh::mech::loft(&local);
    if mesh.num_faces() == 0 || !mesh.validate().is_empty() {
        return Err("these outlines cannot be lofted".into());
    }
    // Holes, when every section has as many: each lofted through the
    // sections — matched by where they sit in their section — and taken
    // out. A hole in some sections and not others has nothing to run to.
    let k = order_holes[0].len();
    let mut mesh = if k == 0 || order_holes.iter().any(|h| h.len() != k) {
        mesh
    } else {
        let mut chains: Vec<Vec<Vec<Vec3>>> = Vec::new();
        for i in 0..k {
            let mut chain = vec![order_holes[0][i].clone()];
            for (s, holes) in order_holes.iter().enumerate().skip(1) {
                let prev = chain.last().expect("one");
                let want = centre(prev) - centre(&order[s - 1]);
                let pick = holes
                    .iter()
                    .min_by(|a, b| {
                        let da = (centre(a) - centre(&order[s])).distance(want);
                        let db = (centre(b) - centre(&order[s])).distance(want);
                        da.total_cmp(&db)
                    })
                    .expect("as many holes");
                chain.push(pick.clone());
            }
            chains.push(
                chain
                    .iter()
                    .map(|ring| ring.iter().map(|p| (*p - start).as_vec3()).collect())
                    .collect(),
            );
        }
        cut_out(
            mesh,
            chains.iter().map(|c| kreate_mesh::mech::loft(c)).collect(),
        )?
    };
    for p in mesh.positions_mut() {
        *p = (p.as_dvec3() + start).as_vec3();
    }
    Ok(mesh)
}

/// A sketch's largest region placed in the world: what a loft takes of it.
pub fn section_world(doc: &Document, id: ObjectId) -> Option<(Vec<DVec3>, Vec<Vec<DVec3>>)> {
    let (outline, holes) = largest_region(doc, id)?;
    let world = doc.world_transform(id);
    let place = |ring: &[Vec2]| -> Vec<DVec3> {
        ring.iter()
            .map(|p| world.transform_point3(DVec3::new(p.x as f64, p.y as f64, 0.0)))
            .collect()
    };
    Some((place(&outline), holes.iter().map(|h| place(h)).collect()))
}
