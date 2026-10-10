//! A mesh, read for what a sculptor would look for: holes and broken
//! surface, faces too thin to bend, bumps where it should be smooth, and
//! where its two halves differ — each said with where it is, so an assistant
//! shaping an organic model knows where to work next.

use glam::DVec3;
use kreate_mesh::MeshData;

/// A place, and how much there is there.
#[derive(Clone, Debug, PartialEq)]
pub struct Spot {
    /// In the world.
    pub at: [f64; 3],
    /// How many vertices (or edges) the spot gathers.
    pub count: usize,
    /// The worst of them, in the measure's own units.
    pub worst: f64,
}

/// What the mesh is like.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub vertices: usize,
    pub triangles: usize,
    pub quads: usize,
    pub ngons: usize,
    /// The typical edge, in metres: what the other measures are scaled by.
    pub edge_length: f64,
    /// Edges on one face only: holes, or an open rim.
    pub open_edges: Vec<Spot>,
    /// Edges on three faces or more: a surface that is not a surface.
    pub non_manifold_edges: Vec<Spot>,
    /// Faces with almost no area.
    pub degenerate_faces: usize,
    /// The share of triangles (after fanning) with an angle under ten
    /// degrees: slivers, which bend badly and shade worse.
    pub sliver_share: f64,
    /// Where the surface is bumpy: a vertex standing off its neighbours'
    /// middle by more than a fifth of an edge, gathered into places.
    pub bumpy: Vec<Spot>,
    /// How far, on average and at worst, a vertex is from its mirror across
    /// X — in edge lengths, so 0 is a symmetric mesh — and where the halves
    /// differ most. `None` when the mesh does not straddle X = 0.
    pub asymmetry: Option<(f64, f64, Vec<Spot>)>,
}

/// Reads `mesh`, its points put in the world by `to_world`.
pub fn analyze(mesh: &MeshData, to_world: &glam::DAffine3) -> Report {
    let points: Vec<DVec3> = mesh
        .positions()
        .iter()
        .map(|p| to_world.transform_point3(p.as_dvec3()))
        .collect();
    let n = points.len();
    let mut lengths: Vec<f64> = mesh
        .edges()
        .iter()
        .map(|e| points[e[0] as usize].distance(points[e[1] as usize]))
        .collect();
    lengths.sort_by(f64::total_cmp);
    let edge_length = lengths
        .get(lengths.len() / 2)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9);

    // Faces: their kinds, their areas and their thinnest angles.
    let (mut triangles, mut quads, mut ngons, mut degenerate) = (0, 0, 0, 0);
    let (mut fans, mut slivers) = (0usize, 0usize);
    let mut edge_faces: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::new();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        match ring.len() {
            3 => triangles += 1,
            4 => quads += 1,
            _ => ngons += 1,
        }
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            *edge_faces.entry((a.min(b), a.max(b))).or_default() += 1;
        }
        let mut area = 0.0;
        for i in 1..ring.len().saturating_sub(1) {
            let (a, b, c) = (
                points[ring[0] as usize],
                points[ring[i] as usize],
                points[ring[i + 1] as usize],
            );
            area += (b - a).cross(c - a).length() * 0.5;
            fans += 1;
            let angle = |p: DVec3, q: DVec3, r: DVec3| (q - p).angle_between(r - p);
            let least = angle(a, b, c).min(angle(b, c, a)).min(angle(c, a, b));
            if least < 10f64.to_radians() {
                slivers += 1;
            }
        }
        if area < edge_length * edge_length * 1e-6 {
            degenerate += 1;
        }
    }
    let mut open = Vec::new();
    let mut tangled = Vec::new();
    for ((a, b), count) in &edge_faces {
        let middle = (points[*a as usize] + points[*b as usize]) * 0.5;
        match count {
            1 => open.push((middle, 1.0)),
            c if *c > 2 => tangled.push((middle, *c as f64)),
            _ => {}
        }
    }

    // Bumps: how differently a vertex stands off its neighbours' middle from
    // how they stand off theirs. A sphere curves everywhere alike, and is
    // smooth; a lump stands out from what is around it.
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in mesh.edges() {
        let (a, b) = (e[0] as usize, e[1] as usize);
        if a < n && b < n && a != b {
            neighbours[a].push(b);
            neighbours[b].push(a);
        }
    }
    let lift: Vec<Option<DVec3>> = neighbours
        .iter()
        .enumerate()
        .map(|(v, near)| {
            // A rim vertex stands off by its nature.
            (near.len() >= 3).then(|| {
                points[v] - near.iter().map(|u| points[*u]).sum::<DVec3>() / near.len() as f64
            })
        })
        .collect();
    let mut bumps = Vec::new();
    for (v, near) in neighbours.iter().enumerate() {
        let Some(own) = lift[v] else { continue };
        let others: Vec<DVec3> = near.iter().filter_map(|u| lift[*u]).collect();
        if others.is_empty() {
            continue;
        }
        let usual = others.iter().sum::<DVec3>() / others.len() as f64;
        let off = own.distance(usual) / edge_length;
        if off > 0.2 {
            bumps.push((points[v], off));
        }
    }

    // Symmetry: each vertex's mirror across X, against the surface.
    let (lo, hi) = points.iter().fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let asymmetry = (n > 0 && lo.x < -edge_length && hi.x > edge_length).then(|| {
        let triangles: Vec<[DVec3; 3]> = (0..mesh.num_faces())
            .flat_map(|f| {
                let ring = mesh.face_verts(f);
                (1..ring.len().saturating_sub(1))
                    .map(|i| {
                        [
                            points[ring[0] as usize],
                            points[ring[i] as usize],
                            points[ring[i + 1] as usize],
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let cell = edge_length * 3.0;
        let key = |p: DVec3| (p / cell).floor().as_ivec3();
        let mut grid: std::collections::HashMap<glam::IVec3, Vec<usize>> =
            std::collections::HashMap::new();
        for (t, tri) in triangles.iter().enumerate() {
            let (a, b) = (
                tri[0].min(tri[1]).min(tri[2]),
                tri[0].max(tri[1]).max(tri[2]),
            );
            let (ka, kb) = (key(a), key(b));
            for x in ka.x..=kb.x {
                for y in ka.y..=kb.y {
                    for z in ka.z..=kb.z {
                        grid.entry(glam::IVec3::new(x, y, z)).or_default().push(t);
                    }
                }
            }
        }
        let far = 3.0;
        let mut total = 0.0;
        let mut worst = 0.0f64;
        let mut off = Vec::new();
        for p in &points {
            let mirror = DVec3::new(-p.x, p.y, p.z);
            let c = key(mirror);
            let mut best = f64::MAX;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(list) = grid.get(&(c + glam::IVec3::new(dx, dy, dz))) {
                            for &t in list {
                                let [a, b, c] = triangles[t];
                                best =
                                    best.min(closest_on_triangle(mirror, a, b, c).distance(mirror));
                            }
                        }
                    }
                }
            }
            // Nothing within a cell: as different as it is measured to be.
            let d = (best / edge_length).min(far);
            total += d;
            worst = worst.max(d);
            if d > 0.5 {
                off.push((*p, d));
            }
        }
        (total / n as f64, worst, gather(&off, edge_length * 8.0))
    });

    Report {
        vertices: n,
        triangles,
        quads,
        ngons,
        edge_length,
        open_edges: gather(&open, edge_length * 8.0),
        non_manifold_edges: gather(&tangled, edge_length * 8.0),
        degenerate_faces: degenerate,
        sliver_share: if fans > 0 {
            slivers as f64 / fans as f64
        } else {
            0.0
        },
        bumpy: gather(&bumps, edge_length * 8.0),
        asymmetry,
    }
}

/// Points into places a few edges across: each place's middle, how many,
/// and the worst — the eight worst, worst first.
fn gather(found: &[(DVec3, f64)], size: f64) -> Vec<Spot> {
    let mut places: std::collections::HashMap<glam::IVec3, (DVec3, usize, f64)> =
        std::collections::HashMap::new();
    for (p, value) in found {
        let entry = places
            .entry((*p / size).floor().as_ivec3())
            .or_insert((DVec3::ZERO, 0, 0.0));
        entry.0 += *p;
        entry.1 += 1;
        entry.2 = entry.2.max(*value);
    }
    let mut spots: Vec<Spot> = places
        .into_values()
        .map(|(sum, count, worst)| Spot {
            at: (sum / count as f64).to_array(),
            count,
            worst,
        })
        .collect();
    spots.sort_by(|a, b| b.worst.total_cmp(&a.worst).then(b.count.cmp(&a.count)));
    spots.truncate(8);
    spots
}

/// The point of triangle `abc` nearest `p` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn closest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_smooth_closed_sphere_is_clean_and_symmetric() {
        let sphere = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
        let report = analyze(&sphere, &glam::DAffine3::IDENTITY);
        assert!(report.open_edges.is_empty(), "{:?}", report.open_edges);
        assert!(report.non_manifold_edges.is_empty());
        assert!(report.bumpy.is_empty(), "{:?}", report.bumpy);
        let (mean, worst, _) = report.asymmetry.expect("it straddles X");
        assert!(mean < 0.05 && worst < 0.2, "{mean} {worst}");
    }

    #[test]
    fn a_bump_a_hole_and_a_lopsided_half_are_found_where_they_are() {
        let mut sphere = kreate_mesh::primitives::uv_sphere(1.0, 48, 24);
        // A bump on the +X side, near (1, 0, 0).
        let bumped = sphere
            .positions()
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.distance(glam::Vec3::X)
                    .total_cmp(&b.1.distance(glam::Vec3::X))
            })
            .map(|(i, _)| i)
            .expect("a vertex");
        sphere.positions_mut()[bumped] *= 1.3;
        let report = analyze(&sphere, &glam::DAffine3::IDENTITY);
        let bump = report.bumpy.first().expect("the bump");
        assert!(bump.at[0] > 0.8, "{bump:?}");
        let (_, worst, off) = report.asymmetry.expect("measured");
        assert!(
            worst > 1.0 && off.first().is_some_and(|s| s.at[0].abs() > 0.8),
            "{off:?}"
        );

        // A hole: a plane is all rim.
        let plane = kreate_mesh::primitives::grid(2.0, 4, 4);
        let report = analyze(&plane, &glam::DAffine3::IDENTITY);
        assert!(report.open_edges.iter().map(|s| s.count).sum::<usize>() == 16);
    }
}
