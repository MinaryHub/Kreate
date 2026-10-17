//! The shrinkwrap modifier: every vertex moved onto another mesh's surface
//! — the nearest point of it, or where a ray along the vertex's normal
//! meets it — and held off it by an offset. A label wrapped onto a bottle,
//! a lid fitted to a pot, clothes laid over a body.

use kreate_math::Vec3;

use crate::MeshData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WrapMode {
    /// The nearest point of the target's surface.
    #[default]
    Nearest,
    /// Where a line along the vertex's normal, either way, meets the
    /// target; a vertex whose line misses stays.
    Project,
}

impl WrapMode {
    pub const ALL: [WrapMode; 2] = [WrapMode::Nearest, WrapMode::Project];

    pub fn tag(self) -> &'static str {
        match self {
            WrapMode::Nearest => "nearest",
            WrapMode::Project => "project",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.tag() == tag)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShrinkwrapParams {
    pub mode: WrapMode,
    /// How far off the surface the vertices are left, along its normal.
    pub offset: f32,
}

impl Default for ShrinkwrapParams {
    fn default() -> Self {
        Self {
            mode: WrapMode::Nearest,
            offset: 0.0,
        }
    }
}

/// The target's triangles, in `input`'s space, in a grid of cells for the
/// nearest-point and ray queries.
pub struct Surface {
    tris: Vec<[Vec3; 3]>,
    normals: Vec<Vec3>,
    cell: f32,
    origin: Vec3,
    dims: [usize; 3],
    cells: Vec<Vec<u32>>,
}

impl Surface {
    /// `target` as a surface to wrap onto, its positions already in the
    /// wrapped mesh's space.
    pub fn new(target: &MeshData) -> Option<Self> {
        let tri = target.triangulate();
        let (idx, _) = tri.indices.as_chunks::<3>();
        let p = target.positions();
        let tris: Vec<[Vec3; 3]> = idx
            .iter()
            .map(|t| [p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]])
            .collect();
        if tris.is_empty() {
            return None;
        }
        let normals: Vec<Vec3> = tris
            .iter()
            .map(|[a, b, c]| (*b - *a).cross(*c - *a).normalize_or_zero())
            .collect();
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for t in &tris {
            for v in t {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        let extent = (hi - lo).max_element().max(1e-6);
        // About as many cells as triangles, on a side the cube root of that.
        let per_side = ((tris.len() as f32).cbrt().ceil() as usize).clamp(1, 64);
        let cell = extent / per_side as f32;
        let dims = [
            ((hi.x - lo.x) / cell).floor() as usize + 1,
            ((hi.y - lo.y) / cell).floor() as usize + 1,
            ((hi.z - lo.z) / cell).floor() as usize + 1,
        ];
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        let at = |p: Vec3| -> [usize; 3] {
            let c = (p - lo) / cell;
            [
                (c.x.floor() as usize).min(dims[0] - 1),
                (c.y.floor() as usize).min(dims[1] - 1),
                (c.z.floor() as usize).min(dims[2] - 1),
            ]
        };
        for (i, t) in tris.iter().enumerate() {
            let (mut tlo, mut thi) = (t[0], t[0]);
            for v in t {
                tlo = tlo.min(*v);
                thi = thi.max(*v);
            }
            let (a, b) = (at(tlo), at(thi));
            for x in a[0]..=b[0] {
                for y in a[1]..=b[1] {
                    for z in a[2]..=b[2] {
                        cells[(z * dims[1] + y) * dims[0] + x].push(i as u32);
                    }
                }
            }
        }
        Some(Self {
            tris,
            normals,
            cell,
            origin: lo,
            dims,
            cells,
        })
    }

    fn cell_of(&self, p: Vec3) -> [i64; 3] {
        let c = (p - self.origin) / self.cell;
        [c.x.floor() as i64, c.y.floor() as i64, c.z.floor() as i64]
    }

    fn cell_tris(&self, c: [i64; 3]) -> &[u32] {
        if c.iter().any(|v| *v < 0)
            || c[0] as usize >= self.dims[0]
            || c[1] as usize >= self.dims[1]
            || c[2] as usize >= self.dims[2]
        {
            return &[];
        }
        &self.cells[(c[2] as usize * self.dims[1] + c[1] as usize) * self.dims[0] + c[0] as usize]
    }

    /// The nearest point of the surface to `p`, and the normal there.
    pub fn nearest(&self, p: Vec3) -> (Vec3, Vec3) {
        let centre = self.cell_of(p);
        let mut best: Option<(f32, Vec3, Vec3)> = None;
        let reach = [self.dims[0], self.dims[1], self.dims[2]]
            .into_iter()
            .max()
            .unwrap_or(1) as i64;
        for ring in 0..=reach {
            let mut seen_any = false;
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for dz in -ring..=ring {
                        if dx.abs() != ring && dy.abs() != ring && dz.abs() != ring {
                            continue;
                        }
                        let c = [centre[0] + dx, centre[1] + dy, centre[2] + dz];
                        for t in self.cell_tris(c) {
                            seen_any = true;
                            let [a, b, cc] = self.tris[*t as usize];
                            let q = closest_on_triangle(p, a, b, cc);
                            let d = q.distance_squared(p);
                            if best.is_none_or(|(bd, _, _)| d < bd) {
                                best = Some((d, q, self.normals[*t as usize]));
                            }
                        }
                    }
                }
            }
            // Past the ring the best is nearer than, nothing nearer waits.
            if let Some((d, _, _)) = best {
                if d.sqrt() <= ring as f32 * self.cell {
                    break;
                }
            }
            let _ = seen_any;
        }
        best.map(|(_, q, n)| (q, n)).unwrap_or((p, Vec3::Z))
    }

    /// Where a line through `p` along `dir`, either way, first meets the
    /// surface, and the normal there.
    pub fn project(&self, p: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        let mut best: Option<(f32, Vec3, Vec3)> = None;
        for (i, [a, b, c]) in self.tris.iter().enumerate() {
            if let Some(t) = ray_triangle(p, dir, *a, *b, *c) {
                if best.is_none_or(|(bt, _, _)| t.abs() < bt.abs()) {
                    best = Some((t, p + dir * t, self.normals[i]));
                }
            }
        }
        best.map(|(_, q, n)| (q, n))
    }
}

fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
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

/// The signed distance along `d` from `o` to triangle `abc`, either way.
fn ray_triangle(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(e2.dot(q) * inv)
}

/// `input` wrapped onto `target`, both in `input`'s space.
pub fn shrinkwrap(input: &MeshData, target: &MeshData, params: &ShrinkwrapParams) -> MeshData {
    let Some(surface) = Surface::new(target) else {
        return input.clone();
    };
    let normals = super::displace::vertex_normals(input);
    let mut out = input.clone();
    for (p, n) in out.positions_mut().iter_mut().zip(&normals) {
        let hit = match params.mode {
            WrapMode::Nearest => Some(surface.nearest(*p)),
            WrapMode::Project => surface.project(*p, *n),
        };
        if let Some((q, sn)) = hit {
            *p = q + sn * params.offset;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_cube_wrapped_onto_a_sphere_lies_on_it() {
        let sphere = primitives::uv_sphere(1.0, 48, 24);
        let cube = primitives::cube(1.0);
        let out = shrinkwrap(&cube, &sphere, &ShrinkwrapParams::default());
        for p in out.positions() {
            assert!((p.length() - 1.0).abs() < 0.01, "{p:?}");
        }
        // Held off by an offset.
        let off = shrinkwrap(
            &cube,
            &sphere,
            &ShrinkwrapParams {
                mode: WrapMode::Nearest,
                offset: 0.1,
            },
        );
        for p in off.positions() {
            assert!((p.length() - 1.1).abs() < 0.02, "{p:?}");
        }
    }

    #[test]
    fn projecting_along_normals_finds_the_surface_either_way() {
        let sphere = primitives::uv_sphere(2.0, 48, 24);
        // A small cube well inside: its normals point out, the sphere is
        // ahead of every vertex.
        let cube = primitives::cube(0.5);
        let out = shrinkwrap(
            &cube,
            &sphere,
            &ShrinkwrapParams {
                mode: WrapMode::Project,
                offset: 0.0,
            },
        );
        for p in out.positions() {
            assert!((p.length() - 2.0).abs() < 0.02, "{p:?}");
        }
    }
}
