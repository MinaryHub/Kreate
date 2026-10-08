//! 2D drawings: the parts seen square from the top, the front or the right,
//! as lines a drafting program can dimension — DXF, or SVG.
//!
//! What is drawn is what a draughtsman draws: the outline (edges with one
//! face), the creases (edges whose faces meet at more than an angle) and the
//! silhouette (edges between a face turned toward the viewer and one turned
//! away — a cylinder's sides). Each line is split where something in front
//! of it starts or stops hiding it; the hidden runs go on a layer of their
//! own, dashed, so they can be shown or turned off.
//!
//! The visibility is sampled along each edge, a ray toward the viewer from
//! each step, which is exact for straight cover and close for curved.

use std::fmt::Write as _;

use kreate_math::{Bvh, DAffine3, DVec3, Ray, Vec3};
use kreate_mesh::MeshData;

/// Which way the drawing looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    /// From above, looking down Z: X across, Y up the page.
    Top,
    /// From the front, looking along +Y: X across, Z up.
    Front,
    /// From the right, looking along -X: Y across, Z up.
    Right,
}

impl View {
    pub const ALL: [View; 3] = [View::Top, View::Front, View::Right];

    /// Toward the viewer, in the world.
    fn toward(self) -> DVec3 {
        match self {
            View::Top => DVec3::Z,
            View::Front => -DVec3::Y,
            View::Right => DVec3::X,
        }
    }

    /// A world point on the page.
    fn project(self, p: DVec3) -> [f64; 2] {
        match self {
            View::Top => [p.x, p.y],
            View::Front => [p.x, p.z],
            View::Right => [p.y, p.z],
        }
    }
}

/// One line on the page, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line2 {
    pub a: [f64; 2],
    pub b: [f64; 2],
    pub hidden: bool,
}

/// The drawing of `parts` (meshes placed in the world) from `view`. Creases
/// are edges whose faces turn by more than `crease` radians.
pub fn project_drawing(parts: &[(&MeshData, DAffine3)], view: View, crease: f64) -> Vec<Line2> {
    // Every part's triangles in the world, about their common middle so the
    // tree's single precision is spent where the parts are.
    let mut world: Vec<Vec<DVec3>> = Vec::new();
    for (mesh, place) in parts {
        world.push(mesh.positions().iter().map(|p| place.transform_point3(p.as_dvec3())).collect());
    }
    let all: Vec<DVec3> = world.iter().flatten().copied().collect();
    if all.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = all.iter().fold((DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)), |(lo, hi), p| {
        (lo.min(*p), hi.max(*p))
    });
    let middle = (lo + hi) * 0.5;
    let size = (hi - lo).length().max(1e-9);
    let mut local: Vec<Vec3> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for ((mesh, _), points) in parts.iter().zip(&world) {
        let base = local.len() as u32;
        local.extend(points.iter().map(|p| (*p - middle).as_vec3()));
        let t = mesh.triangulate();
        tris.extend(t.indices.as_chunks::<3>().0.iter().map(|t| t.map(|v| v + base)));
    }
    let bvh = Bvh::build(&local, &tris);
    let toward = view.toward();
    let lift = size * 1e-5;
    let hidden_at = |p: DVec3| -> bool {
        let start = (p - middle + toward * lift).as_vec3();
        bvh.raycast(&local, &tris, &Ray::new(start, toward.as_vec3()), f32::MAX).is_some()
    };

    let mut lines = Vec::new();
    for ((mesh, place), points) in parts.iter().zip(&world) {
        let flip = place.matrix3.determinant() < 0.0;
        let normals: Vec<DVec3> = (0..mesh.num_faces())
            .map(|f| {
                let ring = mesh.face_verts(f);
                let mut n = DVec3::ZERO;
                for i in 0..ring.len() {
                    n += points[ring[i] as usize].cross(points[ring[(i + 1) % ring.len()] as usize]);
                }
                let n = n.normalize_or_zero();
                if flip { -n } else { n }
            })
            .collect();
        let mut faces_of: Vec<Vec<usize>> = vec![Vec::new(); mesh.num_edges() as usize];
        let offsets = mesh.face_offsets();
        for (c, e) in mesh.corner_edges().iter().enumerate() {
            let f = offsets.partition_point(|o| *o as usize <= c) - 1;
            faces_of[*e as usize].push(f);
        }
        for (e, [a, b]) in mesh.edges().iter().enumerate() {
            let drawn = match faces_of[e].as_slice() {
                [_] => true,
                [f, g] => {
                    let (nf, ng) = (normals[*f], normals[*g]);
                    nf.angle_between(ng) > crease
                        || (nf.dot(toward) > 0.0) != (ng.dot(toward) > 0.0)
                }
                _ => true,
            };
            if !drawn {
                continue;
            }
            let (pa, pb) = (points[*a as usize], points[*b as usize]);
            // Seen end on, an edge is a point: nothing to draw.
            let (sa, sb) = (view.project(pa), view.project(pb));
            if (sa[0] - sb[0]).hypot(sa[1] - sb[1]) < size * 1e-9 {
                continue;
            }
            let steps = ((pa.distance(pb) / (size / 200.0)).ceil() as usize).clamp(1, 64);
            let mut run_start = 0usize;
            let mut run_hidden = hidden_at(pa.lerp(pb, 0.5 / steps as f64));
            for k in 1..=steps {
                let hidden = if k < steps {
                    hidden_at(pa.lerp(pb, (k as f64 + 0.5) / steps as f64))
                } else {
                    !run_hidden
                };
                if hidden != run_hidden || k == steps {
                    let (t0, t1) = (run_start as f64 / steps as f64, k as f64 / steps as f64);
                    lines.push(Line2 {
                        a: view.project(pa.lerp(pb, t0)),
                        b: view.project(pa.lerp(pb, t1)),
                        hidden: run_hidden,
                    });
                    run_start = k;
                    run_hidden = hidden;
                }
            }
        }
    }
    tidy(lines, size * 1e-6)
}

/// Lines that fall on the same place on the page, drawn once: an edge seen
/// straight behind another — a box's bottom under its top — is the same
/// line, and where a visible and a hidden line coincide the visible one is
/// drawn, as a draughtsman would.
fn tidy(lines: Vec<Line2>, tolerance: f64) -> Vec<Line2> {
    let key = |l: &Line2| {
        let q = |v: f64| (v / tolerance).round() as i64;
        let (a, b) = ([q(l.a[0]), q(l.a[1])], [q(l.b[0]), q(l.b[1])]);
        if a <= b { (a, b) } else { (b, a) }
    };
    let mut seen: std::collections::HashMap<([i64; 2], [i64; 2]), usize> = Default::default();
    let mut out: Vec<Line2> = Vec::new();
    for l in lines {
        match seen.get(&key(&l)) {
            Some(&i) => {
                if !l.hidden {
                    out[i].hidden = false;
                }
            }
            None => {
                seen.insert(key(&l), out.len());
                out.push(l);
            }
        }
    }
    out
}

/// AutoCAD's code for a length unit, for `$INSUNITS`.
fn insunits(metres_per_unit: f64) -> u32 {
    [(0.0254, 1), (0.3048, 2), (1e-3, 4), (1e-2, 5), (1.0, 6)]
        .iter()
        .find(|(m, _)| (m - metres_per_unit).abs() < 1e-12)
        .map(|(_, c)| *c)
        .unwrap_or(0)
}

/// The drawing as DXF (R12 entities, with the unit in the header): visible
/// lines on layer VISIBLE, hidden ones on HIDDEN, dashed.
pub fn write_dxf(lines: &[Line2], metres_per_unit: f64) -> String {
    let s = 1.0 / metres_per_unit;
    let mut out = String::new();
    let pair = |out: &mut String, code: u32, value: &str| {
        let _ = write!(out, "{code}\n{value}\n");
    };
    pair(&mut out, 0, "SECTION");
    pair(&mut out, 2, "HEADER");
    pair(&mut out, 9, "$INSUNITS");
    pair(&mut out, 70, &insunits(metres_per_unit).to_string());
    pair(&mut out, 0, "ENDSEC");
    pair(&mut out, 0, "SECTION");
    pair(&mut out, 2, "TABLES");
    pair(&mut out, 0, "TABLE");
    pair(&mut out, 2, "LTYPE");
    pair(&mut out, 70, "2");
    for (name, text, elements) in [
        ("CONTINUOUS", "Solid line", Vec::<f64>::new()),
        ("DASHED", "Dashed __ __", vec![0.5 * s * 0.01, -0.25 * s * 0.01]),
    ] {
        pair(&mut out, 0, "LTYPE");
        pair(&mut out, 2, name);
        pair(&mut out, 70, "0");
        pair(&mut out, 3, text);
        pair(&mut out, 72, "65");
        pair(&mut out, 73, &elements.len().to_string());
        pair(&mut out, 40, &format!("{}", elements.iter().map(|e| e.abs()).sum::<f64>()));
        for e in elements {
            pair(&mut out, 49, &format!("{e}"));
        }
    }
    pair(&mut out, 0, "ENDTAB");
    pair(&mut out, 0, "TABLE");
    pair(&mut out, 2, "LAYER");
    pair(&mut out, 70, "2");
    for (name, colour, ltype) in [("VISIBLE", "7", "CONTINUOUS"), ("HIDDEN", "8", "DASHED")] {
        pair(&mut out, 0, "LAYER");
        pair(&mut out, 2, name);
        pair(&mut out, 70, "0");
        pair(&mut out, 62, colour);
        pair(&mut out, 6, ltype);
    }
    pair(&mut out, 0, "ENDTAB");
    pair(&mut out, 0, "ENDSEC");
    pair(&mut out, 0, "SECTION");
    pair(&mut out, 2, "ENTITIES");
    for l in lines {
        pair(&mut out, 0, "LINE");
        pair(&mut out, 8, if l.hidden { "HIDDEN" } else { "VISIBLE" });
        for (code, v) in [(10, l.a[0]), (20, l.a[1]), (30, 0.0), (11, l.b[0]), (21, l.b[1]), (31, 0.0)] {
            pair(&mut out, code, &format!("{:.6}", v * s));
        }
    }
    pair(&mut out, 0, "ENDSEC");
    pair(&mut out, 0, "EOF");
    out
}

/// The drawing as SVG, one unit of the scene to one user unit, the page's Y
/// running down as SVG's does; hidden lines grey and dashed.
pub fn write_svg(lines: &[Line2], metres_per_unit: f64, unit: &str) -> String {
    let s = 1.0 / metres_per_unit;
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for l in lines {
        for p in [l.a, l.b] {
            lo = [lo[0].min(p[0] * s), lo[1].min(p[1] * s)];
            hi = [hi[0].max(p[0] * s), hi[1].max(p[1] * s)];
        }
    }
    if lines.is_empty() {
        lo = [0.0; 2];
        hi = [1.0; 2];
    }
    let margin = ((hi[0] - lo[0]).max(hi[1] - lo[1]) * 0.05).max(1e-6);
    let (w, h) = (hi[0] - lo[0] + 2.0 * margin, hi[1] - lo[1] + 2.0 * margin);
    let stroke = (w.max(h) / 400.0).max(1e-6);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.4}{unit}\" height=\"{h:.4}{unit}\" viewBox=\"{:.6} {:.6} {w:.6} {h:.6}\">",
        lo[0] - margin,
        -hi[1] - margin
    );
    for (hidden, style) in [
        (true, format!("stroke=\"#888\" stroke-dasharray=\"{:.6} {:.6}\"", stroke * 6.0, stroke * 3.0)),
        (false, "stroke=\"#000\"".to_string()),
    ] {
        let _ = writeln!(out, " <g fill=\"none\" stroke-width=\"{stroke:.6}\" {style}>");
        for l in lines.iter().filter(|l| l.hidden == hidden) {
            let _ = writeln!(
                out,
                "  <line x1=\"{:.6}\" y1=\"{:.6}\" x2=\"{:.6}\" y2=\"{:.6}\"/>",
                l.a[0] * s,
                -l.a[1] * s,
                l.b[0] * s,
                -l.b[1] * s
            );
        }
        out.push_str(" </g>\n");
    }
    out.push_str("</svg>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(lines: &[Line2], hidden: bool) -> f64 {
        lines
            .iter()
            .filter(|l| l.hidden == hidden)
            .map(|l| (l.a[0] - l.b[0]).hypot(l.a[1] - l.b[1]))
            .sum()
    }

    #[test]
    fn a_cube_from_the_top_is_a_square_with_its_bottom_hidden() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let lines = project_drawing(&[(&cube, DAffine3::IDENTITY)], View::Top, 0.5);
        // The top's four edges, seen; the bottom's lie under them and are the
        // same lines; the uprights are seen end on and drawn not at all.
        assert!((total(&lines, false) - 8.0).abs() < 1e-6, "{lines:?}");
        assert_eq!(total(&lines, true), 0.0, "{lines:?}");
    }

    #[test]
    fn a_box_behind_another_is_drawn_hidden() {
        // From the front, a small cube straight behind a big one: its square
        // outline is all behind the big one's face, so all of it is hidden,
        // and the big one's outline is all seen.
        let big = kreate_mesh::primitives::cube(2.0);
        let small = kreate_mesh::primitives::cube(1.0);
        let parts = [
            (&big, DAffine3::IDENTITY),
            (&small, DAffine3::from_translation(DVec3::new(0.0, 3.0, 0.0))),
        ];
        let lines = project_drawing(&parts, View::Front, 0.5);
        assert!((total(&lines, false) - 8.0).abs() < 1e-6, "{lines:?}");
        assert!((total(&lines, true) - 4.0).abs() < 1e-6, "{lines:?}");
    }

    #[test]
    fn dxf_and_svg_carry_the_lines_in_the_unit() {
        let lines = [
            Line2 { a: [0.0, 0.0], b: [0.01, 0.0], hidden: false },
            Line2 { a: [0.0, 0.0], b: [0.0, 0.02], hidden: true },
        ];
        let dxf = write_dxf(&lines, 1e-3);
        assert!(dxf.contains("$INSUNITS\n70\n4\n"));
        assert!(dxf.contains("8\nVISIBLE\n") && dxf.contains("8\nHIDDEN\n"));
        assert!(dxf.contains("11\n10.000000\n"), "10 mm long");
        assert!(dxf.trim_end().ends_with("EOF"));
        let svg = write_svg(&lines, 1e-3, "mm");
        assert!(svg.contains("x2=\"10.000000\""));
        assert!(svg.contains("stroke-dasharray"));
    }
}
