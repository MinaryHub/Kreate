//! STEP (ISO 10303-21, AP214): parts for a CAD program, as faceted solids.
//!
//! A CAD kernel's own STEP carries exact surfaces — a cylinder as a cylinder.
//! Kreate's parts are meshes, so each part goes as a faceted B-rep: every face
//! a planar polygon, the faces a closed shell when the mesh is closed, an
//! open surface model when it is not. Every CAD program reads that, measures
//! it, and can cut and join it; what it cannot do is turn a faceted cylinder
//! back into a true one, which is the honest limit of sending a mesh.
//!
//! Each part is its own product, placed in the world, in the scene's unit.

use std::fmt::Write as _;

use kreate_math::{DAffine3, DVec3};
use kreate_mesh::MeshData;

/// The SI prefix STEP names a unit by, or `None` for a unit that is not
/// metric (inches and feet are written as millimetres).
fn si_prefix(metres_per_unit: f64) -> Option<&'static str> {
    [(1.0, "$"), (1e-2, ".CENTI."), (1e-3, ".MILLI."), (1e-6, ".MICRO.")]
        .iter()
        .find(|(m, _)| (m - metres_per_unit).abs() < 1e-12)
        .map(|(_, p)| *p)
}

/// Writes `parts` as one STEP file in the unit that is `metres_per_unit`
/// metres (a non-metric one becomes millimetres). `name` is the file's.
pub fn write_step(name: &str, parts: &[(String, &MeshData, DAffine3)], metres_per_unit: f64) -> String {
    let (prefix, scale) = match si_prefix(metres_per_unit) {
        Some(p) => (p, metres_per_unit),
        None => (".MILLI.", 1e-3),
    };
    let mut w = Writer::default();
    let app = w.add("APPLICATION_CONTEXT('automotive design')");
    w.add(&format!(
        "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})"
    ));
    let product_context = w.add(&format!("PRODUCT_CONTEXT('',#{app},'mechanical')"));
    let definition_context = w.add(&format!(
        "PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')"
    ));
    let length = w.add(&format!(
        "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT({prefix},.METRE.))"
    ));
    let angle = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))");
    let solid_angle = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())");
    let uncertainty = w.add(&format!(
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{length},'distance_accuracy_value','confusion accuracy')"
    ));
    let context = w.add(&format!(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle}))REPRESENTATION_CONTEXT('Context3D','3D context'))"
    ));
    let origin = w.add("CARTESIAN_POINT('',(0.,0.,0.))");
    let z = w.add("DIRECTION('',(0.,0.,1.))");
    let x = w.add("DIRECTION('',(1.,0.,0.))");
    let placement = w.add(&format!("AXIS2_PLACEMENT_3D('',#{origin},#{z},#{x})"));

    for (part, mesh, world) in parts {
        let part = escape(part);
        let product = w.add(&format!(
            "PRODUCT('{part}','{part}','',(#{product_context}))"
        ));
        let formation = w.add(&format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"));
        let definition = w.add(&format!(
            "PRODUCT_DEFINITION('design','',#{formation},#{definition_context})"
        ));
        let shape = w.add(&format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"));

        // Points, in the world and the unit.
        let points: Vec<usize> = mesh
            .positions()
            .iter()
            .map(|p| {
                let q = world.transform_point3(p.as_dvec3()) / scale;
                w.add(&format!("CARTESIAN_POINT('',({},{},{}))", real(q.x), real(q.y), real(q.z)))
            })
            .collect();
        let flip = world.matrix3.determinant() < 0.0;
        let positions: Vec<DVec3> = mesh.positions().iter().map(|p| p.as_dvec3()).collect();
        let mut faces = Vec::new();
        let mut add_face = |w: &mut Writer, ring: &[u32]| {
            let mut ids: Vec<String> = ring.iter().map(|v| format!("#{}", points[*v as usize])).collect();
            if flip {
                ids.reverse();
            }
            let poly = w.add(&format!("POLY_LOOP('',({}))", ids.join(",")));
            let bound = w.add(&format!("FACE_OUTER_BOUND('',#{poly},.T.)"));
            faces.push(w.add(&format!("FACE('',(#{bound}))")));
        };
        for f in 0..mesh.num_faces() {
            let ring = mesh.face_verts(f);
            if ring.len() < 3 {
                continue;
            }
            // A face that is not flat cannot be one planar polygon: its
            // triangles go instead.
            if ring.len() == 3 || is_planar(&positions, ring) {
                add_face(&mut w, ring);
            } else {
                let points: Vec<_> = ring.iter().map(|v| positions[*v as usize].as_vec3()).collect();
                let mut tris = Vec::new();
                kreate_mesh::tessellate::triangulate_polygon(&points, &mut tris);
                for t in tris {
                    add_face(&mut w, &t.map(|i| ring[i as usize]));
                }
            }
        }
        let list = faces.iter().map(|f| format!("#{f}")).collect::<Vec<_>>().join(",");
        let representation = if kreate_mesh::measure::is_closed(mesh) {
            let shell = w.add(&format!("CLOSED_SHELL('',({list}))"));
            let brep = w.add(&format!("FACETED_BREP('{part}',#{shell})"));
            w.add(&format!(
                "FACETED_BREP_SHAPE_REPRESENTATION('{part}',(#{brep},#{placement}),#{context})"
            ))
        } else {
            let shell = w.add(&format!("OPEN_SHELL('',({list}))"));
            let model = w.add(&format!("SHELL_BASED_SURFACE_MODEL('{part}',(#{shell}))"));
            w.add(&format!(
                "MANIFOLD_SURFACE_SHAPE_REPRESENTATION('{part}',(#{model},#{placement}),#{context})"
            ))
        };
        w.add(&format!("SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})"));
    }

    let mut out = String::new();
    out.push_str("ISO-10303-21;\nHEADER;\n");
    out.push_str("FILE_DESCRIPTION(('Kreate faceted model'),'2;1');\n");
    let _ = writeln!(
        out,
        "FILE_NAME('{}','',(''),(''),'Kreate','Kreate','');",
        escape(name)
    );
    out.push_str("FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n");
    out.push_str(&w.body);
    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    out
}

#[derive(Default)]
struct Writer {
    next: usize,
    body: String,
}

impl Writer {
    fn add(&mut self, entity: &str) -> usize {
        self.next += 1;
        let _ = writeln!(self.body, "#{}={};", self.next, entity);
        self.next
    }
}

/// A real as STEP writes one: always with a point.
fn real(v: f64) -> String {
    let v = if v == 0.0 { 0.0 } else { v };
    // Six places: the points are single precision, and more digits
    // would only write down its rounding.
    let text = format!("{v:.6}");
    let text = text.trim_end_matches('0');
    if text == "-0." {
        "0.".to_string()
    } else if text.contains('.') {
        text.to_string()
    } else {
        format!("{text}.")
    }
}

fn escape(s: &str) -> String {
    s.replace('\'', "''")
}

/// Every corner within a hair of the face's plane.
fn is_planar(points: &[DVec3], ring: &[u32]) -> bool {
    let p: Vec<DVec3> = ring.iter().map(|v| points[*v as usize]).collect();
    let mut n = DVec3::ZERO;
    for i in 0..p.len() {
        n += p[i].cross(p[(i + 1) % p.len()]);
    }
    let n = n.normalize_or_zero();
    if n == DVec3::ZERO {
        return false;
    }
    let size = p.iter().map(|q| (*q - p[0]).length()).fold(0.0, f64::max).max(1e-12);
    p.iter().all(|q| (*q - p[0]).dot(n).abs() <= size * 1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_is_a_closed_faceted_solid_in_millimetres() {
        let cube = kreate_mesh::primitives::cube(0.02);
        let at = DAffine3::from_translation(DVec3::new(0.1, 0.0, 0.0));
        let text = write_step("part.step", &[("Bracket's".to_string(), &cube, at)], 1e-3);
        assert!(text.starts_with("ISO-10303-21;"));
        assert!(text.trim_end().ends_with("END-ISO-10303-21;"));
        assert!(text.contains("SI_UNIT(.MILLI.,.METRE.)"));
        assert!(text.contains("PRODUCT('Bracket''s'"));
        assert!(text.contains("CLOSED_SHELL"));
        assert_eq!(text.matches("POLY_LOOP").count(), 6, "a cube's six quads");
        assert!(text.contains("(110.,10.,10.)"), "100 mm out, 20 mm across");
        // Every reference points at an entity that exists.
        let defined = text.matches("\n#").count();
        for r in text.split('#').skip(1) {
            let n: usize = r.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap();
            assert!(n >= 1 && n <= defined, "#{n} of {defined}");
        }
    }

    #[test]
    fn an_open_surface_is_a_surface_model_and_inches_go_as_millimetres() {
        let plane = kreate_mesh::primitives::plane(1.0);
        let text = write_step("p", &[("Sheet".into(), &plane, DAffine3::IDENTITY)], 0.0254);
        assert!(text.contains("OPEN_SHELL"));
        assert!(text.contains("SHELL_BASED_SURFACE_MODEL"));
        assert!(text.contains(".MILLI."));
    }

    #[test]
    fn reals_always_have_a_point() {
        assert_eq!(real(10.0), "10.");
        assert_eq!(real(-0.5), "-0.5");
        assert_eq!(real(0.0), "0.");
    }
}
