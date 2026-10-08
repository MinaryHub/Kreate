//! 3MF: the 3D printing format. A zip holding an XML model.
//!
//! Unlike STL it says what unit it is in, keeps one object per part with its
//! name, and is always triangles, wound outward, which is what a slicer
//! wants. Kreate writes each part as it stands — modifiers applied, placed in
//! the world — in the scene's unit, so a part drawn as 40 mm arrives as 40 mm
//! whatever the slicer assumes.
//!
//! Only what printing needs: the meshes. No materials, colours or build
//! plate settings.

use std::io::{Cursor, Write};

use kreate_math::DAffine3;
use kreate_mesh::MeshData;

/// The units 3MF knows, by Kreate's length unit tag; anything else is
/// written in millimetres.
pub fn unit_name(metres_per_unit: f64) -> (&'static str, f64) {
    const UNITS: [(&str, f64); 6] = [
        ("micron", 1e-6),
        ("millimeter", 1e-3),
        ("centimeter", 1e-2),
        ("inch", 0.0254),
        ("foot", 0.3048),
        ("meter", 1.0),
    ];
    UNITS
        .iter()
        .find(|(_, m)| (m - metres_per_unit).abs() < 1e-12)
        .copied()
        .unwrap_or(("millimeter", 1e-3))
}

/// Writes `parts` — a name, a mesh and where it is in the world — as a 3MF
/// file in the unit that is `metres_per_unit` metres.
pub fn write_3mf(parts: &[(String, &MeshData, DAffine3)], metres_per_unit: f64) -> Vec<u8> {
    let (unit, scale) = unit_name(metres_per_unit);
    let mut model = String::new();
    model.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    model.push_str(&format!(
        "<model unit=\"{unit}\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
    ));
    model.push_str(" <resources>\n");
    let mut ids = Vec::new();
    for (i, (name, mesh, world)) in parts.iter().enumerate() {
        let id = i + 1;
        let tris = mesh.triangulate();
        if tris.indices.is_empty() {
            continue;
        }
        ids.push(id);
        model.push_str(&format!(
            "  <object id=\"{id}\" type=\"model\" name=\"{}\">\n   <mesh>\n    <vertices>\n",
            escape(name)
        ));
        for p in mesh.positions() {
            let w = world.transform_point3(p.as_dvec3()) / scale;
            model.push_str(&format!(
                "     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                number(w.x),
                number(w.y),
                number(w.z)
            ));
        }
        model.push_str("    </vertices>\n    <triangles>\n");
        // A mirrored placement turns the surface inside out; the triangles are
        // turned back so they still face outward.
        let flip = world.matrix3.determinant() < 0.0;
        for t in tris.indices.as_chunks::<3>().0 {
            let (a, b, c) = if flip { (t[0], t[2], t[1]) } else { (t[0], t[1], t[2]) };
            model.push_str(&format!("     <triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\"/>\n"));
        }
        model.push_str("    </triangles>\n   </mesh>\n  </object>\n");
    }
    model.push_str(" </resources>\n <build>\n");
    for id in ids {
        model.push_str(&format!("  <item objectid=\"{id}\"/>\n"));
    }
    model.push_str(" </build>\n</model>\n");

    let content_types = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n\
 <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n\
 <Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\n\
</Types>\n";
    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n\
 <Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/>\n\
</Relationships>\n";

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .last_modified_time(zip::DateTime::default());
        for (path, body) in [
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", rels),
            ("3D/3dmodel.model", model.as_str()),
        ] {
            // Writing to memory cannot fail but for running out of it.
            zip.start_file(path, options).expect("in memory");
            zip.write_all(body.as_bytes()).expect("in memory");
        }
        zip.finish().expect("in memory");
    }
    buffer.into_inner()
}

/// A coordinate, with what precision a printer can use and no more.
fn number(v: f64) -> String {
    let text = format!("{v:.6}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" || text.is_empty() {
        "0".to_string()
    } else {
        text.to_string()
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn model_of(bytes: &[u8]) -> String {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).expect("a zip");
        let mut text = String::new();
        zip.by_name("3D/3dmodel.model")
            .expect("the model")
            .read_to_string(&mut text)
            .unwrap();
        assert!(zip.by_name("[Content_Types].xml").is_ok());
        assert!(zip.by_name("_rels/.rels").is_ok());
        text
    }

    #[test]
    fn a_part_is_written_in_its_unit_where_it_stands() {
        let cube = kreate_mesh::primitives::cube(0.02); // 20 mm
        let at = DAffine3::from_translation(kreate_math::DVec3::new(0.1, 0.0, 0.0));
        let bytes = write_3mf(&[("Part <1>".to_string(), &cube, at)], 0.001);
        let model = model_of(&bytes);
        assert!(model.contains("unit=\"millimeter\""));
        assert!(model.contains("name=\"Part &lt;1&gt;\""));
        // 100 mm along X, give or take the half width.
        assert!(model.contains("x=\"110\"") && model.contains("x=\"90\""), "{model}");
        assert_eq!(model.matches("<triangle ").count(), 12);
        assert_eq!(model.matches("<vertex ").count(), 8);
        assert!(model.contains("<item objectid=\"1\"/>"));
    }

    #[test]
    fn a_mirrored_part_still_faces_out() {
        let cube = kreate_mesh::primitives::cube(1.0);
        let plain = model_of(&write_3mf(&[("a".into(), &cube, DAffine3::IDENTITY)], 1.0));
        let mirrored = model_of(&write_3mf(
            &[(
                "a".into(),
                &cube,
                DAffine3::from_scale(kreate_math::DVec3::new(-1.0, 1.0, 1.0)),
            )],
            1.0,
        ));
        let first = |m: &str| m.split("<triangle ").nth(1).unwrap().split("/>").next().unwrap().to_string();
        assert_ne!(first(&plain), first(&mirrored), "the winding turned with the mirror");
    }

    #[test]
    fn an_unknown_unit_is_millimetres() {
        assert_eq!(unit_name(0.5).0, "millimeter");
        assert_eq!(unit_name(0.0254).0, "inch");
    }
}
