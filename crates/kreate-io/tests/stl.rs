//! STL, both forms, in and out.
//!
//! What matters here is not the parsing — STL is a bag of triangles and there is
//! nothing to get wrong in the layout. It is everything the file does *not* say:
//! no shared vertices, no units, no orientation, and normals that may or may not
//! agree with the geometry they describe.

use kreate_io::ir::LossyNote;
use kreate_io::stl::{self, StlOptions};
use kreate_mesh::MeshData;

/// A binary STL of `triangles`, each given as three corners.
fn binary(triangles: &[[[f32; 3]; 3]], normals: Option<&[[f32; 3]]>) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out.extend_from_slice(&(triangles.len() as u32).to_le_bytes());
    for (i, t) in triangles.iter().enumerate() {
        let n = normals.map(|n| n[i]).unwrap_or([0.0, 0.0, 0.0]);
        for v in [n, t[0], t[1], t[2]] {
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

/// The two triangles of a unit square, sharing an edge — as soup.
fn square() -> Vec<[[f32; 3]; 3]> {
    vec![
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ]
}

#[test]
fn a_binary_file_is_welded_into_a_shared_surface() {
    // The reason STL needs an importer rather than a reader. Six vertices go in
    // and four come out, and the two triangles now share an edge — without which
    // the mesh comes apart on the first edit.
    let scene = stl::parse_stl(&binary(&square(), None), &StlOptions::default()).expect("parses");
    let mesh = &scene.meshes[0];
    assert_eq!(mesh.positions.len(), 4, "the soup was not welded");
    assert_eq!(mesh.faces.len(), 2);
    let shared = mesh.faces[0]
        .iter()
        .filter(|v| mesh.faces[1].contains(v))
        .count();
    assert_eq!(shared, 2, "the triangles share no edge");
}

#[test]
fn the_assumed_units_and_axis_are_recorded_rather_than_applied_silently() {
    // A model that arrives a thousand times too large is the commonest STL
    // complaint there is, and the answer is always the unit assumption. It has to
    // be somewhere the user can find it.
    let scene = stl::parse_stl(&binary(&square(), None), &StlOptions::default()).expect("parses");
    assert!(
        scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::AssumedUnitScale { .. })),
        "the unit assumption was not recorded"
    );
    assert!(
        scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::AssumedAxisConvention { .. })),
        "the axis assumption was not recorded"
    );
    assert_eq!(scene.source.unit_scale, 0.001, "STL should default to mm");
}

#[test]
fn a_triangle_wound_against_its_stored_normal_is_flipped() {
    // A real defect in real files, and the one legitimate use of the stored
    // normal. Left alone, such a triangle is lit from the inside and the model
    // has patches that look wrong for no visible reason.
    let mut triangles = square();
    // Reverse the first triangle's winding while leaving its normal pointing the
    // original way.
    triangles[0].swap(1, 2);
    let normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
    let scene = stl::parse_stl(&binary(&triangles, Some(&normals)), &StlOptions::default())
        .expect("parses");

    assert!(
        scene.source.notes.iter().any(|n| matches!(
            n,
            LossyNote::Unsupported { what } if what.contains("wound against")
        )),
        "the flip was not recorded: {:?}",
        scene.source.notes
    );

    // And both faces now wind the same way.
    let mesh = &scene.meshes[0];
    let normal_of = |face: &Vec<u32>| {
        let p = |i: usize| mesh.positions[face[i] as usize];
        (p(1) - p(0)).cross(p(2) - p(0)).normalize_or_zero()
    };
    let a = normal_of(&mesh.faces[0]);
    let b = normal_of(&mesh.faces[1]);
    assert!(
        a.dot(b) > 0.9,
        "the two faces still wind oppositely: {a} and {b}"
    );
}

#[test]
fn a_normal_too_small_to_be_a_direction_is_ignored() {
    // Plenty of exporters never fill the normal in, and some fill it with
    // near-zero garbage. Treating such a value as a direction flips triangles at
    // random, which is worse than ignoring the field entirely.
    //
    // The test uses a *tiny* normal rather than an exactly zero one, and that is
    // the whole point: a zero normal has a dot product of exactly zero with any
    // winding, so it is never "opposite" and the guard never fires. A falsifying
    // edit that removed the length check passed a zero-normal test cleanly. Only
    // a small but non-zero value pointing the wrong way can see it.
    let tiny_opposing = [[0.0, 0.0, -1e-30f32], [0.0, 0.0, -1e-30]];
    let scene = stl::parse_stl(
        &binary(&square(), Some(&tiny_opposing)),
        &StlOptions::default(),
    )
    .expect("parses");
    assert!(
        !scene.source.notes.iter().any(|n| matches!(
            n,
            LossyNote::Unsupported { what } if what.contains("wound against")
        )),
        "a normal too small to be a direction was used to flip a triangle: {:?}",
        scene.source.notes
    );
}

#[test]
fn an_exactly_zero_normal_is_also_ignored() {
    // The common case, kept alongside the one above. It passes for a different
    // reason — a zero dot product is not negative — and that difference is
    // exactly why the test above had to be written the way it is.
    let scene = stl::parse_stl(
        &binary(&square(), Some(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        &StlOptions::default(),
    )
    .expect("parses");
    assert!(!scene.source.notes.iter().any(|n| matches!(
        n,
        LossyNote::Unsupported { what } if what.contains("wound against")
    )));
}

#[test]
fn an_ascii_file_parses() {
    let text = "\
solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
";
    let scene = stl::parse_stl(text.as_bytes(), &StlOptions::default()).expect("parses");
    assert_eq!(scene.meshes[0].faces.len(), 2);
    assert_eq!(
        scene.meshes[0].positions.len(),
        4,
        "the ascii soup was not welded"
    );
}

#[test]
fn a_binary_file_whose_header_says_solid_is_not_read_as_ascii() {
    // The trap in the usual test. Plenty of binary writers put a description in
    // the 80-byte header that begins with the word `solid`, and a reader that
    // dispatches on the leading word gets nothing out of such a file.
    let mut bytes = binary(&square(), None);
    let banner = b"solid created by some other program";
    bytes[..banner.len()].copy_from_slice(banner);

    let scene = stl::parse_stl(&bytes, &StlOptions::default()).expect("parses");
    assert_eq!(
        scene.meshes[0].faces.len(),
        2,
        "a binary file was misread as ascii because its header said 'solid'"
    );
}

#[test]
fn a_binary_file_whose_header_contains_the_word_facet_is_still_binary() {
    // The case that makes the size test worth having rather than dispatching on
    // the file's text. A header of "solid faceted bracket" is an entirely
    // ordinary thing for an exporter to write, and it defeats *both* naive
    // tests: it begins with `solid`, and it contains `facet`.
    //
    // Read as ASCII it yields a mesh with no faces — a silent empty import. Only
    // the length rule gets it right, and a falsifying edit that removed that rule
    // passed every other test in this file because the fallback happened to
    // rescue them.
    let mut bytes = binary(&square(), None);
    let banner = b"solid faceted bracket, exported by something";
    bytes[..banner.len()].copy_from_slice(banner);

    let scene = stl::parse_stl(&bytes, &StlOptions::default()).expect("parses");
    assert_eq!(
        scene.meshes[0].faces.len(),
        2,
        "a binary file was read as ascii because its header mentioned facets"
    );
}

#[test]
fn an_impossible_triangle_count_is_refused_rather_than_allocated() {
    // The same defect that fuzzing found in `.kmesh`, guarded against from the
    // start here. The header states a count that nothing else in the file
    // corroborates; four billion triangles asks for 200 GB, and an allocation
    // failure aborts the process rather than returning an error.
    let mut bytes = binary(&square(), None);
    bytes[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
    match stl::parse_stl(&bytes, &StlOptions::default()) {
        Err(stl::StlError::ImpossibleTriangleCount { claimed, .. }) => {
            assert_eq!(claimed, u32::MAX);
        }
        other => panic!(
            "expected a refusal, got {:?}",
            other.map(|s| s.meshes.len())
        ),
    }
}

#[test]
fn every_truncation_of_a_binary_file_is_handled() {
    // Interrupted downloads and partial writes. Some prefixes are legitimately
    // readable — a file with fewer triangles than the header claims is refused,
    // but one truncated inside the header falls through to the ascii reader,
    // which finds nothing. Neither may panic.
    let bytes = binary(&square(), None);
    for len in 0..bytes.len() {
        let _ = stl::parse_stl(&bytes[..len], &StlOptions::default());
    }
    assert!(stl::parse_stl(&bytes, &StlOptions::default()).is_ok());
}

#[test]
fn a_mesh_survives_a_round_trip_through_binary_stl() {
    // Not an identity — STL cannot store n-gons, names or UVs — but the geometry
    // has to come back. Scaled through millimetres and back, so the unit
    // conversion is exercised in both directions rather than cancelling by being
    // skipped.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 16, 8);
    let options = StlOptions::default();
    let (bytes, _) = stl::write_stl(&mesh, &options);
    let scene = stl::parse_stl(&bytes, &options).expect("parses");
    let back = &scene.meshes[0];

    // A sphere's quads become two triangles each, and its poles are triangles.
    let expected_triangles: usize = mesh.faces().map(|f| f.len().saturating_sub(2)).sum();
    assert_eq!(back.faces.len(), expected_triangles, "triangle count");

    // Welded back down to about the original vertex count. Not exactly: a sphere
    // has a seam where the first and last column of vertices coincide, and those
    // weld together on the way back in.
    assert!(
        back.positions.len() <= mesh.num_verts() as usize,
        "welding produced more vertices than the original mesh had: {} against {}",
        back.positions.len(),
        mesh.num_verts()
    );
    assert!(
        back.positions.len() as f32 > mesh.num_verts() as f32 * 0.8,
        "too many vertices were merged: {} against {}",
        back.positions.len(),
        mesh.num_verts()
    );
}

#[test]
fn exporting_an_ngon_says_that_it_was_triangulated() {
    // STL has no n-gons and no extension mechanism to record what was dropped,
    // the way glTF has. So the loss travels back to the caller rather than
    // happening quietly.
    let mesh = MeshData::from_faces(
        vec![
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 1.0, 0.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
        ],
        &[vec![0, 1, 2, 3]],
    );
    let (_, notes) = stl::write_stl(&mesh, &StlOptions::default());
    assert!(
        notes.iter().any(|n| matches!(
            n,
            LossyNote::Unsupported { what } if what.contains("triangulated")
        )),
        "the triangulation was not reported: {notes:?}"
    );
}

#[test]
fn an_exported_file_does_not_begin_with_solid() {
    // So the next program to read it is not caught by the ascii/binary trap this
    // importer has to work around.
    let (bytes, _) = stl::write_stl(&kreate_mesh::primitives::cube(1.0), &StlOptions::default());
    assert!(
        !bytes.starts_with(b"solid"),
        "an exported binary STL begins with the word that makes readers guess wrong"
    );
}

#[test]
fn an_exported_normal_agrees_with_its_winding() {
    // Redundant, but some readers trust it. It should agree with the geometry
    // rather than contradict it.
    let mesh = kreate_mesh::primitives::cube(2.0);
    let (bytes, _) = stl::write_stl(&mesh, &StlOptions::default());
    let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    assert!(count > 0);

    for i in 0..count {
        let at = 84 + i * 50;
        let f = |k: usize| {
            let o = at + k * 4;
            f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
        };
        let n = glam::Vec3::new(f(0), f(1), f(2));
        let a = glam::Vec3::new(f(3), f(4), f(5));
        let b = glam::Vec3::new(f(6), f(7), f(8));
        let c = glam::Vec3::new(f(9), f(10), f(11));
        let wound = (b - a).cross(c - a).normalize_or_zero();
        assert!(
            n.dot(wound) > 0.9,
            "triangle {i}: the stored normal {n} disagrees with the winding {wound}"
        );
    }
}

#[test]
fn an_empty_mesh_exports_and_reimports() {
    let (bytes, _) = stl::write_stl(&MeshData::empty(), &StlOptions::default());
    let scene = stl::parse_stl(&bytes, &StlOptions::default()).expect("parses");
    assert!(scene.meshes[0].faces.is_empty());
}
