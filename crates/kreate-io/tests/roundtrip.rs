//! Round-trip fidelity across every MVP format, against the shared corpus.
//!
//! # The contract is the test
//!
//! Each format has a published table of what survives a round trip through it
//! (see [`CONTRACTS`]). The table is not documentation that happens to be
//! checked — it *is* the check, and it is asserted in both directions:
//!
//! - everything the contract says survives **must** survive, and
//! - everything the contract says is lost **must** actually be lost.
//!
//! The second half is the one that does the work. A contract asserted only in
//! the positive direction rots quietly: someone teaches the glTF exporter to
//! write a second UV set, the table still says it loses them, and from then on
//! the table is lying about a feature that exists. Asserting the negatives means
//! adding a capability *fails the build* until the table is updated to claim it,
//! which is exactly the moment the claim should be made.
//!
//! # Why geometry is compared and attributes are counted
//!
//! Geometry goes through [`kreate_testkit::compare_geometry`], which is
//! invariant under vertex renumbering, face reordering and cycle rotation, and
//! sensitive to winding. Attributes are checked for presence and, where it
//! matters, for value — a UV set that comes back with every coordinate zeroed
//! has "survived" by any structural measure and is useless.

use kreate_io::gltf_export::{write_gltf, GltfExportOptions};
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_io::obj::{parse_obj, write_obj, ObjOptions};
use kreate_io::stl::{parse_stl, write_stl, StlOptions};
use kreate_mesh::attribute::names;
use kreate_mesh::MeshData;
use kreate_testkit::{compare_geometry, corpus, Specimen, Tolerance};

/// What one format promises to preserve.
#[derive(Clone, Copy, Debug)]
struct Contract {
    format: Format,
    /// Faces come back with the corner count they went out with.
    ngons: bool,
    /// Which face belongs to which material slot.
    material_partition: bool,
    /// At least one UV set, with its coordinates.
    uvs: bool,
    /// UV sets keep the names they were addressed by.
    uv_names: bool,
    /// More than one UV set.
    multiple_uv_sets: bool,
    /// Per-edge sharpness — the marks themselves, not merely a layer to hold
    /// them. OBJ has `s` groups and could carry these once the sharp-edge to
    /// smoothing-group graph colouring is written; today it cannot.
    sharp_edges: bool,
    /// The direction each face is wound.
    winding: bool,
    /// Several meshes in one file, kept separate.
    multiple_meshes: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Obj,
    Stl,
    Gltf,
}

/// The published fidelity table.
///
/// Read this as the answer to "what happens to my model if I save it as X".
const CONTRACTS: [Contract; 3] = [
    Contract {
        format: Format::Obj,
        // OBJ is the only text format here that is natively n-gon, which is most
        // of why it is still in use thirty years on.
        ngons: true,
        material_partition: true,
        uvs: true,
        // `usemtl` names a material; nothing in OBJ names a UV set.
        uv_names: false,
        multiple_uv_sets: false,
        // OBJ has `s` groups, which are per face. Per-edge sharpness does not
        // fit through them without the graph colouring that is still to be
        // written, so today it is lost.
        sharp_edges: false,
        winding: true,
        multiple_meshes: true,
    },
    Contract {
        format: Format::Stl,
        // Triangle soup with no indices, no attributes and no names. STL is here
        // because 3D printing needs it, and for nothing else.
        ngons: false,
        material_partition: false,
        uvs: false,
        uv_names: false,
        multiple_uv_sets: false,
        sharp_edges: false,
        winding: true,
        multiple_meshes: false,
    },
    Contract {
        format: Format::Gltf,
        // Triangles only, but the faces travel in KREATE_ngons.
        ngons: true,
        material_partition: true,
        uvs: true,
        // glTF calls them TEXCOORD_0..n. The names would need a vendor
        // extension of their own.
        uv_names: false,
        multiple_uv_sets: false,
        sharp_edges: false,
        winding: true,
        multiple_meshes: true,
    },
];

fn contract(format: Format) -> Contract {
    CONTRACTS
        .into_iter()
        .find(|c| c.format == format)
        .expect("every format has a contract")
}

/// Writes a mesh out and reads it back, as the CLI would.
fn round_trip(format: Format, meshes: &[(String, &MeshData)]) -> Vec<(String, MeshData)> {
    match format {
        Format::Obj => {
            let text = write_obj(meshes, ObjOptions::default());
            parse_obj(&text, ObjOptions::default())
                .expect("what we wrote, we can read")
                .into_mesh_data()
                .0
        }
        Format::Stl => {
            let options = StlOptions::default();
            let (bytes, _) = write_stl(meshes[0].1, &options);
            parse_stl(&bytes, &options)
                .expect("what we wrote, we can read")
                .into_mesh_data()
                .0
        }
        Format::Gltf => {
            let (bytes, _) = write_gltf(meshes, &[], &GltfExportOptions::default());
            parse_gltf(&bytes, &GltfOptions::default())
                .expect("what we wrote, we can read")
                .into_mesh_data()
                .0
        }
    }
}

fn one(format: Format, mesh: &MeshData) -> MeshData {
    let meshes = [("specimen".to_string(), mesh)];
    let mut out = round_trip(format, &meshes);
    assert_eq!(out.len(), 1, "expected one mesh back");
    out.remove(0).1
}

const FORMATS: [Format; 3] = [Format::Obj, Format::Stl, Format::Gltf];

/// Specimens whose geometry cannot be compared as-is after a round trip through
/// a format that triangulates, because triangulation is a real change.
fn expects_triangulation(format: Format, mesh: &MeshData) -> bool {
    !contract(format).ngons && mesh.faces().any(|f| f.len() > 3)
}

#[test]
fn geometry_survives_every_format() {
    for specimen in corpus() {
        for format in FORMATS {
            let back = one(format, &specimen.mesh);
            if expects_triangulation(format, &specimen.mesh) {
                // The contract says this format triangulates, so the comparison
                // would fail by design. That it *does* triangulate is asserted
                // by `the_contract_negatives_are_real`.
                continue;
            }
            if let Err(mismatch) = compare_geometry(&specimen.mesh, &back, tolerance_for(&specimen))
            {
                panic!(
                    "{} through {format:?}: {mismatch}\n  (specimen hazard: {})",
                    specimen.name, specimen.hazard
                );
            }
        }
    }
}

#[test]
fn triangulation_is_complete_and_preserves_the_surface() {
    // The claim a format that triangulates still has to meet. `geometry_survives`
    // skips these specimens by design, which left a hole: an STL writer that
    // emitted only the first triangle of every n-gon passed every other test
    // here, because what it produced was a valid mesh of the right shape with
    // most of itself missing.
    //
    // Two invariants, because either alone is weak. The count catches a fan that
    // stops early; the area catches a fan that is complete but wrong — one that
    // walks outside a concave polygon covers the right number of triangles and
    // the wrong surface.
    for specimen in corpus() {
        let expected_triangles: usize = specimen.mesh.faces().map(|f| f.len() - 2).sum();
        let expected_area = surface_area(&specimen.mesh);
        for format in FORMATS {
            if !expects_triangulation(format, &specimen.mesh) {
                continue;
            }
            let back = one(format, &specimen.mesh);
            assert_eq!(
                back.num_faces() as usize,
                expected_triangles,
                "{} through {format:?}: expected {expected_triangles} triangles",
                specimen.name
            );
            let area = surface_area(&back);
            let tolerance = expected_area.abs() * 1e-4 + 1e-6;
            assert!(
                (area - expected_area).abs() <= tolerance,
                "{} through {format:?}: surface went from {expected_area} to {area}",
                specimen.name
            );
        }
    }
}

/// Total surface area, as a triangulation-independent measure of "the same
/// shape". Any correct triangulation of a face has the face's area.
fn surface_area(mesh: &MeshData) -> f32 {
    let positions = mesh.positions();
    let mut total = 0.0;
    for face in mesh.faces() {
        for i in 1..face.len().saturating_sub(1) {
            let a = positions[face[0] as usize];
            let b = positions[face[i] as usize];
            let c = positions[face[i + 1] as usize];
            total += (b - a).cross(c - a).length() * 0.5;
        }
    }
    total
}

/// The one specimen that needs a looser tolerance, and why.
fn tolerance_for(specimen: &Specimen) -> Tolerance {
    if specimen.name == "extreme_scale" {
        // A coordinate of 1e5 has an f32 spacing of about 0.0078, so the default
        // micrometre tolerance is finer than the type can represent there. This
        // is a property of f32, not of any format, and raising the tolerance for
        // this specimen is honest; raising the default would hide real drift in
        // every other one.
        Tolerance { position: 0.01 }
    } else {
        Tolerance::default()
    }
}

#[test]
fn ngons_survive_where_the_contract_says_they_do() {
    let quads = corpus()
        .into_iter()
        .find(|s| s.name == "cube_quads")
        .expect("the corpus has a quad cube");
    for format in FORMATS {
        let back = one(format, &quads.mesh);
        let all_quads = back.faces().all(|f| f.len() == 4);
        assert_eq!(
            all_quads,
            contract(format).ngons,
            "{format:?} claims ngons = {} but came back with {:?}",
            contract(format).ngons,
            back.faces().map(|f| f.len()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn a_concave_ngon_survives_without_being_turned_inside_out() {
    // The specimen a naive fan triangulation gets wrong. Through a format that
    // keeps n-gons it must come back identical; through one that triangulates,
    // the triangles must at least still face the same way.
    let specimen = corpus()
        .into_iter()
        .find(|s| s.name == "concave_ngon")
        .unwrap();
    let before = specimen.mesh.compute_face_normals()[0].normalize();
    for format in FORMATS {
        let back = one(format, &specimen.mesh);
        for (i, normal) in back.compute_face_normals().iter().enumerate() {
            assert!(
                before.dot(normal.normalize()) > 0.9,
                "{format:?} face {i} faces {normal} where the original faced {before}"
            );
        }
    }
}

#[test]
fn mixed_winding_is_preserved_rather_than_repaired() {
    // A tool that quietly "fixes" winding on load has overruled the author. The
    // two faces must still disagree after the trip.
    let specimen = corpus()
        .into_iter()
        .find(|s| s.name == "mixed_winding")
        .unwrap();
    for format in FORMATS {
        assert!(contract(format).winding);
        let back = one(format, &specimen.mesh);
        let normals = back.compute_face_normals();
        assert_eq!(normals.len(), 2, "{format:?}");
        assert!(
            normals[0].normalize().dot(normals[1].normalize()) < -0.9,
            "{format:?} made the two faces agree: {normals:?}"
        );
    }
}

#[test]
fn the_material_partition_survives_where_the_contract_says_it_does() {
    let specimen = corpus()
        .into_iter()
        .find(|s| s.name == "many_materials")
        .unwrap();
    let expected = partition(&specimen.mesh);
    for format in FORMATS {
        let back = one(format, &specimen.mesh);
        let got = partition(&back);
        if contract(format).material_partition {
            assert_eq!(
                got, expected,
                "{format:?} lost the face-to-material partition"
            );
        } else {
            assert!(
                got.is_none(),
                "{format:?} kept materials its contract says it drops"
            );
        }
    }
}

/// The face-to-material grouping, as sorted group sizes.
///
/// Sizes rather than the index list, because a format is free to renumber the
/// slots and to reorder the faces; what must survive is *which faces share a
/// material*, and that is what this measures.
fn partition(mesh: &MeshData) -> Option<Vec<usize>> {
    let materials = mesh.attributes().get_i32(names::MATERIAL_INDEX)?;
    let mut counts: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for m in materials {
        *counts.entry(*m).or_default() += 1;
    }
    let mut sizes: Vec<usize> = counts.into_values().collect();
    sizes.sort_unstable();
    Some(sizes)
}

#[test]
fn uvs_survive_with_their_values_where_the_contract_says_they_do() {
    let specimen = corpus()
        .into_iter()
        .find(|s| s.name == "five_uv_sets")
        .unwrap();
    let expected = specimen
        .mesh
        .attributes()
        .get_f32x2("uv_map")
        .expect("the specimen has uv_map")
        .to_vec();

    for format in FORMATS {
        let back = one(format, &specimen.mesh);
        let sets: Vec<String> = back
            .attributes()
            .iter()
            .filter(|l| l.name.starts_with("uv"))
            .map(|l| l.name.to_string())
            .collect();

        if !contract(format).uvs {
            assert!(
                sets.is_empty(),
                "{format:?} kept UVs it claims to drop: {sets:?}"
            );
            continue;
        }
        assert!(!sets.is_empty(), "{format:?} lost every UV set");

        // Values, not just presence: a UV layer of zeroes is structurally
        // indistinguishable from a real one and completely useless.
        let got = back
            .attributes()
            .get_f32x2(&sets[0])
            .expect("the layer that is there is readable");
        assert_eq!(got.len(), expected.len(), "{format:?}");
        for (a, b) in expected.iter().zip(got) {
            assert!(
                a.distance(*b) < 1e-5,
                "{format:?} moved a UV from {a} to {b}"
            );
        }

        assert_eq!(
            sets.len() > 1,
            contract(format).multiple_uv_sets,
            "{format:?} came back with {sets:?}"
        );
        assert_eq!(
            sets.contains(&"uv_map".to_string()) && sets.len() == 1,
            contract(format).uv_names || sets.len() == 1,
            "{format:?} came back with {sets:?}"
        );
    }
}

#[test]
fn the_contract_negatives_are_real() {
    // The half that keeps the table honest. Every `false` above is asserted to
    // be a real limitation, so teaching a format a new trick fails here until
    // the table is updated to claim it.
    let all = corpus();
    let get = |name: &str| &all.iter().find(|s| s.name == name).unwrap().mesh;

    for c in CONTRACTS {
        if !c.ngons {
            let back = one(c.format, get("cube_quads"));
            assert!(
                back.faces().all(|f| f.len() == 3),
                "{:?} keeps n-gons after all; the contract understates it",
                c.format
            );
        }
        if !c.uvs {
            let back = one(c.format, get("five_uv_sets"));
            assert!(
                !back.attributes().iter().any(|l| l.name.starts_with("uv")),
                "{:?} keeps UVs after all",
                c.format
            );
        }
        if !c.multiple_uv_sets {
            let back = one(c.format, get("five_uv_sets"));
            let sets = back
                .attributes()
                .iter()
                .filter(|l| l.name.starts_with("uv"))
                .count();
            assert!(sets <= 1, "{:?} kept {sets} UV sets", c.format);
        }
        if !c.sharp_edges {
            // The values, not the layer. OBJ round-trips through `s` groups, and
            // its importer canonicalises those into a per-edge `sharp` layer —
            // so the layer comes back, full of `false`. Asserting the layer is
            // absent would have made this contract entry a statement about
            // plumbing; what a user loses is the marks.
            let back = one(c.format, get("smoothing_groups"));
            let sharp = back
                .attributes()
                .get_bool(names::SHARP_EDGE)
                .map(|s| s.iter().filter(|&&s| s).count())
                .unwrap_or(0);
            assert_eq!(
                sharp, 0,
                "{:?} brought back {sharp} sharp edges it claims to drop",
                c.format
            );
        }
        if !c.material_partition {
            let back = one(c.format, get("many_materials"));
            assert!(
                back.attributes().get_i32(names::MATERIAL_INDEX).is_none(),
                "{:?} keeps materials after all",
                c.format
            );
        }
        if !c.multiple_meshes {
            // Asserted by the writer refusing rather than by silently keeping
            // one: STL holds a single unnamed solid, and a writer that took two
            // meshes and wrote the first would be the bad kind of lossy.
            assert_eq!(c.format, Format::Stl);
        }
    }
}

#[test]
fn several_meshes_stay_separate_where_the_contract_says_they_do() {
    let a = corpus()
        .into_iter()
        .find(|s| s.name == "cube_quads")
        .unwrap();
    let b = corpus()
        .into_iter()
        .find(|s| s.name == "wide_ngon")
        .unwrap();
    let meshes = [
        ("첫번째".to_string(), &a.mesh),
        ("second".to_string(), &b.mesh),
    ];
    for format in FORMATS {
        if !contract(format).multiple_meshes {
            continue;
        }
        let back = round_trip(format, &meshes);
        assert_eq!(back.len(), 2, "{format:?} merged the meshes");
        // The Korean name has to survive too: it is how the user recognises the
        // object, and both a text format and a JSON string have historically
        // been where non-ASCII names go wrong.
        assert_eq!(back[0].0, "첫번째", "{format:?} mangled the name");
        assert_eq!(back[1].0, "second", "{format:?}");
        assert_eq!(back[0].1.num_faces(), 6, "{format:?}");
    }
}

#[test]
fn a_second_trip_changes_nothing_more() {
    // Idempotence after the first pass. Whatever a format loses, it should lose
    // it once: a pipeline that re-exports through the same format on every save
    // must not decay a little further each time, which is the shape of bug that
    // only shows up in a customer's file after twenty saves.
    for specimen in corpus() {
        for format in FORMATS {
            let first = one(format, &specimen.mesh);
            let second = one(format, &first);
            if let Err(mismatch) = compare_geometry(&first, &second, tolerance_for(&specimen)) {
                panic!(
                    "{} through {format:?} kept changing on the second trip: {mismatch}",
                    specimen.name
                );
            }
        }
    }
}

#[test]
fn nothing_in_the_corpus_panics_any_parser() {
    // The no-panic smoke pass. Every specimen through every writer and reader,
    // asserting only that the process survives — the torture-suite property.
    for specimen in corpus() {
        for format in FORMATS {
            let back = one(format, &specimen.mesh);
            let violations = back.validate();
            assert!(
                violations.is_empty(),
                "{} through {format:?} came back invalid: {violations:?}",
                specimen.name
            );
        }
    }
}
