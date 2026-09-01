//! Wavefront OBJ import and export.
//!
//! Written in-house rather than taken from a crate. The mature OBJ crates
//! (`tobj`, `obj-rs`) triangulate on load and flatten groups — which discards
//! precisely the things a polygon modeller cares about: n-gons, `s` smoothing
//! groups, and object/group structure. A parser that throws away the user's quads
//! is not a shortcut, it is a different product.
//!
//! # What is supported
//!
//! `v`, `vt`, `vn`, `f`, `o`, `g`, `s`, `usemtl`, `mtllib`; 1-based and negative
//! indices; all four `f` index forms; n-gons of any size.
//!
//! # What OBJ does not tell us
//!
//! OBJ declares neither units nor axis convention. Both are therefore options
//! with a recorded assumption ([`LossyNote::AssumedAxisConvention`],
//! [`LossyNote::AssumedUnitScale`]) rather than a silent guess. The default is
//! Y-up and metres, matching what almost every OBJ in the wild actually is.

use std::fmt::Write as _;

use ahash::AHashMap;
use glam::{Vec2, Vec3};
use kreate_math::AxisConvention;
use kreate_mesh::attribute::names;
use kreate_mesh::MeshData;

use crate::ir::{ImportMesh, ImportScene, LossyNote, SourceMetadata, UvOrigin};

#[derive(Debug, thiserror::Error)]
pub enum ObjError {
    #[error("line {line}: {msg}")]
    Parse { line: usize, msg: String },
}

fn err<T>(line: usize, msg: impl Into<String>) -> Result<T, ObjError> {
    Err(ObjError::Parse {
        line,
        msg: msg.into(),
    })
}

/// How to interpret the things OBJ leaves unstated.
#[derive(Clone, Copy, Debug)]
pub struct ObjOptions {
    /// Assumed source axis convention. OBJ files are overwhelmingly Y-up.
    pub axis: AxisConvention,
    /// Assumed metres per source unit.
    pub unit_scale: f32,
}

impl Default for ObjOptions {
    fn default() -> Self {
        Self {
            axis: AxisConvention::GLTF,
            unit_scale: 1.0,
        }
    }
}

/// One `f` corner: indices into the file-global v / vt / vn arrays, already
/// resolved to 0-based.
#[derive(Clone, Copy, Debug)]
struct Corner {
    v: u32,
    vt: Option<u32>,
    vn: Option<u32>,
}

#[derive(Default)]
struct GroupBuilder {
    name: String,
    faces: Vec<Vec<Corner>>,
    materials: Vec<i32>,
    smoothing: Vec<u32>,
}

/// Resolves an OBJ index against the current length of its array.
///
/// OBJ indices are 1-based, and negative indices count back from the end *as it
/// stands at that point in the file* — which is why this needs `len` rather than
/// the final count.
fn resolve_index(raw: i64, len: usize, line: usize, what: &str) -> Result<u32, ObjError> {
    let idx = if raw > 0 {
        raw - 1
    } else if raw < 0 {
        len as i64 + raw
    } else {
        return err(
            line,
            format!("{what} index 0 is not valid in OBJ (indices are 1-based)"),
        );
    };
    if idx < 0 || idx as usize >= len {
        return err(
            line,
            format!("{what} index {raw} out of range (have {len})"),
        );
    }
    Ok(idx as u32)
}

/// Parses OBJ text into the neutral IR.
pub fn parse_obj(text: &str, opts: ObjOptions) -> Result<ImportScene, ObjError> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();

    let mut groups: Vec<GroupBuilder> = Vec::new();
    let mut materials: Vec<String> = Vec::new();
    let mut cur_material: i32 = -1;
    let mut cur_smoothing: u32 = 0;
    let mut mtllibs: Vec<String> = Vec::new();

    // Faces before any `o`/`g` belong to an implicit default group. It is only
    // materialised if something actually lands in it, so a well-formed file with
    // an explicit `o` on line 1 does not gain a stray empty mesh.
    let ensure_group = |groups: &mut Vec<GroupBuilder>| -> usize {
        if groups.is_empty() {
            groups.push(GroupBuilder {
                name: "default".into(),
                ..Default::default()
            });
        }
        groups.len() - 1
    };

    for (lineno, raw_line) in text.lines().enumerate() {
        let line = lineno + 1;
        // A backslash line continuation is legal in OBJ but vanishingly rare;
        // comments and blank lines are not.
        let content = raw_line.split('#').next().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let mut it = content.split_whitespace();
        let Some(keyword) = it.next() else { continue };

        match keyword {
            "v" => {
                let c: Vec<&str> = it.collect();
                if c.len() < 3 {
                    return err(line, "v needs at least 3 coordinates");
                }
                positions.push(Vec3::new(
                    parse_f32(c[0], line, "v")?,
                    parse_f32(c[1], line, "v")?,
                    parse_f32(c[2], line, "v")?,
                ));
                // A 4th component (w) and trailing r/g/b vertex colours both
                // exist in the wild; neither is represented here.
            }
            "vt" => {
                let c: Vec<&str> = it.collect();
                if c.is_empty() {
                    return err(line, "vt needs at least 1 coordinate");
                }
                let u = parse_f32(c[0], line, "vt")?;
                let v = if c.len() > 1 {
                    parse_f32(c[1], line, "vt")?
                } else {
                    0.0
                };
                uvs.push(Vec2::new(u, v));
            }
            "vn" => {
                let c: Vec<&str> = it.collect();
                if c.len() < 3 {
                    return err(line, "vn needs 3 components");
                }
                normals.push(Vec3::new(
                    parse_f32(c[0], line, "vn")?,
                    parse_f32(c[1], line, "vn")?,
                    parse_f32(c[2], line, "vn")?,
                ));
            }
            "f" => {
                let mut face = Vec::new();
                for tok in it {
                    face.push(parse_corner(
                        tok,
                        positions.len(),
                        uvs.len(),
                        normals.len(),
                        line,
                    )?);
                }
                if face.len() < 3 {
                    return err(
                        line,
                        format!("f needs at least 3 corners, got {}", face.len()),
                    );
                }
                let g = ensure_group(&mut groups);
                groups[g].faces.push(face);
                groups[g].materials.push(cur_material);
                groups[g].smoothing.push(cur_smoothing);
            }
            "o" | "g" => {
                let name: String = content[keyword.len()..].trim().to_string();
                let name = if name.is_empty() {
                    format!("{keyword}_{line}")
                } else {
                    name
                };
                // `g` with no faces yet just renames the pending group rather
                // than creating an empty one.
                match groups.last_mut() {
                    Some(last) if last.faces.is_empty() => last.name = name,
                    _ => groups.push(GroupBuilder {
                        name,
                        ..Default::default()
                    }),
                }
            }
            "s" => {
                let arg = it.next().unwrap_or("off");
                cur_smoothing = match arg {
                    "off" | "0" => 0,
                    other => other.parse::<u32>().map_err(|_| ObjError::Parse {
                        line,
                        msg: format!("bad smoothing group '{other}'"),
                    })?,
                };
            }
            "usemtl" => {
                let name = content["usemtl".len()..].trim().to_string();
                cur_material = match materials.iter().position(|m| *m == name) {
                    Some(i) => i as i32,
                    None => {
                        materials.push(name);
                        materials.len() as i32 - 1
                    }
                };
            }
            "mtllib" => {
                mtllibs.extend(it.map(str::to_string));
            }
            // Free-form geometry (curves, surfaces) and the rest of the OBJ spec
            // are recognised and skipped rather than treated as errors — a file
            // that mostly parses is more useful than a hard failure.
            _ => {}
        }
    }

    let mut source = SourceMetadata::new("obj");
    source.axis = opts.axis;
    source.unit_scale = opts.unit_scale;
    source.uv_origin = UvOrigin::BottomLeft; // OBJ's native origin
    source.notes.push(LossyNote::AssumedAxisConvention {
        assumed: format!("{:?}", opts.axis),
    });
    source.notes.push(LossyNote::AssumedUnitScale {
        assumed: format!("{} m per unit", opts.unit_scale),
    });
    if !mtllibs.is_empty() {
        source.notes.push(LossyNote::Unsupported {
            what: format!("mtllib not read: {}", mtllibs.join(", ")),
        });
    }

    let mut scene = ImportScene::new(source);
    scene.materials = materials
        .into_iter()
        .map(|name| crate::ir::ImportMaterial {
            name,
            ..Default::default()
        })
        .collect();
    for g in groups {
        if g.faces.is_empty() {
            continue;
        }
        scene.meshes.push(build_mesh(g, &positions, &uvs, &normals));
    }
    Ok(scene)
}

fn parse_f32(s: &str, line: usize, what: &str) -> Result<f32, ObjError> {
    s.parse::<f32>().map_err(|_| ObjError::Parse {
        line,
        msg: format!("bad {what} number '{s}'"),
    })
}

/// Parses one `v`, `v/vt`, `v//vn`, or `v/vt/vn` token.
fn parse_corner(
    tok: &str,
    nv: usize,
    nvt: usize,
    nvn: usize,
    line: usize,
) -> Result<Corner, ObjError> {
    let mut parts = tok.split('/');
    let v_str = parts.next().unwrap_or("");
    let v_raw: i64 = v_str.parse().map_err(|_| ObjError::Parse {
        line,
        msg: format!("bad vertex index '{v_str}'"),
    })?;
    let v = resolve_index(v_raw, nv, line, "vertex")?;

    let read = |part: Option<&str>, len: usize, what: &str| -> Result<Option<u32>, ObjError> {
        match part {
            None | Some("") => Ok(None),
            Some(s) => {
                let raw: i64 = s.parse().map_err(|_| ObjError::Parse {
                    line,
                    msg: format!("bad {what} index '{s}'"),
                })?;
                Ok(Some(resolve_index(raw, len, line, what)?))
            }
        }
    };
    let vt = read(parts.next(), nvt, "texcoord")?;
    let vn = read(parts.next(), nvn, "normal")?;
    Ok(Corner { v, vt, vn })
}

/// Compacts one group's vertices and lifts its per-corner data into the IR.
///
/// This is where OBJ's three parallel index streams collapse. Because UVs and
/// normals live in the Corner domain, a vertex used with two different UVs does
/// **not** need to be split into two vertices — which is what every triangulating
/// importer does, and what leaves users with a mesh that falls apart the moment
/// they move a vertex.
fn build_mesh(g: GroupBuilder, positions: &[Vec3], uvs: &[Vec2], normals: &[Vec3]) -> ImportMesh {
    // Compact in *ascending source order*, not first-encounter order. Both are
    // deterministic, but ascending order also means a file Kreate wrote reads
    // back with identical vertex numbering — first-encounter order would permute
    // the vertices of any mesh whose first face is not wound 0,1,2,..., which
    // silently breaks index-stable round trips.
    let mut used: Vec<u32> = g.faces.iter().flatten().map(|c| c.v).collect();
    used.sort_unstable();
    used.dedup();
    let remap: AHashMap<u32, u32> = used
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    let local_positions: Vec<Vec3> = used.iter().map(|&v| positions[v as usize]).collect();

    let faces: Vec<Vec<u32>> = g
        .faces
        .iter()
        .map(|face| face.iter().map(|c| remap[&c.v]).collect())
        .collect();

    let all_corners = g.faces.iter().flat_map(|f| f.iter());
    let has_uv = !uvs.is_empty() && g.faces.iter().flatten().any(|c| c.vt.is_some());
    let has_normals = !normals.is_empty() && g.faces.iter().flatten().any(|c| c.vn.is_some());

    let corner_uvs = has_uv.then(|| {
        all_corners
            .clone()
            .map(|c| c.vt.map(|i| uvs[i as usize]).unwrap_or(Vec2::ZERO))
            .collect()
    });
    let corner_normals = has_normals.then(|| {
        all_corners
            .map(|c| c.vn.map(|i| normals[i as usize]).unwrap_or(Vec3::ZERO))
            .collect()
    });

    ImportMesh {
        name: g.name,
        positions: local_positions,
        faces,
        corner_uvs,
        corner_normals,
        face_materials: g.materials.iter().any(|&m| m >= 0).then_some(g.materials),
        smoothing_groups: Some(g.smoothing),
    }
}

// --- export -----------------------------------------------------------------

/// Writes meshes as OBJ text.
///
/// N-gons are written as n-gons. `sharp_edge`/`shade_smooth` are collapsed back
/// into `s` groups only at the coarse level — a face is `s 1` if it is smooth and
/// `s off` otherwise. Reconstructing true per-region smoothing groups needs a
/// graph colouring over face adjacency, which is a real algorithm and is not done
/// here; the coarse form round-trips flat-versus-smooth correctly and is what
/// most consumers actually read.
pub fn write_obj(meshes: &[(String, &MeshData)], opts: ObjOptions) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Written by Kreate");
    let inv =
        opts.axis.from_kreate() * glam::Mat3::from_diagonal(Vec3::splat(1.0 / opts.unit_scale));

    // OBJ indices are file-global and 1-based, so every mesh's indices are
    // offset by everything written before it.
    let mut v_base = 1usize;
    let mut vt_base = 1usize;

    for (name, mesh) in meshes {
        let _ = writeln!(out, "o {name}");
        for p in mesh.positions() {
            let p = inv * *p;
            let _ = writeln!(out, "v {} {} {}", fnum(p.x), fnum(p.y), fnum(p.z));
        }

        let uvs = mesh.attributes().get_f32x2("uv_map");
        if let Some(uvs) = uvs {
            for uv in uvs {
                let uv = UvOrigin::BottomLeft.from_kreate(*uv);
                let _ = writeln!(out, "vt {} {}", fnum(uv.x), fnum(uv.y));
            }
        }

        let smooth = mesh.attributes().get_bool(names::SHADE_SMOOTH);
        let mut last_smooth: Option<bool> = None;
        for f in 0..mesh.num_faces() {
            let is_smooth = smooth.map(|s| s[f as usize]).unwrap_or(false);
            if last_smooth != Some(is_smooth) {
                let _ = writeln!(out, "s {}", if is_smooth { "1" } else { "off" });
                last_smooth = Some(is_smooth);
            }
            out.push('f');
            let range = mesh.face_corner_range(f);
            for (k, &v) in mesh.face_verts(f).iter().enumerate() {
                if uvs.is_some() {
                    let _ = write!(
                        out,
                        " {}/{}",
                        v as usize + v_base,
                        range.start + k + vt_base
                    );
                } else {
                    let _ = write!(out, " {}", v as usize + v_base);
                }
            }
            out.push('\n');
        }

        v_base += mesh.num_verts() as usize;
        if let Some(uvs) = uvs {
            vt_base += uvs.len();
        }
    }
    out
}

/// Formats a float compactly but losslessly enough for a round trip.
///
/// `{}` on an `f32` already round-trips through `parse::<f32>()`, and it avoids
/// the trailing-zero noise that a fixed precision would add to every coordinate
/// of a large file.
fn fnum(x: f32) -> String {
    if x == 0.0 {
        // Normalise -0.0, which would otherwise make byte-identical writes of
        // equal meshes differ.
        "0".to_string()
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> ImportScene {
        parse_obj(text, ObjOptions::default()).expect("should parse")
    }

    /// Parses with no axis change, so tests can assert on raw coordinates.
    fn identity_opts() -> ObjOptions {
        ObjOptions {
            axis: AxisConvention::KREATE,
            unit_scale: 1.0,
        }
    }

    const QUAD: &str = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
";

    #[test]
    fn quad_stays_a_quad() {
        let scene = parse(QUAD);
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.meshes[0].faces, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn ngon_is_not_triangulated() {
        let mut text = String::new();
        for i in 0..7 {
            let _ = writeln!(text, "v {i} 0 0");
        }
        text.push_str("f 1 2 3 4 5 6 7\n");
        let scene = parse(&text);
        assert_eq!(scene.meshes[0].faces[0].len(), 7);
    }

    #[test]
    fn all_four_index_forms_parse() {
        let text = "\
v 0 0 0
v 1 0 0
v 1 1 0
vt 0 0
vt 1 0
vt 1 1
vn 0 0 1
f 1 2 3
f 1/1 2/2 3/3
f 1//1 2//1 3//1
f 1/1/1 2/2/1 3/3/1
";
        let scene = parse(text);
        assert_eq!(scene.meshes[0].faces.len(), 4);
        assert_eq!(scene.meshes[0].corner_uvs.as_ref().unwrap().len(), 12);
        assert_eq!(scene.meshes[0].corner_normals.as_ref().unwrap().len(), 12);
    }

    #[test]
    fn negative_indices_count_back_from_the_current_end() {
        let text = "\
v 0 0 0
v 1 0 0
v 1 1 0
f -3 -2 -1
";
        assert_eq!(parse(text).meshes[0].faces, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn objects_become_separate_meshes_with_compacted_vertices() {
        let text = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 5 0 0
v 6 0 0
v 6 1 0
o first
f 1 2 3
o second
f 4 5 6
";
        let scene = parse(text);
        assert_eq!(scene.meshes.len(), 2);
        assert_eq!(scene.meshes[0].name, "first");
        assert_eq!(scene.meshes[1].name, "second");
        // Each mesh keeps only its own vertices, renumbered from zero.
        assert_eq!(scene.meshes[1].positions.len(), 3);
        assert_eq!(scene.meshes[1].faces, vec![vec![0, 1, 2]]);
        assert_eq!(scene.meshes[1].positions[0], Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn a_vertex_used_with_two_uvs_is_not_split() {
        // The whole point of the Corner domain. A triangulating importer would
        // duplicate vertex 1 here.
        let text = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 1
f 1/1 2/1 3/1
f 1/2 3/2 4/2
";
        let scene = parse(text);
        assert_eq!(scene.meshes[0].positions.len(), 4, "no vertex splitting");
        assert_eq!(scene.meshes[0].corner_uvs.as_ref().unwrap().len(), 6);
    }

    #[test]
    fn smoothing_groups_are_recorded_per_face() {
        let text = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
s 1
f 1 2 3
s off
f 1 3 4
";
        assert_eq!(parse(text).meshes[0].smoothing_groups, Some(vec![1, 0]));
    }

    #[test]
    fn usemtl_assigns_face_materials() {
        let text = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
usemtl red
f 1 2 3
usemtl blue
f 1 3 4
usemtl red
f 2 3 4
";
        let scene = parse(text);
        assert_eq!(scene.materials.len(), 2);
        assert_eq!(scene.materials[0].name, "red");
        assert_eq!(scene.meshes[0].face_materials, Some(vec![0, 1, 0]));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let text = "\
# a comment
v 0 0 0   # trailing comment

v 1 0 0
v 1 1 0
f 1 2 3
";
        assert_eq!(parse(text).meshes[0].positions.len(), 3);
    }

    #[test]
    fn unknown_keywords_do_not_fail_the_parse() {
        let text = "\
mtllib scene.mtl
cstype bezier
deg 3
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 3
";
        let scene = parse(text);
        assert_eq!(scene.meshes[0].faces.len(), 1);
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { what } if what.contains("scene.mtl"))));
    }

    #[test]
    fn out_of_range_index_is_an_error_not_a_panic() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 99\n";
        assert!(parse_obj(text, ObjOptions::default()).is_err());
    }

    #[test]
    fn zero_index_is_rejected() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 0 1 2\n";
        let e = parse_obj(text, ObjOptions::default()).unwrap_err();
        assert!(format!("{e}").contains("1-based"));
    }

    #[test]
    fn assumptions_are_recorded() {
        let scene = parse(QUAD);
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::AssumedAxisConvention { .. })));
        assert!(scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::AssumedUnitScale { .. })));
    }

    #[test]
    fn default_options_treat_obj_as_y_up() {
        // A vertex at the source's "up" must arrive at Kreate's +Z.
        let scene = parse("v 0 1 0\nv 1 1 0\nv 1 2 0\nf 1 2 3\n");
        let (meshes, _) = scene.into_mesh_data();
        assert!((meshes[0].1.positions()[0] - Vec3::Z).length() < 1e-6);
    }

    // --- round trip ---------------------------------------------------------

    fn roundtrip(mesh: &MeshData) -> MeshData {
        let text = write_obj(&[("m".into(), mesh)], identity_opts());
        let scene = parse_obj(&text, identity_opts()).expect("written OBJ should re-parse");
        let (out, _) = scene.into_mesh_data();
        out.into_iter().next().unwrap().1
    }

    #[test]
    fn cube_survives_a_round_trip_as_quads() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let back = roundtrip(&cube);
        back.assert_valid();
        assert_eq!(back.num_verts(), cube.num_verts());
        assert_eq!(back.num_faces(), cube.num_faces());
        assert_eq!(back.num_edges(), cube.num_edges());
        for f in 0..cube.num_faces() {
            assert_eq!(back.face_verts(f), cube.face_verts(f), "face {f}");
        }
        for (a, b) in back.positions().iter().zip(cube.positions()) {
            assert!((*a - *b).length() < 1e-5);
        }
    }

    #[test]
    fn ngon_caps_survive_a_round_trip() {
        let cyl = kreate_mesh::primitives::cylinder(1.0, 2.0, 16);
        let back = roundtrip(&cyl);
        back.assert_valid();
        assert_eq!(back.num_faces(), cyl.num_faces());
        assert_eq!(
            back.face_len(16),
            16,
            "cap must not be triangulated by the round trip"
        );
    }

    #[test]
    fn multiple_meshes_get_correct_global_index_offsets() {
        let a = kreate_mesh::primitives::cube(1.0);
        let b = kreate_mesh::primitives::plane(1.0);
        let text = write_obj(&[("a".into(), &a), ("b".into(), &b)], identity_opts());
        let scene = parse_obj(&text, identity_opts()).expect("re-parse");
        let (out, _) = scene.into_mesh_data();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].1.num_faces(), 6);
        assert_eq!(out[1].1.num_faces(), 1);
        // If the offset were wrong, `b` would reference `a`'s vertices and land
        // in the wrong place.
        assert!(out[1].1.bounds().size().length() < 2.0);
    }

    #[test]
    fn writing_is_deterministic() {
        let m = kreate_mesh::primitives::uv_sphere(1.0, 8, 4);
        let a = write_obj(&[("m".into(), &m)], identity_opts());
        let b = write_obj(&[("m".into(), &m)], identity_opts());
        assert_eq!(a, b, "identical meshes must produce byte-identical files");
    }

    #[test]
    fn axis_conversion_roundtrips_through_the_file() {
        // Import assumes Y-up and export must undo it, or every save-load cycle
        // would rotate the model.
        let m = kreate_mesh::primitives::cube(2.0);
        let opts = ObjOptions::default(); // Y-up
        let text = write_obj(&[("m".into(), &m)], opts);
        let (out, _) = parse_obj(&text, opts).unwrap().into_mesh_data();
        for (a, b) in out[0].1.positions().iter().zip(m.positions()) {
            assert!((*a - *b).length() < 1e-5, "{a} != {b}");
        }
    }
}
