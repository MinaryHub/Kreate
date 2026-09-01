//! The neutral import/export intermediate representation.
//!
//! **No importer maps a file format directly onto a document.** Every format goes
//! `format -> ImportScene -> document`, and the reverse on the way out. The
//! reason is that the hard parts of interoperability — n-gons, smoothing groups,
//! split normals, UV origin, axis and unit conventions, winding — are the *same*
//! hard parts for every format. Concentrating them here means fixing each of them
//! once instead of once per importer, and it is the difference between six
//! importers with six subtly different bugs and six thin parsers.
//!
//! The IR uses the same `{Point, Edge, Corner, Face}` domain model as
//! [`MeshData`], which is what makes OBJ's separate `v`/`vt`/`vn` index streams,
//! FBX's `ByPolygonVertex` layers, glTF's split vertices, and USD's `faceVarying`
//! primvars all collapse into one representation instead of four special cases.

use ahash::AHashMap;
use glam::{Vec2, Vec3};
use kreate_math::{AxisConvention, Mat3};
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_mesh::MeshData;
use std::sync::Arc;

/// Where a format puts UV (0, 0).
///
/// Getting this wrong flips every texture vertically, and it is invisible until
/// someone applies a texture — so it is recorded explicitly rather than assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UvOrigin {
    /// OBJ, Blender, and Kreate's own convention.
    BottomLeft,
    /// glTF, and most realtime engines.
    TopLeft,
}

impl UvOrigin {
    /// Converts a UV from this origin into Kreate's (bottom-left).
    #[inline]
    pub fn to_kreate(self, uv: Vec2) -> Vec2 {
        match self {
            UvOrigin::BottomLeft => uv,
            UvOrigin::TopLeft => Vec2::new(uv.x, 1.0 - uv.y),
        }
    }

    /// Converts a Kreate UV into this origin. Self-inverse for both variants.
    #[inline]
    pub fn from_kreate(self, uv: Vec2) -> Vec2 {
        self.to_kreate(uv)
    }
}

/// A lossy decision an importer had to take.
///
/// Surfaced in an import report rather than swallowed. "It imported but looks
/// wrong" is the most expensive kind of support question, and it is almost always
/// answered by one of these.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LossyNote {
    /// A face had fewer than three distinct vertices and was dropped.
    DroppedDegenerateFace { mesh: String, face: usize },
    /// An index was out of range and the face was dropped.
    OutOfRangeIndex { mesh: String, face: usize },
    /// The format does not declare its axis convention, so one was assumed.
    AssumedAxisConvention { assumed: String },
    /// The format does not declare units, so a scale was assumed.
    AssumedUnitScale { assumed: String },
    /// A feature was recognised but is not represented in the IR.
    Unsupported { what: String },
}

/// Where the data came from, and what had to be assumed to read it.
///
/// Attached to imported objects so that re-exporting can restore the source
/// conventions instead of silently rewriting the file in Kreate's own.
#[derive(Clone, Debug)]
pub struct SourceMetadata {
    pub format: String,
    pub generator: Option<String>,
    /// The file's coordinate convention. Applied exactly once, here.
    pub axis: AxisConvention,
    /// Metres per source unit.
    pub unit_scale: f32,
    pub uv_origin: UvOrigin,
    pub notes: Vec<LossyNote>,
}

impl SourceMetadata {
    pub fn new(format: impl Into<String>) -> Self {
        Self {
            format: format.into(),
            generator: None,
            axis: AxisConvention::KREATE,
            unit_scale: 1.0,
            uv_origin: UvOrigin::BottomLeft,
            notes: Vec::new(),
        }
    }

    /// The matrix taking source positions into Kreate space, units included.
    pub fn to_kreate_matrix(&self) -> Mat3 {
        self.axis.to_kreate() * Mat3::from_diagonal(Vec3::splat(self.unit_scale))
    }
}

/// A material as described by an interchange format.
///
/// Deliberately a "standard surface" and nothing more. glTF metal-rough and
/// `UsdPreviewSurface` were designed to match each other, so this maps near-1:1
/// to both. Arbitrary shader graphs are never imported.
#[derive(Clone, Debug)]
pub struct ImportMaterial {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub base_color_texture: Option<String>,
}

impl Default for ImportMaterial {
    fn default() -> Self {
        Self {
            name: "Material".into(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: Vec3::ZERO,
            base_color_texture: None,
        }
    }
}

/// One mesh, in source coordinates, before conversion.
///
/// Corner-domain arrays are flat over every corner of every face, in face order —
/// the same CSR layout [`MeshData`] uses, so conversion is a copy rather than a
/// reindex.
#[derive(Clone, Debug, Default)]
pub struct ImportMesh {
    pub name: String,
    pub positions: Vec<Vec3>,
    /// Vertex indices per face. N-gons are kept as authored.
    pub faces: Vec<Vec<u32>>,
    /// Corner domain. One per corner across all faces.
    pub corner_uvs: Option<Vec<Vec2>>,
    /// Corner domain. Authored split normals; absent means "compute them".
    pub corner_normals: Option<Vec<Vec3>>,
    /// Face domain. Index into [`ImportScene::materials`].
    pub face_materials: Option<Vec<i32>>,
    /// Face domain. `0` means flat shading; equal non-zero values smooth together.
    pub smoothing_groups: Option<Vec<u32>>,
}

impl ImportMesh {
    pub fn num_corners(&self) -> usize {
        self.faces.iter().map(|f| f.len()).sum()
    }

    /// Drops faces that cannot be represented, recording why.
    ///
    /// Never silently: a mesh that loses faces without saying so is worse than
    /// one that fails to import.
    fn sanitize(&mut self, notes: &mut Vec<LossyNote>) {
        let num_verts = self.positions.len() as u32;
        let mut keep = Vec::with_capacity(self.faces.len());
        for (i, f) in self.faces.iter().enumerate() {
            let ok = if f.iter().any(|&v| v >= num_verts) {
                notes.push(LossyNote::OutOfRangeIndex {
                    mesh: self.name.clone(),
                    face: i,
                });
                false
            } else if f.len() < 3 {
                notes.push(LossyNote::DroppedDegenerateFace {
                    mesh: self.name.clone(),
                    face: i,
                });
                false
            } else {
                true
            };
            keep.push(ok);
        }
        if keep.iter().all(|&k| k) {
            return;
        }

        // Corner arrays must be filtered in lockstep with the faces they belong to.
        let mut corner_ranges = Vec::with_capacity(self.faces.len());
        let mut start = 0usize;
        for f in &self.faces {
            corner_ranges.push(start..start + f.len());
            start += f.len();
        }
        let filter_corners = |src: &Vec<Vec2>| -> Vec<Vec2> {
            corner_ranges
                .iter()
                .zip(&keep)
                .filter(|(_, &k)| k)
                .flat_map(|(r, _)| src[r.clone()].iter().copied())
                .collect()
        };
        if let Some(uvs) = &self.corner_uvs {
            self.corner_uvs = Some(filter_corners(uvs));
        }
        if let Some(ns) = &self.corner_normals {
            let filtered: Vec<Vec3> = corner_ranges
                .iter()
                .zip(&keep)
                .filter(|(_, &k)| k)
                .flat_map(|(r, _)| ns[r.clone()].iter().copied())
                .collect();
            self.corner_normals = Some(filtered);
        }
        if let Some(v) = &mut self.face_materials {
            let mut it = keep.iter();
            v.retain(|_| *it.next().unwrap_or(&false));
        }
        if let Some(v) = &mut self.smoothing_groups {
            let mut it = keep.iter();
            v.retain(|_| *it.next().unwrap_or(&false));
        }
        let mut it = keep.iter();
        self.faces.retain(|_| *it.next().unwrap_or(&false));
    }
}

/// A scene as read from a file, before it becomes a document.
#[derive(Clone, Debug)]
pub struct ImportScene {
    pub meshes: Vec<ImportMesh>,
    pub materials: Vec<ImportMaterial>,
    pub source: SourceMetadata,
}

impl ImportScene {
    pub fn new(source: SourceMetadata) -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            source,
        }
    }

    /// Converts every mesh into Kreate space, consuming the scene.
    pub fn into_mesh_data(mut self) -> (Vec<(String, MeshData)>, SourceMetadata) {
        let mut notes = std::mem::take(&mut self.source.notes);
        let out = self
            .meshes
            .iter_mut()
            .map(|m| {
                m.sanitize(&mut notes);
                (m.name.clone(), convert_mesh(m, &self.source))
            })
            .collect();
        self.source.notes = notes;
        (out, self.source)
    }
}

/// Applies the axis and unit conversion, reverses winding if handedness flips,
/// and installs the corner-domain layers.
fn convert_mesh(m: &ImportMesh, src: &SourceMetadata) -> MeshData {
    let xform = src.to_kreate_matrix();
    let positions: Vec<Vec3> = m.positions.iter().map(|&p| xform * p).collect();

    // A handedness flip mirrors the geometry, so face winding must be reversed
    // or every face ends up backfacing. This is the classic silent import bug:
    // the mesh looks correct until backface culling or shading is switched on.
    let flip = src.axis.flips_winding();
    let faces: Vec<Vec<u32>> = if flip {
        m.faces
            .iter()
            .map(|f| f.iter().rev().copied().collect())
            .collect()
    } else {
        m.faces.clone()
    };

    let mut mesh = MeshData::from_faces(positions, &faces);
    let num_corners = mesh.num_corners() as usize;

    // Corner arrays were authored in the source's face order; each face's slice
    // is reversed alongside its winding by `per_face_reorder`.
    if let Some(uvs) = &m.corner_uvs {
        if uvs.len() == num_corners {
            let vals = per_face_reorder(uvs, &m.faces, flip);
            let vals: Vec<Vec2> = vals
                .into_iter()
                .map(|uv| src.uv_origin.to_kreate(uv))
                .collect();
            mesh.attributes_mut()
                .insert("uv_map", Domain::Corner, AttrData::F32x2(Arc::new(vals)));
        }
    }
    if let Some(ns) = &m.corner_normals {
        if ns.len() == num_corners {
            let normal_xform = xform.inverse().transpose();
            let vals: Vec<Vec3> = per_face_reorder(ns, &m.faces, flip)
                .into_iter()
                .map(|n| (normal_xform * n).normalize_or_zero())
                .collect();
            mesh.attributes_mut().insert(
                names::CUSTOM_NORMAL,
                Domain::Corner,
                AttrData::F32x3(Arc::new(vals)),
            );
        }
    }
    if let Some(mats) = &m.face_materials {
        if mats.len() == mesh.num_faces() as usize {
            mesh.attributes_mut().insert(
                names::MATERIAL_INDEX,
                Domain::Face,
                AttrData::I32(Arc::new(mats.clone())),
            );
        }
    }
    if let Some(groups) = &m.smoothing_groups {
        if groups.len() == mesh.num_faces() as usize {
            apply_smoothing_groups(&mut mesh, groups);
        }
    }
    mesh
}

/// Reverses each face's slice of a flat corner array when winding was flipped.
fn per_face_reorder<T: Copy>(vals: &[T], faces: &[Vec<u32>], flip: bool) -> Vec<T> {
    if !flip {
        return vals.to_vec();
    }
    let mut out = Vec::with_capacity(vals.len());
    let mut start = 0usize;
    for f in faces {
        let end = start + f.len();
        out.extend(vals[start..end].iter().rev().copied());
        start = end;
    }
    out
}

/// Converts per-face smoothing groups into Kreate's canonical form: a per-edge
/// `sharp_edge` flag plus a per-face `shade_smooth` flag.
///
/// An edge is sharp when its two faces belong to different groups, or when either
/// face has smoothing off (group 0). Boundary edges are left smooth — they have
/// no second face to disagree with, and marking them sharp would put a hard rim
/// on every open mesh.
fn apply_smoothing_groups(mesh: &mut MeshData, groups: &[u32]) {
    let mut edge_faces: AHashMap<(u32, u32), (u32, Option<u32>)> = AHashMap::new();
    for f in 0..mesh.num_faces() {
        let verts = mesh.face_verts(f);
        for i in 0..verts.len() {
            let a = verts[i];
            let b = verts[(i + 1) % verts.len()];
            if a == b {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            edge_faces
                .entry(key)
                .and_modify(|e| {
                    if e.1.is_none() {
                        e.1 = Some(f)
                    }
                })
                .or_insert((f, None));
        }
    }

    let mut sharp = vec![false; mesh.num_edges() as usize];
    for (e, &[a, b]) in mesh.edges().iter().enumerate() {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&(f0, Some(f1))) = edge_faces.get(&key) {
            let (g0, g1) = (groups[f0 as usize], groups[f1 as usize]);
            sharp[e] = g0 == 0 || g1 == 0 || g0 != g1;
        }
    }
    let smooth: Vec<bool> = groups.iter().map(|&g| g != 0).collect();

    mesh.attributes_mut().insert(
        names::SHARP_EDGE,
        Domain::Edge,
        AttrData::Bool(Arc::new(sharp)),
    );
    mesh.attributes_mut().insert(
        names::SHADE_SMOOTH,
        Domain::Face,
        AttrData::Bool(Arc::new(smooth)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::{Handedness, UpAxis};

    fn quad_mesh() -> ImportMesh {
        ImportMesh {
            name: "quad".into(),
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            faces: vec![vec![0, 1, 2, 3]],
            ..Default::default()
        }
    }

    #[test]
    fn yup_source_becomes_zup() {
        let mut src = SourceMetadata::new("test");
        src.axis = AxisConvention::GLTF;
        let mut scene = ImportScene::new(src);
        scene.meshes.push(ImportMesh {
            positions: vec![Vec3::Y], // "up" in the source
            faces: vec![],
            ..Default::default()
        });
        let (out, _) = scene.into_mesh_data();
        assert!((out[0].1.positions()[0] - Vec3::Z).length() < 1e-6);
    }

    #[test]
    fn unit_scale_is_applied() {
        let mut src = SourceMetadata::new("test");
        src.unit_scale = 0.001; // millimetres
        let mut scene = ImportScene::new(src);
        scene.meshes.push(ImportMesh {
            positions: vec![Vec3::new(1000.0, 0.0, 0.0)],
            ..Default::default()
        });
        let (out, _) = scene.into_mesh_data();
        assert!((out[0].1.positions()[0] - Vec3::X).length() < 1e-6);
    }

    #[test]
    fn left_handed_source_reverses_winding() {
        let mut src = SourceMetadata::new("test");
        src.axis = AxisConvention {
            up: UpAxis::YUp,
            handedness: Handedness::Left,
        };
        let mut scene = ImportScene::new(src);
        scene.meshes.push(quad_mesh());
        let (out, _) = scene.into_mesh_data();
        assert_eq!(
            out[0].1.face_verts(0),
            &[3, 2, 1, 0],
            "a handedness flip must reverse winding or every face is backfacing"
        );
    }

    #[test]
    fn uv_origin_is_flipped_for_top_left_formats() {
        let mut src = SourceMetadata::new("test");
        src.uv_origin = UvOrigin::TopLeft;
        let mut scene = ImportScene::new(src);
        let mut m = quad_mesh();
        m.corner_uvs = Some(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 0.25),
            Vec2::new(0.0, 0.25),
        ]);
        scene.meshes.push(m);
        let (out, _) = scene.into_mesh_data();
        let uvs = out[0].1.attributes().get_f32x2("uv_map").unwrap();
        assert_eq!(uvs[0], Vec2::new(0.0, 1.0));
        assert_eq!(uvs[2], Vec2::new(1.0, 0.75));
    }

    #[test]
    fn corner_uvs_land_in_the_corner_domain() {
        let mut scene = ImportScene::new(SourceMetadata::new("test"));
        let mut m = quad_mesh();
        m.corner_uvs = Some(vec![Vec2::ZERO; 4]);
        scene.meshes.push(m);
        let (out, _) = scene.into_mesh_data();
        let layer = out[0].1.attributes().get("uv_map").unwrap();
        assert_eq!(layer.domain, Domain::Corner);
        out[0].1.assert_valid();
    }

    #[test]
    fn degenerate_faces_are_dropped_with_a_note() {
        let mut scene = ImportScene::new(SourceMetadata::new("test"));
        scene.meshes.push(ImportMesh {
            name: "m".into(),
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            faces: vec![vec![0, 1, 2], vec![0, 1]],
            ..Default::default()
        });
        let (out, src) = scene.into_mesh_data();
        assert_eq!(out[0].1.num_faces(), 1);
        assert!(src
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::DroppedDegenerateFace { face: 1, .. })));
        out[0].1.assert_valid();
    }

    #[test]
    fn out_of_range_indices_are_dropped_not_panicked_on() {
        let mut scene = ImportScene::new(SourceMetadata::new("test"));
        scene.meshes.push(ImportMesh {
            name: "m".into(),
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            faces: vec![vec![0, 1, 99]],
            ..Default::default()
        });
        let (out, src) = scene.into_mesh_data();
        assert_eq!(out[0].1.num_faces(), 0);
        assert!(src
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::OutOfRangeIndex { .. })));
    }

    #[test]
    fn dropping_a_face_keeps_corner_arrays_aligned() {
        let mut scene = ImportScene::new(SourceMetadata::new("test"));
        scene.meshes.push(ImportMesh {
            name: "m".into(),
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)],
            // The bad face sits first, so a naive filter would shift every UV.
            faces: vec![vec![0, 1], vec![0, 1, 3, 2]],
            corner_uvs: Some(vec![
                Vec2::new(9.0, 9.0),
                Vec2::new(9.0, 9.0),
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ]),
            ..Default::default()
        });
        let (out, _) = scene.into_mesh_data();
        let uvs = out[0].1.attributes().get_f32x2("uv_map").unwrap();
        assert_eq!(uvs.len(), 4);
        assert_eq!(uvs[0], Vec2::ZERO, "the dropped face's UVs must go with it");
        out[0].1.assert_valid();
    }

    #[test]
    fn smoothing_groups_become_sharp_edges() {
        // Two quads sharing one edge, in different smoothing groups.
        let mut scene = ImportScene::new(SourceMetadata::new("test"));
        scene.meshes.push(ImportMesh {
            name: "m".into(),
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
            ],
            faces: vec![vec![0, 1, 2, 3], vec![1, 4, 5, 2]],
            smoothing_groups: Some(vec![1, 2]),
            ..Default::default()
        });
        let (out, _) = scene.into_mesh_data();
        let mesh = &out[0].1;
        let sharp = mesh.attributes().get_bool(names::SHARP_EDGE).unwrap();
        let shared = mesh
            .edges()
            .iter()
            .position(|e| *e == [1, 2])
            .expect("shared edge");
        assert!(sharp[shared], "an edge between two groups must be sharp");
        assert_eq!(
            sharp.iter().filter(|&&s| s).count(),
            1,
            "boundary edges must stay smooth"
        );
        assert_eq!(
            mesh.attributes().get_bool(names::SHADE_SMOOTH).unwrap(),
            &[true, true]
        );
    }

    #[test]
    fn smoothing_group_zero_means_flat() {
        let mut scene = ImportScene::new(SourceMetadata::new("test"));
        scene.meshes.push(ImportMesh {
            name: "m".into(),
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            faces: vec![vec![0, 1, 2]],
            smoothing_groups: Some(vec![0]),
            ..Default::default()
        });
        let (out, _) = scene.into_mesh_data();
        assert_eq!(
            out[0].1.attributes().get_bool(names::SHADE_SMOOTH).unwrap(),
            &[false]
        );
    }

    #[test]
    fn uv_origin_conversion_roundtrips() {
        for origin in [UvOrigin::BottomLeft, UvOrigin::TopLeft] {
            let uv = Vec2::new(0.3, 0.7);
            assert!((origin.from_kreate(origin.to_kreate(uv)) - uv).length() < 1e-6);
        }
    }
}
