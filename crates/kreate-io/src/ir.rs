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
use glam::{Quat, Vec2, Vec3, Vec4};
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
    /// Coincident vertices were merged. The count is how many disappeared.
    Welded { mesh: String, merged: usize },
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
    /// Index into the scene's `images`, for each of the three maps a standard
    /// surface has. An index rather than a name: two images in one file may
    /// share a name, and a name the importer cannot resolve is a texture that
    /// silently never arrives.
    pub base_color_texture: Option<usize>,
    /// glTF's packing: roughness in green, metallic in blue.
    pub metallic_roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    /// glTF's `normalTexture.scale`.
    pub normal_scale: f32,
    /// What `emissive` is multiplied by: `KHR_materials_emissive_strength`.
    pub emissive_strength: f32,
    /// `KHR_materials_transmission`: zero is solid, one is glass.
    pub transmission: f32,
    /// `KHR_materials_ior`.
    pub ior: f32,
    /// glTF's `MASK`: cut away below this opacity. `None` for `OPAQUE`, and
    /// for `BLEND`, which arrives as glass — see the glTF reader.
    pub alpha_clip: Option<f32>,
}

/// An image as a file carries it: eight bits a channel, four channels.
///
/// **The bytes as stored**, not decoded. Whether they are sRGB-encoded colour
/// or plain numbers is not the image's to say in glTF — it is the slot's, and
/// the same image can sit in two — so the space is decided where a material
/// takes it up, which is the rule the document follows too.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA, `width * height * 4` bytes, top row first.
    pub pixels: Vec<u8>,
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
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            emissive_strength: 1.0,
            transmission: 0.0,
            ior: 1.5,
            alpha_clip: None,
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

    /// Merges vertices that share a position, and remaps the faces onto them.
    ///
    /// The highest-leverage step in any importer, and the one that separates a
    /// toy from a usable one. Formats that store triangle soup — STL above all,
    /// but also glTF after a vertex split — give every triangle its own copies
    /// of its corners. Such a mesh looks perfect and survives no edit at all: it
    /// has no shared edges, so extruding a face moves one triangle, subdividing
    /// leaves cracks, and the whole thing comes apart on the first operation.
    ///
    /// # Corner attributes are not touched, and that is the point
    ///
    /// Welding changes which *vertex* a corner refers to; it does not change the
    /// corners. UVs and split normals live in the corner domain precisely so
    /// that a vertex shared by two faces can still carry a different UV in each,
    /// which is what makes a seam possible. So they stay exactly as authored —
    /// this is the pay-off for the domain model, and the reason a position-only
    /// weld is safe here where in a vertex-attribute model it would destroy every
    /// seam in the mesh.
    ///
    /// # Determinism
    ///
    /// Welding with a tolerance is not transitive: `a` may be within `epsilon`
    /// of `b`, and `b` of `c`, without `a` being within `epsilon` of `c`. Any
    /// greedy merge therefore depends on the order vertices are visited, and a
    /// hash grid visited in map order would give different geometry on different
    /// runs of the same import.
    ///
    /// So vertices are visited in index order and each one joins the
    /// **lowest-indexed** representative it is close enough to, or becomes a new
    /// one. That is a definite rule that produces the same mesh every time, and
    /// the file's own vertex order is the tie-breaker.
    pub fn weld(&mut self, epsilon: f32, notes: &mut Vec<LossyNote>) {
        self.weld_mapped(epsilon, notes);
    }

    /// [`ImportMesh::weld`], saying where each vertex went: `Some(map)` with
    /// `map[old] == new` when anything merged. For data kept per vertex
    /// outside the mesh — a skin's weights — which has to follow.
    pub fn weld_mapped(&mut self, epsilon: f32, notes: &mut Vec<LossyNote>) -> Option<Vec<u32>> {
        if self.positions.is_empty() || epsilon <= 0.0 {
            return None;
        }
        let before = self.positions.len();

        // A grid whose cell is the tolerance, so anything within `epsilon` of a
        // point lies in its cell or one of the twenty-six around it.
        let cell = |p: Vec3| -> [i64; 3] {
            [
                (p.x as f64 / epsilon as f64).floor() as i64,
                (p.y as f64 / epsilon as f64).floor() as i64,
                (p.z as f64 / epsilon as f64).floor() as i64,
            ]
        };
        let mut grid: ahash::AHashMap<[i64; 3], Vec<u32>> = ahash::AHashMap::new();

        // Original index -> index of the representative it merged into.
        let mut representative: Vec<u32> = Vec::with_capacity(before);
        // Representative original index -> its index in the welded positions.
        let mut welded: Vec<Vec3> = Vec::new();
        let mut slot_of: ahash::AHashMap<u32, u32> = ahash::AHashMap::new();

        let epsilon_sq = epsilon * epsilon;
        for (i, position) in self.positions.iter().enumerate() {
            let c = cell(*position);
            let mut found: Option<u32> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = [c[0] + dx, c[1] + dy, c[2] + dz];
                        let Some(bucket) = grid.get(&key) else {
                            continue;
                        };
                        // Buckets hold representatives in index order, so the
                        // first hit in a bucket is that cell's lowest — but the
                        // search does not stop there. Taking the minimum across
                        // all twenty-seven cells is what makes the result
                        // independent of the order the neighbours are visited,
                        // rather than merely repeatable for this loop nesting.
                        for candidate in bucket {
                            if self.positions[*candidate as usize].distance_squared(*position)
                                <= epsilon_sq
                            {
                                found = Some(match found {
                                    Some(best) => best.min(*candidate),
                                    None => *candidate,
                                });
                                break;
                            }
                        }
                    }
                }
            }
            match found {
                Some(rep) => representative.push(slot_of[&rep]),
                None => {
                    let slot = welded.len() as u32;
                    welded.push(*position);
                    slot_of.insert(i as u32, slot);
                    grid.entry(c).or_default().push(i as u32);
                    representative.push(slot);
                }
            }
        }

        if welded.len() == before {
            return None;
        }
        notes.push(LossyNote::Welded {
            mesh: self.name.clone(),
            merged: before - welded.len(),
        });

        // Remap the faces, collapsing corners that have become the same vertex.
        //
        // A quad with two coincident corners is a triangle, not a quad with a
        // zero-length edge — and a triangle whose corners all merged is nothing
        // at all. Both have to be handled here rather than left for `sanitize`,
        // because the corner arrays have to lose exactly the corners the faces
        // lose.
        let mut kept_corner: Vec<bool> = Vec::with_capacity(self.num_corners());
        let mut faces: Vec<Vec<u32>> = Vec::with_capacity(self.faces.len());
        for face in &self.faces {
            let mut out: Vec<u32> = Vec::with_capacity(face.len());
            for (position_in_face, corner) in face.iter().enumerate() {
                let v = representative[*corner as usize];
                // Compared against the previous kept corner, and against the
                // first when this is the last: a face is a cycle, so its ends are
                // adjacent too.
                let duplicate = out.last() == Some(&v)
                    || (position_in_face + 1 == face.len() && out.first() == Some(&v));
                kept_corner.push(!duplicate);
                if !duplicate {
                    out.push(v);
                }
            }
            faces.push(out);
        }

        self.positions = welded;
        self.faces = faces;
        retain_corners(&mut self.corner_uvs, &kept_corner);
        retain_corners(&mut self.corner_normals, &kept_corner);

        // Faces that collapsed below three corners are dropped by `sanitize`,
        // which also keeps the corner arrays in step — one place that decides
        // what an unrepresentable face is.
        self.sanitize(notes);
        Some(representative)
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

/// Keeps only the corners a weld did not collapse.
///
/// Generic over the element so UVs and normals cannot drift apart: they are the
/// same operation on the same mask, and writing it twice is how one of them ends
/// up a corner short.
fn retain_corners<T: Copy>(corners: &mut Option<Vec<T>>, keep: &[bool]) {
    let Some(values) = corners else { return };
    if values.len() != keep.len() {
        // The array does not describe these corners, so filtering it would
        // silently misalign it. Leaving it is wrong too, but it is wrong
        // visibly — and `sanitize` will not have produced this state.
        return;
    }
    let mut it = keep.iter();
    values.retain(|_| *it.next().unwrap_or(&false));
}

/// Where a node is, as the source file wrote it.
///
/// Kept as translation, rotation and scale rather than as a matrix, because
/// that is what every format writes and what the document stores — and because
/// decomposing a matrix back into the three is lossy in exactly the cases
/// (mirrored and sheared transforms) where somebody would notice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImportTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for ImportTransform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl ImportTransform {
    /// The same placement, in Kreate space.
    ///
    /// The axis change is a *conjugation* of the linear part and a plain
    /// multiplication of the translation, which is the whole of the subtlety
    /// here. Rotating a Y-up rotation by the Y-up-to-Z-up matrix once gives a
    /// rotation about the wrong axis; it has to be undone on the other side as
    /// well, or a turntable animation spins about the floor.
    ///
    /// The unit scale cancels out of the conjugation, which is correct: metres
    /// per unit changes where a thing is, not how it is turned. It applies to
    /// the translation exactly once.
    pub fn to_kreate(&self, source: &SourceMetadata) -> kreate_math::DAffine3 {
        let axis = source.axis.to_kreate();
        let inverse = axis.transpose();
        let linear =
            axis * Mat3::from_quat(self.rotation) * Mat3::from_diagonal(self.scale) * inverse;
        let translation = axis * (self.translation * source.unit_scale);
        kreate_math::DAffine3::from_mat3_translation(linear.as_dmat3(), translation.as_dvec3())
    }

    /// The same placement, back in the file's space.
    ///
    /// Written as the *inverse* of [`ImportTransform::to_kreate`] rather than as
    /// a second conversion derived from the same rules. Two independently
    /// written axis conversions that disagree is the failure this module exists
    /// to prevent, and a round trip through a file is exactly where the
    /// disagreement would first show — so the test asserts the identity
    /// `from_kreate(to_kreate(t)) == t` rather than any particular matrix.
    ///
    /// **This is where an export loses something, and it loses it for a real
    /// reason.** Every interchange format writes a translation, a rotation and a
    /// scale, so a sheared transform cannot be written at all; it comes out as
    /// the nearest rotation and scale. Nothing the document can currently
    /// produce is sheared — a `DTransform` is itself a TRS — but a matrix
    /// arriving from somewhere else would be, and silently rounding it is worth
    /// saying out loud.
    pub fn from_kreate(affine: kreate_math::DAffine3, source: &SourceMetadata) -> Self {
        let axis = source.axis.to_kreate();
        let inverse = axis.transpose();
        // The conjugation undone: `axis⁻¹ · (axis · M · axis⁻¹) · axis` is `M`.
        let linear = inverse * affine.matrix3.as_mat3() * axis;
        // A unit scale of zero is not a scale — it collapses the file onto the
        // origin, and no inverse of that exists. Treated as one rather than
        // writing infinities into a file that would then open as nothing.
        let unit = if source.unit_scale.is_normal() {
            source.unit_scale
        } else {
            1.0
        };
        let translation = (inverse * affine.translation.as_vec3()) / unit;
        let (scale, rotation, _) =
            glam::Affine3A::from_mat3(linear).to_scale_rotation_translation();
        Self {
            translation,
            rotation,
            scale,
        }
    }
}

/// What a node carries.
///
/// A node with nothing is not a mistake: a file's hierarchy is mostly empties
/// that exist to group and to be animated, and dropping them would flatten a
/// rig into a pile of parts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ImportNodeData {
    #[default]
    Empty,
    /// Index into [`ImportScene::meshes`].
    Mesh(usize),
    /// Index into [`ImportScene::lights`].
    Light(usize),
    /// Index into [`ImportScene::cameras`].
    Camera(usize),
}

/// One node of the source file's hierarchy.
///
/// The thing the IR did not have, and the reason a light could not be imported
/// at all: a light is a placement and a datablock, and there was nowhere to put
/// the placement. Meshes came through as `(name, MeshData)` pairs with their
/// position silently dropped — which was survivable only because everything
/// exported so far had its transform baked in.
#[derive(Clone, Debug)]
pub struct ImportNode {
    pub name: String,
    /// Index into [`ImportScene::nodes`], or `None` for a root.
    ///
    /// An index and not a child list, because a cycle is then not expressible
    /// in the common direction — a node can only name one parent — and because
    /// the document stores parents the same way.
    pub parent: Option<usize>,
    pub transform: ImportTransform,
    pub data: ImportNodeData,
    /// Index into [`ImportScene::skins`], for a mesh the joints deform.
    pub skin: Option<usize>,
}

impl ImportNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            transform: ImportTransform::default(),
            data: ImportNodeData::Empty,
            skin: None,
        }
    }
}

/// A skin as the file wrote it: which nodes are its joints and where each
/// stood when the mesh was bound.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportSkin {
    pub name: String,
    /// Indices into [`ImportScene::nodes`], in the order a vertex's joint
    /// numbers count them.
    pub joints: Vec<usize>,
    /// Per joint, in the file's space: from the mesh to the joint's own space
    /// at the bind. `None` when the file left them out, which glTF reads as
    /// the identity — the joints' placements in the file are then the bind.
    pub inverse_bind: Option<Vec<glam::Mat4>>,
}

/// One mesh's joints and weights per vertex, as glTF carries them: four of
/// each. Parallel to the mesh's positions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VertexSkin {
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
}

/// What kind of light, in the units the format writes.
///
/// glTF's `KHR_lights_punctual` exactly, because §2.2 chose those units for the
/// datablock for this reason: the boundary is a copy rather than a conversion,
/// and a conversion is where rounding accumulates across a round trip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImportLightKind {
    /// Candela.
    Point,
    /// Lux.
    Directional,
    /// Candela, with a cone in radians: the inner and outer half-angles.
    Spot { inner: f32, outer: f32 },
}

/// A light as described by an interchange format.
#[derive(Clone, Debug)]
pub struct ImportLight {
    pub name: String,
    pub kind: ImportLightKind,
    /// Linear, bounded by one. A tint and not an amount of light.
    pub color: [f32; 3],
    /// Candela or lux, per [`ImportLight::kind`].
    pub intensity: f32,
    /// Metres past which the format says the light stops. `None` is the usual
    /// case and means "it does not".
    pub range: Option<f32>,
}

/// A camera as described by an interchange format.
///
/// glTF writes a vertical field of view and an aspect; the datablock stores a
/// focal length and a sensor. The conversion happens at the boundary rather
/// than in the document, because a file that says "forty degrees" is not saying
/// anything about a sensor size and inventing one silently would make the
/// export write back a different number.
#[derive(Clone, Debug)]
pub struct ImportCamera {
    pub name: String,
    /// Radians. `None` for an orthographic camera.
    pub fov_y: Option<f32>,
    /// The width of an orthographic view, in source units.
    pub ortho_width: Option<f32>,
    pub clip_near: f32,
    pub clip_far: Option<f32>,
}

/// One node on the way out, in Kreate space.
///
/// Its own type rather than a reused [`ImportNode`], and the difference is the
/// one that matters: an `ImportNode`'s transform is in the *file's* space and
/// this one's is in the document's. Sharing the type would mean a field whose
/// meaning depends on which direction the value is travelling, which is the
/// kind of thing that is right until somebody writes a function taking either.
///
/// The axis change therefore happens in the writer, exactly once — which is the
/// rule this module's own documentation opens with.
#[derive(Clone, Debug)]
pub struct ExportNode {
    pub name: String,
    /// Index into [`ExportScene::nodes`], or `None` for a root.
    pub parent: Option<usize>,
    /// Relative to the parent, in Kreate space.
    pub transform: kreate_math::DTransform,
    pub data: ImportNodeData,
    /// Index into [`ExportScene::skins`], for a mesh node the bones deform.
    pub skin: Option<usize>,
    /// How far each of its mesh's morph targets is turned up, in the order
    /// [`ExportScene::morph_targets`] lists them.
    pub morph_weights: Vec<f32>,
}

impl ExportNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            transform: kreate_math::DTransform::IDENTITY,
            data: ImportNodeData::Empty,
            skin: None,
            morph_weights: Vec::new(),
        }
    }
}

/// One node's morph target weights over time: a list of weights per time,
/// one weight per target.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportWeights {
    /// Index into [`ExportScene::nodes`].
    pub node: usize,
    /// Seconds, increasing.
    pub times: Vec<f32>,
    pub values: Vec<Vec<f32>>,
}

/// A skin on the way out: the nodes that are its joints, in the order a
/// vertex's joint numbers count them, and where each stood when the mesh was
/// bound.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportSkin {
    pub name: String,
    /// Indices into [`ExportScene::nodes`].
    pub joints: Vec<usize>,
    /// The bone each joint is, by name: a mesh's weight layer
    /// `weight:<name>` is that joint's weights.
    pub joint_names: Vec<String>,
    /// Per joint, in Kreate space: the joint's world frame at rest, inverted,
    /// times the mesh's world transform. What takes a vertex of the mesh into
    /// the joint's own space, which is glTF's inverse bind matrix with the
    /// mesh's placement folded in — glTF ignores a skinned mesh node's own.
    pub inverse_bind: Vec<kreate_math::DAffine3>,
}

/// How a channel travels between its keys, as glTF says it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelInterpolation {
    /// Holds each key until the next.
    Step,
    /// Straight lines, and the shorter arc for a rotation.
    #[default]
    Linear,
    /// Hermite, with an in and an out tangent stored around every value.
    CubicSpline,
}

/// Which part of a node's placement a channel drives.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelValues {
    Translation(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Scale(Vec<Vec3>),
}

/// One animated part of one node, in the *file's* space and time.
///
/// As the file wrote it, keys and all, rather than already turned into curves:
/// the axis change mixes the components of a translation (Y-up's Y is Z-up's
/// Z) and the rotation is a quaternion where the document keys Euler angles, so
/// the conversion needs the whole placement at an instant — which is the
/// builder's job, with every channel of the node in hand.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportChannel {
    /// Index into [`ImportScene::nodes`].
    pub node: usize,
    pub interpolation: ChannelInterpolation,
    /// Seconds, increasing.
    pub times: Vec<f32>,
    /// One per time — or, for a cubic spline, three: in-tangent, value,
    /// out-tangent, which is how glTF lays them out.
    pub values: ChannelValues,
}

impl ImportChannel {
    /// How many keys the channel has: the shorter of its two lists, so a file
    /// whose lists disagree is read as far as they agree rather than indexed
    /// out of bounds.
    pub fn key_count(&self) -> usize {
        let len = match &self.values {
            ChannelValues::Translation(v) | ChannelValues::Scale(v) => v.len(),
            ChannelValues::Rotation(v) => v.len(),
        };
        let per = if self.interpolation == ChannelInterpolation::CubicSpline {
            3
        } else {
            1
        };
        (len / per).min(self.times.len())
    }

    /// Where the key `i`'s value sits in the values list.
    fn at(&self, i: usize) -> usize {
        if self.interpolation == ChannelInterpolation::CubicSpline {
            i * 3 + 1
        } else {
            i
        }
    }

    /// The span `t` falls in and how far along it: `(i, i + 1, u)`, clamped to
    /// the first and last key, as glTF holds a channel outside its range.
    fn span(&self, t: f32) -> Option<(usize, usize, f32)> {
        let n = self.key_count();
        if n == 0 {
            return None;
        }
        if n == 1 || t <= self.times[0] {
            return Some((0, 0, 0.0));
        }
        if t >= self.times[n - 1] {
            return Some((n - 1, n - 1, 0.0));
        }
        let next = self.times[..n].partition_point(|k| *k <= t).clamp(1, n - 1);
        let prev = next - 1;
        let width = self.times[next] - self.times[prev];
        let u = if width > 0.0 {
            (t - self.times[prev]) / width
        } else {
            0.0
        };
        Some((prev, next, u))
    }

    /// The vector value at `t` — a translation's or a scale's.
    pub fn vec3_at(&self, t: f32) -> Option<Vec3> {
        let (ChannelValues::Translation(v) | ChannelValues::Scale(v)) = &self.values else {
            return None;
        };
        let (a, b, u) = self.span(t)?;
        let (va, vb) = (v[self.at(a)], v[self.at(b)]);
        Some(match self.interpolation {
            _ if a == b => va,
            ChannelInterpolation::Step => va,
            ChannelInterpolation::Linear => va.lerp(vb, u),
            ChannelInterpolation::CubicSpline => {
                let width = self.times[b] - self.times[a];
                hermite(
                    Vec4::from((va, 0.0)),
                    Vec4::from((v[self.at(a) + 1], 0.0)) * width,
                    Vec4::from((vb, 0.0)),
                    Vec4::from((v[self.at(b) - 1], 0.0)) * width,
                    u,
                )
                .truncate()
            }
        })
    }

    /// The rotation at `t`.
    pub fn quat_at(&self, t: f32) -> Option<Quat> {
        let ChannelValues::Rotation(v) = &self.values else {
            return None;
        };
        let (a, b, u) = self.span(t)?;
        let (qa, qb) = (v[self.at(a)], v[self.at(b)]);
        Some(match self.interpolation {
            _ if a == b => qa.normalize(),
            ChannelInterpolation::Step => qa.normalize(),
            ChannelInterpolation::Linear => qa.normalize().slerp(qb.normalize(), u),
            ChannelInterpolation::CubicSpline => {
                // glTF's own rule: the spline on the four numbers, then
                // normalised.
                let width = self.times[b] - self.times[a];
                let h = hermite(
                    Vec4::from(qa),
                    Vec4::from(v[self.at(a) + 1]) * width,
                    Vec4::from(qb),
                    Vec4::from(v[self.at(b) - 1]) * width,
                    u,
                );
                Quat::from_vec4(h).normalize()
            }
        })
    }
}

fn hermite(p: Vec4, m: Vec4, q: Vec4, n: Vec4, u: f32) -> Vec4 {
    let u2 = u * u;
    let u3 = u2 * u;
    p * (2.0 * u3 - 3.0 * u2 + 1.0)
        + m * (u3 - 2.0 * u2 + u)
        + q * (-2.0 * u3 + 3.0 * u2)
        + n * (u3 - u2)
}

/// One clip: glTF's `animation`, a set of channels played together.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportAnimation {
    pub name: String,
    pub channels: Vec<ImportChannel>,
}

/// One node's movement on the way out, in Kreate space.
///
/// Already sampled into placements, not curves: the document keys Euler angles
/// and a file keys quaternions in another axis convention, and the only
/// conversion between the two that cannot disagree with how the document plays
/// is to ask the document where the object is. The writer changes the axes,
/// once, the way it does for a node's rest placement.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportTrack {
    /// Index into [`ExportScene::nodes`].
    pub node: usize,
    /// Seconds, increasing.
    pub times: Vec<f32>,
    /// Relative to the parent, as [`ExportNode::transform`] is, one per time.
    pub transforms: Vec<kreate_math::DTransform>,
    /// Hold each until the next rather than drawing lines between them.
    pub step: bool,
    /// Which parts move: a channel is written for each, and only for each.
    pub translation: bool,
    pub rotation: bool,
    pub scale: bool,
}

/// One clip on the way out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExportAnimation {
    pub name: String,
    pub tracks: Vec<ExportTrack>,
    /// Shape keys moving, as morph target weights.
    pub weights: Vec<ExportWeights>,
}

/// A scene on its way into a file.
///
/// The other half of [`ImportScene`], and it did not exist: an export took a
/// list of `(name, &MeshData)` pairs, so a lit scene could be *imported* and
/// could not be written back. §5.7's fidelity contract was open at one end, and
/// a contract that only holds in one direction is not one.
///
/// Meshes are borrowed because the document already owns them and an export is
/// a read. Everything else is owned, because it is a conversion of what the
/// document holds rather than a view of it.
pub struct ExportScene<'a> {
    pub meshes: Vec<(String, &'a MeshData)>,
    pub materials: Vec<ImportMaterial>,
    /// What the materials' texture indices point at.
    pub images: Vec<ImportImage>,
    /// Which material each mesh's slots refer to, parallel to `meshes`.
    ///
    /// A face carries a *slot* index and a slot is per-mesh (§7.5), so slot 1
    /// of one mesh and slot 1 of another are not the same material. Without
    /// this table an exporter keyed on the face's own number would quietly give
    /// the second mesh the first one's materials.
    ///
    /// Empty — for the scene as a whole or for one mesh — means the slot number
    /// *is* the material number, which is what a caller handing over a bare
    /// mesh list means.
    pub mesh_slots: Vec<Vec<Option<usize>>>,
    pub lights: Vec<ImportLight>,
    pub cameras: Vec<ImportCamera>,
    /// The hierarchy. Empty means the format should place each mesh at the
    /// origin, which is what OBJ and STL can say and what a bare mesh list is.
    pub nodes: Vec<ExportNode>,
    /// What moves. Empty for a still scene and for every format without time.
    pub animations: Vec<ExportAnimation>,
    /// The skins the mesh nodes' `skin` indices name.
    pub skins: Vec<ExportSkin>,
    /// Per mesh, parallel to `meshes` (and possibly shorter): the shape keys
    /// it carries as morph targets, by name — each a `shape:<name>` layer.
    pub morph_targets: Vec<Vec<String>>,
    /// Meshes whose shape changes over time — skinned, shape keys playing,
    /// cloth baked — as their points at each frame. For a format that can
    /// say "these points, at this time" and has no rig of its own to carry.
    pub point_samples: Vec<PointSamples>,
}

/// One mesh node's points over time, in its own space.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointSamples {
    /// Index into [`ExportScene::nodes`].
    pub node: usize,
    /// Seconds, increasing.
    pub times: Vec<f32>,
    /// Every point, at each time; as many as the mesh has.
    pub points: Vec<Vec<Vec3>>,
}

impl<'a> ExportScene<'a> {
    pub fn new(meshes: Vec<(String, &'a MeshData)>) -> Self {
        Self {
            meshes,
            materials: Vec::new(),
            images: Vec::new(),
            mesh_slots: Vec::new(),
            lights: Vec::new(),
            cameras: Vec::new(),
            nodes: Vec::new(),
            animations: Vec::new(),
            skins: Vec::new(),
            morph_targets: Vec::new(),
            point_samples: Vec::new(),
        }
    }

    /// Which material a face of `mesh` means by slot `slot`.
    ///
    /// `None` for a slot nothing is assigned to, which is ordinary: a mesh can
    /// have three slots with the middle one empty.
    pub fn material_of(&self, mesh: usize, slot: i32) -> Option<usize> {
        if slot < 0 {
            return None;
        }
        match self.mesh_slots.get(mesh) {
            // No table at all: the face's number is the material's number.
            None => Some(slot as usize),
            Some(slots) if slots.is_empty() => Some(slot as usize),
            Some(slots) => slots.get(slot as usize).copied().flatten(),
        }
    }
}

/// A scene as read from a file, before it becomes a document.
#[derive(Clone, Debug)]
pub struct ImportScene {
    pub meshes: Vec<ImportMesh>,
    pub materials: Vec<ImportMaterial>,
    /// What the materials' texture indices point at.
    pub images: Vec<ImportImage>,
    pub lights: Vec<ImportLight>,
    pub cameras: Vec<ImportCamera>,
    /// The file's hierarchy, parents before children is *not* guaranteed — a
    /// node's parent is an index and may point either way.
    ///
    /// Empty for a format that has no hierarchy at all, which OBJ and STL do
    /// not: their meshes are placed at the origin because that is where the
    /// file puts them, and inventing a node for each would claim a structure
    /// the file does not have.
    pub nodes: Vec<ImportNode>,
    /// What moves, clip by clip. Empty for a format without time.
    pub animations: Vec<ImportAnimation>,
    pub skins: Vec<ImportSkin>,
    /// Per mesh, parallel to `meshes` (and possibly shorter): the joints and
    /// weights of its vertices, for a mesh a skin deforms.
    pub vertex_skins: Vec<Option<VertexSkin>>,
    /// Per mesh, parallel to `meshes` (and possibly shorter): its morph
    /// targets, each a name and a displacement per vertex in the file's space.
    pub vertex_morphs: Vec<Vec<(String, Vec<Vec3>)>>,
    /// Per node, parallel to `nodes` (and possibly shorter): its morph target
    /// weights as the file stands.
    pub node_weights: Vec<Vec<f32>>,
    /// Morph target weights moving: per channel, the node and its keys.
    pub morph_channels: Vec<ImportMorphChannel>,
    pub source: SourceMetadata,
}

/// One node's morph target weights over time, as the file keyed them.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportMorphChannel {
    pub animation: String,
    /// Index into [`ImportScene::nodes`].
    pub node: usize,
    pub interpolation: ChannelInterpolation,
    /// Seconds, increasing.
    pub times: Vec<f32>,
    /// Flat: the targets' weights for the first time, then the next — and,
    /// for a cubic spline, each key's in-tangents, values and out-tangents.
    pub values: Vec<f32>,
}

impl ImportScene {
    pub fn new(source: SourceMetadata) -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            images: Vec::new(),
            lights: Vec::new(),
            cameras: Vec::new(),
            nodes: Vec::new(),
            animations: Vec::new(),
            skins: Vec::new(),
            vertex_skins: Vec::new(),
            vertex_morphs: Vec::new(),
            node_weights: Vec::new(),
            morph_channels: Vec::new(),
            source,
        }
    }

    /// A matrix in the file's space, in Kreate's: the axis change on both
    /// sides of the linear part, the unit on the translation — what
    /// [`ImportTransform::to_kreate`] does to a placement, for a matrix that
    /// need not be one.
    pub fn matrix_to_kreate(&self, m: glam::Mat4) -> kreate_math::DAffine3 {
        let axis = self.source.axis.to_kreate();
        let linear = axis * Mat3::from_mat4(m) * axis.transpose();
        let translation = axis * (m.w_axis.truncate() * self.source.unit_scale);
        kreate_math::DAffine3::from_mat3_translation(linear.as_dmat3(), translation.as_dvec3())
    }

    /// Where a node ends up, with its parents applied.
    ///
    /// `None` when the chain is broken or loops — a file can say anything, and
    /// a loop here would be an infinite one. A broken chain is reported rather
    /// than guessed at, because the honest answer is "this file's hierarchy is
    /// not one" and the caller can place the node at the root and say so.
    pub fn world_transform(&self, node: usize) -> Option<kreate_math::DAffine3> {
        let mut out = kreate_math::DAffine3::IDENTITY;
        let mut at = node;
        // Bounded by the node count: a chain longer than that has revisited
        // something, which is the loop.
        for _ in 0..=self.nodes.len() {
            let n = self.nodes.get(at)?;
            out = n.transform.to_kreate(&self.source) * out;
            match n.parent {
                Some(p) => at = p,
                None => return Some(out),
            }
        }
        None
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
            mesh.attributes_mut().insert(
                names::UV,
                Domain::Corner,
                AttrData::F32x2(Arc::new(vals)),
            );
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
mod weld_tests {
    use super::*;

    /// Two triangles sharing an edge, stored as soup: six vertices where four
    /// would do. This is what STL gives, every time.
    fn soup() -> ImportMesh {
        ImportMesh {
            name: "soup".to_string(),
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                // The second triangle repeats two of them, bit for bit.
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![vec![0, 1, 2], vec![3, 4, 5]],
            corner_uvs: Some(vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, 1.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ]),
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        }
    }

    #[test]
    fn triangle_soup_becomes_a_shared_surface() {
        // The whole point. Before welding these two triangles share no vertex, so
        // they share no edge — extruding one would tear the mesh in half.
        let mut mesh = soup();
        let mut notes = Vec::new();
        mesh.weld(1e-5, &mut notes);

        assert_eq!(mesh.positions.len(), 4, "the duplicates were not merged");
        assert_eq!(mesh.faces.len(), 2, "a face was lost");
        // The two faces now name two vertices in common, which is what an edge is.
        let shared: Vec<u32> = mesh.faces[0]
            .iter()
            .filter(|v| mesh.faces[1].contains(v))
            .copied()
            .collect();
        assert_eq!(shared.len(), 2, "the triangles still share no edge");
        assert!(matches!(
            notes.as_slice(),
            [LossyNote::Welded { merged: 2, .. }]
        ));
    }

    #[test]
    fn corner_attributes_are_left_exactly_as_authored() {
        // The pay-off for putting UVs in the corner domain. A vertex shared by
        // two faces can still carry a different UV in each — that is what a seam
        // is — so a position-only weld must not touch them.
        //
        // In a vertex-attribute model this same weld would have to pick one UV
        // per merged vertex and would destroy every seam in the mesh.
        let mut mesh = soup();
        let before = mesh.corner_uvs.clone();
        let mut notes = Vec::new();
        mesh.weld(1e-5, &mut notes);
        assert_eq!(mesh.corner_uvs, before, "the UVs were disturbed");
    }

    #[test]
    fn vertices_further_apart_than_the_tolerance_are_left_alone() {
        let mut mesh = soup();
        mesh.positions[3] = Vec3::new(1.5, 0.0, 0.0);
        let mut notes = Vec::new();
        mesh.weld(1e-5, &mut notes);
        assert_eq!(
            mesh.positions.len(),
            5,
            "a vertex was merged that should not be"
        );
    }

    #[test]
    fn a_weld_that_changes_nothing_records_nothing() {
        // A mesh that is already shared must come through untouched and unnoted:
        // a note on every import would train the reader to ignore them.
        let mut mesh = ImportMesh {
            name: "clean".to_string(),
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![vec![0, 1, 2]],
            corner_uvs: None,
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        let mut notes = Vec::new();
        mesh.weld(1e-5, &mut notes);
        assert_eq!(mesh.positions.len(), 3);
        assert!(notes.is_empty());
    }

    #[test]
    fn a_quad_with_two_coincident_corners_becomes_a_triangle() {
        // Not a quad with a zero-length edge. The corner arrays have to lose
        // exactly the corner the face loses, or every UV after it shifts by one —
        // which shows up as texturing that is subtly wrong from that face onward
        // and is very hard to trace back here.
        let mut mesh = ImportMesh {
            name: "pinched".to_string(),
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                // Coincident with the previous one.
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![vec![0, 1, 2, 3]],
            corner_uvs: Some(vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(9.0, 9.0),
                Vec2::new(0.0, 1.0),
            ]),
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        let mut notes = Vec::new();
        mesh.weld(1e-5, &mut notes);

        assert_eq!(mesh.faces, vec![vec![0, 1, 2]]);
        assert_eq!(
            mesh.corner_uvs.as_ref().unwrap().len(),
            3,
            "the corner arrays did not follow the face"
        );
        assert_eq!(
            mesh.corner_uvs.as_ref().unwrap()[2],
            Vec2::new(0.0, 1.0),
            "the wrong corner was dropped"
        );
    }

    #[test]
    fn a_face_whose_corners_all_merge_is_dropped() {
        let mut mesh = ImportMesh {
            name: "collapsed".to_string(),
            positions: vec![
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![vec![0, 1, 2], vec![0, 3, 4]],
            corner_uvs: None,
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        let mut notes = Vec::new();
        mesh.weld(1e-5, &mut notes);
        assert_eq!(mesh.faces.len(), 1, "the collapsed face survived");
        assert!(
            notes
                .iter()
                .any(|n| matches!(n, LossyNote::DroppedDegenerateFace { .. })),
            "a face vanished without a note"
        );
    }

    #[test]
    fn the_first_and_last_corner_of_a_face_are_adjacent() {
        // A face is a cycle. If only consecutive pairs were compared, a triangle
        // whose first and last corners merged would come out as a two-corner
        // face rather than being dropped.
        let mut mesh = ImportMesh {
            name: "wrapped".to_string(),
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
            ],
            faces: vec![vec![0, 1, 2]],
            corner_uvs: None,
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        let mut notes = Vec::new();
        mesh.weld(1e-5, &mut notes);
        assert!(
            mesh.faces.is_empty(),
            "a two-corner face survived: {:?}",
            mesh.faces
        );
    }

    #[test]
    fn welding_is_deterministic_regardless_of_how_the_points_are_spread() {
        // Welding with a tolerance is not transitive — `a` may be within epsilon
        // of `b`, and `b` of `c`, without `a` being within epsilon of `c` — so a
        // greedy merge depends on the order vertices are visited. Visiting in
        // index order and joining the lowest-indexed representative is what makes
        // it a definite rule.
        //
        // Run repeatedly because the grid is a hash map: an implementation that
        // depended on its iteration order would be stable within one run and
        // differ between them.
        let make = || ImportMesh {
            name: "chain".to_string(),
            positions: (0..64)
                .map(|i| Vec3::new(i as f32 * 0.4, (i % 7) as f32 * 0.4, 0.0))
                .collect(),
            faces: (0..20).map(|i| vec![i, i + 1, i + 2]).collect(),
            corner_uvs: None,
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        let reference = {
            let mut m = make();
            m.weld(0.5, &mut Vec::new());
            (m.positions.clone(), m.faces.clone())
        };
        for _ in 0..8 {
            let mut m = make();
            m.weld(0.5, &mut Vec::new());
            assert_eq!(
                (m.positions, m.faces),
                reference,
                "the weld gave a different mesh on a repeat run"
            );
        }
    }

    #[test]
    fn points_within_the_tolerance_but_in_different_cells_still_merge() {
        // The reason the search covers twenty-seven cells and not one.
        //
        // The grid's cell is the tolerance, so two points closer than the
        // tolerance can still straddle a cell boundary — and a search that only
        // looked in the point's own cell would leave them unmerged. The mesh
        // would come out looking welded almost everywhere, with a scattering of
        // seams wherever a shared vertex happened to fall near a multiple of the
        // tolerance. That is far worse than not welding at all: it is a bug that
        // depends on the model's absolute coordinates.
        //
        // A falsification pass that narrowed the search to one cell passed every
        // other test in this file, because they all use points that are either
        // bit-identical or far apart.
        let epsilon = 0.1f32;
        let mut mesh = ImportMesh {
            name: "straddle".to_string(),
            positions: vec![
                // Either side of the boundary at x = 10 * epsilon, and a
                // hundredth of the tolerance apart.
                Vec3::new(epsilon * 10.0 - 0.0005, 0.0, 0.0),
                Vec3::new(epsilon * 10.0 + 0.0005, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
            faces: vec![vec![0, 2, 3], vec![1, 3, 2]],
            corner_uvs: None,
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        let mut notes = Vec::new();
        mesh.weld(epsilon, &mut notes);
        assert_eq!(
            mesh.positions.len(),
            3,
            "two points a hundredth of the tolerance apart were not merged, so \
             the search is not covering neighbouring cells"
        );
    }

    #[test]
    fn a_point_near_two_representatives_joins_the_lower_indexed_one() {
        // Welding with a tolerance is not transitive, so a point can be close
        // enough to two separate representatives at once. Which one it joins has
        // to be decided by a rule, not by whichever cell the search happened to
        // look in first — otherwise the geometry depends on the loop nesting in
        // the search, and changing that nesting for speed would silently change
        // people's meshes.
        //
        // The rule is the lowest index, and this is the case that can see it: a
        // falsification pass that took the last match instead passed everything
        // else here, including the determinism test — because taking the last
        // match is perfectly deterministic too.
        let mut mesh = ImportMesh {
            name: "between".to_string(),
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                // Too far from the first to merge, so it becomes a second
                // representative.
                Vec3::new(1.0, 0.0, 0.0),
                // Within tolerance of both.
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::new(0.0, 5.0, 0.0),
            ],
            faces: vec![vec![2, 3, 0]],
            corner_uvs: None,
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        let mut notes = Vec::new();
        mesh.weld(0.7, &mut notes);

        // Vertex 2 joined vertex 0, which is slot 0 — not vertex 1, which is
        // slot 1. The face names slot 0 twice, so the wrap-around collapse
        // reduces it to two corners and it is dropped.
        assert_eq!(
            mesh.positions.len(),
            3,
            "the merge did not happen as expected: {:?}",
            mesh.positions
        );
        assert!(
            mesh.faces.is_empty(),
            "vertex 2 joined the wrong representative: the face is {:?}",
            mesh.faces
        );
    }

    #[test]
    fn a_zero_tolerance_does_nothing() {
        // Guards the division by the tolerance in the grid, and states the
        // policy: no tolerance means no welding, not welding by exact equality.
        let mut mesh = soup();
        let mut notes = Vec::new();
        mesh.weld(0.0, &mut notes);
        assert_eq!(mesh.positions.len(), 6);
        assert!(notes.is_empty());
    }

    #[test]
    fn an_empty_mesh_welds_without_panicking() {
        let mut mesh = ImportMesh {
            name: "empty".to_string(),
            positions: Vec::new(),
            faces: Vec::new(),
            corner_uvs: None,
            corner_normals: None,
            face_materials: None,
            smoothing_groups: None,
        };
        mesh.weld(1e-5, &mut Vec::new());
        assert!(mesh.positions.is_empty());
    }
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
        let uvs = out[0].1.attributes().get_f32x2(names::UV).unwrap();
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
        let layer = out[0].1.attributes().get(names::UV).unwrap();
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
        let uvs = out[0].1.attributes().get_f32x2(names::UV).unwrap();
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

    #[test]
    fn a_transform_comes_back_from_kreate_space_unchanged() {
        // The whole claim `from_kreate` makes. Written as the identity rather
        // than against an expected matrix on purpose: an expected matrix is a
        // second statement of the conversion, and if the two ever disagree the
        // test says which is "right" by fiat rather than by arithmetic.
        //
        // The cases are chosen to break the two ways this goes wrong. A rotation
        // about an axis that is *not* the up axis catches a conjugation applied
        // on one side only; a non-uniform scale catches one applied to the wrong
        // factor, because a uniform scale commutes with everything and hides it.
        let mut source = SourceMetadata::new("test");
        source.axis = AxisConvention::GLTF;
        source.unit_scale = 0.01;

        for original in [
            ImportTransform::default(),
            ImportTransform {
                translation: Vec3::new(1.0, -2.0, 3.5),
                rotation: Quat::from_rotation_x(0.7),
                scale: Vec3::new(2.0, 0.5, 1.25),
            },
            ImportTransform {
                translation: Vec3::new(-40.0, 0.25, 0.0),
                rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 2.2),
                scale: Vec3::splat(3.0),
            },
        ] {
            let there = original.to_kreate(&source);
            let back = ImportTransform::from_kreate(there, &source);
            assert!(
                (back.translation - original.translation).length() < 1e-4,
                "{:?} came back at {:?}",
                original.translation,
                back.translation
            );
            assert!(
                (back.scale - original.scale).length() < 1e-5,
                "{:?} came back as {:?}",
                original.scale,
                back.scale
            );
            // Compared by what the rotation *does*: `q` and `-q` are the same
            // rotation, so comparing the four numbers would fail on a correct
            // answer half the time.
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                let turned = back.rotation * axis;
                let wanted = original.rotation * axis;
                assert!(
                    (turned - wanted).length() < 1e-5,
                    "{axis:?} was sent to {turned:?} instead of {wanted:?}"
                );
            }
        }
    }

    #[test]
    fn a_kreate_file_is_its_own_space_in_both_directions() {
        // The convention that is the identity, which is the one a conversion
        // written backwards still passes — so it is asserted separately and the
        // case above carries the axis change.
        let source = SourceMetadata::new("test");
        let original = ImportTransform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_z(0.4),
            scale: Vec3::new(1.0, 2.0, 1.0),
        };
        let there = original.to_kreate(&source);
        assert!((there.translation.as_vec3() - original.translation).length() < 1e-6);
        let back = ImportTransform::from_kreate(there, &source);
        assert!((back.translation - original.translation).length() < 1e-6);
    }

    #[test]
    fn a_mesh_with_no_slot_table_reads_its_faces_number_as_the_material() {
        // What a bare mesh list means, and the behaviour every existing caller
        // depends on. The `-1` is "no material", which faces carry when nothing
        // has been assigned.
        let mesh = MeshData::empty();
        let scene = ExportScene::new(vec![("Thing".into(), &mesh)]);
        assert_eq!(scene.material_of(0, 0), Some(0));
        assert_eq!(scene.material_of(0, 3), Some(3));
        assert_eq!(scene.material_of(0, -1), None);
    }

    #[test]
    fn two_meshes_do_not_share_a_slot_number() {
        // The reason the table exists. Slot 0 of one mesh and slot 0 of another
        // are different materials, and an exporter keyed on the face's own
        // number would give the second mesh the first one's.
        let mesh = MeshData::empty();
        let mut scene = ExportScene::new(vec![("A".into(), &mesh), ("B".into(), &mesh)]);
        scene.mesh_slots = vec![vec![Some(2)], vec![Some(5), None]];
        assert_eq!(scene.material_of(0, 0), Some(2));
        assert_eq!(scene.material_of(1, 0), Some(5));
        // An empty slot is ordinary: a mesh can have three of them with the
        // middle one unassigned.
        assert_eq!(scene.material_of(1, 1), None);
        // And a face pointing past the table is a face pointing at nothing,
        // rather than at whatever material happens to be there.
        assert_eq!(scene.material_of(1, 9), None);
    }

    #[test]
    fn a_file_with_no_size_is_read_as_one_metre_per_unit() {
        // A unit scale of zero collapses every position onto the origin, and no
        // inverse of that exists. The guard turns it into one rather than
        // writing infinities into a file that would then open as nothing.
        //
        // Nothing in the codebase produces a zero — `SourceMetadata` starts at
        // one — so this is a claim about a value a *file* can carry, and the
        // only way to check it is to say it directly.
        let mut source = SourceMetadata::new("test");
        source.unit_scale = 0.0;
        let back = ImportTransform::from_kreate(
            kreate_math::DAffine3::from_translation(kreate_math::DVec3::new(1.0, 2.0, 3.0)),
            &source,
        );
        assert!(
            back.translation.is_finite(),
            "a zero unit scale produced {:?}",
            back.translation
        );
        assert!((back.translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }
}
