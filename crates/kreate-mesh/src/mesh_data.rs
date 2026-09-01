//! The document / storage / IO / render form of a mesh.
//!
//! `MeshData` is a compact, CSR n-gon, structure-of-arrays representation. It is
//! **not** an editable structure — there is no adjacency here, so answering "what
//! faces touch this vertex" is a linear scan. Interactive editing builds a
//! `BMesh` (loop + radial + disk cycles) on entering edit mode and discards it on
//! exit.
//!
//! That split is not legacy cruft, it is arithmetic. A `BMesh` costs roughly
//! 200 bytes per vertex; `MeshData` costs roughly 50. At the scale this
//! application targets, storing the edit structure as the document form would
//! mean a 2 GB process for a 10 M vertex mesh.
//!
//! # Copy-on-write
//!
//! Every array is behind its own [`Arc`]. Cloning a `MeshData` is a handful of
//! refcount bumps, and mutating one array through `Arc::make_mut` copies only
//! that array. Three things fall out of this and nothing else in the design has
//! to work for them:
//!
//! - undo snapshots cost only what an operator actually touched,
//! - `Arc::ptr_eq` is a free and exact cache-invalidation test,
//! - the renderer can hold an immutable snapshot with no locking.

use std::sync::Arc;

use ahash::AHashMap;
use glam::Vec3;

use crate::attribute::{names, AttrData, AttrType, AttributeStore, Domain};

/// A structural problem found by [`MeshData::validate`].
#[derive(Clone, Debug, PartialEq)]
pub enum Violation {
    /// `face_offsets` must be non-empty, start at 0, and be non-decreasing.
    MalformedFaceOffsets { detail: String },
    /// The last face offset must equal the corner count.
    CornerCountMismatch { offsets_end: u32, corners: usize },
    /// A face has fewer than 3 corners.
    DegenerateFace { face: u32, corner_count: u32 },
    /// A corner references a vertex that does not exist.
    CornerVertOutOfRange {
        corner: u32,
        vert: u32,
        num_verts: u32,
    },
    /// An edge references a vertex that does not exist.
    EdgeVertOutOfRange {
        edge: u32,
        vert: u32,
        num_verts: u32,
    },
    /// An edge joins a vertex to itself.
    DegenerateEdge { edge: u32, vert: u32 },
    /// Two edges join the same pair of vertices.
    DuplicateEdge { edge: u32, other: u32 },
    /// A face repeats a vertex.
    RepeatedVertInFace { face: u32, vert: u32 },
    /// An attribute layer's length does not match its domain's element count.
    AttrLengthMismatch {
        layer: String,
        domain: Domain,
        len: usize,
        expected: usize,
    },
    /// A required layer is missing.
    MissingLayer { layer: String },
    /// A position is not finite.
    NonFinitePosition { vert: u32, value: Vec3 },
}

/// A polygon mesh in storage form.
#[derive(Clone, Debug)]
pub struct MeshData {
    num_verts: u32,
    /// CSR offsets, `len == num_faces + 1`. Face `f` owns corners
    /// `face_offsets[f] .. face_offsets[f + 1]`.
    face_offsets: Arc<Vec<u32>>,
    /// Vertex index per corner.
    corner_verts: Arc<Vec<u32>>,
    /// Vertex pairs. Derived from the faces, but stored because edges carry
    /// their own attribute domain (crease, seam, sharp, selection).
    edges: Arc<Vec<[u32; 2]>>,
    attributes: AttributeStore,
}

impl MeshData {
    /// An empty mesh that still has the mandatory `position` layer.
    pub fn empty() -> Self {
        let mut attributes = AttributeStore::new();
        attributes.insert(
            names::POSITION,
            Domain::Point,
            AttrData::zeroed(AttrType::F32x3, 0),
        );
        Self {
            num_verts: 0,
            face_offsets: Arc::new(vec![0]),
            corner_verts: Arc::new(Vec::new()),
            edges: Arc::new(Vec::new()),
            attributes,
        }
    }

    /// Builds a mesh from positions and per-face vertex lists, deriving edges.
    ///
    /// Faces may be n-gons; they are stored as authored and never triangulated.
    /// Triangulation happens only when handing geometry to the GPU or to a
    /// triangle-only file format.
    pub fn from_faces(positions: Vec<Vec3>, faces: &[Vec<u32>]) -> Self {
        let mut face_offsets = Vec::with_capacity(faces.len() + 1);
        let mut corner_verts = Vec::new();
        face_offsets.push(0u32);
        for f in faces {
            corner_verts.extend_from_slice(f);
            face_offsets.push(corner_verts.len() as u32);
        }

        let mut attributes = AttributeStore::new();
        let num_verts = positions.len() as u32;
        attributes.insert(
            names::POSITION,
            Domain::Point,
            AttrData::F32x3(Arc::new(positions)),
        );

        let mut mesh = Self {
            num_verts,
            face_offsets: Arc::new(face_offsets),
            corner_verts: Arc::new(corner_verts),
            edges: Arc::new(Vec::new()),
            attributes,
        };
        mesh.rebuild_edges();
        mesh
    }

    // --- counts -------------------------------------------------------------

    #[inline]
    pub fn num_verts(&self) -> u32 {
        self.num_verts
    }

    #[inline]
    pub fn num_edges(&self) -> u32 {
        self.edges.len() as u32
    }

    #[inline]
    pub fn num_faces(&self) -> u32 {
        (self.face_offsets.len() - 1) as u32
    }

    #[inline]
    pub fn num_corners(&self) -> u32 {
        self.corner_verts.len() as u32
    }

    /// Element count in a domain, used to size and validate attribute layers.
    pub fn domain_len(&self, domain: Domain) -> usize {
        match domain {
            Domain::Point => self.num_verts as usize,
            Domain::Edge => self.edges.len(),
            Domain::Corner => self.corner_verts.len(),
            Domain::Face => self.face_offsets.len() - 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_verts == 0 && self.num_faces() == 0
    }

    // --- topology -----------------------------------------------------------

    #[inline]
    pub fn face_offsets(&self) -> &[u32] {
        &self.face_offsets
    }

    #[inline]
    pub fn corner_verts(&self) -> &[u32] {
        &self.corner_verts
    }

    #[inline]
    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    /// The corner index range owned by `face`.
    #[inline]
    pub fn face_corner_range(&self, face: u32) -> std::ops::Range<usize> {
        let f = face as usize;
        self.face_offsets[f] as usize..self.face_offsets[f + 1] as usize
    }

    /// The vertices of `face`, in winding order.
    #[inline]
    pub fn face_verts(&self, face: u32) -> &[u32] {
        &self.corner_verts[self.face_corner_range(face)]
    }

    #[inline]
    pub fn face_len(&self, face: u32) -> u32 {
        let f = face as usize;
        self.face_offsets[f + 1] - self.face_offsets[f]
    }

    pub fn faces(&self) -> impl Iterator<Item = &[u32]> + '_ {
        (0..self.num_faces()).map(|f| self.face_verts(f))
    }

    // --- attributes ---------------------------------------------------------

    #[inline]
    pub fn attributes(&self) -> &AttributeStore {
        &self.attributes
    }

    /// Direct mutable access to the attribute store.
    ///
    /// Adding or removing layers is fine here. Resizing a layer out of step with
    /// its domain is not, and [`MeshData::validate`] will catch it.
    #[inline]
    pub fn attributes_mut(&mut self) -> &mut AttributeStore {
        &mut self.attributes
    }

    /// Vertex positions. Always present.
    #[inline]
    pub fn positions(&self) -> &[Vec3] {
        self.attributes
            .get_f32x3(names::POSITION)
            .expect("every MeshData has a position layer")
    }

    /// Copy-on-write access to positions.
    ///
    /// This is the hot path for interactive edits, and it copies only the
    /// position array — topology and every other layer stay shared with any
    /// outstanding snapshot.
    #[inline]
    pub fn positions_mut(&mut self) -> &mut Vec<Vec3> {
        self.attributes
            .make_mut_f32x3(names::POSITION)
            .expect("every MeshData has a position layer")
    }

    /// Adds the three selection layers if absent.
    pub fn ensure_selection_layers(&mut self) {
        for (name, domain) in [
            (names::SELECT_VERT, Domain::Point),
            (names::SELECT_EDGE, Domain::Edge),
            (names::SELECT_FACE, Domain::Face),
        ] {
            let len = self.domain_len(domain);
            self.attributes.ensure(name, domain, AttrType::Bool, len);
        }
    }

    // --- derived ------------------------------------------------------------

    /// Rebuilds the edge array from the faces and resizes edge-domain layers.
    ///
    /// Edge order is first-encounter order over faces and corners, so it is fully
    /// deterministic and independent of hash iteration order. Golden tests and
    /// reproducible file writes both depend on that.
    ///
    /// Edge *attributes* do not survive this, because edge identity may have
    /// changed. Callers that reorder topology are responsible for remapping.
    pub fn rebuild_edges(&mut self) {
        let mut map: AHashMap<(u32, u32), u32> = AHashMap::new();
        let mut edges: Vec<[u32; 2]> = Vec::new();
        for f in 0..self.num_faces() {
            let verts = self.face_verts(f);
            for i in 0..verts.len() {
                let a = verts[i];
                let b = verts[(i + 1) % verts.len()];
                if a == b {
                    continue; // degenerate; reported by validate()
                }
                let key = if a < b { (a, b) } else { (b, a) };
                map.entry(key).or_insert_with(|| {
                    edges.push([key.0, key.1]);
                    (edges.len() - 1) as u32
                });
            }
        }
        self.edges = Arc::new(edges);
        let len = self.edges.len();
        self.attributes.resize_domain(Domain::Edge, len);
    }

    /// Area-weighted vertex normals, ignoring sharp edges and custom normals.
    ///
    /// Area weighting comes for free from *not* normalising the cross product,
    /// and it is what makes normals stable when a mesh has one huge face meeting
    /// several slivers.
    pub fn compute_vertex_normals(&self) -> Vec<Vec3> {
        let pos = self.positions();
        let mut normals = vec![Vec3::ZERO; self.num_verts as usize];
        for f in 0..self.num_faces() {
            let verts = self.face_verts(f);
            if verts.len() < 3 {
                continue;
            }
            // Newell's method: correct for non-planar n-gons, where any single
            // cross product would depend on which corner you happened to pick.
            let mut n = Vec3::ZERO;
            for i in 0..verts.len() {
                let a = pos[verts[i] as usize];
                let b = pos[verts[(i + 1) % verts.len()] as usize];
                n += (a - b).cross(a + b);
            }
            // n is twice the area-weighted normal; the factor is uniform across
            // faces, so it cancels on normalisation and is not worth removing.
            for &v in verts {
                normals[v as usize] += n;
            }
        }
        for n in &mut normals {
            *n = n.normalize_or_zero();
        }
        normals
    }

    pub fn bounds(&self) -> kreate_math::Aabb {
        kreate_math::Aabb::from_points(self.positions().iter().copied())
    }

    // --- validation ---------------------------------------------------------

    /// Checks every structural invariant.
    ///
    /// Wired into the operator dispatcher under `debug_assertions`, this catches
    /// the large majority of mesh-kernel bugs at the moment they are introduced
    /// rather than three operators later, when the cause is unattributable.
    pub fn validate(&self) -> Vec<Violation> {
        let mut v = Vec::new();

        if self.face_offsets.is_empty() {
            v.push(Violation::MalformedFaceOffsets {
                detail: "empty".into(),
            });
            return v; // nothing below is meaningful without offsets
        }
        if self.face_offsets[0] != 0 {
            v.push(Violation::MalformedFaceOffsets {
                detail: format!("starts at {}", self.face_offsets[0]),
            });
        }
        if let Some(i) =
            (1..self.face_offsets.len()).find(|&i| self.face_offsets[i] < self.face_offsets[i - 1])
        {
            v.push(Violation::MalformedFaceOffsets {
                detail: format!("decreases at index {i}"),
            });
            return v;
        }
        let end = *self.face_offsets.last().unwrap();
        if end as usize != self.corner_verts.len() {
            v.push(Violation::CornerCountMismatch {
                offsets_end: end,
                corners: self.corner_verts.len(),
            });
        }

        for f in 0..self.num_faces() {
            let n = self.face_len(f);
            if n < 3 {
                v.push(Violation::DegenerateFace {
                    face: f,
                    corner_count: n,
                });
                continue;
            }
            let verts = self.face_verts(f);
            for (i, &a) in verts.iter().enumerate() {
                if verts[i + 1..].contains(&a) {
                    v.push(Violation::RepeatedVertInFace { face: f, vert: a });
                    break;
                }
            }
        }

        for (c, &vert) in self.corner_verts.iter().enumerate() {
            if vert >= self.num_verts {
                v.push(Violation::CornerVertOutOfRange {
                    corner: c as u32,
                    vert,
                    num_verts: self.num_verts,
                });
            }
        }

        let mut seen: AHashMap<(u32, u32), u32> = AHashMap::new();
        for (e, &[a, b]) in self.edges.iter().enumerate() {
            let e = e as u32;
            for vert in [a, b] {
                if vert >= self.num_verts {
                    v.push(Violation::EdgeVertOutOfRange {
                        edge: e,
                        vert,
                        num_verts: self.num_verts,
                    });
                }
            }
            if a == b {
                v.push(Violation::DegenerateEdge { edge: e, vert: a });
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&other) = seen.get(&key) {
                v.push(Violation::DuplicateEdge { edge: e, other });
            } else {
                seen.insert(key, e);
            }
        }

        if !self.attributes.contains(names::POSITION) {
            v.push(Violation::MissingLayer {
                layer: names::POSITION.into(),
            });
        }
        for layer in self.attributes.iter() {
            let expected = self.domain_len(layer.domain);
            if layer.data.len() != expected {
                v.push(Violation::AttrLengthMismatch {
                    layer: layer.name.to_string(),
                    domain: layer.domain,
                    len: layer.data.len(),
                    expected,
                });
            }
        }

        if let Some(pos) = self.attributes.get_f32x3(names::POSITION) {
            for (i, p) in pos.iter().enumerate() {
                if !p.is_finite() {
                    v.push(Violation::NonFinitePosition {
                        vert: i as u32,
                        value: *p,
                    });
                }
            }
        }

        v
    }

    /// Panics if the mesh is invalid. Intended for `debug_assert`-style use in
    /// tests and in the operator dispatcher.
    #[track_caller]
    pub fn assert_valid(&self) {
        let v = self.validate();
        assert!(v.is_empty(), "mesh failed validation: {v:#?}");
    }
}

impl Default for MeshData {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit quad plus a triangle sharing one edge.
    fn quad_and_tri() -> MeshData {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.5, 0.0),
        ];
        MeshData::from_faces(positions, &[vec![0, 1, 2, 3], vec![1, 4, 2]])
    }

    #[test]
    fn empty_mesh_is_valid() {
        MeshData::empty().assert_valid();
    }

    #[test]
    fn counts_and_csr_ranges() {
        let m = quad_and_tri();
        assert_eq!(m.num_verts(), 5);
        assert_eq!(m.num_faces(), 2);
        assert_eq!(m.num_corners(), 7);
        assert_eq!(m.face_verts(0), &[0, 1, 2, 3]);
        assert_eq!(m.face_verts(1), &[1, 4, 2]);
        assert_eq!(m.face_len(0), 4);
        m.assert_valid();
    }

    #[test]
    fn shared_edge_is_not_duplicated() {
        let m = quad_and_tri();
        // quad: 01 12 23 30, tri: 14 42 and the shared 12 -> 6 unique edges.
        assert_eq!(m.num_edges(), 6);
        m.assert_valid();
    }

    #[test]
    fn edge_order_is_deterministic() {
        let a = quad_and_tri();
        let b = quad_and_tri();
        assert_eq!(a.edges(), b.edges());
        assert_eq!(
            a.edges()[0],
            [0, 1],
            "first-encounter order, not hash order"
        );
    }

    #[test]
    fn ngons_are_preserved_not_triangulated() {
        let m = MeshData::from_faces(
            (0..6).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect(),
            &[vec![0, 1, 2, 3, 4, 5]],
        );
        assert_eq!(m.num_faces(), 1);
        assert_eq!(m.face_len(0), 6);
        m.assert_valid();
    }

    #[test]
    fn position_edit_shares_topology_with_snapshot() {
        let mut m = quad_and_tri();
        let snapshot = m.clone();
        m.positions_mut()[0] = Vec3::splat(9.0);

        assert!(
            Arc::ptr_eq(&m.corner_verts, &snapshot.corner_verts),
            "topology must not be copied by a position edit"
        );
        assert!(Arc::ptr_eq(&m.face_offsets, &snapshot.face_offsets));
        assert!(Arc::ptr_eq(&m.edges, &snapshot.edges));
        assert_eq!(snapshot.positions()[0], Vec3::ZERO);
        assert_eq!(m.positions()[0], Vec3::splat(9.0));
    }

    #[test]
    fn selection_layers_are_sized_to_their_domains() {
        let mut m = quad_and_tri();
        m.ensure_selection_layers();
        assert_eq!(
            m.attributes().get_bool(names::SELECT_VERT).unwrap().len(),
            5
        );
        assert_eq!(
            m.attributes().get_bool(names::SELECT_EDGE).unwrap().len(),
            6
        );
        assert_eq!(
            m.attributes().get_bool(names::SELECT_FACE).unwrap().len(),
            2
        );
        m.assert_valid();
    }

    #[test]
    fn quad_normals_point_up() {
        let m = MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3]],
        );
        for n in m.compute_vertex_normals() {
            assert!((n - Vec3::Z).length() < 1e-5, "expected +Z, got {n}");
        }
    }

    #[test]
    fn normals_of_a_nonplanar_ngon_are_finite() {
        // Newell's method must not depend on which corner is sampled.
        let m = MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.3),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, -0.3),
            ],
            &[vec![0, 1, 2, 3]],
        );
        for n in m.compute_vertex_normals() {
            assert!(n.is_finite() && n.length() > 0.9);
        }
    }

    // --- validator ----------------------------------------------------------

    #[test]
    fn detects_out_of_range_corner() {
        let mut m = quad_and_tri();
        Arc::make_mut(&mut m.corner_verts)[0] = 99;
        assert!(m
            .validate()
            .iter()
            .any(|v| matches!(v, Violation::CornerVertOutOfRange { vert: 99, .. })));
    }

    #[test]
    fn detects_degenerate_face() {
        let m = MeshData::from_faces(vec![Vec3::ZERO, Vec3::X], &[vec![0, 1]]);
        assert!(m.validate().iter().any(|v| matches!(
            v,
            Violation::DegenerateFace {
                corner_count: 2,
                ..
            }
        )));
    }

    #[test]
    fn detects_repeated_vert_in_face() {
        let m = MeshData::from_faces(vec![Vec3::ZERO, Vec3::X, Vec3::Y], &[vec![0, 1, 0, 2]]);
        assert!(m
            .validate()
            .iter()
            .any(|v| matches!(v, Violation::RepeatedVertInFace { vert: 0, .. })));
    }

    #[test]
    fn detects_attribute_length_mismatch() {
        let mut m = quad_and_tri();
        m.attributes_mut().insert(
            names::CREASE,
            Domain::Edge,
            AttrData::zeroed(AttrType::F32, 2),
        );
        assert!(m.validate().iter().any(|v| matches!(
            v,
            Violation::AttrLengthMismatch {
                domain: Domain::Edge,
                len: 2,
                expected: 6,
                ..
            }
        )));
    }

    #[test]
    fn detects_non_finite_position() {
        let mut m = quad_and_tri();
        m.positions_mut()[2] = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(m
            .validate()
            .iter()
            .any(|v| matches!(v, Violation::NonFinitePosition { vert: 2, .. })));
    }

    #[test]
    fn detects_duplicate_edge() {
        let mut m = quad_and_tri();
        let dup = m.edges[0];
        Arc::make_mut(&mut m.edges).push(dup);
        let n = m.edges.len();
        m.attributes_mut().resize_domain(Domain::Edge, n);
        assert!(m
            .validate()
            .iter()
            .any(|v| matches!(v, Violation::DuplicateEdge { .. })));
    }
}
