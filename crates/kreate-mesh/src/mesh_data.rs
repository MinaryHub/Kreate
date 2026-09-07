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

/// What can be wrong with a set of CSR topology arrays.
///
/// Named cases rather than one opaque error, because these arrive from files and
/// a user whose file will not open deserves to know which part of it is wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TopologyError {
    #[error("face offsets must start at zero")]
    FaceOffsetsMustStartAtZero,
    #[error("face offsets decrease at face {face}")]
    FaceOffsetsNotAscending { face: usize },
    #[error("face offsets describe {declared} corners but {actual} were supplied")]
    CornerCountMismatch { declared: usize, actual: usize },
    #[error("corner {corner} refers to vertex {vertex}, but there are only {num_verts}")]
    CornerOutOfRange {
        corner: usize,
        vertex: u32,
        num_verts: u32,
    },
    #[error("edge {edge:?} refers past the {num_verts} vertices")]
    EdgeOutOfRange { edge: [u32; 2], num_verts: u32 },
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

    /// Builds a mesh straight from its CSR arrays, validating them first.
    ///
    /// What a deserialiser needs. The alternative, going through
    /// [`Self::from_faces`], means materialising a `Vec` per face — which for a
    /// million-face mesh is a million allocations to rebuild arrays the file
    /// already holds in exactly the right form.
    ///
    /// Validation lives here rather than in each caller because these are the
    /// invariants every later operation assumes without checking: a corner index
    /// past the vertex count is not a wrong picture, it is an out-of-bounds read
    /// somewhere three layers away, and the file that caused it will be long out
    /// of scope by then.
    ///
    /// `edges` is trusted as a *hint* and rebuilt if it does not describe the
    /// faces, so a file cannot introduce an edge list inconsistent with its own
    /// topology.
    pub fn from_csr(
        num_verts: u32,
        face_offsets: Vec<u32>,
        corner_verts: Vec<u32>,
        edges: Vec<[u32; 2]>,
    ) -> Result<Self, TopologyError> {
        if face_offsets.first() != Some(&0) {
            return Err(TopologyError::FaceOffsetsMustStartAtZero);
        }
        for (i, w) in face_offsets.windows(2).enumerate() {
            if w[1] < w[0] {
                return Err(TopologyError::FaceOffsetsNotAscending { face: i });
            }
        }
        let last = face_offsets.last().copied().unwrap_or(0) as usize;
        if last != corner_verts.len() {
            return Err(TopologyError::CornerCountMismatch {
                declared: last,
                actual: corner_verts.len(),
            });
        }
        for (corner, vertex) in corner_verts.iter().enumerate() {
            if *vertex >= num_verts {
                return Err(TopologyError::CornerOutOfRange {
                    corner,
                    vertex: *vertex,
                    num_verts,
                });
            }
        }
        for [a, b] in &edges {
            if *a >= num_verts || *b >= num_verts {
                return Err(TopologyError::EdgeOutOfRange {
                    edge: [*a, *b],
                    num_verts,
                });
            }
        }

        let mut attributes = AttributeStore::new();
        attributes.insert(
            names::POSITION,
            Domain::Point,
            AttrData::zeroed(AttrType::F32x3, num_verts as usize),
        );
        let mut mesh = Self {
            num_verts,
            face_offsets: Arc::new(face_offsets),
            corner_verts: Arc::new(corner_verts),
            edges: Arc::new(edges),
            attributes,
        };
        // Rebuilt when the supplied list does not match what the faces imply.
        // Trusting it would let a file declare edge attributes against edges
        // that do not exist, and every edge-domain layer would then be the wrong
        // length or, worse, the right length against the wrong edges.
        mesh.rebuild_edges();
        Ok(mesh)
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

    /// The edge each corner walks along, to the next corner of its face.
    ///
    /// `u32::MAX` for a corner whose edge is missing, which a valid mesh does not
    /// have — the validator forbids it — but a half-built one can.
    ///
    /// The bridge from CSR faces to the edge list, and what anything walking
    /// "the edges of this face" needs. Built with a sorted pair table and binary
    /// search rather than a hash map: several callers derive output geometry from
    /// it, and a hasher with a per-run seed would make the answer depend on which
    /// run it was.
    pub fn corner_edges(&self) -> Vec<u32> {
        let mut pairs: Vec<([u32; 2], u32)> = self
            .edges
            .iter()
            .enumerate()
            .map(|(i, [a, b])| {
                let mut key = [*a, *b];
                key.sort_unstable();
                (key, i as u32)
            })
            .collect();
        pairs.sort_unstable();

        let mut out = vec![u32::MAX; self.corner_verts.len()];
        for face in 0..self.num_faces() {
            let ring = self.face_verts(face);
            let start = self.face_corner_range(face).start;
            for (i, v) in ring.iter().enumerate() {
                let next = ring[(i + 1) % ring.len()];
                let mut key = [*v, next];
                key.sort_unstable();
                if let Ok(found) = pairs.binary_search_by_key(&key, |(k, _)| *k) {
                    out[start + i] = pairs[found].1;
                }
            }
        }
        out
    }

    /// Whether `other` has the same *geometry* arrays as this mesh.
    ///
    /// Topology and positions, by pointer identity. Attribute layers are not
    /// compared: a selection click rewrites one of those and leaves everything
    /// else shared, and the thing asking this question — the edit session — holds
    /// a decoded copy that a selection change does not invalidate.
    ///
    /// The distinction is load-bearing. Comparing whole meshes instead makes
    /// every click invalidate the session, so the click *after* a click pays a
    /// full decode: at a million vertices that is two seconds, which is worse
    /// than having no session at all.
    pub fn same_geometry(&self, other: &MeshData) -> bool {
        self.num_verts == other.num_verts
            && Arc::ptr_eq(&self.face_offsets, &other.face_offsets)
            && Arc::ptr_eq(&self.corner_verts, &other.corner_verts)
            && Arc::ptr_eq(&self.edges, &other.edges)
            && match (
                self.attributes.get(names::POSITION),
                other.attributes.get(names::POSITION),
            ) {
                (Some(a), Some(b)) => a.data.ptr_eq(&b.data),
                (None, None) => true,
                _ => false,
            }
    }

    /// Whether `other` differs from this mesh in nothing but its selection.
    ///
    /// The question a renderer has to be able to answer cheaply. Selection is
    /// stored as ordinary attribute layers — which is what makes it survive the
    /// round trip through edit mode and share structurally under undo — but it
    /// also means a click produces a whole new `MeshData`, and a cache that can
    /// only ask "is this the same mesh" will re-upload the geometry of a million
    /// vertices because one face was clicked.
    ///
    /// Answered by pointer identity per array, so it costs a handful of
    /// comparisons regardless of how large the mesh is. Copy-on-write is what
    /// makes that valid: an array nobody wrote to keeps its pointer.
    ///
    /// A selection layer *appearing* counts, which is not a nicety: the layers
    /// are created when the user enters edit mode, so without it the first click
    /// on every mesh is a full upload — the one moment when the mesh is largest
    /// and the user is most attentive.
    ///
    /// Conservative in every other direction. Any non-selection layer that
    /// changed, appeared or vanished means a full upload, which is correct but
    /// slower, rather than risking a buffer that quietly stays stale.
    pub fn only_selection_differs(&self, other: &MeshData) -> bool {
        if self.num_verts != other.num_verts
            || !Arc::ptr_eq(&self.face_offsets, &other.face_offsets)
            || !Arc::ptr_eq(&self.corner_verts, &other.corner_verts)
            || !Arc::ptr_eq(&self.edges, &other.edges)
        {
            return false;
        }

        // Both directions, so a layer that vanished is caught as well as one
        // that changed. The stores are kept sorted by name, so this is a walk
        // over two short sorted lists either way round.
        let unmatched = |a: &AttributeStore, b: &AttributeStore| {
            a.iter().any(|layer| match b.get(&layer.name) {
                None => !names::is_selection(&layer.name),
                Some(theirs) => {
                    theirs.domain != layer.domain
                        || (!layer.data.ptr_eq(&theirs.data) && !names::is_selection(&layer.name))
                }
            })
        };
        !unmatched(&self.attributes, &other.attributes)
            && !unmatched(&other.attributes, &self.attributes)
    }

    /// Roughly how many heap bytes this mesh occupies.
    ///
    /// The topology arrays plus every attribute layer. Shared arrays are counted
    /// in full by each holder, so summing this over several meshes that share
    /// arrays overstates the total — which is the safe direction for a budget,
    /// and the honest one for asking "what would this cost if it were not
    /// shared".
    pub fn memory_bytes(&self) -> usize {
        self.face_offsets.len() * 4
            + self.corner_verts.len() * 4
            + self.edges.len() * 8
            + self.attributes.memory_bytes()
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

    /// Area-weighted face normals, one per face.
    ///
    /// Newell's method, so a non-planar n-gon gets a normal that does not depend
    /// on which corner happened to be sampled. Left unnormalised: the magnitude
    /// is twice the area, which is exactly the weight wanted when averaging.
    pub fn compute_face_normals(&self) -> Vec<Vec3> {
        let pos = self.positions();
        (0..self.num_faces())
            .map(|f| {
                let verts = self.face_verts(f);
                let mut n = Vec3::ZERO;
                for i in 0..verts.len() {
                    let a = pos[verts[i] as usize];
                    let b = pos[verts[(i + 1) % verts.len()] as usize];
                    n += (a - b).cross(a + b);
                }
                n
            })
            .collect()
    }

    /// Per-corner normals, splitting shading across sharp edges.
    ///
    /// Averaging every face that meets at a vertex — which is what a plain
    /// per-vertex normal does — shades a cube as a rounded lump: each of its
    /// eight vertices ends up with a diagonal normal, and interpolating those
    /// across a flat face makes it curve. On hard-surface geometry that is not a
    /// subtle inaccuracy, it is the wrong picture.
    ///
    /// So a corner averages only the neighbouring faces whose own normal is
    /// within `angle_threshold` radians of this face's. A cube's faces meet at
    /// 90 degrees and stay flat; a sphere's meet at a few degrees and stay smooth.
    /// A face marked `shade_smooth = false` never averages with anything.
    ///
    /// Returned per *corner*, not per vertex, because that is the domain the
    /// distinction lives in — the same reason UVs and split normals live there.
    /// Per-corner normals, honouring smoothing groups.
    ///
    /// A corner's normal averages the faces in its **smooth group**: the faces
    /// reachable from its own, walking the fan around that vertex, without
    /// crossing a sharp edge or a crease sharper than `angle_threshold`.
    ///
    /// Reachability rather than a pairwise comparison is the whole point. A
    /// pairwise test — "average every face at this vertex that is within the
    /// angle" — cannot see a sharp edge standing between two faces that do not
    /// touch each other, so smoothing leaks straight across it. The symptom is a
    /// shading seam that ignores the sharp edge the user just marked, which
    /// reads as the mark not working rather than as an averaging bug.
    pub fn compute_corner_normals(&self, angle_threshold: f32) -> Vec<Vec3> {
        let face_normals = self.compute_face_normals();
        let unit: Vec<Vec3> = face_normals.iter().map(|n| n.normalize_or_zero()).collect();
        let cos_threshold = angle_threshold.cos();
        let smooth = self.attributes.get_bool(names::SHADE_SMOOTH);

        // Sharp edges, as sorted vertex pairs for binary search. Sorted rather
        // than hashed because this decides the output geometry's shading and a
        // per-run hash seed would make it depend on which run it was.
        let mut sharp: Vec<[u32; 2]> = Vec::new();
        if let Some(flags) = self.attributes.get_bool(names::SHARP_EDGE) {
            for (edge, &is_sharp) in self.edges.iter().zip(flags) {
                if is_sharp {
                    let mut key = *edge;
                    key.sort_unstable();
                    sharp.push(key);
                }
            }
            sharp.sort_unstable();
        }
        let is_sharp = |a: u32, b: u32| -> bool {
            if sharp.is_empty() {
                return false;
            }
            let mut key = [a, b];
            key.sort_unstable();
            sharp.binary_search(&key).is_ok()
        };

        // Which faces touch each vertex, as CSR so the inner loop is a slice.
        let mut counts = vec![0u32; self.num_verts() as usize + 1];
        for &v in self.corner_verts.iter() {
            counts[v as usize + 1] += 1;
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let offsets = counts.clone();
        let mut cursor = counts;
        let mut adjacent = vec![0u32; self.corner_verts.len()];
        for f in 0..self.num_faces() {
            for &v in self.face_verts(f) {
                adjacent[cursor[v as usize] as usize] = f;
                cursor[v as usize] += 1;
            }
        }

        /// The two vertices a face's ring reaches from `v`, i.e. the far ends of
        /// its two edges at that corner.
        fn neighbours_at(ring: &[u32], v: u32) -> Option<[u32; 2]> {
            let n = ring.len();
            let i = ring.iter().position(|&x| x == v)?;
            Some([ring[(i + n - 1) % n], ring[(i + 1) % n]])
        }

        let mut out = Vec::with_capacity(self.corner_verts.len());
        // Reused across corners so the walk allocates once rather than per
        // corner; a vertex fan is small but there is one per corner.
        let mut group: Vec<u32> = Vec::new();
        let mut queue: Vec<u32> = Vec::new();

        for f in 0..self.num_faces() {
            let face_is_smooth = smooth.map(|s| s[f as usize]).unwrap_or(true);
            for &v in self.face_verts(f) {
                if !face_is_smooth {
                    out.push(unit[f as usize]);
                    continue;
                }
                let fan = &adjacent[offsets[v as usize] as usize..offsets[v as usize + 1] as usize];

                group.clear();
                queue.clear();
                group.push(f);
                queue.push(f);
                while let Some(current) = queue.pop() {
                    let Some(from) = neighbours_at(self.face_verts(current), v) else {
                        continue;
                    };
                    for &g in fan {
                        if group.contains(&g) {
                            continue;
                        }
                        if !smooth.map(|s| s[g as usize]).unwrap_or(true) {
                            continue;
                        }
                        // A crease as steep as the threshold breaks the group
                        // just as a sharp edge does, so an auto-smooth angle
                        // needs no marks on the mesh to take effect.
                        if unit[current as usize].dot(unit[g as usize]) < cos_threshold {
                            continue;
                        }
                        let Some(to) = neighbours_at(self.face_verts(g), v) else {
                            continue;
                        };
                        // Fan-adjacent: they share one of the edges at `v`.
                        let shared = to.iter().find(|w| from.contains(w));
                        if let Some(&w) = shared {
                            if !is_sharp(v, w) {
                                group.push(g);
                                queue.push(g);
                            }
                        }
                    }
                }

                let mut n = Vec3::ZERO;
                for &g in &group {
                    // Unnormalised, so the average is area-weighted: a strip of
                    // slivers beside one large quad should not outvote the quad.
                    n += face_normals[g as usize];
                }
                // Falling back to the face's own normal keeps a degenerate
                // neighbourhood from producing a zero normal and a black face.
                out.push(if n == Vec3::ZERO {
                    unit[f as usize]
                } else {
                    n.normalize_or_zero()
                });
            }
        }
        out
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
    // --- selection-only change detection ------------------------------------

    /// A copy of `mesh` with the three selection layers present and empty.
    ///
    /// Derived from the given mesh rather than built fresh, because the whole
    /// comparison is pointer identity: two independently generated cubes have
    /// equal contents in different allocations and are correctly *not* the same
    /// mesh.
    fn with_selection_layers(mesh: &MeshData) -> MeshData {
        let mut mesh = mesh.clone();
        let counts = [
            (names::SELECT_VERT, Domain::Point),
            (names::SELECT_EDGE, Domain::Edge),
            (names::SELECT_FACE, Domain::Face),
        ];
        for (name, domain) in counts {
            let len = mesh.domain_len(domain);
            mesh.attributes_mut().insert(
                name,
                domain,
                AttrData::Bool(std::sync::Arc::new(vec![false; len])),
            );
        }
        mesh
    }

    fn selectable_cube() -> MeshData {
        with_selection_layers(&crate::primitives::cube(2.0))
    }

    #[test]
    fn a_mesh_only_differs_from_itself_by_nothing() {
        let mesh = selectable_cube();
        let copy = mesh.clone();
        assert!(mesh.only_selection_differs(&copy));
    }

    #[test]
    fn a_selection_change_is_recognised_as_one() {
        // The whole point: a click on one face must not make the renderer
        // re-upload the geometry of a million vertices.
        let mesh = selectable_cube();
        let mut clicked = mesh.clone();
        clicked
            .attributes_mut()
            .make_mut_bool(names::SELECT_FACE)
            .expect("the layer is there")[2] = true;
        assert!(
            mesh.only_selection_differs(&clicked),
            "a selection change was mistaken for a geometry change"
        );
    }

    #[test]
    fn moving_a_vertex_is_not_a_selection_change() {
        let mesh = selectable_cube();
        let mut moved = mesh.clone();
        moved.positions_mut()[0].y += 1.0;
        assert!(!mesh.only_selection_differs(&moved));
    }

    #[test]
    fn a_uv_selection_change_is_recognised_as_one() {
        // The UV editor's equivalent of the click test above, and it matters
        // for the same reason: without it every click in the UV editor is a
        // full geometry re-upload of the mesh, which is exactly the coarse
        // invalidation the plan names as the top cause of DCC hitching.
        //
        // Nothing checked this until a falsification sweep removed
        // `SELECT_UV` from `is_selection` and no test noticed.
        let mesh = selectable_cube();
        let mut with_uv_selection = mesh.clone();
        let corners = with_uv_selection.num_corners() as usize;
        with_uv_selection.attributes_mut().ensure(
            names::SELECT_UV,
            Domain::Corner,
            AttrType::Bool,
            corners,
        );
        assert!(
            mesh.only_selection_differs(&with_uv_selection),
            "the UV selection layer appearing was mistaken for a geometry change"
        );

        let mut clicked = with_uv_selection.clone();
        clicked
            .attributes_mut()
            .make_mut_bool(names::SELECT_UV)
            .expect("the layer is there")[3] = true;
        assert!(
            with_uv_selection.only_selection_differs(&clicked),
            "a UV click was mistaken for a geometry change"
        );
    }

    #[test]
    fn changing_a_uv_is_not_a_selection_change() {
        // Any layer that is not a selection layer means a full upload. A
        // renderer applying a UV edit as a flags write would show the old
        // texture coordinates until something else forced a rebuild.
        let mut mesh = selectable_cube();
        let corners = mesh.num_corners() as usize;
        mesh.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(std::sync::Arc::new(vec![kreate_math::Vec2::ZERO; corners])),
        );
        let mut edited = mesh.clone();
        edited
            .attributes_mut()
            .make_mut_f32x2("uv")
            .expect("the layer is there")[0] = kreate_math::Vec2::ONE;
        assert!(!mesh.only_selection_differs(&edited));
    }

    #[test]
    fn a_topology_change_is_not_a_selection_change() {
        let mesh = selectable_cube();
        let other = crate::primitives::uv_sphere(1.0, 8, 4);
        assert!(!mesh.only_selection_differs(&other));
    }

    #[test]
    fn a_selection_layer_appearing_counts_as_a_selection_change() {
        // The layers are created when the user enters edit mode, so without this
        // the first click on every mesh is a full upload — at the one moment the
        // mesh is largest and the user is watching most closely.
        let bare = crate::primitives::cube(2.0);
        let selectable = with_selection_layers(&bare);
        assert!(bare.only_selection_differs(&selectable));
        assert!(
            selectable.only_selection_differs(&bare),
            "and the other way"
        );
    }

    #[test]
    fn a_non_selection_layer_appearing_forces_a_full_upload() {
        // Conservative in the safe direction: a mesh that gained a UV set needs a
        // real upload, not a flags write that leaves the coordinates stale.
        let mesh = selectable_cube();
        let mut textured = mesh.clone();
        let corners = textured.num_corners() as usize;
        textured.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(std::sync::Arc::new(vec![kreate_math::Vec2::ZERO; corners])),
        );
        assert!(!mesh.only_selection_differs(&textured));
        assert!(!textured.only_selection_differs(&mesh), "and the other way");
    }

    #[test]
    fn a_layer_that_changed_domain_forces_a_full_upload() {
        // Same name, different domain: a different array with a different length,
        // and writing it into a buffer sized for the old one is a corruption
        // rather than a stale value.
        let mesh = selectable_cube();
        let mut moved = mesh.clone();
        moved.attributes_mut().remove(names::SELECT_FACE);
        let verts = moved.num_verts() as usize;
        moved.attributes_mut().insert(
            names::SELECT_FACE,
            Domain::Point,
            AttrData::Bool(std::sync::Arc::new(vec![false; verts])),
        );
        assert!(!mesh.only_selection_differs(&moved));
    }

    #[test]
    fn the_comparison_does_not_scan_the_arrays() {
        // Two meshes with equal *contents* but separate allocations must still
        // come back as different, because the test is pointer identity — which is
        // what makes it cost the same on a million vertices as on eight.
        let mesh = selectable_cube();
        let rebuilt = MeshData::from_faces(
            mesh.positions().to_vec(),
            &mesh.faces().map(|f| f.to_vec()).collect::<Vec<_>>(),
        );
        assert!(!mesh.only_selection_differs(&rebuilt));
    }

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
    fn a_cube_stays_flat_shaded() {
        // The defect this exists to prevent: averaging every face at a vertex
        // gives a cube's corners diagonal normals, and interpolating those across
        // a flat face shades it as if it were rounded.
        let cube = crate::primitives::cube(2.0);
        let corners = cube.compute_corner_normals(30f32.to_radians());
        assert_eq!(corners.len(), cube.num_corners() as usize);

        for f in 0..cube.num_faces() {
            let range = cube.face_corner_range(f);
            let first = corners[range.start];
            for c in range {
                assert!(
                    (corners[c] - first).length() < 1e-5,
                    "face {f} has varying normals; it will shade as curved"
                );
            }
            // And it must be an axis direction, not a diagonal.
            let biggest = first.abs().max_element();
            assert!(
                biggest > 0.99,
                "face {f} normal {first} is not axis-aligned"
            );
        }
    }

    #[test]
    fn a_sphere_stays_smooth_shaded() {
        // The other half: a threshold that flattened everything would make a
        // sphere faceted, which is just as wrong in the other direction.
        let sphere = crate::primitives::uv_sphere(1.0, 48, 24);
        let corners = sphere.compute_corner_normals(30f32.to_radians());
        let pos = sphere.positions();

        // On a unit sphere the correct normal at a vertex is its position.
        let mut worst: f32 = 0.0;
        for f in 0..sphere.num_faces() {
            for (i, &v) in sphere.face_verts(f).iter().enumerate() {
                let n = corners[sphere.face_corner_range(f).start + i];
                let expected = pos[v as usize].normalize();
                worst = worst.max(n.dot(expected).clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
        assert!(
            worst < 5.0,
            "worst deviation {worst} degrees; the sphere is faceted"
        );
    }

    #[test]
    fn a_face_marked_flat_never_averages_with_its_neighbours() {
        let mut m = crate::primitives::uv_sphere(1.0, 16, 8);
        let n = m.num_faces() as usize;
        let mut smooth = vec![true; n];
        smooth[0] = false;
        m.attributes_mut().insert(
            names::SHADE_SMOOTH,
            Domain::Face,
            AttrData::Bool(Arc::new(smooth)),
        );

        let corners = m.compute_corner_normals(30f32.to_radians());
        let range = m.face_corner_range(0);
        let first = corners[range.start];
        for c in range {
            assert!(
                (corners[c] - first).length() < 1e-5,
                "face 0 should be flat"
            );
        }
    }

    #[test]
    fn the_threshold_controls_where_shading_splits() {
        let cube = crate::primitives::cube(2.0);
        // A threshold above 90 degrees smooths even a cube's corners.
        let smoothed = cube.compute_corner_normals(120f32.to_radians());
        let range = cube.face_corner_range(0);
        let first = smoothed[range.start];
        assert!(
            first.abs().max_element() < 0.99,
            "at a 120 degree threshold a cube corner should be diagonal, got {first}"
        );
    }

    #[test]
    fn corner_normals_are_finite_for_degenerate_geometry() {
        // A zero-area face has no direction to contribute; the result must still
        // be a usable normal rather than a black surface.
        let m = MeshData::from_faces(
            vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0, Vec3::Y],
            &[vec![0, 1, 2], vec![0, 2, 3]],
        );
        for n in m.compute_corner_normals(30f32.to_radians()) {
            assert!(n.is_finite(), "got {n}");
        }
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
