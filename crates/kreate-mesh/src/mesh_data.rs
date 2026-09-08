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
        // A counting sort keyed by the *lower* endpoint, not a sort of pairs
        // followed by a binary search per corner.
        //
        // The normalisation on both sides is symmetry rather than necessity:
        // `rebuild_edges` already stores `[low, high]`, so keying on `a`
        // directly would give the same answer, and a falsification sweep
        // proved it by doing exactly that. Normalising both sides means the
        // build and the lookup cannot disagree even if that ever changes. The keys are dense vertex
        // indices, so counting is linear where sorting is not — and each
        // corner then scans one vertex's handful of edges instead of paying a
        // logarithm against two million of them.
        //
        // At a million faces the old shape was two million pair comparisons to
        // sort plus four million binary searches, each about twenty-one probes
        // into an array far larger than any cache. This is two counting passes
        // and a short scan.
        let num_verts = self.num_verts as usize;
        let mut offsets = vec![0u32; num_verts + 1];
        for [a, b] in self.edges.iter() {
            let low = (*a).min(*b) as usize;
            if low < num_verts {
                offsets[low + 1] += 1;
            }
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut cursor = offsets.clone();
        // `(the other endpoint, the edge's index)`, grouped by lower endpoint.
        let mut entries: Vec<(u32, u32)> = vec![(0, 0); offsets[num_verts] as usize];
        for (index, [a, b]) in self.edges.iter().enumerate() {
            let (low, high) = if a <= b { (*a, *b) } else { (*b, *a) };
            if (low as usize) < num_verts {
                let slot = &mut cursor[low as usize];
                entries[*slot as usize] = (high, index as u32);
                *slot += 1;
            }
        }

        let mut out = vec![u32::MAX; self.corner_verts.len()];
        for face in 0..self.num_faces() {
            let ring = self.face_verts(face);
            let start = self.face_corner_range(face).start;
            for (i, v) in ring.iter().enumerate() {
                let next = ring[(i + 1) % ring.len()];
                let (low, high) = if *v <= next { (*v, next) } else { (next, *v) };
                let from = offsets[low as usize] as usize;
                let to = offsets[low as usize + 1] as usize;
                if let Some(found) = entries[from..to].iter().find(|(o, _)| *o == high) {
                    out[start + i] = found.1;
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
        self.same_topology(other)
            && match (
                self.attributes.get(names::POSITION),
                other.attributes.get(names::POSITION),
            ) {
                (Some(a), Some(b)) => a.data.ptr_eq(&b.data),
                (None, None) => true,
                _ => false,
            }
    }

    /// Whether `other` has the same connectivity as this mesh.
    ///
    /// Positions excluded, deliberately. Moving a vertex changes no
    /// relationship between elements: the triangulation, the edge list, the
    /// index buffers and the adjacency are all still correct, which is exactly
    /// what §4.5 asks a positions-only change to leave alone. Answered by
    /// pointer identity per array, so it costs four comparisons regardless of
    /// the size of the mesh.
    ///
    /// The vertex count is compared too, inherited from `same_geometry` and
    /// kept rather than removed. **No test reaches it, and a falsification
    /// sweep that deleted it found nothing that failed** — which is what one
    /// would expect: two meshes reach this line sharing all three arrays by
    /// pointer, and nothing in the crate produces two of those with different
    /// vertex counts. It stays because it costs one integer comparison and
    /// describes the intent, not because it is known to be load-bearing.
    pub fn same_topology(&self, other: &MeshData) -> bool {
        self.num_verts == other.num_verts
            && Arc::ptr_eq(&self.face_offsets, &other.face_offsets)
            && Arc::ptr_eq(&self.corner_verts, &other.corner_verts)
            && Arc::ptr_eq(&self.edges, &other.edges)
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
        self.only_these_layers_differ(other, names::is_selection)
    }

    /// Whether `other` differs from this mesh in nothing but where its vertices
    /// are.
    ///
    /// The second question a renderer has to answer cheaply, and the one a drag
    /// asks. A transform rewrites the positions array and leaves every other
    /// array shared — that is what `DocMut::edit_positions` is for — but a
    /// cache that can only ask "is this the same mesh, or is it only the
    /// selection" falls through to a full upload: new buffers for the indices,
    /// the triangle-to-face map, the edge list and the flags, every one of them
    /// byte-for-byte what was already there, on every pointer position.
    ///
    /// §4.5 states the requirement directly: a change to positions must not
    /// invalidate the triangulation, the edge list or the index buffers. This is
    /// the predicate that lets a caller act on it.
    ///
    /// Normals are recomputed from positions rather than compared, so they are
    /// not a layer this asks about — the renderer derives them either way.
    pub fn only_positions_differ(&self, other: &MeshData) -> bool {
        self.only_these_layers_differ(other, |name| name == names::POSITION)
    }

    /// The shared shape of both predicates: identical connectivity, and no
    /// attribute layer changed except the ones `allowed` names.
    ///
    /// Answered by pointer identity per array, so it costs a handful of
    /// comparisons regardless of how large the mesh is. Copy-on-write is what
    /// makes that valid: an array nobody wrote to keeps its pointer.
    ///
    /// An allowed layer *appearing* counts as an allowed difference, which is
    /// not a nicety: the selection layers are created when the user enters edit
    /// mode, so without it the first click on every mesh is a full upload — the
    /// one moment when the mesh is largest and the user is most attentive.
    ///
    /// Conservative in every other direction. Any other layer that changed,
    /// appeared or vanished means a full upload, which is correct but slower,
    /// rather than risking a buffer that quietly stays stale.
    fn only_these_layers_differ(
        &self,
        other: &MeshData,
        allowed: impl Fn(&str) -> bool + Copy,
    ) -> bool {
        if !self.same_topology(other) {
            return false;
        }

        // Both directions, so a layer that vanished is caught as well as one
        // that changed. The stores are kept sorted by name, so this is a walk
        // over two short sorted lists either way round.
        let unmatched = |a: &AttributeStore, b: &AttributeStore| {
            a.iter().any(|layer| match b.get(&layer.name) {
                None => !allowed(&layer.name),
                Some(theirs) => {
                    theirs.domain != layer.domain
                        || (!layer.data.ptr_eq(&theirs.data) && !allowed(&layer.name))
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
        // A bucket per lower endpoint rather than a hash map. The keys are
        // dense vertex indices, so counting is linear and the dedup within a
        // bucket is a scan over the four to six edges a vertex has — against
        // one hash and one probe per corner, four million times on a
        // million-face mesh. Measured at 463 ms there; this is the single
        // largest cost in writing an edit back to the document.
        //
        // The counting pass bounds each bucket by its *occurrences*, which
        // over-counts: a shared edge is met twice. Over-allocating a few
        // million bytes briefly is the cheap side of that trade, and it means
        // the fill pass can dedup as it goes.
        //
        // First-encounter order is preserved exactly, and that is not
        // cosmetic: edge attributes are indexed by it, golden files record it,
        // and `edge_gather` pairs against it. The fill pass walks corners in
        // face order and pushes on first sight, which is the same rule the
        // hash map's `or_insert_with` implemented.
        let num_verts = self.num_verts as usize;
        let mut offsets = vec![0u32; num_verts + 2];
        for f in 0..self.num_faces() {
            let verts = self.face_verts(f);
            for i in 0..verts.len() {
                let a = verts[i];
                let b = verts[(i + 1) % verts.len()];
                if a == b {
                    continue; // degenerate; reported by validate()
                }
                let low = a.min(b) as usize;
                if low < num_verts {
                    offsets[low + 1] += 1;
                }
            }
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        let total = offsets[num_verts] as usize;
        let mut highs = vec![u32::MAX; total];
        // How much of each bucket is filled: `offsets[v]..filled[v]` holds the
        // distinct higher endpoints seen at `v` so far.
        let mut filled: Vec<u32> = offsets[..=num_verts].to_vec();

        let mut edges: Vec<[u32; 2]> = Vec::new();
        for f in 0..self.num_faces() {
            let verts = self.face_verts(f);
            for i in 0..verts.len() {
                let a = verts[i];
                let b = verts[(i + 1) % verts.len()];
                if a == b {
                    continue;
                }
                let (low, high) = if a < b { (a, b) } else { (b, a) };
                let low_index = low as usize;
                if low_index >= num_verts {
                    continue;
                }
                let from = offsets[low_index] as usize;
                let to = filled[low_index] as usize;
                if highs[from..to].contains(&high) {
                    continue;
                }
                highs[to] = high;
                filled[low_index] += 1;
                edges.push([low, high]);
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
    /// A normal per corner, split where shading must break.
    ///
    /// # The shape of the problem
    ///
    /// A corner's normal is the average of the face normals in its *smooth
    /// group*: the faces around its vertex reachable without crossing a break.
    /// So the question is one of connected components, and the thing that
    /// breaks a group is a property of an **edge** — marked sharp, creased past
    /// the threshold, or with anything other than two faces on it.
    ///
    /// Written as such, the whole computation is a union-find over corners with
    /// one pass per edge. It used to be a flood fill *per corner*, which asked
    /// the same question once for every corner of every vertex and answered it
    /// by searching each candidate face's ring for the vertex. **401 ms on a
    /// million faces, measured, of a 559 ms render-mesh extraction.**
    ///
    /// # Why corners rather than faces
    ///
    /// A component has to be per (vertex, face) pair, not per face: two faces
    /// can be smooth across one shared edge and split across another, at a
    /// different vertex. A corner *is* that pair, and a face never repeats a
    /// vertex — the validator enforces it — so each face contributes exactly one
    /// corner, and therefore exactly one normal, to each group it is in.
    ///
    /// # Non-manifold edges break the group
    ///
    /// An edge with three or more faces has no well-defined "the other side",
    /// and the old flood fill joined across it in whatever order it happened to
    /// walk. Breaking is the deterministic answer, it is what other tools of
    /// this kind do, and a non-manifold edge is usually a mistake the user
    /// wants to see rather than have smoothed over.
    pub fn compute_corner_normals(&self, angle_threshold: f32) -> Vec<Vec3> {
        let face_normals = self.compute_face_normals();
        let unit: Vec<Vec3> = face_normals.iter().map(|n| n.normalize_or_zero()).collect();
        let cos_threshold = angle_threshold.cos();
        let smooth = self.attributes.get_bool(names::SHADE_SMOOTH);
        // Read straight off the edge domain. The old version rebuilt a sorted
        // list of vertex pairs and binary-searched it once per candidate,
        // because the flood fill only knew the two vertices; an edge index
        // knows the edge.
        let sharp = self.attributes.get_bool(names::SHARP_EDGE);
        let corner_edges = self.corner_edges();
        let corners = self.corner_verts.len();

        // Which face each corner belongs to, and which corner comes next round
        // that face. Both in one pass, because the union loop below reads them
        // millions of times and neither is worth working out again there: the
        // first version asked `face_of_corner`, a binary search over the CSR
        // offsets, twice per edge.
        let mut corner_face = vec![0u32; corners];
        let mut next_corner = vec![0u32; corners];
        for f in 0..self.num_faces() {
            let range = self.face_corner_range(f);
            for c in range.clone() {
                corner_face[c] = f;
                next_corner[c] = if c + 1 < range.end {
                    c as u32 + 1
                } else {
                    range.start as u32
                };
            }
        }

        // Corners grouped by the edge they walk along, by counting sort. For a
        // manifold edge there are exactly two.
        let num_edges = self.edges.len();
        let mut offsets = vec![0u32; num_edges + 1];
        for e in corner_edges.iter() {
            if (*e as usize) < num_edges {
                offsets[*e as usize + 1] += 1;
            }
        }
        for e in 0..num_edges {
            offsets[e + 1] += offsets[e];
        }
        let mut cursor = offsets.clone();
        let mut by_edge = vec![0u32; offsets[num_edges] as usize];
        for (corner, e) in corner_edges.iter().enumerate() {
            if (*e as usize) < num_edges {
                let slot = &mut cursor[*e as usize];
                by_edge[*slot as usize] = corner as u32;
                *slot += 1;
            }
        }

        let mut groups = Components::new(corners);
        for e in 0..num_edges {
            if sharp.is_some_and(|s| s.get(e).copied().unwrap_or(false)) {
                continue;
            }
            let here = &by_edge[offsets[e] as usize..offsets[e + 1] as usize];
            // Exactly two faces, or there is nothing to join across: one is a
            // boundary, three or more is non-manifold and breaks the group.
            let [c0, c1] = match here {
                [a, b] => [*a as usize, *b as usize],
                _ => continue,
            };
            let (f0, f1) = (corner_face[c0] as usize, corner_face[c1] as usize);
            let is_smooth = |f: usize| smooth.is_some_and(|s| !s[f]).eq(&false);
            if !is_smooth(f0) || !is_smooth(f1) {
                continue;
            }
            // A crease as steep as the threshold breaks the group just as a
            // sharp edge does, so an auto-smooth angle needs no marks on the
            // mesh to take effect.
            if unit[f0].dot(unit[f1]) < cos_threshold {
                continue;
            }
            // The two corners at each end of the edge. `c` sits at the edge's
            // first vertex as its own face traverses it, and the next corner
            // round that face sits at the other end — so which pairs with which
            // depends on whether the two faces wind the same way.
            let (n0, n1) = (next_corner[c0] as usize, next_corner[c1] as usize);
            if self.corner_verts[c0] == self.corner_verts[c1] {
                groups.union(c0, c1);
                groups.union(n0, n1);
            } else {
                groups.union(c0, n1);
                groups.union(n0, c1);
            }
        }

        // Unnormalised, so the average is area-weighted: a strip of slivers
        // beside one large quad should not outvote the quad.
        let mut sums = vec![Vec3::ZERO; corners];
        for c in 0..corners {
            sums[groups.find(c)] += face_normals[corner_face[c] as usize];
        }
        (0..corners)
            .map(|c| {
                let n = sums[groups.find(c)];
                // Falling back to the face's own normal keeps a degenerate
                // neighbourhood from producing a zero normal and a black face.
                if n == Vec3::ZERO {
                    unit[corner_face[c] as usize]
                } else {
                    n.normalize_or_zero()
                }
            })
            .collect()
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

/// Disjoint sets over `0..n`, for grouping corners into smooth groups.
///
/// Union by size with path halving. Written here rather than taken from a crate
/// because it is twenty lines and the mesh kernel's dependency list is a thing
/// the plan asks to keep short.
struct Components {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Components {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    /// The root of `i`'s set.
    ///
    /// **The loop is not reachable by anything here, and that is worth saying
    /// rather than leaving to look covered.** Union by size and path halving
    /// between them keep the trees so shallow that a sweep replacing this
    /// `while` with a single `if` failed no test: unioning a line of a thousand
    /// elements produces a *star*, because each merge attaches a singleton to a
    /// root, and halving jumps two levels whenever it is deeper than that.
    ///
    /// It stays a loop because that is what makes `find` correct on its own,
    /// independently of the two things that happen to keep the trees flat. A
    /// version that depends on them is one that breaks when either is changed
    /// for a reason unrelated to it.
    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] as usize != i {
            // Path halving: point at the grandparent as we go, which flattens
            // the tree without a second pass.
            let grand = self.parent[self.parent[i] as usize];
            self.parent[i] = grand;
            i = grand as usize;
        }
        i
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return;
        }
        // The smaller tree hangs off the larger, which is what keeps `find`
        // shallow. It changes no answer — a sweep removing it failed nothing,
        // correctly — only how deep the trees get.
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a as u32;
        self.size[a] += self.size[b];
    }
}

#[cfg(test)]
mod component_tests {
    use super::Components;

    #[test]
    fn a_chain_of_unions_still_finds_one_root() {
        // The disjoint sets on their own, because the shading fixtures barely
        // exercise them: a vertex fan is four or five faces.
        //
        // What this does *not* reach is `find`'s loop. Unioning in a line looks
        // like the shape that makes a tree deep and is not: each merge attaches
        // a singleton to a root, so the result is a star. See `find`, which says
        // so and says why the loop stays anyway.
        let n = 1000;
        let mut c = Components::new(n);
        for i in 1..n {
            c.union(i - 1, i);
        }
        let root = c.find(0);
        for i in 0..n {
            assert_eq!(c.find(i), root, "{i} is in a different set from 0");
        }
    }

    #[test]
    fn the_order_of_the_unions_does_not_change_the_sets() {
        // Two sets built the same way in opposite orders have to come out the
        // same. Union by size chooses which root survives, so an implementation
        // that let the order matter would give a different *root* — which is
        // fine — but must never give different *membership*.
        let mut forwards = Components::new(64);
        for i in 1..32 {
            forwards.union(i - 1, i);
        }
        for i in 33..64 {
            forwards.union(i - 1, i);
        }

        let mut backwards = Components::new(64);
        for i in (1..32).rev() {
            backwards.union(i, i - 1);
        }
        for i in (33..64).rev() {
            backwards.union(i, i - 1);
        }

        for a in 0..64 {
            for b in 0..64 {
                assert_eq!(
                    forwards.find(a) == forwards.find(b),
                    backwards.find(a) == backwards.find(b),
                    "{a} and {b} disagree about being in one set"
                );
            }
        }
        // And the two sets really are two, split at the element nothing joined.
        assert_ne!(forwards.find(0), forwards.find(63));
        assert_ne!(forwards.find(0), forwards.find(32));
    }

    #[test]
    fn an_element_starts_in_a_set_of_its_own() {
        let mut c = Components::new(4);
        let roots: Vec<usize> = (0..4).map(|i| c.find(i)).collect();
        assert_eq!(roots, vec![0, 1, 2, 3]);
    }

    #[test]
    fn uniting_a_set_with_itself_changes_nothing() {
        let mut c = Components::new(8);
        c.union(3, 5);
        let before: Vec<usize> = (0..8).map(|i| c.find(i)).collect();
        c.union(3, 5);
        c.union(5, 3);
        c.union(3, 3);
        let after: Vec<usize> = (0..8).map(|i| c.find(i)).collect();
        assert_eq!(before, after);
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
    fn the_edge_order_is_first_encounter_over_faces_and_corners() {
        // What every edge-domain layer is indexed by, what golden files record,
        // and what `edge_gather` pairs against. `rebuild_edges` was a hash map
        // and is now a bucket-per-vertex counting pass; this is the test that
        // says the two agree, computed here from the definition rather than
        // copied from a run.
        let reference = |mesh: &MeshData| -> Vec<[u32; 2]> {
            let mut seen: Vec<[u32; 2]> = Vec::new();
            for f in 0..mesh.num_faces() {
                let verts = mesh.face_verts(f);
                for i in 0..verts.len() {
                    let a = verts[i];
                    let b = verts[(i + 1) % verts.len()];
                    if a == b {
                        continue;
                    }
                    let pair = if a < b { [a, b] } else { [b, a] };
                    if !seen.contains(&pair) {
                        seen.push(pair);
                    }
                }
            }
            seen
        };

        for (name, mesh) in [
            ("cube", crate::primitives::cube(2.0)),
            ("sphere", crate::primitives::uv_sphere(1.0, 9, 5)),
            ("grid", crate::primitives::grid(2.0, 4, 3)),
            ("cylinder", crate::primitives::cylinder(1.0, 2.0, 7)),
            ("cone", crate::primitives::cone(1.0, 2.0, 6)),
            ("torus", crate::primitives::torus(1.0, 0.4, 9, 7)),
        ] {
            assert_eq!(
                mesh.edges(),
                reference(&mesh).as_slice(),
                "{name}: the edge order is not first-encounter"
            );
        }

        // An n-gon whose corners revisit a vertex the winding already passed,
        // and a face sharing two edges with its neighbour: the cases where a
        // bucket holds more than one candidate and the dedup has to pick.
        let awkward = MeshData::from_faces(
            vec![
                glam::Vec3::ZERO,
                glam::Vec3::X,
                glam::Vec3::new(1.0, 1.0, 0.0),
                glam::Vec3::Y,
                glam::Vec3::new(2.0, 0.0, 0.0),
                glam::Vec3::new(2.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3], vec![1, 4, 5, 2]],
        );
        assert_eq!(awkward.edges(), reference(&awkward).as_slice());
        // Seven distinct edges: four plus four minus the one they share.
        assert_eq!(awkward.num_edges(), 7);
    }

    #[test]
    fn the_edge_array_is_always_normalised_in_range_and_non_degenerate() {
        // The invariant three other pieces of code rest on: the bulk `BMesh`
        // construction skips a degenerate or dangling edge, and `corner_edges`
        // keys its lookup table on the lower endpoint. A falsification sweep
        // removed both and nothing failed, which is the right answer — every
        // constructor ends in `rebuild_edges`, which derives the array from the
        // faces, drops `a == b`, and stores `[low, high]`.
        //
        // So the guards are unreachable today, and this is what says why. If
        // this test ever fails, those guards are the reason the kernel skips
        // instead of panicking, and they stop being dead code.
        let check = |mesh: &MeshData, what: &str| {
            for [a, b] in mesh.edges() {
                assert!(a < b, "{what}: edge [{a}, {b}] is not normalised");
                assert!(
                    *b < mesh.num_verts(),
                    "{what}: edge [{a}, {b}] is out of range"
                );
            }
        };

        for (name, mesh) in [
            ("cube", crate::primitives::cube(2.0)),
            ("sphere", crate::primitives::uv_sphere(1.0, 8, 4)),
            ("grid", crate::primitives::grid(2.0, 3, 3)),
            ("cylinder", crate::primitives::cylinder(1.0, 2.0, 7)),
        ] {
            check(&mesh, name);
        }

        // And through `from_csr`, which is the path a file takes: a supplied
        // array that is unnormalised, self-referential and in the wrong order
        // comes back derived from the faces instead.
        let mesh = MeshData::from_csr(
            4,
            vec![0, 4],
            vec![0, 1, 2, 3],
            // Reversed, degenerate, and one that no face implies.
            vec![[1, 0], [2, 2], [3, 0], [0, 3]],
        )
        .expect("the faces are sound even if the edges are not");
        check(&mesh, "from_csr");
        assert_eq!(
            mesh.num_edges(),
            4,
            "the supplied array was trusted rather than rebuilt"
        );

        // An out-of-range *corner* is a different matter and is refused, not
        // repaired: a face referencing a vertex that does not exist has no
        // sound interpretation.
        assert!(MeshData::from_csr(2, vec![0, 3], vec![0, 1, 9], Vec::new()).is_err());
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

    /// The corner normals of every corner at `vertex`, one per face.
    fn normals_at(m: &MeshData, vertex: u32, threshold: f32) -> Vec<(u32, Vec3)> {
        let corners = m.compute_corner_normals(threshold);
        let mut out = Vec::new();
        for f in 0..m.num_faces() {
            let range = m.face_corner_range(f);
            for (i, &v) in m.face_verts(f).iter().enumerate() {
                if v == vertex {
                    out.push((f, corners[range.start + i]));
                }
            }
        }
        out
    }

    /// How many distinct directions are in a set, to within a degree or so.
    fn distinct(normals: &[(u32, Vec3)]) -> usize {
        let mut groups: Vec<Vec3> = Vec::new();
        for (_, n) in normals {
            if !groups.iter().any(|g| g.dot(*n) > 0.999) {
                groups.push(*n);
            }
        }
        groups.len()
    }

    fn edge_index(m: &MeshData, a: u32, b: u32) -> usize {
        m.edges()
            .iter()
            .position(|e| (e[0] == a && e[1] == b) || (e[0] == b && e[1] == a))
            .expect("the mesh has that edge")
    }

    fn mark_sharp(m: &mut MeshData, edges: &[usize]) {
        let mut flags = vec![false; m.num_edges() as usize];
        for e in edges {
            flags[*e] = true;
        }
        m.attributes_mut().insert(
            names::SHARP_EDGE,
            Domain::Edge,
            AttrData::Bool(Arc::new(flags)),
        );
    }

    /// Four triangles round an apex, each facing a different way.
    ///
    /// The fan is a *cycle*: cutting one edge of it leaves every face still
    /// reachable the long way round, so it takes two cuts to split the group.
    /// That is the case a naive "a sharp edge splits the vertex" would get
    /// wrong, and the reason this fixture is a closed fan rather than a strip.
    fn tent() -> MeshData {
        let apex = Vec3::new(0.0, 0.0, 1.0);
        MeshData::from_faces(
            vec![
                apex,
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
            ],
            &[vec![0, 1, 2], vec![0, 2, 3], vec![0, 3, 4], vec![0, 4, 1]],
        )
    }

    #[test]
    fn the_average_is_weighted_by_area() {
        // Stated in the code as "a strip of slivers beside one large quad
        // should not outvote the quad", and until now not tested: a sweep
        // summing *unit* normals instead of the unnormalised ones broke
        // nothing.
        //
        // A fan of one large triangle and two slivers, connected so they are
        // one smooth group. Both candidate answers are computed here from the
        // face normals, so the test cannot drift with the fixture and says
        // exactly which of the two the code produces.
        let m = MeshData::from_faces(
            vec![
                Vec3::ZERO,
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(0.0, 10.0, 0.0),
                Vec3::new(-0.1, 10.0, 4.0),
                Vec3::new(-0.2, 10.0, 8.0),
            ],
            &[vec![0, 1, 2], vec![0, 2, 3], vec![0, 3, 4]],
        );
        let faces = m.compute_face_normals();
        assert_eq!(faces.len(), 3);

        let weighted = (faces[0] + faces[1] + faces[2]).normalize();
        let unweighted = (faces[0].normalize_or_zero()
            + faces[1].normalize_or_zero()
            + faces[2].normalize_or_zero())
        .normalize();
        // The fixture only means something if the two answers differ.
        let apart = weighted
            .dot(unweighted)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        assert!(
            apart > 10.0,
            "the two averages are only {apart} degrees apart; the fixture \
             cannot tell them apart"
        );

        let at_origin = normals_at(&m, 0, std::f32::consts::PI);
        assert_eq!(at_origin.len(), 3, "all three faces meet at the origin");
        for (face, n) in &at_origin {
            assert!(
                n.dot(weighted) > 0.9999,
                "face {face} got {n}, which is the unweighted average"
            );
        }
    }

    #[test]
    fn cutting_a_closed_fan_once_does_not_split_it() {
        // A cycle cut in one place is still connected. A version that split the
        // vertex on the first sharp edge it met would answer four groups here,
        // or two, and both would be wrong.
        let mut m = tent();
        let smooth_threshold = std::f32::consts::PI;
        assert_eq!(
            distinct(&normals_at(&m, 0, smooth_threshold)),
            1,
            "the apex should have one normal before anything is marked"
        );

        let cut = edge_index(&m, 0, 1);
        mark_sharp(&mut m, &[cut]);
        assert_eq!(
            distinct(&normals_at(&m, 0, smooth_threshold)),
            1,
            "one cut in a cycle still leaves one group"
        );
    }

    #[test]
    fn cutting_a_closed_fan_twice_splits_it_along_the_cuts() {
        // Two cuts on opposite sides give two arcs of two faces. Which faces
        // land together is the part that pins the component logic: a version
        // that grouped by anything other than reachability could produce two
        // groups of the wrong two.
        let mut m = tent();
        let cuts = [edge_index(&m, 0, 1), edge_index(&m, 0, 3)];
        mark_sharp(&mut m, &cuts);

        let at_apex = normals_at(&m, 0, std::f32::consts::PI);
        assert_eq!(at_apex.len(), 4, "four faces meet at the apex");
        assert_eq!(distinct(&at_apex), 2, "two cuts should give two arcs");

        let of = |face: u32| {
            at_apex
                .iter()
                .find(|(f, _)| *f == face)
                .expect("that face is at the apex")
                .1
        };
        // The cuts are at edges 0-1 (between faces 3 and 0) and 0-3 (between
        // faces 1 and 2), so the arcs are {0, 1} and {2, 3}.
        assert!(of(0).dot(of(1)) > 0.999, "faces 0 and 1 should agree");
        assert!(of(2).dot(of(3)) > 0.999, "faces 2 and 3 should agree");
        assert!(
            of(0).dot(of(2)) < 0.999,
            "the two arcs should not agree with each other"
        );
    }

    #[test]
    fn two_fans_meeting_at_a_point_do_not_average() {
        // The tempting wrong shortcut, written down so it cannot be taken: "if
        // nothing at this vertex breaks, the group is the whole fan". Two cones
        // tip to tip share a vertex and no edge, so nothing breaks and the fan
        // is still two groups. Shading them together would round off a point
        // that is geometrically sharp.
        let m = MeshData::from_faces(
            vec![
                Vec3::ZERO,
                // A triangle in the XY plane.
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                // And one in the XZ plane, sharing only the origin.
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 1.0),
            ],
            &[vec![0, 1, 2], vec![0, 3, 4]],
        );
        let at_tip = normals_at(&m, 0, std::f32::consts::PI);
        assert_eq!(at_tip.len(), 2);
        assert_eq!(
            distinct(&at_tip),
            2,
            "the two fans were shaded as one: {at_tip:?}"
        );
    }

    #[test]
    fn a_non_manifold_edge_breaks_the_smooth_group() {
        // The behaviour this rewrite chose, so it is written down. An edge with
        // three faces has no well-defined "the other side", and the flood fill
        // it replaced joined across one in whatever order it happened to walk.
        // Breaking is deterministic, it is what other tools of this kind do, and
        // a non-manifold edge is usually a mistake the user wants to see rather
        // than have smoothed over.
        let m = MeshData::from_faces(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, -1.0, 1.0),
            ],
            // Three faces on the edge 0-1.
            &[vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4]],
        );
        for vertex in [0, 1] {
            let at = normals_at(&m, vertex, std::f32::consts::PI);
            assert_eq!(at.len(), 3, "three faces meet at vertex {vertex}");
            assert_eq!(
                distinct(&at),
                3,
                "vertex {vertex} averaged across a non-manifold edge: {at:?}"
            );
        }
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
