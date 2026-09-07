//! `BMesh`: the edit-mode topology structure.
//!
//! Three intrusive doubly-linked cycles, following Blender's design:
//!
//! - **loop cycle** — the corners around one face, in winding order.
//! - **radial cycle** — the loops that use one edge, i.e. the faces on it.
//! - **disk cycle** — the edges that touch one vertex.
//!
//! # Why not half-edge
//!
//! A half-edge's `twin` pointer encodes "at most two faces per edge" in the type
//! itself. Users violate that on day three, and `knife`, `dissolve` and boolean
//! all produce non-manifold intermediate states internally even when both their
//! input and their output are manifold. A radial cycle is a list, so three faces
//! on an edge is an ordinary case rather than a special one, and the code that
//! walks it does not change.
//!
//! Wire edges (no faces) and isolated vertices (no edges) are likewise ordinary
//! here. They are what a user has mid-operation, and a structure that cannot
//! represent them forces every operator to reach its final state in one step.
//!
//! # Why this is not the document form
//!
//! Roughly 190 bytes per vertex on a quad mesh, against about 50 for
//! [`MeshData`](crate::MeshData). At 10 M vertices that is 2 GB versus 500 MB.
//! `BMesh` is built on entering edit mode and dropped on leaving it; the
//! document never holds one.
//!
//! # Positions
//!
//! [`Vert::co`] is stored inline rather than in the attribute store. This is a
//! deliberate exception to "no hardcoded fields": position is mandatory, always
//! `Vec3`, and read by every single algorithm, so the indirection buys nothing
//! and costs a cache miss in the hottest loops there are. Every *other* layer —
//! UVs, creases, selection, custom normals — is generic, which is the part that
//! is painful to retrofit.

mod convert;
mod select;
mod validate;

pub use select::SelectMode;
pub use validate::BmViolation;

use kreate_math::Vec3;

use crate::arena::{Arena, ArenaIndex, EdgeId, FaceId, LoopId, VertId};
use crate::attribute::{AttributeStore, Domain};

/// A vertex.
#[derive(Clone, Copy, Debug)]
pub struct Vert {
    pub co: Vec3,
    /// An arbitrary edge of this vertex's disk cycle, or `None` when isolated.
    edge: Option<EdgeId>,
}

/// One vertex's link in an edge's two disk cycles.
#[derive(Clone, Copy, Debug)]
struct DiskLink {
    prev: EdgeId,
    next: EdgeId,
}

/// An edge between two vertices.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    verts: [VertId; 2],
    /// Disk links, one per endpoint: `disk[i]` threads this edge into the cycle
    /// around `verts[i]`. Two links rather than one is what lets a single edge
    /// belong to two different vertices' cycles at once.
    disk: [DiskLink; 2],
    /// An arbitrary loop of this edge's radial cycle, or `None` when the edge is
    /// a wire.
    first_loop: Option<LoopId>,
}

/// A face-corner: one face's use of one vertex.
///
/// The element that carries per-corner attributes (UVs, split normals, vertex
/// colours) and the node both the loop and radial cycles are built from.
#[derive(Clone, Copy, Debug)]
pub struct Loop {
    vert: VertId,
    /// The edge from [`Loop::vert`] to `next.vert`.
    edge: EdgeId,
    face: FaceId,
    next: LoopId,
    prev: LoopId,
    radial_next: LoopId,
    radial_prev: LoopId,
}

/// A face.
#[derive(Clone, Copy, Debug)]
pub struct Face {
    /// An arbitrary loop of this face's loop cycle.
    first_loop: LoopId,
    len: u32,
}

impl Face {
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Why a structural operation could not be performed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BmError {
    /// A face needs at least three distinct corners.
    FaceTooShort { given: usize },
    /// A face repeated a vertex. Legal in some kernels, but it makes the loop
    /// and radial cycles ambiguous and every downstream operator wrong.
    RepeatedVert { vert: VertId },
    /// A handle did not refer to a live element.
    DeadHandle,
    /// An edge would join a vertex to itself.
    SelfEdge { vert: VertId },
}

impl std::fmt::Display for BmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BmError::FaceTooShort { given } => {
                write!(f, "a face needs at least 3 corners, got {given}")
            }
            BmError::RepeatedVert { vert } => write!(f, "face repeats {vert:?}"),
            BmError::DeadHandle => write!(f, "stale or dangling handle"),
            BmError::SelfEdge { vert } => write!(f, "edge joins {vert:?} to itself"),
        }
    }
}

impl std::error::Error for BmError {}

/// A handle to one mesh element, whatever domain it is in.
///
/// Produced by [`BMesh::handle_at`], which is the single place an index that came
/// from outside the kernel — a GPU id buffer, a script, a file — turns into
/// something safe to look up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementId {
    Vert(VertId),
    Edge(EdgeId),
    Face(FaceId),
}

/// The edit-mode mesh.
#[derive(Clone, Debug, Default)]
pub struct BMesh {
    verts: Arena<VertId, Vert>,
    edges: Arena<EdgeId, Edge>,
    loops: Arena<LoopId, Loop>,
    faces: Arena<FaceId, Face>,
    /// Generic layers, indexed by *slot* so they stay parallel to the arenas.
    ///
    /// Slot rather than a dense index: a dense index would have to be rebuilt on
    /// every removal, and edit-mode operators remove constantly.
    attributes: AttributeStore,
}

impl BMesh {
    pub fn new() -> Self {
        Self::default()
    }

    // --- counts -------------------------------------------------------------

    #[inline]
    pub fn num_verts(&self) -> usize {
        self.verts.len()
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    #[inline]
    pub fn num_loops(&self) -> usize {
        self.loops.len()
    }

    #[inline]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }

    /// Slot counts per domain. The length every parallel attribute array must
    /// have — not the same as the element count once anything has been removed.
    pub fn slot_count(&self, domain: Domain) -> usize {
        match domain {
            Domain::Point => self.verts.slot_count(),
            Domain::Edge => self.edges.slot_count(),
            Domain::Corner => self.loops.slot_count(),
            Domain::Face => self.faces.slot_count(),
        }
    }

    /// The live handle for a slot in one domain, if that slot is occupied.
    ///
    /// The bridge from an outside index to a handle. A click resolves to an
    /// element *index* — from the GPU id buffer, which knows nothing about
    /// arena generations — and this is what turns it into something safe to
    /// look up. A corner is not one of the three things a 3D click can hit, so
    /// it is not part of this enum — see [`Self::corner_at`], which the UV
    /// editor uses instead.
    pub fn handle_at(&self, domain: Domain, index: u32) -> Option<ElementId> {
        match domain {
            Domain::Point => self.verts.handle_at(index).map(ElementId::Vert),
            Domain::Edge => self.edges.handle_at(index).map(ElementId::Edge),
            Domain::Face => self.faces.handle_at(index).map(ElementId::Face),
            Domain::Corner => None,
        }
    }

    /// The live handle for a corner slot, if that slot is occupied.
    ///
    /// Separate from [`Self::handle_at`] because a corner is not something a
    /// 3D click can land on — the id buffer knows vertices, edges and faces.
    /// The UV editor is the one place that picks corners, because a UV *is* a
    /// corner, and it needs the same generation check: a slot index captured
    /// before a topology edit must be dropped rather than aliasing whatever
    /// now lives in that slot.
    pub fn corner_at(&self, index: u32) -> Option<LoopId> {
        self.loops.handle_at(index)
    }

    /// Bytes held by the four arenas and the attribute layers.
    ///
    /// Exposed so the per-vertex memory budget can be a test. Excludes `Vec`
    /// growth slack, which is real but depends on allocation history rather than
    /// on any decision in this file.
    pub fn memory_bytes(&self) -> usize {
        self.verts.memory_bytes()
            + self.edges.memory_bytes()
            + self.loops.memory_bytes()
            + self.faces.memory_bytes()
            + self
                .attributes
                .iter()
                .map(|l| l.data.len() * l.data.ty().size())
                .sum::<usize>()
    }

    // --- element access -----------------------------------------------------

    #[inline]
    pub fn vert(&self, v: VertId) -> Option<&Vert> {
        self.verts.get(v)
    }

    #[inline]
    pub fn edge(&self, e: EdgeId) -> Option<&Edge> {
        self.edges.get(e)
    }

    #[inline]
    pub fn loop_(&self, l: LoopId) -> Option<&Loop> {
        self.loops.get(l)
    }

    #[inline]
    pub fn face(&self, f: FaceId) -> Option<&Face> {
        self.faces.get(f)
    }

    #[inline]
    pub fn position(&self, v: VertId) -> Option<Vec3> {
        self.verts.get(v).map(|x| x.co)
    }

    /// Moves a vertex. The only mutation that never changes topology, and by far
    /// the most frequent one.
    pub fn set_position(&mut self, v: VertId, co: Vec3) -> bool {
        match self.verts.get_mut(v) {
            Some(x) => {
                x.co = co;
                true
            }
            None => false,
        }
    }

    #[inline]
    pub fn edge_verts(&self, e: EdgeId) -> Option<[VertId; 2]> {
        self.edges.get(e).map(|x| x.verts)
    }

    #[inline]
    pub fn loop_vert(&self, l: LoopId) -> Option<VertId> {
        self.loops.get(l).map(|x| x.vert)
    }

    #[inline]
    pub fn loop_edge(&self, l: LoopId) -> Option<EdgeId> {
        self.loops.get(l).map(|x| x.edge)
    }

    #[inline]
    pub fn loop_face(&self, l: LoopId) -> Option<FaceId> {
        self.loops.get(l).map(|x| x.face)
    }

    /// The next corner around the same face, in winding order.
    #[inline]
    pub fn loop_next(&self, l: LoopId) -> Option<LoopId> {
        self.loops.get(l).map(|x| x.next)
    }

    /// The previous corner around the same face.
    #[inline]
    pub fn loop_prev(&self, l: LoopId) -> Option<LoopId> {
        self.loops.get(l).map(|x| x.prev)
    }

    /// The next corner around the same *edge* — the neighbouring face.
    #[inline]
    pub fn loop_radial_next(&self, l: LoopId) -> Option<LoopId> {
        self.loops.get(l).map(|x| x.radial_next)
    }

    pub fn verts(&self) -> impl Iterator<Item = (VertId, &Vert)> + '_ {
        self.verts.iter()
    }

    pub fn vert_ids(&self) -> impl Iterator<Item = VertId> + '_ {
        self.verts.ids()
    }

    pub fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edges.ids()
    }

    pub fn face_ids(&self) -> impl Iterator<Item = FaceId> + '_ {
        self.faces.ids()
    }

    pub fn loop_ids(&self) -> impl Iterator<Item = LoopId> + '_ {
        self.loops.ids()
    }

    #[inline]
    pub fn attributes(&self) -> &AttributeStore {
        &self.attributes
    }

    /// Mutable attribute access.
    ///
    /// Resizing a layer through this handle desynchronises it from the arenas;
    /// [`BMesh::validate`] reports that rather than letting it corrupt an
    /// operator much later.
    #[inline]
    pub fn attributes_mut(&mut self) -> &mut AttributeStore {
        &mut self.attributes
    }

    // --- cycle walks --------------------------------------------------------

    /// The loops of a face, in winding order.
    ///
    /// Bounded by the recorded face length rather than by "walk until you get
    /// back to the start": on a corrupted mesh the latter is an infinite loop,
    /// and the validator is precisely the code that runs on corrupted meshes.
    pub fn face_loops(&self, f: FaceId) -> Vec<LoopId> {
        let Some(face) = self.faces.get(f) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(face.len as usize);
        let mut l = face.first_loop;
        for _ in 0..face.len {
            out.push(l);
            let Some(node) = self.loops.get(l) else { break };
            l = node.next;
        }
        out
    }

    /// The vertices of a face, in winding order.
    pub fn face_verts(&self, f: FaceId) -> Vec<VertId> {
        self.face_loops(f)
            .into_iter()
            .filter_map(|l| self.loop_vert(l))
            .collect()
    }

    /// The loops on an edge — one per face using it.
    ///
    /// Length 0 for a wire, 1 for a boundary, 2 for the manifold case, more for
    /// a non-manifold fan.
    pub fn edge_loops(&self, e: EdgeId) -> Vec<LoopId> {
        let Some(edge) = self.edges.get(e) else {
            return Vec::new();
        };
        let Some(first) = edge.first_loop else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut l = first;
        // A radial cycle can be any length, so the guard is the loop arena's
        // size: no valid cycle can be longer, and a corrupt one stops here
        // instead of hanging the editor.
        for _ in 0..=self.loops.slot_count() {
            out.push(l);
            let Some(node) = self.loops.get(l) else { break };
            l = node.radial_next;
            if l == first {
                break;
            }
        }
        out
    }

    /// The faces on an edge.
    pub fn edge_faces(&self, e: EdgeId) -> Vec<FaceId> {
        self.edge_loops(e)
            .into_iter()
            .filter_map(|l| self.loop_face(l))
            .collect()
    }

    /// How many faces use an edge.
    #[inline]
    pub fn edge_face_count(&self, e: EdgeId) -> usize {
        self.edge_loops(e).len()
    }

    /// No faces at all.
    #[inline]
    pub fn is_wire_edge(&self, e: EdgeId) -> bool {
        self.edges.get(e).is_some_and(|x| x.first_loop.is_none())
    }

    /// Exactly one face — the outline of an open surface.
    #[inline]
    pub fn is_boundary_edge(&self, e: EdgeId) -> bool {
        self.edge_face_count(e) == 1
    }

    /// At most two faces. Wire and boundary edges count as manifold, matching
    /// what the word means to a modeller looking at an open surface.
    #[inline]
    pub fn is_manifold_edge(&self, e: EdgeId) -> bool {
        self.edge_face_count(e) <= 2
    }

    /// The edges touching a vertex, in disk order.
    pub fn vert_edges(&self, v: VertId) -> Vec<EdgeId> {
        let Some(vert) = self.verts.get(v) else {
            return Vec::new();
        };
        let Some(first) = vert.edge else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut e = first;
        for _ in 0..=self.edges.slot_count() {
            out.push(e);
            let Some(i) = self.disk_index(e, v) else {
                break;
            };
            let Some(edge) = self.edges.get(e) else { break };
            e = edge.disk[i].next;
            if e == first {
                break;
            }
        }
        out
    }

    /// The faces touching a vertex, each listed once.
    pub fn vert_faces(&self, v: VertId) -> Vec<FaceId> {
        let mut out = Vec::new();
        for e in self.vert_edges(v) {
            for f in self.edge_faces(e) {
                if !out.contains(&f) {
                    out.push(f);
                }
            }
        }
        out
    }

    #[inline]
    pub fn vert_edge_count(&self, v: VertId) -> usize {
        self.vert_edges(v).len()
    }

    /// The edge between two vertices, if there is one.
    ///
    /// Walks the shorter disk. There is at most one — duplicate edges are
    /// rejected on creation and reported by the validator.
    pub fn edge_between(&self, a: VertId, b: VertId) -> Option<EdgeId> {
        self.vert_edges(a).into_iter().find(|&e| {
            self.edges
                .get(e)
                .is_some_and(|x| x.verts == [a, b] || x.verts == [b, a])
        })
    }

    /// The other endpoint of an edge.
    pub fn edge_other_vert(&self, e: EdgeId, v: VertId) -> Option<VertId> {
        let edge = self.edges.get(e)?;
        if edge.verts[0] == v {
            Some(edge.verts[1])
        } else if edge.verts[1] == v {
            Some(edge.verts[0])
        } else {
            None
        }
    }

    /// Newell's normal for a face, unnormalised.
    ///
    /// Newell rather than a cross product of the first three corners: an n-gon
    /// is rarely planar in practice, and three corners of a saddle can be nearly
    /// collinear even when the face as a whole has a perfectly good orientation.
    /// The length is twice the projected area, which is exactly the weight an
    /// area-weighted average of face normals wants.
    pub fn face_normal(&self, f: FaceId) -> Vec3 {
        let loops = self.face_loops(f);
        let mut n = Vec3::ZERO;
        for i in 0..loops.len() {
            let (Some(a), Some(b)) = (
                self.loop_vert(loops[i]).and_then(|v| self.position(v)),
                self.loop_vert(loops[(i + 1) % loops.len()])
                    .and_then(|v| self.position(v)),
            ) else {
                continue;
            };
            n.x += (a.y - b.y) * (a.z + b.z);
            n.y += (a.z - b.z) * (a.x + b.x);
            n.z += (a.x - b.x) * (a.y + b.y);
        }
        n
    }

    /// The area-weighted average normal of a set of faces, normalised.
    ///
    /// Area-weighted because that is what "the direction this region faces"
    /// means when the region is uneven: a strip of tiny triangles beside one
    /// large quad should not outvote the quad.
    pub fn region_normal(&self, faces: &[FaceId]) -> Vec3 {
        let n: Vec3 = faces.iter().map(|f| self.face_normal(*f)).sum();
        n.normalize_or_zero()
    }

    // --- construction -------------------------------------------------------

    /// Adds an isolated vertex.
    pub fn add_vert(&mut self, co: Vec3) -> VertId {
        let id = self.verts.insert(Vert { co, edge: None });
        self.grow_domain(Domain::Point, id.index());
        id
    }

    /// Adds an edge, or returns the existing one.
    ///
    /// Find-or-create rather than always-create: two edges between the same pair
    /// of vertices make the disk cycle ambiguous, and every operator that walks
    /// it then picks one arbitrarily and produces a different answer depending on
    /// insertion order.
    pub fn add_edge(&mut self, a: VertId, b: VertId) -> Result<EdgeId, BmError> {
        if a == b {
            return Err(BmError::SelfEdge { vert: a });
        }
        if !self.verts.contains(a) || !self.verts.contains(b) {
            return Err(BmError::DeadHandle);
        }
        if let Some(existing) = self.edge_between(a, b) {
            return Ok(existing);
        }
        // The disk links are placeholders until `disk_insert` splices the edge
        // in; it is the only code allowed to leave them meaningful.
        let placeholder = EdgeId::from_parts(0, 0);
        let id = self.edges.insert(Edge {
            verts: [a, b],
            disk: [DiskLink {
                prev: placeholder,
                next: placeholder,
            }; 2],
            first_loop: None,
        });
        self.grow_domain(Domain::Edge, id.index());
        self.disk_insert(id, a);
        self.disk_insert(id, b);
        Ok(id)
    }

    /// Adds a face through the given vertices, creating any missing edges.
    ///
    /// The vertex order is the winding order, and it is what decides which way
    /// the face's normal points.
    pub fn add_face(&mut self, verts: &[VertId]) -> Result<FaceId, BmError> {
        if verts.len() < 3 {
            return Err(BmError::FaceTooShort { given: verts.len() });
        }
        for (i, v) in verts.iter().enumerate() {
            if !self.verts.contains(*v) {
                return Err(BmError::DeadHandle);
            }
            if verts[..i].contains(v) {
                return Err(BmError::RepeatedVert { vert: *v });
            }
        }

        // Edges first, so a failure leaves no half-built face behind. Creating an
        // edge that already exists is a no-op, so this is safe to retry.
        let n = verts.len();
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            edges.push(self.add_edge(verts[i], verts[(i + 1) % n])?);
        }

        let placeholder = LoopId::from_parts(0, 0);
        let face = self.faces.insert(Face {
            first_loop: placeholder,
            len: n as u32,
        });
        self.grow_domain(Domain::Face, face.index());

        let mut loop_ids = Vec::with_capacity(n);
        for i in 0..n {
            let id = self.loops.insert(Loop {
                vert: verts[i],
                edge: edges[i],
                face,
                next: placeholder,
                prev: placeholder,
                radial_next: placeholder,
                radial_prev: placeholder,
            });
            self.grow_domain(Domain::Corner, id.index());
            loop_ids.push(id);
        }

        for i in 0..n {
            let l = loop_ids[i];
            self.loops[l].next = loop_ids[(i + 1) % n];
            self.loops[l].prev = loop_ids[(i + n - 1) % n];
        }
        self.faces[face].first_loop = loop_ids[0];

        for i in 0..n {
            self.radial_insert(loop_ids[i], edges[i]);
        }
        Ok(face)
    }

    /// Convenience for building a face from positions in one call.
    pub fn add_face_from_positions(&mut self, positions: &[Vec3]) -> Result<FaceId, BmError> {
        let verts: Vec<VertId> = positions.iter().map(|p| self.add_vert(*p)).collect();
        self.add_face(&verts)
    }

    /// Reverses a face's winding, flipping its normal.
    ///
    /// In place, keeping every loop. That is what preserves per-corner data: a
    /// corner's UV belongs to (this face, this vertex), and reversing changes
    /// only the order the corners are visited and which edge each one runs
    /// along — not which vertex it is. Rebuilding the face instead would be
    /// simpler and would silently clear its UVs and split normals, which the
    /// user discovers when the texture on one flipped face goes blank.
    pub fn reverse_face(&mut self, f: FaceId) -> bool {
        let loops = self.face_loops(f);
        let n = loops.len();
        if n < 3 || loops.iter().any(|l| self.loops.get(*l).is_none()) {
            return false;
        }
        let edges: Vec<EdgeId> = loops.iter().filter_map(|l| self.loop_edge(*l)).collect();
        if edges.len() != n {
            return false;
        }

        // Out of the radial cycles first: each loop is about to sit on a
        // different edge, and leaving it linked to the old one is how a radial
        // cycle ends up containing a loop that no longer references it.
        for (l, e) in loops.iter().zip(&edges) {
            self.radial_remove(*l, *e);
        }

        for (i, &l) in loops.iter().enumerate() {
            let previous = (i + n - 1) % n;
            self.loops[l].next = loops[previous];
            self.loops[l].prev = loops[(i + 1) % n];
            // Loop `i` used to run from `v_i` to `v_{i+1}`; reversed it runs
            // from `v_i` to `v_{i-1}`, which is the edge the previous loop had.
            self.loops[l].edge = edges[previous];
        }
        for (i, &l) in loops.iter().enumerate() {
            self.radial_insert(l, edges[(i + n - 1) % n]);
        }
        true
    }

    /// Moves one of a face's corners onto a different vertex.
    ///
    /// The primitive behind split and rip. In place, keeping the loop: a corner's
    /// UV belongs to (this face, this corner), so reassigning which vertex the
    /// corner sits on preserves it. Rebuilding the face instead would be simpler
    /// and would clear the per-corner data of everything it touches — which on a
    /// split means every face along the seam loses its UVs, and the user finds
    /// out when the texture tears.
    ///
    /// Returns `false` and changes nothing if the result would be degenerate:
    /// the new vertex already being a neighbour in this face would fold the
    /// corner onto an edge.
    pub fn replace_vert_in_face(&mut self, f: FaceId, old: VertId, new: VertId) -> bool {
        if old == new || self.verts.get(new).is_none() || self.faces.get(f).is_none() {
            return false;
        }
        let loops = self.face_loops(f);
        let Some(&l) = loops.iter().find(|&&l| self.loop_vert(l) == Some(old)) else {
            return false;
        };
        let (Some(previous), Some(next)) = (self.loop_prev(l), self.loop_next(l)) else {
            return false;
        };
        let (Some(before), Some(after)) = (self.loop_vert(previous), self.loop_vert(next)) else {
            return false;
        };
        if before == new || after == new {
            return false;
        }

        let leaving = self.loops[l].edge;
        let arriving = self.loops[previous].edge;
        let Ok(new_leaving) = self.add_edge(new, after) else {
            return false;
        };
        let Ok(new_arriving) = self.add_edge(before, new) else {
            return false;
        };

        // Out of the old radial cycles before the edges change, or a cycle ends
        // up holding a loop that no longer references it.
        self.radial_remove(l, leaving);
        self.radial_remove(previous, arriving);
        self.loops[l].vert = new;
        self.loops[l].edge = new_leaving;
        self.loops[previous].edge = new_arriving;
        self.radial_insert(l, new_leaving);
        self.radial_insert(previous, new_arriving);
        true
    }

    // --- removal ------------------------------------------------------------

    /// Removes a face, leaving its edges and vertices behind.
    ///
    /// The leftovers are usually wanted — deleting one face of a cube should not
    /// take the hole's border with it — and [`BMesh::remove_loose`] cleans them
    /// up when they are not.
    pub fn remove_face(&mut self, f: FaceId) -> bool {
        if !self.faces.contains(f) {
            return false;
        }
        for l in self.face_loops(f) {
            let Some(node) = self.loops.get(l).copied() else {
                continue;
            };
            self.radial_remove(l, node.edge);
            self.loops.remove(l);
        }
        self.faces.remove(f).is_some()
    }

    /// Removes an edge and every face that used it.
    pub fn remove_edge(&mut self, e: EdgeId) -> bool {
        if !self.edges.contains(e) {
            return false;
        }
        for f in self.edge_faces(e) {
            self.remove_face(f);
        }
        let verts = self.edges[e].verts;
        self.disk_remove(e, verts[0]);
        self.disk_remove(e, verts[1]);
        self.edges.remove(e).is_some()
    }

    /// Removes a vertex, every edge on it, and every face on those edges.
    pub fn remove_vert(&mut self, v: VertId) -> bool {
        if !self.verts.contains(v) {
            return false;
        }
        for e in self.vert_edges(v) {
            self.remove_edge(e);
        }
        self.verts.remove(v).is_some()
    }

    /// Removes wire edges and isolated vertices.
    ///
    /// Returns how many of each went. Separate from the removal operations
    /// because "delete these faces and tidy up" and "delete these faces" are
    /// both things a user asks for, and which one they meant is the caller's
    /// business.
    pub fn remove_loose(&mut self) -> (usize, usize) {
        let wire: Vec<EdgeId> = self
            .edge_ids()
            .filter(|&e| self.is_wire_edge(e))
            .collect::<Vec<_>>();
        for e in &wire {
            self.remove_edge(*e);
        }
        let isolated: Vec<VertId> = self
            .vert_ids()
            .filter(|&v| self.verts[v].edge.is_none())
            .collect::<Vec<_>>();
        for v in &isolated {
            self.verts.remove(*v);
        }
        (wire.len(), isolated.len())
    }

    // --- cycle maintenance --------------------------------------------------

    /// Which of an edge's two disk links belongs to `v`.
    fn disk_index(&self, e: EdgeId, v: VertId) -> Option<usize> {
        let edge = self.edges.get(e)?;
        if edge.verts[0] == v {
            Some(0)
        } else if edge.verts[1] == v {
            Some(1)
        } else {
            None
        }
    }

    fn disk_insert(&mut self, e: EdgeId, v: VertId) {
        let Some(i) = self.disk_index(e, v) else {
            return;
        };
        match self.verts[v].edge {
            None => {
                self.verts[v].edge = Some(e);
                self.edges[e].disk[i] = DiskLink { prev: e, next: e };
            }
            Some(first) => {
                let Some(j) = self.disk_index(first, v) else {
                    return;
                };
                let next = self.edges[first].disk[j].next;
                let Some(k) = self.disk_index(next, v) else {
                    return;
                };
                // Written through the arena one field at a time so that the
                // single-edge case, where `first == next` and `j == k`, updates
                // both halves of the same link rather than one clobbering the
                // other.
                self.edges[first].disk[j].next = e;
                self.edges[next].disk[k].prev = e;
                self.edges[e].disk[i] = DiskLink { prev: first, next };
            }
        }
    }

    fn disk_remove(&mut self, e: EdgeId, v: VertId) {
        let Some(i) = self.disk_index(e, v) else {
            return;
        };
        let link = self.edges[e].disk[i];
        if link.next == e {
            // The only edge on this vertex.
            self.verts[v].edge = None;
            return;
        }
        if let Some(j) = self.disk_index(link.prev, v) {
            self.edges[link.prev].disk[j].next = link.next;
        }
        if let Some(k) = self.disk_index(link.next, v) {
            self.edges[link.next].disk[k].prev = link.prev;
        }
        if self.verts[v].edge == Some(e) {
            self.verts[v].edge = Some(link.next);
        }
    }

    fn radial_insert(&mut self, l: LoopId, e: EdgeId) {
        match self.edges[e].first_loop {
            None => {
                self.edges[e].first_loop = Some(l);
                self.loops[l].radial_next = l;
                self.loops[l].radial_prev = l;
            }
            Some(first) => {
                let next = self.loops[first].radial_next;
                self.loops[first].radial_next = l;
                self.loops[next].radial_prev = l;
                self.loops[l].radial_prev = first;
                self.loops[l].radial_next = next;
            }
        }
    }

    fn radial_remove(&mut self, l: LoopId, e: EdgeId) {
        let Some(node) = self.loops.get(l).copied() else {
            return;
        };
        if node.radial_next == l {
            self.edges[e].first_loop = None;
            return;
        }
        self.loops[node.radial_prev].radial_next = node.radial_next;
        self.loops[node.radial_next].radial_prev = node.radial_prev;
        if self.edges[e].first_loop == Some(l) {
            self.edges[e].first_loop = Some(node.radial_next);
        }
    }

    /// Keeps the attribute arrays parallel to the arenas after an insert.
    ///
    /// A reused slot is reset to the layer's default, so a new element never
    /// inherits the UVs or the selection state of whatever was deleted there —
    /// a bug that only shows up after a delete and reads as random corruption.
    fn grow_domain(&mut self, domain: Domain, slot: u32) {
        let needed = self.slot_count(domain);
        self.attributes.resize_domain(domain, needed);
        self.attributes.reset_at(domain, slot as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles sharing one edge.
    fn two_tris() -> (BMesh, Vec<VertId>) {
        let mut m = BMesh::new();
        let v: Vec<VertId> = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
        .iter()
        .map(|p| m.add_vert(*p))
        .collect();
        m.add_face(&[v[0], v[1], v[2]]).unwrap();
        m.add_face(&[v[0], v[2], v[3]]).unwrap();
        (m, v)
    }

    #[test]
    fn an_empty_mesh_is_valid() {
        let m = BMesh::new();
        assert!(m.is_empty());
        assert_eq!(m.validate(), Vec::new());
    }

    #[test]
    fn a_single_triangle_wires_up_every_cycle() {
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::ZERO);
        let b = m.add_vert(Vec3::X);
        let c = m.add_vert(Vec3::Y);
        let f = m.add_face(&[a, b, c]).unwrap();

        assert_eq!(m.num_verts(), 3);
        assert_eq!(m.num_edges(), 3);
        assert_eq!(m.num_loops(), 3);
        assert_eq!(m.num_faces(), 1);
        assert_eq!(m.face_verts(f), vec![a, b, c]);
        assert_eq!(m.vert_edge_count(a), 2);
        for e in m.edge_ids() {
            assert!(m.is_boundary_edge(e), "a lone triangle is all boundary");
        }
        m.assert_valid();
    }

    #[test]
    fn a_shared_edge_holds_two_loops() {
        let (m, v) = two_tris();
        let shared = m.edge_between(v[0], v[2]).expect("the diagonal");
        assert_eq!(m.edge_face_count(shared), 2);
        assert!(m.is_manifold_edge(shared));
        assert!(!m.is_boundary_edge(shared));
        assert_eq!(m.num_edges(), 5, "four sides and one diagonal");
        m.assert_valid();
    }

    #[test]
    fn a_third_face_on_one_edge_is_an_ordinary_case() {
        // The reason for a radial cycle instead of a half-edge twin: this is not
        // an error state, it is what `knife` and `dissolve` produce halfway
        // through, and a structure that cannot hold it forces every operator to
        // reach its final state in a single step.
        let (mut m, v) = two_tris();
        let d = m.add_vert(Vec3::new(0.5, 0.5, 1.0));
        m.add_face(&[v[0], v[2], d]).unwrap();

        let shared = m.edge_between(v[0], v[2]).unwrap();
        assert_eq!(m.edge_face_count(shared), 3);
        assert!(!m.is_manifold_edge(shared));
        m.assert_valid();
    }

    #[test]
    fn an_edge_is_never_duplicated() {
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::ZERO);
        let b = m.add_vert(Vec3::X);
        let first = m.add_edge(a, b).unwrap();
        let second = m.add_edge(b, a).unwrap();
        assert_eq!(first, second, "reversed endpoints are the same edge");
        assert_eq!(m.num_edges(), 1);
        m.assert_valid();
    }

    #[test]
    fn a_wire_edge_and_an_isolated_vertex_are_representable() {
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::ZERO);
        let b = m.add_vert(Vec3::X);
        let _lonely = m.add_vert(Vec3::Y);
        let e = m.add_edge(a, b).unwrap();
        assert!(m.is_wire_edge(e));
        assert_eq!(m.edge_face_count(e), 0);
        m.assert_valid();

        let (wires, isolated) = m.remove_loose();
        assert_eq!((wires, isolated), (1, 3), "the wire takes its ends with it");
        assert!(m.is_empty());
    }

    #[test]
    fn degenerate_faces_are_refused_rather_than_stored() {
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::ZERO);
        let b = m.add_vert(Vec3::X);
        assert_eq!(m.add_face(&[a, b]), Err(BmError::FaceTooShort { given: 2 }));
        assert_eq!(
            m.add_face(&[a, b, a]),
            Err(BmError::RepeatedVert { vert: a })
        );
        assert_eq!(m.add_edge(a, a), Err(BmError::SelfEdge { vert: a }));
        assert_eq!(m.num_faces(), 0);
        assert_eq!(
            m.num_edges(),
            0,
            "the checks run before anything is created, so a refusal leaves no              half-built face and no stray edges behind"
        );
        m.assert_valid();
    }

    #[test]
    fn removing_a_face_leaves_the_hole_bordered() {
        let (mut m, v) = two_tris();
        let f = m.face_ids().next().unwrap();
        assert!(m.remove_face(f));

        assert_eq!(m.num_faces(), 1);
        assert_eq!(m.num_verts(), 4, "no vertex is taken with it");
        assert_eq!(m.num_edges(), 5, "nor any edge");
        let shared = m.edge_between(v[0], v[2]).unwrap();
        assert_eq!(m.edge_face_count(shared), 1, "the diagonal is now boundary");
        m.assert_valid();

        let (wires, isolated) = m.remove_loose();
        assert_eq!(
            (wires, isolated),
            (2, 1),
            "two sides of the deleted triangle, and the corner they alone held"
        );
        m.assert_valid();
    }

    #[test]
    fn removing_an_edge_takes_its_faces_with_it() {
        let (mut m, v) = two_tris();
        let shared = m.edge_between(v[0], v[2]).unwrap();
        assert!(m.remove_edge(shared));
        assert_eq!(m.num_faces(), 0);
        assert_eq!(m.num_loops(), 0);
        assert_eq!(m.num_edges(), 4);
        assert_eq!(m.num_verts(), 4);
        m.assert_valid();
    }

    #[test]
    fn removing_a_vertex_unravels_only_what_touches_it() {
        let (mut m, v) = two_tris();
        // v[1] belongs to one triangle only.
        assert!(m.remove_vert(v[1]));
        assert_eq!(m.num_verts(), 3);
        assert_eq!(m.num_faces(), 1, "the far triangle survives");
        assert!(m.edge_between(v[0], v[2]).is_some());
        m.assert_valid();
    }

    #[test]
    fn the_disk_cycle_stays_closed_through_removals() {
        // The disk splice is the easiest thing here to get subtly wrong, and a
        // broken one shows up as an operator silently missing edges rather than
        // as a crash. Grow a fan and then take it apart one edge at a time.
        let mut m = BMesh::new();
        let hub = m.add_vert(Vec3::ZERO);
        let spokes: Vec<VertId> = (0..8)
            .map(|i| m.add_vert(Vec3::new(i as f32, 1.0, 0.0)))
            .collect();
        for s in &spokes {
            m.add_edge(hub, *s).unwrap();
        }
        assert_eq!(m.vert_edge_count(hub), 8);
        m.assert_valid();

        for (i, s) in spokes.iter().enumerate() {
            let e = m.edge_between(hub, *s).unwrap();
            m.remove_edge(e);
            assert_eq!(m.vert_edge_count(hub), 7 - i);
            m.assert_valid();
        }
        assert!(m.vert(hub).is_some(), "the hub itself is untouched");
    }

    #[test]
    fn a_reused_slot_does_not_inherit_the_old_element_s_attributes() {
        // Arenas reuse freed slots, and attribute layers are indexed by slot. A
        // new vertex landing in a deleted one must not arrive pre-selected.
        use crate::attribute::{names, AttrType};
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::ZERO);
        let points = m.slot_count(Domain::Point);
        m.attributes_mut()
            .ensure(names::SELECT_VERT, Domain::Point, AttrType::Bool, points);
        m.attributes_mut()
            .make_mut_bool(names::SELECT_VERT)
            .unwrap()[a.index() as usize] = true;

        m.remove_vert(a);
        let b = m.add_vert(Vec3::X);
        assert_eq!(b.index(), a.index(), "the slot really was reused");
        let sel = m.attributes().get_bool(names::SELECT_VERT).unwrap();
        assert!(!sel[b.index() as usize], "the new vertex must start clean");
        assert!(matches!(
            m.attributes().get(names::SELECT_VERT).map(|l| l.data.ty()),
            Some(AttrType::Bool)
        ));
        m.assert_valid();
    }

    #[test]
    fn a_cube_has_the_topology_a_cube_should() {
        let m = BMesh::from_mesh_data(&crate::primitives::cube(2.0));
        assert_eq!(m.num_verts(), 8);
        assert_eq!(m.num_edges(), 12);
        assert_eq!(m.num_faces(), 6);
        assert_eq!(m.num_loops(), 24);
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "a closed solid has no boundary");
        }
        for v in m.vert_ids() {
            assert_eq!(m.vert_edge_count(v), 3);
            assert_eq!(m.vert_faces(v).len(), 3);
        }
        // Euler: V - E + F = 2 for a genus-0 closed surface.
        assert_eq!(
            m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64,
            2
        );
        m.assert_valid();
    }
}
