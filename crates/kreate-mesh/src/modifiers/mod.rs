//! Modifiers: functions from a mesh to a mesh.
//!
//! Each one is `MeshData -> MeshData` and nothing else. No document, no
//! parameters beyond its own, no cache — the caller owns all of that. That is
//! what lets a modifier be tested against a mesh literal and what makes the
//! eventual node graph a different *scheduler* over the same functions rather
//! than a rewrite of them.
//!
//! # Attributes are the hard part, not the geometry
//!
//! Mirroring a cube is arithmetic. Carrying its UVs, creases, seams and material
//! indices through the mirror is where a modifier is either right or quietly
//! destructive, and it is invisible until someone textures the result. Every
//! modifier here builds an index map per domain and gathers the layers through
//! it, so a layer type added later needs no work in this module — which is the
//! whole argument for the domain model.
//!
//! # Edge attributes need a pairing, not a copy
//!
//! `MeshData` derives its edge list from its faces in a canonical order, so the
//! output's edge *indices* have nothing to do with the input's. Copying edge
//! layers positionally puts every crease and seam on the wrong edge — and looks
//! right on any mesh whose orders happen to agree, which is most small ones.
//! [`gather_edges_by_pair`] does the pairing instead.

pub mod array;
pub mod bevel;
pub mod boolean;
pub mod catmull_clark;
pub mod mirror;
pub mod path_array;
pub mod solidify;
pub mod subdivide;

pub use array::{array, ArrayParams};
pub use bevel::{bevel, BevelParams};
pub use boolean::{boolean, BooleanOp};
pub use mirror::{mirror, MirrorParams};
pub use path_array::{path_array, path_chain, PathArrayParams};
pub use solidify::{solidify, SolidifyParams};
pub use subdivide::{subdivide, SubdivideParams};

use crate::attribute::{AttributeStore, Domain};
use crate::MeshData;

/// Copies edge layers from `source` onto `output`, matching edges by the
/// vertices they join.
///
/// `out_to_source` takes an output vertex index to the source one it came from,
/// or `u32::MAX` for a vertex the modifier invented. That direction is the one
/// that works: a modifier can give one source vertex many output vertices — an
/// array's copies, a mirror's reflection — and then a source-to-output map can
/// only name one of them, silently creasing the first copy and leaving the rest
/// smooth. Read the other way round every copy of an edge finds its source, so a
/// crease on the half mirrors along with the geometry.
///
/// An output edge with no source keeps the layer's default: an edge the modifier
/// invented gets a zero crease rather than someone else's.
pub fn gather_edges_by_pair(output: &mut MeshData, source: &MeshData, out_to_source: &[u32]) {
    if source
        .attributes()
        .iter_domain(Domain::Edge)
        .next()
        .is_none()
    {
        return;
    }
    // Sorted pairs plus binary search rather than a hash map: this decides which
    // edge is creased, and a hasher with a per-run seed would make the answer
    // depend on which run it was.
    let mut pairs: Vec<([u32; 2], u32)> = source
        .edges()
        .iter()
        .enumerate()
        .map(|(i, [a, b])| {
            let mut key = [*a, *b];
            key.sort_unstable();
            (key, i as u32)
        })
        .collect();
    pairs.sort_unstable();
    // Relied on below: an output edge joining two copies of one source vertex
    // looks up `[v, v]` and finds nothing, so it needs no guard of its own.
    debug_assert!(
        pairs.iter().all(|([a, b], _)| a != b),
        "a MeshData edge comes from a face ring with distinct corners, so it is \
         never a self-loop"
    );

    let order: Vec<u32> = output
        .edges()
        .iter()
        .map(|[a, b]| {
            let source_a = out_to_source.get(*a as usize).copied().unwrap_or(u32::MAX);
            let source_b = out_to_source.get(*b as usize).copied().unwrap_or(u32::MAX);
            if source_a == u32::MAX || source_b == u32::MAX {
                return u32::MAX;
            }
            let mut key = [source_a, source_b];
            key.sort_unstable();
            match pairs.binary_search_by_key(&key, |(k, _)| *k) {
                Ok(i) => pairs[i].1,
                Err(_) => u32::MAX,
            }
        })
        .collect();

    for layer in source.attributes().iter_domain(Domain::Edge) {
        let gathered = layer.data.gather(&order);
        output
            .attributes_mut()
            .insert(&layer.name, Domain::Edge, gathered);
    }
}

/// Copies the layers of one domain through an index map.
///
/// `order[i]` is the source element the output's element `i` came from.
pub fn gather_domain_from(
    output: &mut AttributeStore,
    source: &AttributeStore,
    domain: Domain,
    order: &[u32],
) {
    for layer in source.iter_domain(domain) {
        let gathered = layer.data.gather(order);
        output.insert(&layer.name, domain, gathered);
    }
}

/// Copies Point, Corner and Face layers through their maps.
///
/// Edge layers are deliberately not here: they need a pairing rather than an
/// order, and mixing the two is how a crease ends up on the wrong edge.
pub fn carry_attributes(
    output: &mut MeshData,
    source: &MeshData,
    points: &[u32],
    corners: &[u32],
    faces: &[u32],
) {
    // Position is regenerated by every modifier, so it must not be gathered over
    // the top of what was just computed.
    let mut source_store = source.attributes().clone();
    source_store.remove(crate::attribute::names::POSITION);

    let store = output.attributes_mut();
    gather_domain_from(store, &source_store, Domain::Point, points);
    gather_domain_from(store, &source_store, Domain::Corner, corners);
    gather_domain_from(store, &source_store, Domain::Face, faces);
}
