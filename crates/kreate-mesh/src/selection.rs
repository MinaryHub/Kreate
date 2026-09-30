//! Element selection, on the document mesh.
//!
//! Selection lives in attribute layers — `.select_vert`, `.select_edge`,
//! `.select_face` — which is what makes it survive the round trip through edit
//! mode and share structurally under undo. This module reads and writes those
//! layers directly on a [`MeshData`], without building a [`crate::BMesh`].
//!
//! # Why not through the edit structure
//!
//! `BMesh` has all of this already, and going through it costs a full decode and
//! encode of the mesh: every array rebuilt, twice, for a click.
//!
//! That is not merely slow. The renderer decides what to re-upload by comparing
//! attribute arrays for pointer identity, and a rebuilt mesh shares *nothing* —
//! so a round trip through `BMesh` turns "three bits changed" into "re-upload a
//! million vertices", and defeats the one optimisation that makes selection on a
//! dense mesh affordable. Clicking and rubber-band dragging are the most
//! frequent things a user does in edit mode, and they were paying for a
//! conversion neither of them needs.
//!
//! Loop, ring and face-loop selection are a different matter: they walk radial
//! and disk cycles, which is exactly what `BMesh` is for, and they stay there.
//!
//! # The flush rules are the same rules
//!
//! Selection is expressed in one domain and has to be reflected in the others,
//! and the rules here mirror [`crate::BMesh::flush_selection`] exactly — because
//! two implementations that disagree would make a mesh's selection depend on
//! which operator last touched it.

use crate::attribute::{names, AttrData, Domain};
use crate::bmesh::SelectMode;
use crate::MeshData;
use std::sync::Arc;

/// The layer name for a domain's selection, if it has one.
fn layer_of(domain: Domain) -> Option<&'static str> {
    match domain {
        Domain::Point => Some(names::SELECT_VERT),
        Domain::Edge => Some(names::SELECT_EDGE),
        Domain::Face => Some(names::SELECT_FACE),
        // Corners are not selectable: a selection is of vertices, edges or
        // faces, and a corner is an implementation of a face's ring.
        Domain::Corner => None,
    }
}

/// Creates any missing selection layer, sized to the mesh.
///
/// Called by everything here, so a caller never has to. A layer that is missing
/// and a layer that is all `false` mean the same thing, and code that has to
/// know which it is gets one of the two cases wrong.
pub fn ensure_layers(mesh: &mut MeshData) {
    for domain in [Domain::Point, Domain::Edge, Domain::Face] {
        let Some(name) = layer_of(domain) else {
            continue;
        };
        let wanted = mesh.domain_len(domain);
        let present = mesh
            .attributes()
            .get_bool(name)
            .is_some_and(|layer| layer.len() == wanted);
        if !present {
            mesh.attributes_mut().insert(
                name,
                domain,
                AttrData::Bool(Arc::new(vec![false; wanted])),
            );
        }
    }
}

/// Whether one element is selected.
pub fn is_selected(mesh: &MeshData, domain: Domain, index: u32) -> bool {
    layer_of(domain)
        .and_then(|name| mesh.attributes().get_bool(name))
        .and_then(|layer| layer.get(index as usize).copied())
        .unwrap_or(false)
}

/// Selects or deselects one element.
///
/// An index the mesh does not have is ignored: an id from the GPU buffer can be
/// a frame or two old, and a stale click is normal rather than an error.
pub fn set_selected(mesh: &mut MeshData, domain: Domain, index: u32, on: bool) {
    ensure_layers(mesh);
    let Some(name) = layer_of(domain) else { return };
    if let Some(layer) = mesh.attributes_mut().make_mut_bool(name) {
        if let Some(slot) = layer.get_mut(index as usize) {
            *slot = on;
        }
    }
}

/// Every selected element of a domain, in index order.
pub fn selected(mesh: &MeshData, domain: Domain) -> Vec<u32> {
    layer_of(domain)
        .and_then(|name| mesh.attributes().get_bool(name))
        .map(|layer| {
            layer
                .iter()
                .enumerate()
                .filter(|(_, on)| **on)
                .map(|(i, _)| i as u32)
                .collect()
        })
        .unwrap_or_default()
}

/// Whether anything at all is selected.
pub fn any_selected(mesh: &MeshData) -> bool {
    [Domain::Point, Domain::Edge, Domain::Face].iter().any(|d| {
        layer_of(*d)
            .and_then(|name| mesh.attributes().get_bool(name))
            .is_some_and(|layer| layer.iter().any(|on| *on))
    })
}

/// Sets every element of every domain — every *visible* one: a hidden
/// element is never selected.
fn set_all(mesh: &mut MeshData, on: bool) {
    ensure_layers(mesh);
    for domain in [Domain::Point, Domain::Edge, Domain::Face] {
        let Some(name) = layer_of(domain) else {
            continue;
        };
        if let Some(layer) = mesh.attributes_mut().make_mut_bool(name) {
            layer.fill(on);
        }
    }
    if on {
        deselect_hidden(mesh);
    }
}

pub fn select_all(mesh: &mut MeshData) {
    set_all(mesh, true);
}

pub fn select_none(mesh: &mut MeshData) {
    set_all(mesh, false);
}

/// Inverts the selection in one domain, then reflects it into the others.
pub fn select_invert(mesh: &mut MeshData, mode: SelectMode) {
    let corner_edges = mesh.corner_edges();
    select_invert_with(mesh, mode, &corner_edges);
}

/// The same, with the corner-to-edge map supplied.
pub fn select_invert_with(mesh: &mut MeshData, mode: SelectMode, corner_edges: &[u32]) {
    ensure_layers(mesh);
    let domain = mode.domain();
    let hidden = hidden_flags(mesh, domain);
    if let Some(name) = layer_of(domain) {
        if let Some(layer) = mesh.attributes_mut().make_mut_bool(name) {
            for (i, slot) in layer.iter_mut().enumerate() {
                // What is out of sight is not part of "everything else".
                *slot = !*slot && !hidden.get(i).copied().unwrap_or(false);
            }
        }
    }
    flush_with(mesh, mode, corner_edges);
}

/// Reflects a selection made in `mode` into the other two domains.
///
/// Mirrors [`crate::BMesh::flush_selection`] rule for rule. Two implementations
/// that disagree would make a mesh's selection depend on which operator last
/// touched it.
///
/// Derives the corner-to-edge map itself, which costs a pass over the mesh. A
/// caller that flushes repeatedly — which is every click and every frame of a
/// rubber-band drag — should hold the map and use [`flush_with`]: at a million
/// faces, rebuilding it is four hundred milliseconds and the flush itself is
/// two.
pub fn flush(mesh: &mut MeshData, mode: SelectMode) {
    let corner_edges = mesh.corner_edges();
    flush_with(mesh, mode, &corner_edges);
}

/// The same, with the corner-to-edge map supplied.
///
/// `corner_edges` must be [`MeshData::corner_edges`] for this mesh. It is
/// topology, so it survives any number of selection changes — which is the
/// whole reason it is worth holding on to.
pub fn flush_with(mesh: &mut MeshData, mode: SelectMode, corner_edges: &[u32]) {
    ensure_layers(mesh);
    match mode {
        SelectMode::Vert => flush_from_verts(mesh),
        SelectMode::Edge => flush_from_edges(mesh, corner_edges),
        SelectMode::Face => flush_from_faces(mesh, corner_edges),
    }
    deselect_hidden(mesh);
}

/// An edge is selected when both endpoints are; a face when all its corners are.
fn flush_from_verts(mesh: &mut MeshData) {
    let verts = selected_flags(mesh, Domain::Point);
    let on = |v: u32| verts.get(v as usize).copied().unwrap_or(false);

    let edges: Vec<bool> = mesh.edges().iter().map(|[a, b]| on(*a) && on(*b)).collect();
    let faces: Vec<bool> = (0..mesh.num_faces())
        .map(|f| {
            let ring = mesh.face_verts(f);
            !ring.is_empty() && ring.iter().all(|v| on(*v))
        })
        .collect();

    write(mesh, Domain::Edge, edges);
    write(mesh, Domain::Face, faces);
}

/// Endpoints follow the edges, and faces follow the *edges* — never back through
/// the vertices.
///
/// Re-deriving edges from vertices here is the bug that looks like a
/// simplification: selecting a ring of edges around a tube selects every vertex
/// on it, and every perpendicular edge then has both endpoints selected too. The
/// user asked for a ring and gets a band, and the tool that runs next acts on a
/// shape nobody chose.
fn flush_from_edges(mesh: &mut MeshData, corner_edges: &[u32]) {
    let edges = selected_flags(mesh, Domain::Edge);
    let edge_on = |e: u32| edges.get(e as usize).copied().unwrap_or(false);

    let mut verts = vec![false; mesh.num_verts() as usize];
    for (e, [a, b]) in mesh.edges().iter().enumerate() {
        if edge_on(e as u32) {
            verts[*a as usize] = true;
            verts[*b as usize] = true;
        }
    }

    let faces: Vec<bool> = (0..mesh.num_faces())
        .map(|f| {
            let corners = mesh.face_corner_range(f);
            !corners.is_empty()
                && corners
                    .clone()
                    .all(|c| corner_edges.get(c).is_some_and(|e| edge_on(*e)))
        })
        .collect();

    write(mesh, Domain::Point, verts);
    write(mesh, Domain::Face, faces);
}

/// A selected face marks its own vertices and edges, and nothing is cleared.
///
/// Additive, matching the edit structure: deselecting a face does not deselect
/// the vertices its neighbours still use.
fn flush_from_faces(mesh: &mut MeshData, corner_edges: &[u32]) {
    let faces = selected_flags(mesh, Domain::Face);

    let mut verts = selected_flags(mesh, Domain::Point);
    let mut edges = selected_flags(mesh, Domain::Edge);
    verts.resize(mesh.num_verts() as usize, false);
    edges.resize(mesh.num_edges() as usize, false);

    for f in 0..mesh.num_faces() {
        if !faces.get(f as usize).copied().unwrap_or(false) {
            continue;
        }
        for v in mesh.face_verts(f) {
            if let Some(slot) = verts.get_mut(*v as usize) {
                *slot = true;
            }
        }
        for c in mesh.face_corner_range(f) {
            if let Some(e) = corner_edges.get(c) {
                if let Some(slot) = edges.get_mut(*e as usize) {
                    *slot = true;
                }
            }
        }
    }

    write(mesh, Domain::Point, verts);
    write(mesh, Domain::Edge, edges);
}

fn selected_flags(mesh: &MeshData, domain: Domain) -> Vec<bool> {
    layer_of(domain)
        .and_then(|name| mesh.attributes().get_bool(name))
        .map(|layer| layer.to_vec())
        .unwrap_or_default()
}

fn write(mesh: &mut MeshData, domain: Domain, values: Vec<bool>) {
    let Some(name) = layer_of(domain) else { return };
    mesh.attributes_mut()
        .insert(name, domain, AttrData::Bool(Arc::new(values)));
}

// --- hiding -------------------------------------------------------------------

/// The hide layer for a domain, if it has one.
fn hide_layer_of(domain: Domain) -> Option<&'static str> {
    match domain {
        Domain::Point => Some(names::HIDE_VERT),
        Domain::Edge => Some(names::HIDE_EDGE),
        Domain::Face => Some(names::HIDE_FACE),
        Domain::Corner => None,
    }
}

/// Which elements of a domain are hidden, one flag each; all `false` when the
/// mesh has hidden nothing.
pub fn hidden_flags(mesh: &MeshData, domain: Domain) -> Vec<bool> {
    let n = mesh.domain_len(domain);
    hide_layer_of(domain)
        .and_then(|name| mesh.attributes().get_bool(name))
        .filter(|layer| layer.len() == n)
        .map(|layer| layer.to_vec())
        .unwrap_or_else(|| vec![false; n])
}

/// Whether anything is hidden.
pub fn any_hidden(mesh: &MeshData) -> bool {
    [Domain::Point, Domain::Edge, Domain::Face].iter().any(|d| {
        hide_layer_of(*d)
            .and_then(|name| mesh.attributes().get_bool(name))
            .is_some_and(|layer| layer.iter().any(|on| *on))
    })
}

/// Takes hidden elements out of the selection. A no-op on a mesh that hides
/// nothing, which is nearly every mesh, and costs nothing there.
pub fn deselect_hidden(mesh: &mut MeshData) {
    if !any_hidden(mesh) {
        return;
    }
    for domain in [Domain::Point, Domain::Edge, Domain::Face] {
        let hidden = hidden_flags(mesh, domain);
        let Some(name) = layer_of(domain) else { continue };
        if let Some(layer) = mesh.attributes_mut().make_mut_bool(name) {
            for (slot, h) in layer.iter_mut().zip(&hidden) {
                *slot &= !*h;
            }
        }
    }
}

fn write_hidden(mesh: &mut MeshData, domain: Domain, values: Vec<bool>) {
    let Some(name) = hide_layer_of(domain) else { return };
    mesh.attributes_mut().insert(name, domain, AttrData::Bool(Arc::new(values)));
}

/// Hides the selection — or with `unselected`, everything else — as `H` and
/// `Shift+H` do, in the terms of the select mode. `false` when that hides
/// nothing new.
///
/// Blender's rules, which keep what is left a whole mesh: hiding vertices
/// hides every edge and face that touches them; hiding edges hides the faces
/// they bound and the vertices left with no visible edge; hiding faces hides
/// the edges left with no visible face and the vertices left with no visible
/// edge. What is hidden is also deselected.
pub fn hide(mesh: &mut MeshData, mode: SelectMode, unselected: bool, corner_edges: &[u32]) -> bool {
    ensure_layers(mesh);
    let domain = mode.domain();
    let was = hidden_flags(mesh, domain);
    let chosen: Vec<bool> = selected_flags(mesh, domain)
        .iter()
        .zip(&was)
        .map(|(s, h)| *h || (*s != unselected))
        .collect();
    if chosen == was {
        return false;
    }
    let mut verts = hidden_flags(mesh, Domain::Point);
    let mut edges = hidden_flags(mesh, Domain::Edge);
    let mut faces = hidden_flags(mesh, Domain::Face);
    let nf = mesh.num_faces();
    match mode {
        SelectMode::Vert => {
            verts = chosen;
            for (e, [a, b]) in mesh.edges().iter().enumerate() {
                edges[e] |= verts[*a as usize] || verts[*b as usize];
            }
            for f in 0..nf {
                faces[f as usize] |= mesh.face_verts(f).iter().any(|v| verts[*v as usize]);
            }
        }
        SelectMode::Edge => {
            edges = chosen;
            for f in 0..nf {
                faces[f as usize] |= mesh.face_corner_range(f).any(|c| edges[corner_edges[c] as usize]);
            }
            verts = hide_bare_verts(mesh, &edges, verts);
        }
        SelectMode::Face => {
            faces = chosen;
            // An edge is left in sight while any face on it is, or when it
            // bounds no face at all.
            let mut visible_face = vec![false; edges.len()];
            let mut any_face = vec![false; edges.len()];
            for f in 0..nf {
                for c in mesh.face_corner_range(f) {
                    let e = corner_edges[c] as usize;
                    any_face[e] = true;
                    visible_face[e] |= !faces[f as usize];
                }
            }
            for e in 0..edges.len() {
                edges[e] |= any_face[e] && !visible_face[e];
            }
            verts = hide_bare_verts(mesh, &edges, verts);
        }
    }
    write_hidden(mesh, Domain::Point, verts);
    write_hidden(mesh, Domain::Edge, edges);
    write_hidden(mesh, Domain::Face, faces);
    deselect_hidden(mesh);
    // What is left selected is what the mode's own elements still say: the
    // corners a hidden face shared with nothing selected go with it.
    if mode == SelectMode::Face {
        let (nv, ne) = (mesh.num_verts() as usize, mesh.num_edges() as usize);
        write(mesh, Domain::Point, vec![false; nv]);
        write(mesh, Domain::Edge, vec![false; ne]);
    }
    flush_with(mesh, mode, corner_edges);
    true
}

/// Hides the vertices whose every edge is hidden. A vertex on no edge at all is
/// left alone.
fn hide_bare_verts(mesh: &MeshData, edges: &[bool], mut verts: Vec<bool>) -> Vec<bool> {
    let n = mesh.num_verts() as usize;
    let mut visible = vec![false; n];
    let mut any = vec![false; n];
    for (e, [a, b]) in mesh.edges().iter().enumerate() {
        for v in [*a as usize, *b as usize] {
            any[v] = true;
            visible[v] |= !edges[e];
        }
    }
    for v in 0..n {
        verts[v] |= any[v] && !visible[v];
    }
    verts
}

/// Brings everything hidden back, selected — `Alt+H`, which in Blender leaves
/// what came back selected so the next key acts on it. `false` when nothing
/// was hidden.
pub fn reveal(mesh: &mut MeshData) -> bool {
    if !any_hidden(mesh) {
        return false;
    }
    ensure_layers(mesh);
    for domain in [Domain::Point, Domain::Edge, Domain::Face] {
        let hidden = hidden_flags(mesh, domain);
        if let Some(name) = layer_of(domain) {
            if let Some(layer) = mesh.attributes_mut().make_mut_bool(name) {
                for (slot, h) in layer.iter_mut().zip(&hidden) {
                    *slot |= *h;
                }
            }
        }
        if let Some(name) = hide_layer_of(domain) {
            mesh.attributes_mut().remove(name);
        }
    }
    true
}

/// The mesh without its hide layers: what object mode draws.
pub fn without_hidden(mesh: &MeshData) -> MeshData {
    let mut out = mesh.clone();
    for domain in [Domain::Point, Domain::Edge, Domain::Face] {
        if let Some(name) = hide_layer_of(domain) {
            out.attributes_mut().remove(name);
        }
    }
    out
}

// --- growing and shrinking ----------------------------------------------------

/// Grows the selection by one step — `Ctrl+Numpad+`. Blender's "face step":
/// every face touching a selected vertex comes in whole, and so does the far
/// end of every edge that does, so a wire mesh grows too. Hidden elements stay
/// out. `false` when nothing was added.
pub fn select_more(mesh: &mut MeshData, mode: SelectMode, corner_edges: &[u32]) -> bool {
    ensure_layers(mesh);
    let verts = seed_verts(mesh, mode);
    let hidden_faces = hidden_flags(mesh, Domain::Face);
    let hidden_edges = hidden_flags(mesh, Domain::Edge);
    let before = (selected(mesh, Domain::Point).len(), selected(mesh, Domain::Face).len());
    match mode {
        SelectMode::Face => {
            let mut faces = selected_flags(mesh, Domain::Face);
            for f in 0..mesh.num_faces() {
                if !hidden_faces[f as usize] && mesh.face_verts(f).iter().any(|v| verts[*v as usize]) {
                    faces[f as usize] = true;
                }
            }
            write(mesh, Domain::Face, faces);
            flush_with(mesh, SelectMode::Face, corner_edges);
        }
        _ => {
            let mut grown = verts.clone();
            for f in 0..mesh.num_faces() {
                let ring = mesh.face_verts(f);
                if !hidden_faces[f as usize] && ring.iter().any(|v| verts[*v as usize]) {
                    for v in ring {
                        grown[*v as usize] = true;
                    }
                }
            }
            for (e, [a, b]) in mesh.edges().iter().enumerate() {
                if !hidden_edges[e] && (verts[*a as usize] || verts[*b as usize]) {
                    grown[*a as usize] = true;
                    grown[*b as usize] = true;
                }
            }
            write(mesh, Domain::Point, grown);
            flush_with(mesh, SelectMode::Vert, corner_edges);
        }
    }
    (selected(mesh, Domain::Point).len(), selected(mesh, Domain::Face).len()) != before
}

/// The vertices a grow or a shrink measures from: in face mode, the corners
/// of the selected faces — what face mode means by the selection, whatever the
/// vertex layer happens to say.
fn seed_verts(mesh: &MeshData, mode: SelectMode) -> Vec<bool> {
    if mode != SelectMode::Face {
        let mut verts = selected_flags(mesh, Domain::Point);
        verts.resize(mesh.num_verts() as usize, false);
        return verts;
    }
    let faces = selected_flags(mesh, Domain::Face);
    let mut verts = vec![false; mesh.num_verts() as usize];
    for f in 0..mesh.num_faces() {
        if faces.get(f as usize).copied().unwrap_or(false) {
            for v in mesh.face_verts(f) {
                verts[*v as usize] = true;
            }
        }
    }
    verts
}

/// Shrinks the selection by one step — `Ctrl+Numpad-`: what is on its edge
/// leaves. A selected vertex stays only if every face and edge around it is
/// wholly selected and it is not on the mesh's open border; in face mode a
/// face stays only if every face sharing a vertex with it is selected. `false` when nothing left.
pub fn select_less(mesh: &mut MeshData, mode: SelectMode, corner_edges: &[u32]) -> bool {
    ensure_layers(mesh);
    let verts = seed_verts(mesh, mode);
    let before = (selected(mesh, Domain::Point).len(), selected(mesh, Domain::Face).len());
    // A vertex on the rim: some face or edge around it has an unselected
    // vertex. Hidden neighbours count as outside, as in Blender.
    let mut rim = vec![false; verts.len()];
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        if ring.iter().any(|v| !verts[*v as usize]) {
            for v in ring {
                rim[*v as usize] = true;
            }
        }
    }
    // The mesh's own open border is a rim too, so a whole plane shrinks.
    let mut uses = vec![0u32; mesh.num_edges() as usize];
    for e in corner_edges {
        if let Some(u) = uses.get_mut(*e as usize) {
            *u += 1;
        }
    }
    for (e, [a, b]) in mesh.edges().iter().enumerate() {
        if verts[*a as usize] != verts[*b as usize] || uses[e] == 1 {
            rim[*a as usize] = true;
            rim[*b as usize] = true;
        }
    }
    match mode {
        SelectMode::Face => {
            // In faces: the rim is every corner of an unselected face.
            let selected_faces = selected_flags(mesh, Domain::Face);
            // The open border shrinks too, as in vertices.
            let mut rim: Vec<bool> = vec![false; verts.len()];
            for (e, [a, b]) in mesh.edges().iter().enumerate() {
                if uses[e] == 1 {
                    rim[*a as usize] = true;
                    rim[*b as usize] = true;
                }
            }
            for f in 0..mesh.num_faces() {
                if !selected_faces.get(f as usize).copied().unwrap_or(false) {
                    for v in mesh.face_verts(f) {
                        rim[*v as usize] = true;
                    }
                }
            }
            let faces: Vec<bool> = selected_faces
                .iter()
                .enumerate()
                .map(|(f, on)| *on && mesh.face_verts(f as u32).iter().all(|v| !rim[*v as usize]))
                .collect();
            let (nv, ne) = (mesh.num_verts() as usize, mesh.num_edges() as usize);
            write(mesh, Domain::Point, vec![false; nv]);
            write(mesh, Domain::Edge, vec![false; ne]);
            write(mesh, Domain::Face, faces);
            flush_with(mesh, SelectMode::Face, corner_edges);
        }
        _ => {
            let kept: Vec<bool> = verts.iter().zip(&rim).map(|(s, r)| *s && !*r).collect();
            write(mesh, Domain::Point, kept);
            flush_with(mesh, SelectMode::Vert, corner_edges);
        }
    }
    (selected(mesh, Domain::Point).len(), selected(mesh, Domain::Face).len()) != before
}

/// Selects every piece the selection touches — `Ctrl+L`: all that is joined to
/// a selected vertex by edges, through visible edges only. `false` when
/// nothing was added.
pub fn select_linked(mesh: &mut MeshData, corner_edges: &[u32]) -> bool {
    ensure_layers(mesh);
    let verts = selected_flags(mesh, Domain::Point);
    let hidden_edges = hidden_flags(mesh, Domain::Edge);
    let n = verts.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn root(parent: &mut [u32], mut v: u32) -> u32 {
        while parent[v as usize] != v {
            parent[v as usize] = parent[parent[v as usize] as usize];
            v = parent[v as usize];
        }
        v
    }
    for (e, [a, b]) in mesh.edges().iter().enumerate() {
        if hidden_edges[e] {
            continue;
        }
        let (ra, rb) = (root(&mut parent, *a), root(&mut parent, *b));
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }
    let mut reached = vec![false; n];
    for (v, on) in verts.iter().enumerate() {
        if *on {
            let r = root(&mut parent, v as u32);
            reached[r as usize] = true;
        }
    }
    let before = selected(mesh, Domain::Point).len();
    let grown: Vec<bool> = (0..n).map(|v| reached[root(&mut parent, v as u32) as usize]).collect();
    write(mesh, Domain::Point, grown);
    flush_with(mesh, SelectMode::Vert, corner_edges);
    selected(mesh, Domain::Point).len() != before
}

#[cfg(test)]
mod hide_tests {
    use super::*;
    use crate::primitives;

    fn grid() -> MeshData {
        primitives::grid(3.0, 3, 3)
    }

    #[test]
    fn hiding_a_face_hides_only_what_nothing_else_shows() {
        let mut mesh = primitives::cube(2.0);
        let ce = mesh.corner_edges();
        set_selected(&mut mesh, Domain::Face, 0, true);
        flush_with(&mut mesh, SelectMode::Face, &ce);
        assert!(hide(&mut mesh, SelectMode::Face, false, &ce));
        assert_eq!(hidden_flags(&mesh, Domain::Face).iter().filter(|h| **h).count(), 1);
        // A cube's face shares every edge and vertex with its neighbours.
        assert!(!hidden_flags(&mesh, Domain::Edge).iter().any(|h| *h));
        assert!(!hidden_flags(&mesh, Domain::Point).iter().any(|h| *h));
        assert!(!any_selected(&mesh), "what was hidden stayed selected");
        // Hidden elements do not come back with select all or invert.
        select_all(&mut mesh);
        assert!(!is_selected(&mesh, Domain::Face, 0));
        select_none(&mut mesh);
        select_invert_with(&mut mesh, SelectMode::Face, &ce);
        assert!(!is_selected(&mesh, Domain::Face, 0));
        assert_eq!(selected(&mesh, Domain::Face).len(), 5);
        // Alt+H brings it back, selected.
        assert!(reveal(&mut mesh));
        assert!(is_selected(&mesh, Domain::Face, 0));
        assert!(!any_hidden(&mesh));
        assert!(!reveal(&mut mesh));
    }

    #[test]
    fn hiding_a_vertex_hides_what_touches_it_and_shift_h_hides_the_rest() {
        let mut mesh = primitives::cube(2.0);
        let ce = mesh.corner_edges();
        set_selected(&mut mesh, Domain::Point, 0, true);
        assert!(hide(&mut mesh, SelectMode::Vert, false, &ce));
        assert_eq!(hidden_flags(&mesh, Domain::Edge).iter().filter(|h| **h).count(), 3);
        assert_eq!(hidden_flags(&mesh, Domain::Face).iter().filter(|h| **h).count(), 3);

        let mut mesh = primitives::cube(2.0);
        set_selected(&mut mesh, Domain::Face, 0, true);
        flush_with(&mut mesh, SelectMode::Face, &ce);
        assert!(hide(&mut mesh, SelectMode::Face, true, &ce));
        let hidden = hidden_flags(&mesh, Domain::Face);
        assert!(!hidden[0] && hidden.iter().filter(|h| **h).count() == 5);
        assert!(is_selected(&mesh, Domain::Face, 0), "the kept face lost its selection");
    }

    #[test]
    fn more_and_less_step_out_and_back() {
        let mut mesh = grid();
        let ce = mesh.corner_edges();
        // The middle face of a 3 by 3 grid.
        let middle = (0..mesh.num_faces())
            .find(|f| mesh.face_verts(*f).iter().all(|v| mesh.positions()[*v as usize].x.abs() < 1.0 && mesh.positions()[*v as usize].y.abs() < 1.0))
            .expect("a middle face");
        set_selected(&mut mesh, Domain::Face, middle, true);
        flush_with(&mut mesh, SelectMode::Face, &ce);
        assert!(select_more(&mut mesh, SelectMode::Face, &ce));
        assert_eq!(selected(&mesh, Domain::Face).len(), 9);
        assert!(!select_more(&mut mesh, SelectMode::Face, &ce), "there was nothing more");
        assert!(select_less(&mut mesh, SelectMode::Face, &ce));
        assert_eq!(selected(&mesh, Domain::Face), vec![middle]);
        assert_eq!(selected(&mesh, Domain::Point).len(), 4);
        // In vertex mode too: one vertex grows to its faces' corners.
        select_none(&mut mesh);
        let centre_vert = mesh.face_verts(middle)[0];
        set_selected(&mut mesh, Domain::Point, centre_vert, true);
        select_more(&mut mesh, SelectMode::Vert, &ce);
        assert_eq!(selected(&mesh, Domain::Point).len(), 9);
        select_less(&mut mesh, SelectMode::Vert, &ce);
        assert_eq!(selected(&mesh, Domain::Point), vec![centre_vert]);
    }

    #[test]
    fn linked_takes_the_whole_piece_and_nothing_else() {
        let mut a = primitives::cube(1.0);
        let b = primitives::cube(1.0);
        let n = a.num_verts();
        let part = |mesh| crate::edit::join::Part { mesh, transform: glam::DAffine3::IDENTITY };
        a = crate::edit::join::join(&[part(&a), part(&b)]).0;
        let ce = a.corner_edges();
        set_selected(&mut a, Domain::Point, 0, true);
        assert!(select_linked(&mut a, &ce));
        let sel = selected(&a, Domain::Point);
        assert_eq!(sel.len() as u32, n, "{sel:?}");
        assert!(sel.iter().all(|v| *v < n));
        assert_eq!(selected(&a, Domain::Face).len(), 6);
        assert!(!select_linked(&mut a, &ce));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn selecting_a_face_marks_its_vertices_and_edges() {
        let mut mesh = primitives::cube(2.0);
        set_selected(&mut mesh, Domain::Face, 0, true);
        flush(&mut mesh, SelectMode::Face);

        assert_eq!(selected(&mesh, Domain::Point).len(), 4, "a quad has four");
        assert_eq!(selected(&mesh, Domain::Edge).len(), 4);
        assert_eq!(selected(&mesh, Domain::Face), vec![0]);
    }

    #[test]
    fn selecting_vertices_closes_up_to_the_faces_they_bound() {
        let mut mesh = primitives::cube(2.0);
        for v in mesh.face_verts(0).to_vec() {
            set_selected(&mut mesh, Domain::Point, v, true);
        }
        flush(&mut mesh, SelectMode::Vert);
        assert_eq!(selected(&mesh, Domain::Face), vec![0]);
        assert_eq!(selected(&mesh, Domain::Edge).len(), 4);
    }

    #[test]
    fn a_partly_selected_face_is_not_selected() {
        // All its corners, or none. Otherwise an extrude acts on a face the user
        // only half-chose.
        let mut mesh = primitives::cube(2.0);
        let ring = mesh.face_verts(0).to_vec();
        for v in &ring[..2] {
            set_selected(&mut mesh, Domain::Point, *v, true);
        }
        flush(&mut mesh, SelectMode::Vert);
        assert!(selected(&mesh, Domain::Face).is_empty());
    }

    #[test]
    fn an_edge_selection_does_not_spread_sideways() {
        // The rule worth stating twice. Flushing edges down to vertices and then
        // *back up* would select every perpendicular edge too: the user asks for
        // a ring around a tube and gets a band.
        let mut mesh = primitives::cube(2.0);
        // The four edges of one face.
        let corner_edges = mesh.corner_edges();
        let ring: Vec<u32> = mesh.face_corner_range(0).map(|c| corner_edges[c]).collect();
        for e in &ring {
            set_selected(&mut mesh, Domain::Edge, *e, true);
        }
        flush(&mut mesh, SelectMode::Edge);

        assert_eq!(
            selected(&mesh, Domain::Edge).len(),
            4,
            "the four chosen edges, and no others: {:?}",
            selected(&mesh, Domain::Edge)
        );
        assert_eq!(selected(&mesh, Domain::Point).len(), 4);
        assert_eq!(selected(&mesh, Domain::Face), vec![0]);
    }

    #[test]
    fn select_all_and_none_reach_every_domain() {
        let mut mesh = primitives::cube(2.0);
        select_all(&mut mesh);
        assert_eq!(selected(&mesh, Domain::Point).len(), 8);
        assert_eq!(selected(&mesh, Domain::Edge).len(), 12);
        assert_eq!(selected(&mesh, Domain::Face).len(), 6);

        select_none(&mut mesh);
        assert!(!any_selected(&mesh));
    }

    #[test]
    fn inverting_leaves_the_other_domains_consistent() {
        let mut mesh = primitives::cube(2.0);
        set_selected(&mut mesh, Domain::Face, 0, true);
        flush(&mut mesh, SelectMode::Face);
        select_invert(&mut mesh, SelectMode::Face);

        assert_eq!(selected(&mesh, Domain::Face).len(), 5);
        // Every vertex of the cube touches at least one of the other five faces.
        assert_eq!(selected(&mesh, Domain::Point).len(), 8);
    }

    #[test]
    fn an_index_the_mesh_does_not_have_is_ignored() {
        // An id from the GPU buffer can be a frame or two old.
        let mut mesh = primitives::cube(2.0);
        set_selected(&mut mesh, Domain::Face, 9_999, true);
        assert!(!any_selected(&mesh));
    }

    #[test]
    fn the_layers_are_created_on_demand_and_sized_to_the_mesh() {
        let mut mesh = primitives::cube(2.0);
        assert!(mesh.attributes().get_bool(names::SELECT_VERT).is_none());
        ensure_layers(&mut mesh);
        assert_eq!(
            mesh.attributes()
                .get_bool(names::SELECT_VERT)
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            mesh.attributes()
                .get_bool(names::SELECT_EDGE)
                .unwrap()
                .len(),
            12
        );
        assert_eq!(
            mesh.attributes()
                .get_bool(names::SELECT_FACE)
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn a_layer_of_the_wrong_length_is_replaced() {
        // A mesh whose topology changed under a stale selection layer. Left
        // alone, every later index would be reading the wrong element.
        let mut mesh = primitives::cube(2.0);
        mesh.attributes_mut().insert(
            names::SELECT_FACE,
            Domain::Face,
            AttrData::Bool(Arc::new(vec![true; 2])),
        );
        ensure_layers(&mut mesh);
        assert_eq!(
            mesh.attributes()
                .get_bool(names::SELECT_FACE)
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn selecting_touches_only_the_selection_layers() {
        // The whole point of not going through the edit structure: the renderer
        // decides what to re-upload by comparing arrays for pointer identity, so
        // a click has to leave the geometry alone.
        let mesh = primitives::cube(2.0);
        let mut clicked = mesh.clone();
        set_selected(&mut clicked, Domain::Face, 2, true);
        flush(&mut clicked, SelectMode::Face);
        assert!(
            mesh.only_selection_differs(&clicked),
            "a click changed something other than the selection"
        );
    }

    #[test]
    fn the_rules_match_the_edit_structure() {
        // Two implementations of the same rules would drift, and a mesh's
        // selection would depend on which operator last touched it. Checked
        // against `BMesh` for every mode rather than trusted.
        for mode in [SelectMode::Vert, SelectMode::Edge, SelectMode::Face] {
            let base = primitives::cube(2.0);
            let domain = mode.domain();

            let mut direct = base.clone();
            ensure_layers(&mut direct);
            set_selected(&mut direct, domain, 0, true);
            flush(&mut direct, mode);

            let mut bm = base.to_bmesh();
            bm.ensure_selection_layers();
            match bm.handle_at(domain, 0).expect("element zero exists") {
                crate::bmesh::ElementId::Vert(v) => bm.set_vert_selected(v, true),
                crate::bmesh::ElementId::Edge(e) => bm.set_edge_selected(e, true),
                crate::bmesh::ElementId::Face(f) => bm.set_face_selected(f, true),
            }
            bm.flush_selection(mode);
            let through_bmesh = bm.to_mesh_data();

            for d in [Domain::Point, Domain::Edge, Domain::Face] {
                assert_eq!(
                    selected(&direct, d),
                    selected(&through_bmesh, d),
                    "{mode:?} disagreed in {d:?}"
                );
            }
        }
    }
}
