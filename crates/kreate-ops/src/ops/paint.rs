//! Vertex groups and weight painting.
//!
//! A group is a point attribute on the active object's mesh, named as
//! [`kreate_scene::skin::weight_layer`] names it: a group per bone is what
//! makes a mesh follow an armature, and a group made here for a bone's name is
//! one the bone picks up.
//!
//! # A stroke is one operator
//!
//! Blender's brush, on the operator model: a stroke is the vertices it passed
//! over, the weight it painted towards and how strongly — replayed, it paints
//! the same vertices the same way. The application runs it as a modal, so the
//! weights change under the brush while it moves and the whole stroke is one
//! step to undo.

use kreate_ops_derive::OpParams;
use kreate_mesh::attribute::{AttrData, Domain};
use kreate_scene::skin::{weight_layer, WEIGHT_PREFIX};
use kreate_scene::{DocMut, Document, MeshId, ObjectId};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddGroup>();
    registry.register::<RemoveGroup>();
    registry.register::<SetActiveGroup>();
    registry.register::<Stroke>();
    registry.register::<Normalize>();
}

/// The active object and its mesh, if it has one.
fn active_mesh(doc: &Document) -> Option<(ObjectId, MeshId)> {
    let id = doc.selection.active?;
    Some((id, doc.object(id)?.mesh()?))
}

/// The vertex groups of a mesh, by name without the prefix, in name order.
pub fn groups(doc: &Document, mesh: MeshId) -> Vec<String> {
    doc.mesh(mesh)
        .map(|m| {
            m.attributes()
                .iter()
                .filter(|l| l.domain == Domain::Point)
                .filter_map(|l| l.name.strip_prefix(WEIGHT_PREFIX).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The weights of one group of `mesh`, one per vertex; zeros when the mesh has
/// no such group.
pub fn weights(doc: &Document, mesh: MeshId, group: &str) -> Vec<f32> {
    let Some(m) = doc.mesh(mesh) else { return Vec::new() };
    let n = m.num_verts() as usize;
    match m.attributes().get(&weight_layer(group)).map(|l| &l.data) {
        Some(AttrData::F32(w)) if w.len() == n => w.to_vec(),
        _ => vec![0.0; n],
    }
}

fn has_mesh(_ctx: &Context, doc: &Document) -> bool {
    active_mesh(doc).is_some()
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddGroupParams {
    /// The group's name; empty for "Group", numbered if taken.
    pub name: String,
}

/// Adds an empty vertex group to the active mesh and makes it the active one.
pub struct AddGroup;

impl Operator for AddGroup {
    type Params = AddGroupParams;
    fn id() -> &'static str {
        "object.vertex_group_add"
    }
    fn label() -> &'static str {
        "Add Vertex Group"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((object, mesh)) = active_mesh(doc.document()) else { return Ok(OpResult::Cancelled) };
        let taken = groups(doc.document(), mesh);
        let base = if params.name.trim().is_empty() { "Group".to_string() } else { params.name.trim().to_string() };
        let name = if taken.contains(&base) {
            (1..).map(|i| format!("{base}.{i:03}")).find(|n| !taken.contains(n)).expect("names run out after integers")
        } else {
            base
        };
        let d = doc.document_mut();
        let Some(m) = d.mesh_mut(mesh) else { return Ok(OpResult::Cancelled) };
        let n = m.num_verts() as usize;
        m.attributes_mut().insert(&weight_layer(&name), Domain::Point, AttrData::F32(std::sync::Arc::new(vec![0.0; n])));
        if let Some(o) = d.object_mut(object) {
            o.active_group = Some(name);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct GroupParams {
    pub name: String,
}

/// Removes a vertex group, and with it that group's pull on the vertices.
pub struct RemoveGroup;

impl Operator for RemoveGroup {
    type Params = GroupParams;
    fn id() -> &'static str {
        "object.vertex_group_remove"
    }
    fn label() -> &'static str {
        "Remove Vertex Group"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((object, mesh)) = active_mesh(doc.document()) else { return Ok(OpResult::Cancelled) };
        if !groups(doc.document(), mesh).contains(&params.name) {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        if let Some(m) = d.mesh_mut(mesh) {
            m.attributes_mut().remove(&weight_layer(&params.name));
        }
        let left = groups(d, mesh);
        if let Some(o) = d.object_mut(object) {
            if o.active_group.as_deref() == Some(params.name.as_str()) {
                o.active_group = left.first().cloned();
            }
        }
        Ok(OpResult::Finished)
    }
}

/// Makes a vertex group the one painted and shown.
pub struct SetActiveGroup;

impl Operator for SetActiveGroup {
    type Params = GroupParams;
    fn id() -> &'static str {
        "object.vertex_group_set_active"
    }
    fn label() -> &'static str {
        "Set Active Vertex Group"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((object, mesh)) = active_mesh(doc.document()) else { return Ok(OpResult::Cancelled) };
        if !groups(doc.document(), mesh).contains(&params.name)
            || doc.document().object(object).and_then(|o| o.active_group.as_deref()) == Some(params.name.as_str())
        {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else { return Ok(OpResult::Cancelled) };
        o.active_group = Some(params.name.clone());
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct StrokeParams {
    /// The group painted.
    pub group: String,
    /// The vertices the brush passed over.
    pub vertices: Vec<u32>,
    /// What they are painted towards.
    #[param(min = 0.0, max = 1.0)]
    pub weight: f64,
    /// How far towards it: one sets it, a half goes halfway.
    #[param(min = 0.0, max = 1.0)]
    pub strength: f64,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self { group: String::new(), vertices: Vec::new(), weight: 1.0, strength: 1.0 }
    }
}

/// Paints the weight of one group on some vertices: one brush stroke.
///
/// Each vertex moves towards the weight once however often the brush crossed
/// it, which is what makes the stroke a function of its parameters. A group
/// not there yet is made, so painting a bone's name that has no group is how
/// the group starts.
pub struct Stroke;

impl Operator for Stroke {
    type Params = StrokeParams;
    fn id() -> &'static str {
        "paint.weight_stroke"
    }
    fn label() -> &'static str {
        "Weight Paint"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((_, mesh)) = active_mesh(doc.document()) else { return Ok(OpResult::Cancelled) };
        if params.group.is_empty() || !params.weight.is_finite() || !params.strength.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let mut w = weights(doc.document(), mesh, &params.group);
        let (target, k) = (params.weight.clamp(0.0, 1.0) as f32, params.strength.clamp(0.0, 1.0) as f32);
        let mut once: Vec<u32> = params.vertices.clone();
        once.sort_unstable();
        once.dedup();
        for v in once {
            if let Some(x) = w.get_mut(v as usize) {
                *x += (target - *x) * k;
            }
        }
        let Some(m) = doc.document_mut().mesh_mut(mesh) else { return Ok(OpResult::Cancelled) };
        m.attributes_mut().insert(&weight_layer(&params.group), Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct NormalizeParams {}

/// Scales every vertex's weights so they sum to one — Blender's Normalize All.
/// A vertex no group weighs on is left at nothing.
pub struct Normalize;

impl Operator for Normalize {
    type Params = NormalizeParams;
    fn id() -> &'static str {
        "paint.weights_normalize"
    }
    fn label() -> &'static str {
        "Normalize All"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(_params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((_, mesh)) = active_mesh(doc.document()) else { return Ok(OpResult::Cancelled) };
        let names = groups(doc.document(), mesh);
        if names.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let mut all: Vec<Vec<f32>> = names.iter().map(|g| weights(doc.document(), mesh, g)).collect();
        let n = all[0].len();
        let mut changed = false;
        for v in 0..n {
            let total: f32 = all.iter().map(|w| w[v]).sum();
            if total > 0.0 && (total - 1.0).abs() > 1e-6 {
                for w in &mut all {
                    w[v] /= total;
                }
                changed = true;
            }
        }
        if !changed {
            return Ok(OpResult::Cancelled);
        }
        let Some(m) = doc.document_mut().mesh_mut(mesh) else { return Ok(OpResult::Cancelled) };
        for (name, w) in names.iter().zip(all) {
            m.attributes_mut().insert(&weight_layer(name), Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
        }
        Ok(OpResult::Finished)
    }
}

/// A copy of `mesh` for the screen, with `group`'s weights where the
/// renderer's weight ramp reads them: the first surface coordinate of every
/// corner. The document's mesh is not touched — its real UVs are its own.
pub fn weight_display(mesh: &kreate_mesh::MeshData, group: Option<&str>) -> kreate_mesh::MeshData {
    let n = mesh.num_verts() as usize;
    let w: Vec<f32> = match group.and_then(|g| mesh.attributes().get(&weight_layer(g))).map(|l| &l.data) {
        Some(AttrData::F32(w)) if w.len() == n => w.to_vec(),
        _ => vec![0.0; n],
    };
    let uv: Vec<glam::Vec2> = mesh.corner_verts().iter().map(|v| glam::Vec2::new(w[*v as usize], 0.0)).collect();
    let mut out = mesh.clone();
    out.attributes_mut().insert(
        kreate_mesh::attribute::names::UV,
        Domain::Corner,
        AttrData::F32x2(std::sync::Arc::new(uv)),
    );
    out
}
