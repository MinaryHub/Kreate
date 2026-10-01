//! Shape key operators: adding, removing, naming, turning up, and the trade
//! in and out of edit mode that lets edit mode edit a key.
//!
//! All on the active object. A key's offsets are a layer on the mesh and its
//! name and value are on the object — see `kreate_scene::shape`.

use kreate_scene::document::ObjectId;
use kreate_scene::shape::{self, ShapeKey};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddKey>();
    registry.register::<RemoveKey>();
    registry.register::<SetKey>();
    registry.register::<SetActiveKey>();
    registry.register::<EditKey>();
}

/// The active object, when it is a mesh and edit mode is not trading a key.
fn target(doc: &Document) -> Option<ObjectId> {
    let id = doc.selection.active?;
    doc.object(id)?.mesh()?;
    Some(id)
}

fn has_mesh(_ctx: &Context, doc: &Document) -> bool {
    target(doc).is_some()
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddKeyParams {
    /// The new key's name; empty for "Key 1", "Key 2"…
    pub name: String,
    /// Start from the shape the keys make now, rather than from the basis.
    pub from_mix: bool,
}

/// Adds a shape key and makes it the active one — Blender's `+`.
pub struct AddKey;

impl Operator for AddKey {
    type Params = AddKeyParams;
    fn id() -> &'static str {
        "object.shape_key_add"
    }
    fn label() -> &'static str {
        "Add Shape Key"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc) && ctx.mode.acts_on() == crate::mode::ActsOn::Objects
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else { return Ok(OpResult::Cancelled) };
        let o = doc.document().object(id).expect("target");
        let taken: Vec<String> = o.shape_keys.iter().map(|k| k.name.clone()).collect();
        let name = if params.name.trim().is_empty() {
            (1..).map(|i| format!("Key {i}")).find(|n| !taken.contains(n)).expect("names run out after integers")
        } else if taken.contains(&params.name) {
            (1..).map(|i| format!("{}.{i:03}", params.name)).find(|n| !taken.contains(n)).expect("names")
        } else {
            params.name.clone()
        };
        let mix = if params.from_mix { o.shape_keys.clone() } else { Vec::new() };
        let mesh = o.mesh().expect("target");
        let d = doc.document_mut();
        let Some(m) = d.mesh_mut(mesh) else { return Ok(OpResult::Cancelled) };
        shape::add_layer(m, &name, &mix);
        let o = d.object_mut(id).expect("target");
        o.shape_keys.push(ShapeKey { name: name.clone(), value: 0.0 });
        o.active_shape = Some(name);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct KeyNameParams {
    /// The key, by name; empty for the basis.
    pub name: String,
}

/// Removes a shape key and its offsets.
pub struct RemoveKey;

impl Operator for RemoveKey {
    type Params = KeyNameParams;
    fn id() -> &'static str {
        "object.shape_key_remove"
    }
    fn label() -> &'static str {
        "Remove Shape Key"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc) && ctx.mode.acts_on() == crate::mode::ActsOn::Objects
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else { return Ok(OpResult::Cancelled) };
        let o = doc.document().object(id).expect("target");
        if !o.shape_keys.iter().any(|k| k.name == params.name) {
            return Ok(OpResult::Cancelled);
        }
        let mesh = o.mesh().expect("target");
        let d = doc.document_mut();
        if let Some(m) = d.mesh_mut(mesh) {
            m.attributes_mut().remove(&shape::layer(&params.name));
        }
        let o = d.object_mut(id).expect("target");
        o.shape_keys.retain(|k| k.name != params.name);
        if o.active_shape.as_deref() == Some(params.name.as_str()) {
            o.active_shape = o.shape_keys.last().map(|k| k.name.clone());
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetKeyParams {
    pub name: String,
    /// How far the key is turned up.
    #[param(min = 0.0, max = 1.0, step = 0.01)]
    pub value: f64,
    /// A new name; empty to keep it.
    pub new_name: String,
}

/// Turns a key up or down, or renames it.
pub struct SetKey;

impl Operator for SetKey {
    type Params = SetKeyParams;
    fn id() -> &'static str {
        "object.shape_key_set"
    }
    fn label() -> &'static str {
        "Set Shape Key"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else { return Ok(OpResult::Cancelled) };
        let o = doc.document().object(id).expect("target");
        let Some(key) = o.shape_keys.iter().find(|k| k.name == params.name).cloned() else {
            return Ok(OpResult::Cancelled);
        };
        let value = params.value.clamp(0.0, 1.0) as f32;
        let rename = (!params.new_name.trim().is_empty() && params.new_name != params.name
            && !o.shape_keys.iter().any(|k| k.name == params.new_name))
            .then(|| params.new_name.clone());
        if rename.is_none() && key.value == value {
            return Ok(OpResult::Cancelled);
        }
        let mesh = o.mesh().expect("target");
        let d = doc.document_mut();
        if let Some(new) = &rename {
            if let Some(m) = d.mesh_mut(mesh) {
                if let Some(layer) = m.attributes().get(&shape::layer(&params.name)).cloned() {
                    m.attributes_mut().remove(&shape::layer(&params.name));
                    m.attributes_mut().insert(&shape::layer(new), layer.domain, layer.data);
                }
            }
        }
        let o = d.object_mut(id).expect("target");
        if let Some(k) = o.shape_keys.iter_mut().find(|k| k.name == params.name) {
            k.value = value;
            if let Some(new) = &rename {
                k.name = new.clone();
            }
        }
        if let Some(new) = rename {
            if o.active_shape.as_deref() == Some(params.name.as_str()) {
                o.active_shape = Some(new);
            }
        }
        Ok(OpResult::Finished)
    }
}

/// Makes a key the one shown and edited; empty for the basis.
pub struct SetActiveKey;

impl Operator for SetActiveKey {
    type Params = KeyNameParams;
    fn id() -> &'static str {
        "object.shape_key_set_active"
    }
    fn label() -> &'static str {
        "Set Active Shape Key"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc) && ctx.mode.acts_on() == crate::mode::ActsOn::Objects
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else { return Ok(OpResult::Cancelled) };
        let o = doc.document().object(id).expect("target");
        let wanted = (!params.name.is_empty()).then(|| params.name.clone());
        if wanted.as_ref().is_some_and(|n| !o.shape_keys.iter().any(|k| &k.name == n)) || o.active_shape == wanted {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().object_mut(id).expect("target").active_shape = wanted;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct EditKeyParams {
    /// In: the active key's shape into the mesh for edit mode. Out: back.
    pub enter: bool,
}

/// The trade that lets edit mode edit a shape key. Run by the application on
/// the way into and out of edit mode, as a step of its own so that undo
/// across the mode change finds the mesh as it was.
pub struct EditKey;

impl Operator for EditKey {
    type Params = EditKeyParams;
    fn id() -> &'static str {
        "object.shape_key_edit"
    }
    fn label() -> &'static str {
        "Edit Shape Key"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else { return Ok(OpResult::Cancelled) };
        let o = doc.document().object(id).expect("target");
        let mesh = o.mesh().expect("target");
        let key = if params.enter {
            match (&o.active_shape, &o.editing_shape) {
                (Some(k), None) => k.clone(),
                _ => return Ok(OpResult::Cancelled),
            }
        } else {
            match &o.editing_shape {
                Some(k) => k.clone(),
                None => return Ok(OpResult::Cancelled),
            }
        };
        let d = doc.document_mut();
        let Some(m) = d.mesh_mut(mesh) else { return Ok(OpResult::Cancelled) };
        let traded = if params.enter { shape::trade_in(m, &key) } else { shape::trade_out(m, &key) };
        if !traded {
            return Ok(OpResult::Cancelled);
        }
        d.object_mut(id).expect("target").editing_shape = params.enter.then_some(key);
        Ok(OpResult::Finished)
    }
}
