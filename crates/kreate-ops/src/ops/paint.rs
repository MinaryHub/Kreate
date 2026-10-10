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

use kreate_mesh::attribute::{AttrData, Domain};
use kreate_ops_derive::OpParams;
use kreate_scene::skin::{weight_layer, WEIGHT_PREFIX};
use kreate_scene::{DocMut, Document, MeshId, ObjectId};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddGroup>();
    registry.register::<RemoveGroup>();
    registry.register::<SetActiveGroup>();
    registry.register::<Stroke>();
    registry.register::<Normalize>();
    registry.register::<SetWeights>();
    registry.register::<SmoothWeights>();
    registry.register::<TextureStroke>();
    registry.register::<NewTexture>();
}

// --- texture painting ------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct TextureStrokeParams {
    /// The image painted, by its identity.
    pub image: String,
    /// The dabs, four numbers each: where in the UV square, and the brush's
    /// radius there, in UV units — and how strongly, zero to one.
    pub dabs: Vec<f64>,
    /// The colour, as the image stores it: an sRGB image's own encoding, so
    /// the swatch picked is the colour that lands.
    pub color: [f64; 3],
}

impl Default for TextureStrokeParams {
    fn default() -> Self {
        Self {
            image: String::new(),
            dabs: Vec::new(),
            color: [1.0, 1.0, 1.0],
        }
    }
}

/// A brush stroke on an image, through the UVs: Blender's Texture Paint.
///
/// The stroke is its dabs in UV space — replayed, it paints the same pixels
/// the same way — and undo keeps only the tiles it touched (§3-11). The
/// application runs it as a modal, so the image changes under the brush and
/// the whole stroke is one step to undo.
pub struct TextureStroke;

impl Operator for TextureStroke {
    type Params = TextureStrokeParams;
    fn id() -> &'static str {
        "paint.texture_stroke"
    }
    fn label() -> &'static str {
        "Texture Paint Stroke"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = image_by_uid(doc.document(), &params.image) else {
            return Err(OpError::Failed("no such image".into()));
        };
        let image = doc.document().image(id).expect("found");
        if image.format != kreate_scene::PixelFormat::Rgba8 {
            return Err(OpError::Failed(
                "only eight-bit colour images can be painted".into(),
            ));
        }
        let (w, h) = (image.width(), image.height());
        let dabs: Vec<[f64; 4]> = params
            .dabs
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|d| d.iter().all(|x| x.is_finite()) && d[2] > 0.0)
            .copied()
            .collect();
        if dabs.is_empty() || w == 0 || h == 0 {
            return Ok(OpResult::Cancelled);
        }
        // Every tile any dab reaches, once.
        let mut tiles: Vec<(u32, u32)> = Vec::new();
        for d in &dabs {
            let (cx, cy) = (
                d[0].rem_euclid(1.0) * w as f64,
                d[1].rem_euclid(1.0) * h as f64,
            );
            let r = d[2] * w.max(h) as f64;
            let x0 = (cx - r).floor().max(0.0) as u32;
            let y0 = (cy - r).floor().max(0.0) as u32;
            let x1 = ((cx + r).ceil() as u32).min(w);
            let y1 = ((cy + r).ceil() as u32).min(h);
            for t in image.tiles_covering(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)) {
                if !tiles.contains(&t) {
                    tiles.push(t);
                }
            }
        }
        if tiles.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let target = params.color.map(|c| (c.clamp(0.0, 1.0) * 255.0) as f32);
        doc.edit_image(id, &tiles, |img| {
            img.source = kreate_scene::image::Source::Painted;
            let tile = kreate_scene::image::TILE;
            for d in &dabs {
                let (cx, cy) = (
                    d[0].rem_euclid(1.0) * w as f64,
                    d[1].rem_euclid(1.0) * h as f64,
                );
                let r = (d[2] * w.max(h) as f64).max(0.5);
                let strength = d[3].clamp(0.0, 1.0);
                let x0 = (cx - r).floor().max(0.0) as u32;
                let y0 = (cy - r).floor().max(0.0) as u32;
                let x1 = ((cx + r).ceil() as u32).min(w);
                let y1 = ((cy + r).ceil() as u32).min(h);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let dist = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2))
                            .sqrt()
                            / r;
                        if dist >= 1.0 {
                            continue;
                        }
                        // Blender's smooth falloff.
                        let t = 1.0 - dist;
                        let a = (strength * t * t * (3.0 - 2.0 * t)) as f32;
                        let Some(px) = img.tile_mut(x / tile, y / tile) else {
                            continue;
                        };
                        let at = (((y % tile) * tile + (x % tile)) * 4) as usize;
                        let bytes = &mut px.bytes_mut()[at..at + 4];
                        for c in 0..3 {
                            let old = bytes[c] as f32;
                            bytes[c] =
                                (old + (target[c] - old) * a).round().clamp(0.0, 255.0) as u8;
                        }
                        let old = bytes[3] as f32;
                        bytes[3] = (old + (255.0 - old) * a).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        });
        Ok(OpResult::Finished)
    }
}

/// An image by the identity a parameter names it with.
fn image_by_uid(doc: &Document, uid: &str) -> Option<kreate_scene::document::ImageId> {
    let raw: u64 = uid.trim().parse().ok()?;
    doc.images()
        .find(|(id, _)| doc.image_uid(*id).is_some_and(|u| u.raw() == raw))
        .map(|(id, _)| id)
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct NewTextureParams {
    #[param(min = 16, max = 8192, unit = "count")]
    pub width: u32,
    #[param(min = 16, max = 8192, unit = "count")]
    pub height: u32,
    /// What it starts as, in its own sRGB encoding.
    pub color: [f64; 3],
}

impl Default for NewTextureParams {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            color: [1.0, 1.0, 1.0],
        }
    }
}

/// A new image to paint on, as the active object's base colour: Blender's
/// "Add Texture Paint Slot". The object's first material takes it, and gets
/// made when there is none.
pub struct NewTexture;

impl Operator for NewTexture {
    type Params = NewTextureParams;
    fn id() -> &'static str {
        "paint.new_texture"
    }
    fn label() -> &'static str {
        "New Paint Texture"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .and_then(|id| doc.object(id))
            .is_some_and(|o| o.mesh().is_some())
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document_mut();
        let Some(object) = d.selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let Some(mesh) = d.object(object).and_then(|o| o.mesh()) else {
            return Ok(OpResult::Cancelled);
        };
        let name = format!(
            "{} Paint",
            d.object(object).map(|o| o.name.clone()).unwrap_or_default()
        );
        let (w, h) = (params.width.clamp(16, 8192), params.height.clamp(16, 8192));
        let mut image =
            kreate_scene::Image::new(name.clone(), w, h, kreate_scene::PixelFormat::Rgba8);
        image.color_space = kreate_scene::ColorSpace::Srgb;
        image.source = kreate_scene::image::Source::Painted;
        let fill = params
            .color
            .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
        let (across, down) = image.tile_counts();
        for ty in 0..down {
            for tx in 0..across {
                if let Some(t) = image.tile_mut(tx, ty) {
                    for px in t.bytes_mut().as_chunks_mut::<4>().0 {
                        *px = [fill[0], fill[1], fill[2], 255];
                    }
                }
            }
        }
        let image = d.add_image(image);
        let material = match d.mesh_slots(mesh).first().copied().flatten() {
            Some(m) => m,
            None => {
                let m = d.add_material(kreate_scene::Material {
                    name: format!("{name} Material"),
                    ..Default::default()
                });
                let mut slots = d.mesh_slots(mesh).to_vec();
                if slots.is_empty() {
                    slots.push(Some(m));
                } else {
                    slots[0] = Some(m);
                }
                d.set_mesh_slots(mesh, slots);
                m
            }
        };
        if let Some(m) = d.material_mut(material) {
            m.base_color_image = Some(image);
            // White, so the texture shows as painted rather than tinted.
            m.base_color = kreate_math::SceneRgb::new(1.0, 1.0, 1.0);
        }
        Ok(OpResult::Finished)
    }
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
    let Some(m) = doc.mesh(mesh) else {
        return Vec::new();
    };
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((object, mesh)) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let taken = groups(doc.document(), mesh);
        let base = if params.name.trim().is_empty() {
            "Group".to_string()
        } else {
            params.name.trim().to_string()
        };
        let name = if taken.contains(&base) {
            (1..)
                .map(|i| format!("{base}.{i:03}"))
                .find(|n| !taken.contains(n))
                .expect("names run out after integers")
        } else {
            base
        };
        let d = doc.document_mut();
        let Some(m) = d.mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        let n = m.num_verts() as usize;
        m.attributes_mut().insert(
            &weight_layer(&name),
            Domain::Point,
            AttrData::F32(std::sync::Arc::new(vec![0.0; n])),
        );
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((object, mesh)) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((object, mesh)) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        if !groups(doc.document(), mesh).contains(&params.name)
            || doc
                .document()
                .object(object)
                .and_then(|o| o.active_group.as_deref())
                == Some(params.name.as_str())
        {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
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
    /// How much of the strength each vertex gets, parallel to `vertices` —
    /// the brush's falloff, less towards its edge. Empty for all of it.
    #[param(hidden)]
    pub amounts: Vec<f64>,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self {
            group: String::new(),
            vertices: Vec::new(),
            weight: 1.0,
            strength: 1.0,
            amounts: Vec::new(),
        }
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((_, mesh)) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        if params.group.is_empty() || !params.weight.is_finite() || !params.strength.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let mut w = weights(doc.document(), mesh, &params.group);
        let (target, k) = (
            params.weight.clamp(0.0, 1.0) as f32,
            params.strength.clamp(0.0, 1.0) as f32,
        );
        // Each vertex once, with the most the brush gave it.
        let mut once: Vec<(u32, f32)> = params
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    *v,
                    params
                        .amounts
                        .get(i)
                        .map(|a| a.clamp(0.0, 1.0) as f32)
                        .unwrap_or(1.0),
                )
            })
            .collect();
        once.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        once.dedup_by_key(|(v, _)| *v);
        for (v, amount) in once {
            if let Some(x) = w.get_mut(v as usize) {
                *x += (target - *x) * k * amount;
            }
        }
        let Some(m) = doc.document_mut().mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        m.attributes_mut().insert(
            &weight_layer(&params.group),
            Domain::Point,
            AttrData::F32(std::sync::Arc::new(w)),
        );
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
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((_, mesh)) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let names = groups(doc.document(), mesh);
        if names.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let mut all: Vec<Vec<f32>> = names
            .iter()
            .map(|g| weights(doc.document(), mesh, g))
            .collect();
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
        let Some(m) = doc.document_mut().mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        for (name, w) in names.iter().zip(all) {
            m.attributes_mut().insert(
                &weight_layer(name),
                Domain::Point,
                AttrData::F32(std::sync::Arc::new(w)),
            );
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetWeightsParams {
    /// The group — a bone's name, for a mesh an armature moves. Made if the
    /// mesh has none by that name.
    pub group: String,
    /// Which vertices, by index. Empty: those inside `from`..`to` when a box
    /// is given, else the selected ones.
    pub vertices: Vec<u32>,
    /// A box in the mesh's own coordinates, low corner and high: every vertex
    /// inside. Unused while the two corners are the same.
    #[param(unit = "length")]
    pub from: [f64; 3],
    #[param(unit = "length")]
    pub to: [f64; 3],
    #[param(min = 0.0, max = 1.0)]
    pub weight: f64,
    /// Put it there, add it, or take it away.
    #[param(variants("replace", "add", "subtract"))]
    pub mode: u32,
    /// These vertices follow this group alone: their weight in every other
    /// group is taken away — a hand's vertices off the forearm.
    pub exclusive: bool,
}

impl Default for SetWeightsParams {
    fn default() -> Self {
        Self {
            group: String::new(),
            vertices: Vec::new(),
            from: [0.0; 3],
            to: [0.0; 3],
            weight: 1.0,
            mode: 0,
            exclusive: false,
        }
    }
}

/// Sets how much chosen vertices follow a group, exactly: by their numbers,
/// by a box around them, or as selected. What a brush cannot say — "these
/// two hundred vertices belong to the left hand" — said in one step.
pub struct SetWeights;

impl Operator for SetWeights {
    type Params = SetWeightsParams;
    fn id() -> &'static str {
        "object.vertex_weights_set"
    }
    fn label() -> &'static str {
        "Set Vertex Weights"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let group = params.group.trim();
        if group.is_empty() {
            return Err(OpError::Failed("name the group, or the bone".into()));
        }
        let Some((object, mesh)) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(m) = doc.document().mesh(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        let n = m.num_verts() as usize;
        let chosen: Vec<usize> = if !params.vertices.is_empty() {
            params
                .vertices
                .iter()
                .map(|v| *v as usize)
                .filter(|v| *v < n)
                .collect()
        } else if params.from != params.to {
            let lo = glam::DVec3::from_array(params.from).min(glam::DVec3::from_array(params.to));
            let hi = glam::DVec3::from_array(params.from).max(glam::DVec3::from_array(params.to));
            m.positions()
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    let p = p.as_dvec3();
                    p.cmpge(lo).all() && p.cmple(hi).all()
                })
                .map(|(i, _)| i)
                .collect()
        } else {
            crate::ops::mesh::selection_vertices(m)
                .into_iter()
                .map(|v| v as usize)
                .collect()
        };
        if chosen.is_empty() {
            return Err(OpError::Failed("no vertices chosen".into()));
        }
        let weight = params.weight.clamp(0.0, 1.0) as f32;
        let mut mine = weights(doc.document(), mesh, group);
        for &v in &chosen {
            mine[v] = match params.mode {
                1 => (mine[v] + weight).min(1.0),
                2 => (mine[v] - weight).max(0.0),
                _ => weight,
            };
        }
        let others: Vec<(String, Vec<f32>)> = if params.exclusive {
            groups(doc.document(), mesh)
                .into_iter()
                .filter(|g| g != group)
                .map(|g| {
                    let mut w = weights(doc.document(), mesh, &g);
                    for &v in &chosen {
                        w[v] = 0.0;
                    }
                    (g, w)
                })
                .collect()
        } else {
            Vec::new()
        };
        let d = doc.document_mut();
        let Some(m) = d.mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        m.attributes_mut().insert(
            &weight_layer(group),
            Domain::Point,
            AttrData::F32(std::sync::Arc::new(mine)),
        );
        for (g, w) in others {
            m.attributes_mut().insert(
                &weight_layer(&g),
                Domain::Point,
                AttrData::F32(std::sync::Arc::new(w)),
            );
        }
        if let Some(o) = d.object_mut(object) {
            o.active_group = Some(group.to_string());
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SmoothWeightsParams {
    /// One group; empty for every group.
    pub group: String,
    /// How many passes.
    #[param(min = 1, max = 100, unit = "count")]
    pub iterations: u32,
    /// How far each pass moves a weight toward its neighbours' mean.
    #[param(min = 0.0, max = 1.0)]
    pub factor: f64,
}

impl Default for SmoothWeightsParams {
    fn default() -> Self {
        Self {
            group: String::new(),
            iterations: 3,
            factor: 0.5,
        }
    }
}

/// Blurs weights along the edges — Blender's Smooth Vertex Weights — so a
/// joint bends as a sleeve does, not as a hinge.
pub struct SmoothWeights;

impl Operator for SmoothWeights {
    type Params = SmoothWeightsParams;
    fn id() -> &'static str {
        "object.vertex_weights_smooth"
    }
    fn label() -> &'static str {
        "Smooth Vertex Weights"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((_, mesh)) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let names: Vec<String> = if params.group.trim().is_empty() {
            groups(doc.document(), mesh)
        } else {
            vec![params.group.trim().to_string()]
        };
        let Some(m) = doc.document().mesh(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        if names.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let n = m.num_verts() as usize;
        let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in m.edges() {
            let (a, b) = (e[0] as usize, e[1] as usize);
            if a < n && b < n {
                neighbours[a].push(b);
                neighbours[b].push(a);
            }
        }
        let factor = params.factor.clamp(0.0, 1.0) as f32;
        let smoothed: Vec<(String, Vec<f32>)> = names
            .into_iter()
            .map(|g| {
                let mut w = weights(doc.document(), mesh, &g);
                for _ in 0..params.iterations.clamp(1, 100) {
                    let before = w.clone();
                    for (v, near) in neighbours.iter().enumerate() {
                        if near.is_empty() {
                            continue;
                        }
                        let mean = near.iter().map(|u| before[*u]).sum::<f32>() / near.len() as f32;
                        w[v] = before[v] + (mean - before[v]) * factor;
                    }
                }
                (g, w)
            })
            .collect();
        let Some(m) = doc.document_mut().mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        for (g, w) in smoothed {
            m.attributes_mut().insert(
                &weight_layer(&g),
                Domain::Point,
                AttrData::F32(std::sync::Arc::new(w)),
            );
        }
        Ok(OpResult::Finished)
    }
}

/// A copy of `mesh` for the screen, with `group`'s weights where the
/// renderer's weight ramp reads them: the first surface coordinate of every
/// corner. The document's mesh is not touched — its real UVs are its own.
pub fn weight_display(mesh: &kreate_mesh::MeshData, group: Option<&str>) -> kreate_mesh::MeshData {
    let n = mesh.num_verts() as usize;
    let w: Vec<f32> = match group
        .and_then(|g| mesh.attributes().get(&weight_layer(g)))
        .map(|l| &l.data)
    {
        Some(AttrData::F32(w)) if w.len() == n => w.to_vec(),
        _ => vec![0.0; n],
    };
    let uv: Vec<glam::Vec2> = mesh
        .corner_verts()
        .iter()
        .map(|v| glam::Vec2::new(w[*v as usize], 0.0))
        .collect();
    let mut out = mesh.clone();
    out.attributes_mut().insert(
        kreate_mesh::attribute::names::UV,
        Domain::Corner,
        AttrData::F32x2(std::sync::Arc::new(uv)),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_doc() -> (kreate_scene::Document, ObjectId, MeshId) {
        // A column of points up Z, one per unit, joined by edges.
        let mut d = kreate_scene::Document::new();
        let id = d.add_mesh_object("Arm", kreate_mesh::primitives::grid(1.0, 1, 4));
        d.selection.clear();
        d.selection.add(id);
        d.selection.active = Some(id);
        let mesh = d.object(id).and_then(|o| o.mesh()).expect("a mesh");
        (d, id, mesh)
    }

    fn run<O: Operator>(d: &mut kreate_scene::Document, params: &O::Params) -> OpResult {
        let mut edit = DocMut::new(d);
        O::execute(params, &mut edit, &Context::default()).expect("ran")
    }

    #[test]
    fn weights_are_set_by_number_by_box_and_alone() {
        let (mut d, _, mesh) = strip_doc();
        let n = d.mesh(mesh).expect("mesh").num_verts() as usize;
        run::<SetWeights>(
            &mut d,
            &SetWeightsParams {
                group: "Upper".into(),
                vertices: (0..n as u32).collect(),
                ..Default::default()
            },
        );
        assert!(weights(&d, mesh, "Upper").iter().all(|w| *w == 1.0));
        // The far half by a box, to a second bone, alone.
        let ys: Vec<f32> = d
            .mesh(mesh)
            .expect("mesh")
            .positions()
            .iter()
            .map(|p| p.y)
            .collect();
        let top = ys.iter().copied().fold(f32::MIN, f32::max) as f64;
        run::<SetWeights>(
            &mut d,
            &SetWeightsParams {
                group: "Lower".into(),
                from: [-10.0, 0.01, -10.0],
                to: [10.0, top + 1.0, 10.0],
                exclusive: true,
                ..Default::default()
            },
        );
        let (upper, lower) = (weights(&d, mesh, "Upper"), weights(&d, mesh, "Lower"));
        for (v, y) in ys.iter().enumerate() {
            if *y > 0.01 {
                assert_eq!((upper[v], lower[v]), (0.0, 1.0), "vertex {v} at {y}");
            } else {
                assert_eq!((upper[v], lower[v]), (1.0, 0.0), "vertex {v} at {y}");
            }
        }
        // Subtracting half.
        run::<SetWeights>(
            &mut d,
            &SetWeightsParams {
                group: "Lower".into(),
                vertices: vec![n as u32 - 1],
                weight: 0.5,
                mode: 2,
                ..Default::default()
            },
        );
        assert_eq!(weights(&d, mesh, "Lower")[n - 1], 0.5);
    }

    #[test]
    fn smoothing_softens_a_hard_edge_between_bones() {
        let (mut d, _, mesh) = strip_doc();
        let ys: Vec<f32> = d
            .mesh(mesh)
            .expect("mesh")
            .positions()
            .iter()
            .map(|p| p.y)
            .collect();
        run::<SetWeights>(
            &mut d,
            &SetWeightsParams {
                group: "Lower".into(),
                from: [-10.0, 0.01, -10.0],
                to: [10.0, 10.0, 10.0],
                ..Default::default()
            },
        );
        let hard = weights(&d, mesh, "Lower");
        run::<SmoothWeights>(&mut d, &SmoothWeightsParams::default());
        let soft = weights(&d, mesh, "Lower");
        // Next to the step, a vertex that had none now has some.
        let (v, _) = ys
            .iter()
            .enumerate()
            .filter(|(_, y)| **y <= 0.01)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("one below");
        assert_eq!(hard[v], 0.0);
        assert!(soft[v] > 0.05 && soft[v] < 0.95, "{}", soft[v]);
    }
}
