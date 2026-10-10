//! Sculpting: one operator per stroke of the brush, and the mask and face set
//! operators around it.
//!
//! The dabs ride in the parameters — where the brush touched, in the mesh's
//! own space, how wide, and how hard the pen pressed — so a stroke replays
//! without the pointer and the whole of it is one step to undo. See
//! `kreate_mesh::sculpt`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use kreate_math::Vec3;
use kreate_mesh::sculpt::{self, Alpha, Brush, Dab, Falloff, MaskAction, Stroke, StrokeCache};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

/// The stroke being drawn. While the pointer moves the stroke is re-run on
/// its starting mesh with one more dab each time; the cache lets that run do
/// only the new dab. It checks it was given the mesh it began on, so a stale
/// one is never used — at worst it is rebuilt.
static STROKE: Mutex<Option<StrokeCache>> = Mutex::new(None);

/// The same, for a stroke with dynamic topology.
static DYN: Mutex<Option<kreate_mesh::dyntopo::DynCache>> = Mutex::new(None);

/// Alpha pictures made so far, by what named them and the image's revision:
/// the same `Arc` frame after frame, which is what lets a stroke's cache see
/// the same brush.
static ALPHAS: Mutex<Option<Alphas>> = Mutex::new(None);

/// Alphas by the name that asked for them and the image's revision.
type Alphas = HashMap<(String, u64), Arc<Alpha>>;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<SculptStroke>();
    registry.register::<SculptMask>();
    registry.register::<FaceSetFromMask>();
    registry.register::<FaceSetVisibility>();
    registry.register::<VoxelRemesh>();
    registry.register::<DeformRegion>();
    registry.register::<CreasePath>();
}

fn active_mesh(doc: &Document) -> Option<kreate_scene::MeshId> {
    doc.selection
        .active
        .and_then(|a| doc.object(a))
        .and_then(|o| o.mesh())
}

fn has_mesh(_ctx: &Context, doc: &Document) -> bool {
    active_mesh(doc).is_some()
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct StrokeParams {
    #[param(variants(
        "draw",
        "smooth",
        "grab",
        "inflate",
        "clay_strips",
        "crease",
        "flatten",
        "scrape",
        "fill",
        "pinch",
        "layer",
        "snake_hook",
        "mask",
        "face_set"
    ))]
    pub brush: u32,
    /// Where the brush touched: x, y, z and radius per dab, in the mesh's space.
    pub dabs: Vec<f64>,
    /// How hard the pen pressed at each dab, 0 to 1. Missing ones are 1.
    pub pressures: Vec<f64>,
    #[param(min = 0.0, max = 1.0)]
    pub strength: f64,
    /// Carve instead of raise; shrink instead of swell.
    pub invert: bool,
    /// For a grab, how far it was dragged, in the mesh's space.
    pub grab: [f64; 3],
    /// Mirrored across X.
    pub mirror_x: bool,
    /// Mirrored across Y.
    pub mirror_y: bool,
    /// Mirrored across Z.
    pub mirror_z: bool,
    /// Repeated round the Z axis this many times; 1 is once.
    #[param(min = 1, max = 64)]
    pub radial: u32,
    /// How the brush fades to its rim.
    #[param(variants("smooth", "sphere", "root", "sharp", "linear", "constant"))]
    pub falloff: u32,
    /// The way the view looked, in the mesh's space.
    pub view: [f64; 3],
    /// Only what faces the view.
    pub front_only: bool,
    /// How much each dab also smooths, 0 to 1.
    #[param(min = 0.0, max = 1.0)]
    pub auto_smooth: f64,
    /// A picture stamped through the brush: `noise`, `stripes`, `dots`, or
    /// an image by name or uid. Empty for none.
    pub alpha: String,
    /// The face set the face set brush paints.
    pub face_set: i32,
    /// Dynamic topology: the surface cut finer where the brush goes.
    pub dyntopo: bool,
    /// Dynamic topology's edge length, as a fraction of the brush's radius.
    #[param(min = 0.01, max = 1.0)]
    pub detail: f64,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self {
            brush: 0,
            dabs: Vec::new(),
            pressures: Vec::new(),
            strength: 0.5,
            invert: false,
            grab: [0.0; 3],
            mirror_x: false,
            mirror_y: false,
            mirror_z: false,
            radial: 1,
            falloff: 0,
            view: [0.0; 3],
            front_only: false,
            auto_smooth: 0.0,
            alpha: String::new(),
            face_set: 1,
            dyntopo: false,
            detail: 0.2,
        }
    }
}

/// A built-in alpha, 128 square: what a brush can stamp with no image.
fn builtin_alpha(name: &str) -> Option<Alpha> {
    const N: u32 = 128;
    let at = |f: &dyn Fn(f32, f32) -> f32| Alpha {
        width: N,
        height: N,
        values: (0..N * N)
            .map(|i| f((i % N) as f32 / N as f32, (i / N) as f32 / N as f32).clamp(0.0, 1.0))
            .collect(),
    };
    match name {
        "noise" => {
            // Value noise in two octaves, tiling.
            let hash = |x: i32, y: i32| {
                let mut h = (x.rem_euclid(16) as u32).wrapping_mul(374_761_393)
                    ^ (y.rem_euclid(16) as u32).wrapping_mul(668_265_263);
                h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
                (h ^ (h >> 16)) as f32 / u32::MAX as f32
            };
            let octave = |u: f32, v: f32, cells: f32| {
                let (x, y) = (u * cells, v * cells);
                let (x0, y0) = (x.floor(), y.floor());
                let (fx, fy) = (x - x0, y - y0);
                let s = |t: f32| t * t * (3.0 - 2.0 * t);
                let (ix, iy, k) = (x0 as i32, y0 as i32, 16 / cells as i32);
                let g = |a: i32, b: i32| hash(a * k, b * k);
                let top = g(ix, iy) + (g(ix + 1, iy) - g(ix, iy)) * s(fx);
                let bottom = g(ix, iy + 1) + (g(ix + 1, iy + 1) - g(ix, iy + 1)) * s(fx);
                top + (bottom - top) * s(fy)
            };
            Some(at(&|u, v| {
                0.65 * octave(u, v, 4.0) + 0.35 * octave(u, v, 8.0)
            }))
        }
        "stripes" => Some(at(&|u, _| {
            0.5 + 0.5 * (u * std::f32::consts::TAU * 6.0).sin()
        })),
        "dots" => Some(at(&|u, v| {
            let (x, y) = ((u * 6.0).fract() - 0.5, (v * 6.0).fract() - 0.5);
            1.0 - ((x * x + y * y).sqrt() * 3.0).min(1.0)
        })),
        _ => None,
    }
}

/// An image as an alpha: its brightness, at most 256 across.
fn image_alpha(image: &kreate_scene::Image) -> Alpha {
    use kreate_scene::PixelFormat;
    let (w, h) = (image.width().max(1), image.height().max(1));
    let step = (w.max(h) as f32 / 256.0).max(1.0);
    let (aw, ah) = (
        ((w as f32 / step) as u32).max(1),
        ((h as f32 / step) as u32).max(1),
    );
    let tile = kreate_scene::TILE;
    let bpp = image.format.bytes_per_pixel();
    let half = |b: [u8; 2]| -> f32 {
        let bits = u16::from_le_bytes(b);
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 10) & 0x1f) as i32;
        let frac = (bits & 0x3ff) as f32;
        sign * match exp {
            0 => frac / 1024.0 * 2f32.powi(-14),
            31 => 1.0,
            e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
        }
    };
    let mut values = Vec::with_capacity((aw * ah) as usize);
    for ay in 0..ah {
        for ax in 0..aw {
            let x = ((ax as f32 + 0.5) * step) as u32;
            let y = ((ay as f32 + 0.5) * step) as u32;
            let (x, y) = (x.min(w - 1), y.min(h - 1));
            let value = image
                .tile(x / tile, y / tile)
                .map(|t| {
                    let at = (((y % tile) * tile + (x % tile)) as usize) * bpp;
                    let b = t.bytes();
                    match image.format {
                        PixelFormat::R8 => b[at] as f32 / 255.0,
                        PixelFormat::Rgba8 => {
                            (0.2126 * b[at] as f32
                                + 0.7152 * b[at + 1] as f32
                                + 0.0722 * b[at + 2] as f32)
                                / 255.0
                        }
                        PixelFormat::Rgba16F => {
                            let c = |k: usize| half([b[at + 2 * k], b[at + 2 * k + 1]]);
                            0.2126 * c(0) + 0.7152 * c(1) + 0.0722 * c(2)
                        }
                    }
                })
                .unwrap_or(0.0);
            values.push(value.clamp(0.0, 1.0));
        }
    }
    Alpha {
        width: aw,
        height: ah,
        values,
    }
}

/// The alpha a stroke names, made once and kept.
fn alpha_for(doc: &Document, name: &str) -> Result<Option<Arc<Alpha>>, OpError> {
    let name = name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    let image = doc.images().find(|(id, img)| {
        img.name == name
            || doc
                .image_uid(*id)
                .is_some_and(|u| u.raw().to_string() == name)
    });
    let revision = image.map_or(0, |(_, img)| img.revision());
    let mut held = ALPHAS.lock().unwrap_or_else(|e| e.into_inner());
    let map = held.get_or_insert_with(HashMap::new);
    if let Some(a) = map.get(&(name.to_string(), revision)) {
        return Ok(Some(a.clone()));
    }
    let alpha = match image {
        Some((_, img)) => image_alpha(img),
        None => builtin_alpha(name).ok_or_else(|| {
            OpError::Failed(format!(
                "no alpha called \"{name}\": use noise, stripes, dots, or an image's name"
            ))
        })?,
    };
    let alpha = Arc::new(alpha);
    // One picture per name: an edited image replaces the old one.
    map.retain(|(n, _), _| n != name);
    map.insert((name.to_string(), revision), alpha.clone());
    Ok(Some(alpha))
}

/// The stroke the parameters describe, its dabs, and the alpha it names.
pub fn stroke_of(doc: &Document, params: &StrokeParams) -> Result<(Stroke, Vec<Dab>), OpError> {
    let dabs: Vec<Dab> = params
        .dabs
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(k, c)| Dab {
            centre: Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32),
            radius: c[3] as f32,
            pressure: params
                .pressures
                .get(k)
                .copied()
                .unwrap_or(1.0)
                .clamp(0.0, 1.0) as f32,
        })
        .filter(|d| d.centre.is_finite() && d.radius.is_finite() && d.radius > 0.0)
        .collect();
    let stroke = Stroke {
        brush: Brush::from_index(params.brush),
        strength: params.strength as f32,
        invert: params.invert,
        grab: kreate_math::DVec3::from_array(params.grab).as_vec3(),
        mirror: [params.mirror_x, params.mirror_y, params.mirror_z],
        radial: params.radial.clamp(1, 64),
        falloff: Falloff::from_index(params.falloff),
        view: kreate_math::DVec3::from_array(params.view)
            .as_vec3()
            .normalize_or_zero(),
        front_only: params.front_only,
        auto_smooth: params.auto_smooth.clamp(0.0, 1.0) as f32,
        alpha: alpha_for(doc, &params.alpha)?,
        face_set: params.face_set,
    };
    Ok((stroke, dabs))
}

/// One stroke of a sculpt brush on the active mesh.
pub struct SculptStroke;

impl Operator for SculptStroke {
    type Params = StrokeParams;
    fn id() -> &'static str {
        "sculpt.stroke"
    }
    fn label() -> &'static str {
        "Sculpt"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let (stroke, dabs) = stroke_of(doc.document(), params)?;
        if dabs.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let Some(m) = doc.document_mut().mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        // Dynamic topology re-cuts the surface; a grab, the mask and face
        // sets need the surface to stay as it is, as in Blender.
        if params.dyntopo && !matches!(stroke.brush, Brush::Grab | Brush::Mask | Brush::FaceSet) {
            let detail = (params.detail.clamp(0.01, 1.0) as f32) * dabs[0].radius;
            let mut held = DYN.lock().unwrap_or_else(|e| e.into_inner());
            let cache = match held.take() {
                Some(cache) if cache.fits(m, &stroke, detail, &dabs) => cache,
                _ => kreate_mesh::dyntopo::DynCache::new(m, &stroke, detail),
            };
            let cache = held.insert(cache);
            return Ok(if cache.run(m, &dabs, &stroke, detail) {
                OpResult::Finished
            } else {
                OpResult::Cancelled
            });
        }
        let mut held = STROKE.lock().unwrap_or_else(|e| e.into_inner());
        let cache = match held.take() {
            Some(cache) if cache.fits(m, &stroke, &dabs) => cache,
            _ => StrokeCache::new(m, &stroke, dabs[0].radius),
        };
        let cache = held.insert(cache);
        Ok(if cache.run(m, &dabs, &stroke) {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MaskParams {
    /// Clear it, turn it inside out, or mask everything.
    #[param(variants("clear", "invert", "fill"))]
    pub action: u32,
}

/// Blender's Alt+M and Ctrl+I in sculpt: the mask cleared, inverted or
/// filled.
pub struct SculptMask;

impl Operator for SculptMask {
    type Params = MaskParams;
    fn id() -> &'static str {
        "sculpt.mask"
    }
    fn label() -> &'static str {
        "Mask"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let action = match params.action {
            1 => MaskAction::Invert,
            2 => MaskAction::Fill,
            _ => MaskAction::Clear,
        };
        let mut m = doc.document().mesh(mesh).cloned().unwrap_or_default();
        if !sculpt::edit_mask(&mut m, action) {
            return Ok(OpResult::Cancelled);
        }
        if let Some(target) = doc.document_mut().mesh_mut(mesh) {
            *target = m;
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct FaceSetFromMaskParams {}

/// The faces whose corners are all masked become a new face set.
pub struct FaceSetFromMask;

impl Operator for FaceSetFromMask {
    type Params = FaceSetFromMaskParams;
    fn id() -> &'static str {
        "sculpt.face_set_from_mask"
    }
    fn label() -> &'static str {
        "Face Set from Mask"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(m) = doc.document().mesh(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        let mask = sculpt::mask_of(m);
        let mut sets = sculpt::face_sets_of(m);
        let next = sets.iter().copied().max().unwrap_or(0) + 1;
        let mut any = false;
        for (f, set) in sets.iter_mut().enumerate() {
            if m.face_verts(f as u32)
                .iter()
                .all(|v| mask[*v as usize] > 0.5)
            {
                *set = next;
                any = true;
            }
        }
        if !any {
            return Err(OpError::Failed("nothing is masked".into()));
        }
        if let Some(m) = doc.document_mut().mesh_mut(mesh) {
            m.attributes_mut().insert(
                sculpt::FACE_SET,
                kreate_mesh::Domain::Face,
                kreate_mesh::AttrData::I32(Arc::new(sets)),
            );
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct FaceSetVisibilityParams {
    /// Hide the face set, show only it, or show everything.
    #[param(variants("hide", "isolate", "show_all"))]
    pub action: u32,
    /// A face of the face set meant: the one under the pointer.
    pub face: u32,
}

/// Blender's H, Shift+H and Alt+H in sculpt: a face set hidden, shown alone,
/// or everything shown again. Hidden is the edit-mode hiding, so a brush
/// leaves it alone and edit mode sees it hidden too.
pub struct FaceSetVisibility;

impl Operator for FaceSetVisibility {
    type Params = FaceSetVisibilityParams;
    fn id() -> &'static str {
        "sculpt.face_set_visibility"
    }
    fn label() -> &'static str {
        "Face Set Visibility"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::selection;
        use kreate_mesh::SelectMode;
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(base) = doc.document().mesh(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        let mut m = base.clone();
        let changed = if params.action == 2 {
            let shown = selection::reveal(&mut m);
            // What came back is not left selected: sculpt has no selection.
            selection::select_none(&mut m);
            shown
        } else {
            let sets = sculpt::face_sets_of(&m);
            let Some(&set) = sets.get(params.face as usize) else {
                return Err(OpError::Failed("no face there".into()));
            };
            // The face set is chosen as a face selection, then hidden through
            // the edit-mode rules, which keep what is left a whole mesh.
            selection::ensure_layers(&mut m);
            selection::select_none(&mut m);
            for (f, s) in sets.iter().enumerate() {
                if *s == set {
                    selection::set_selected(&mut m, kreate_mesh::Domain::Face, f as u32, true);
                }
            }
            let corner_edges = m.corner_edges();
            let hid = selection::hide(&mut m, SelectMode::Face, params.action == 1, &corner_edges);
            selection::select_none(&mut m);
            hid
        };
        if !changed {
            return Ok(OpResult::Cancelled);
        }
        if let Some(target) = doc.document_mut().mesh_mut(mesh) {
            *target = m;
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct VoxelRemeshParams {
    /// The size of a voxel, in the mesh's own units: about the length of an
    /// edge afterwards.
    #[param(min = 0.0001, soft_max = 1.0, unit = "length")]
    pub voxel_size: f64,
}

impl Default for VoxelRemeshParams {
    fn default() -> Self {
        Self { voxel_size: 0.05 }
    }
}

/// Blender's Ctrl+R in sculpt: the active mesh rebuilt as even squares.
pub struct VoxelRemesh;

impl Operator for VoxelRemesh {
    type Params = VoxelRemeshParams;
    fn id() -> &'static str {
        "sculpt.voxel_remesh"
    }
    fn label() -> &'static str {
        "Voxel Remesh"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(m) = doc.document().mesh(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        let out = kreate_mesh::remesh::voxel_remesh(m, params.voxel_size as f32)
            .map_err(|e| OpError::Failed(e.to_string()))?;
        if let Some(target) = doc.document_mut().mesh_mut(mesh) {
            *target = out;
        }
        Ok(OpResult::Finished)
    }
}
// --- shaping by number ---------------------------------------------------------

/// How much of a change a vertex `t` of the way out from the middle takes.
fn fade(t: f64, curve: u32) -> f64 {
    let t = t.clamp(0.0, 1.0);
    match curve {
        1 => 1.0 - t,
        2 => 1.0,
        3 => (1.0 - t).powi(3),
        _ => {
            let s = 1.0 - t;
            s * s * (3.0 - 2.0 * s)
        }
    }
}

/// Each vertex's normal, from the faces round it, weighted by their area.
fn vertex_normals(mesh: &kreate_mesh::MeshData) -> Vec<Vec3> {
    let p = mesh.positions();
    let mut out = vec![Vec3::ZERO; p.len()];
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        for i in 1..ring.len().saturating_sub(1) {
            let (a, b, c) = (ring[0] as usize, ring[i] as usize, ring[i + 1] as usize);
            let n = (p[b] - p[a]).cross(p[c] - p[a]);
            for v in [a, b, c] {
                out[v] += n;
            }
        }
    }
    out.iter().map(|n| n.normalize_or_zero()).collect()
}

/// Each vertex's neighbours along the edges.
fn neighbours(mesh: &kreate_mesh::MeshData) -> Vec<Vec<usize>> {
    let n = mesh.num_verts() as usize;
    let mut out = vec![Vec::new(); n];
    for e in mesh.edges() {
        let (a, b) = (e[0] as usize, e[1] as usize);
        if a < n && b < n && a != b {
            out[a].push(b);
            out[b].push(a);
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DeformRegionParams {
    #[param(variants("push", "inflate", "move", "scale", "flatten", "smooth"))]
    pub action: u32,
    /// The region's middle, in the mesh's own coordinates.
    #[param(unit = "length")]
    pub center: [f64; 3],
    #[param(unit = "length", min = 0.0)]
    pub radius: f64,
    /// The region stretched along each axis: (1, 1, 1) is a ball, (1, 1, 3)
    /// a tall egg — a limb, a jaw line.
    pub stretch: [f64; 3],
    /// Push and inflate: metres along the normal, out positive. Scale: the
    /// factor. Flatten and smooth: how much, 0 to 1.
    pub amount: f64,
    /// Move: how far, in the mesh's coordinates.
    #[param(unit = "length")]
    pub offset: [f64; 3],
    #[param(variants("smooth", "linear", "constant", "sharp"))]
    pub falloff: u32,
    /// The same on the other side of X.
    pub mirror_x: bool,
}

impl Default for DeformRegionParams {
    fn default() -> Self {
        Self {
            action: 0,
            center: [0.0; 3],
            radius: 0.1,
            stretch: [1.0; 3],
            amount: 0.01,
            offset: [0.0; 3],
            falloff: 0,
            mirror_x: false,
        }
    }
}

/// Shapes a region of the active mesh by number: pushes it out or in by so
/// many millimetres, swells it, moves it, scales it, flattens or smooths it,
/// fading to nothing at the region's rim. The sculpt mask holds what it
/// covers. What an assistant sculpts with, where a brush is a gesture.
pub struct DeformRegion;

impl Operator for DeformRegion {
    type Params = DeformRegionParams;
    fn id() -> &'static str {
        "sculpt.deform_region"
    }
    fn label() -> &'static str {
        "Deform Region"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        if params.radius <= 0.0 || !params.radius.is_finite() {
            return Err(OpError::Failed("the region needs a radius".into()));
        }
        let Some(m) = doc.document().mesh(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        let stretch = Vec3::new(
            params.stretch[0].max(1e-6) as f32,
            params.stretch[1].max(1e-6) as f32,
            params.stretch[2].max(1e-6) as f32,
        );
        let mask = sculpt::mask_of(m);
        let normals = vertex_normals(m);
        let near = neighbours(m);
        let before = m.positions().to_vec();
        let mut after = before.clone();
        let centers: Vec<Vec3> = {
            let c = Vec3::new(
                params.center[0] as f32,
                params.center[1] as f32,
                params.center[2] as f32,
            );
            if params.mirror_x && c.x.abs() > 1e-6 {
                vec![c, Vec3::new(-c.x, c.y, c.z)]
            } else {
                vec![c]
            }
        };
        let radius = params.radius as f32;
        let mut touched = 0;
        for (side, center) in centers.iter().enumerate() {
            let mirrored = side == 1;
            // How much of the change each vertex takes.
            let share: Vec<f32> = before
                .iter()
                .zip(&mask)
                .map(|(p, held)| {
                    let t = ((*p - *center) / stretch).length() / radius;
                    if t >= 1.0 {
                        0.0
                    } else {
                        fade(t as f64, params.falloff) as f32 * (1.0 - held.clamp(0.0, 1.0))
                    }
                })
                .collect();
            let inside: Vec<usize> = (0..before.len()).filter(|v| share[*v] > 0.0).collect();
            if inside.is_empty() {
                continue;
            }
            touched += inside.len();
            let amount = params.amount as f32;
            match params.action {
                // Out along the region's own facing, as a lump rises.
                0 => {
                    let facing = inside
                        .iter()
                        .map(|v| normals[*v] * share[*v])
                        .sum::<Vec3>()
                        .normalize_or_zero();
                    for &v in &inside {
                        after[v] += facing * amount * share[v];
                    }
                }
                1 => {
                    for &v in &inside {
                        after[v] += normals[v] * amount * share[v];
                    }
                }
                2 => {
                    let mut o = Vec3::new(
                        params.offset[0] as f32,
                        params.offset[1] as f32,
                        params.offset[2] as f32,
                    );
                    if mirrored {
                        o.x = -o.x;
                    }
                    for &v in &inside {
                        after[v] += o * share[v];
                    }
                }
                3 => {
                    for &v in &inside {
                        let scaled = *center + (before[v] - *center) * amount;
                        after[v] += (scaled - before[v]) * share[v];
                    }
                }
                // Onto the plane through the region's weighted middle, square
                // to its facing.
                4 => {
                    let total: f32 = inside.iter().map(|v| share[*v]).sum();
                    let middle = inside.iter().map(|v| before[*v] * share[*v]).sum::<Vec3>()
                        / total.max(1e-9);
                    let facing = inside
                        .iter()
                        .map(|v| normals[*v] * share[*v])
                        .sum::<Vec3>()
                        .normalize_or_zero();
                    for &v in &inside {
                        let lift = (before[v] - middle).dot(facing);
                        after[v] -= facing * lift * amount.clamp(0.0, 1.0) * share[v];
                    }
                }
                _ => {
                    let strength = amount.clamp(0.0, 1.0);
                    for _ in 0..4 {
                        let now = after.clone();
                        for &v in &inside {
                            if near[v].is_empty() {
                                continue;
                            }
                            let mean = near[v].iter().map(|u| now[*u]).sum::<Vec3>()
                                / near[v].len() as f32;
                            after[v] += (mean - now[v]) * strength * share[v];
                        }
                    }
                }
            }
        }
        if touched == 0 {
            return Err(OpError::Failed("no vertex is inside the region".into()));
        }
        let Some(m) = doc.document_mut().mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        m.positions_mut().copy_from_slice(&after);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct CreasePathParams {
    /// The path, three numbers a point, in the mesh's own coordinates.
    pub points: Vec<f64>,
    /// How far either side of the path the crease reaches.
    #[param(unit = "length", min = 0.0)]
    pub radius: f64,
    /// Metres into the surface along the path; negative raises a ridge.
    #[param(unit = "length")]
    pub depth: f64,
    /// How sharp the bottom of the groove is, 0 (round) to 1 (a cut).
    #[param(min = 0.0, max = 1.0)]
    pub sharpness: f64,
    pub mirror_x: bool,
}

impl Default for CreasePathParams {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            radius: 0.01,
            depth: 0.003,
            sharpness: 0.5,
            mirror_x: false,
        }
    }
}

/// A groove or a ridge along a path of points — a lip line, a brow, the
/// edge of a muscle — so many millimetres deep, fading out to `radius`.
pub struct CreasePath;

impl Operator for CreasePath {
    type Params = CreasePathParams;
    fn id() -> &'static str {
        "sculpt.crease_path"
    }
    fn label() -> &'static str {
        "Crease Along a Path"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        has_mesh(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(mesh) = active_mesh(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let path: Vec<Vec3> = params
            .points
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect();
        if path.is_empty() || params.radius <= 0.0 {
            return Err(OpError::Failed(
                "a path of points and a radius are needed".into(),
            ));
        }
        let Some(m) = doc.document().mesh(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        let mask = sculpt::mask_of(m);
        let normals = vertex_normals(m);
        let before = m.positions().to_vec();
        let mut after = before.clone();
        let radius = params.radius as f32;
        // Sharper means the depth stays near the path and falls off fast.
        let power = 1.0 + params.sharpness.clamp(0.0, 1.0) as f32 * 4.0;
        let mut paths = vec![path.clone()];
        if params.mirror_x {
            paths.push(path.iter().map(|p| Vec3::new(-p.x, p.y, p.z)).collect());
        }
        let mut touched = 0;
        for path in &paths {
            for (v, p) in before.iter().enumerate() {
                let d = if path.len() == 1 {
                    p.distance(path[0])
                } else {
                    path.windows(2)
                        .map(|w| {
                            let (a, b) = (w[0], w[1]);
                            let ab = b - a;
                            let t = if ab.length_squared() > 0.0 {
                                ((*p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            p.distance(a + ab * t)
                        })
                        .fold(f32::MAX, f32::min)
                };
                if d >= radius {
                    continue;
                }
                let share = (1.0 - d / radius).powf(power) * (1.0 - mask[v].clamp(0.0, 1.0));
                after[v] -= normals[v] * params.depth as f32 * share;
                touched += 1;
            }
        }
        if touched == 0 {
            return Err(OpError::Failed(
                "the path does not pass near the surface".into(),
            ));
        }
        let Some(m) = doc.document_mut().mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
        m.positions_mut().copy_from_slice(&after);
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ball() -> (kreate_scene::Document, kreate_scene::MeshId) {
        let mut d = kreate_scene::Document::new();
        let id = d.add_mesh_object("Ball", kreate_mesh::primitives::uv_sphere(1.0, 48, 24));
        d.selection.add(id);
        d.selection.active = Some(id);
        let mesh = d.object(id).and_then(|o| o.mesh()).expect("mesh");
        (d, mesh)
    }

    fn nearest(d: &kreate_scene::Document, mesh: kreate_scene::MeshId, to: Vec3) -> Vec3 {
        let p = d.mesh(mesh).expect("mesh").positions();
        *p.iter()
            .min_by(|a, b| a.distance(to).total_cmp(&b.distance(to)))
            .expect("one")
    }

    #[test]
    fn a_region_is_pushed_out_by_so_much_and_its_mirror_with_it() {
        let (mut d, mesh) = ball();
        let mut edit = DocMut::new(&mut d);
        DeformRegion::execute(
            &DeformRegionParams {
                center: [1.0, 0.0, 0.0],
                radius: 0.4,
                amount: 0.1,
                mirror_x: true,
                ..Default::default()
            },
            &mut edit,
            &Context::default(),
        )
        .expect("ran");
        let out = nearest(&d, mesh, Vec3::new(1.2, 0.0, 0.0));
        assert!((out.x - 1.1).abs() < 0.01, "pushed out a tenth: {out}");
        let mirrored = nearest(&d, mesh, Vec3::new(-1.2, 0.0, 0.0));
        assert!(
            (mirrored.x + 1.1).abs() < 0.01,
            "and its mirror: {mirrored}"
        );
        let top = nearest(&d, mesh, Vec3::new(0.0, 0.0, 1.2));
        assert!(
            (top.z - 1.0).abs() < 1e-5,
            "outside the region, untouched: {top}"
        );
    }

    #[test]
    fn a_crease_carves_along_its_path_and_the_mask_holds() {
        let (mut d, mesh) = ball();
        // The +Y half masked.
        {
            let m = d.mesh_mut(mesh).expect("mesh");
            let held: Vec<f32> = m
                .positions()
                .iter()
                .map(|p| if p.y > 0.0 { 1.0 } else { 0.0 })
                .collect();
            m.attributes_mut().insert(
                kreate_mesh::sculpt::MASK,
                kreate_mesh::attribute::Domain::Point,
                kreate_mesh::attribute::AttrData::F32(Arc::new(held)),
            );
        }
        // Along the surface, up and down through the vertex nearest -Y.
        let start = nearest(&d, mesh, Vec3::new(0.0, -1.2, 0.0));
        let points: Vec<f64> = [-0.4f32, 0.0, 0.4]
            .iter()
            .flat_map(|t| {
                let p = (start + Vec3::new(0.0, 0.0, *t)).normalize();
                [p.x as f64, p.y as f64, p.z as f64]
            })
            .collect();
        let mut edit = DocMut::new(&mut d);
        CreasePath::execute(
            &CreasePathParams {
                points,
                radius: 0.15,
                depth: 0.05,
                sharpness: 1.0,
                mirror_x: false,
            },
            &mut edit,
            &Context::default(),
        )
        .expect("ran");
        let carved = d.mesh(mesh).expect("mesh").positions()[d
            .mesh(mesh)
            .expect("mesh")
            .positions()
            .iter()
            .position(|p| p.distance(start) < 0.06)
            .expect("near")];
        assert!(
            carved.length() < 0.97,
            "carved in on the open side: {carved}"
        );
        let held = nearest(&d, mesh, Vec3::new(0.0, 1.2, 0.0));
        assert!((held.y - 1.0).abs() < 1e-5, "held by the mask: {held}");
    }
}
