//! The sculpt stroke: one operator per stroke of the brush.
//!
//! The dabs ride in the parameters — where the brush touched, in the mesh's
//! own space, and how wide — so a stroke replays without the pointer and the
//! whole of it is one step to undo. See `kreate_mesh::sculpt`.

use kreate_math::Vec3;
use std::sync::Mutex;

use kreate_mesh::sculpt::{Brush, Dab, Stroke, StrokeCache};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

/// The stroke being drawn. While the pointer moves the stroke is re-run on
/// its starting mesh with one more dab each time; the cache lets that run do
/// only the new dab. It checks it was given the mesh it began on, so a stale
/// one is never used — at worst it is rebuilt.
static STROKE: Mutex<Option<StrokeCache>> = Mutex::new(None);

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<SculptStroke>();
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct StrokeParams {
    #[param(variants("draw", "smooth", "grab", "inflate"))]
    pub brush: u32,
    /// Where the brush touched: x, y, z and radius per dab, in the mesh's space.
    pub dabs: Vec<f64>,
    #[param(min = 0.0, max = 1.0)]
    pub strength: f64,
    /// Carve instead of raise; shrink instead of swell.
    pub invert: bool,
    /// For a grab, how far it was dragged, in the mesh's space.
    pub grab: [f64; 3],
    /// Mirrored across X.
    pub mirror_x: bool,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self {
            brush: 0,
            dabs: Vec::new(),
            strength: 0.5,
            invert: false,
            grab: [0.0; 3],
            mirror_x: false,
        }
    }
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
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .and_then(|a| doc.object(a))
            .and_then(|o| o.mesh())
            .is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(mesh) = d
            .selection
            .active
            .and_then(|a| d.object(a))
            .and_then(|o| o.mesh())
        else {
            return Ok(OpResult::Cancelled);
        };
        let dabs: Vec<Dab> = params
            .dabs
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| Dab {
                centre: Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32),
                radius: c[3] as f32,
            })
            .filter(|d| d.centre.is_finite() && d.radius.is_finite() && d.radius > 0.0)
            .collect();
        if dabs.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let stroke = Stroke {
            brush: match params.brush {
                1 => Brush::Smooth,
                2 => Brush::Grab,
                3 => Brush::Inflate,
                _ => Brush::Draw,
            },
            strength: params.strength as f32,
            invert: params.invert,
            grab: kreate_math::DVec3::from_array(params.grab).as_vec3(),
            mirror_x: params.mirror_x,
        };
        let Some(m) = doc.document_mut().mesh_mut(mesh) else {
            return Ok(OpResult::Cancelled);
        };
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
